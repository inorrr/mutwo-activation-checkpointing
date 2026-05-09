from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Union
from utils import SPMD_DECOMP_TABLE

import torch

# We need to import _functional_collectives to trigger op registration
import torch.distributed._functional_collectives
import torch.nn as nn
import torch.optim as optim
import torch.utils._pytree as pytree
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph import CodeGen, _PyTreeCodeGen, _PyTreeInfo
from torch.nn.utils import stateless
from torch.utils.hooks import RemovableHandle


# Identity op used as a forward marker. Registering it as a torch library op
# makes it appear as a recognizable node in the traced FX graph.
def sep(x: torch.Tensor) -> torch.Tensor:
    return x


# Identity op used as the matching backward marker.
def sep_backward(grad: torch.Tensor) -> torch.Tensor:
    return grad


# Define the custom separator ops so make_fx can record them by target.
separator_lib = torch.library.Library("separator", "DEF")
separator_lib.define("sep(Tensor x) -> Tensor")
separator_lib.impl("sep", sep, "CompositeExplicitAutograd")
separator_lib.define("sep_backward(Tensor x) -> Tensor")
separator_lib.impl("sep_backward", sep_backward, "CompositeExplicitAutograd")


# DTensor sharding rules say the separator ops preserve the input layout.
# This keeps the marker compatible with distributed/SPMD tracing contexts.
def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
    (x,) = op_schema.args_schema
    assert isinstance(x, DTensorSpec), f"expecting DTensorSpec but got {x}"

    return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))

def _prop_sepm(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)

def _prop_sepm_backward(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)

DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(torch.ops.separator.sep.default, _prop_sepm)
DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(torch.ops.separator.sep_backward.default, _prop_sepm_backward)



# Autograd wrapper that inserts sep in forward and sep_backward during backward.
class SEPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.separator.sep(x)

    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:
        return torch.ops.separator.sep_backward(grad_x)


# Dummy op used by data parallel to tag gradients.
_spmd_lib_def = torch.library.Library("dummy", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("dummy", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")


# Custom codegen keeps the GraphModule callable with already-flattened inputs.
# The wrapper outside the graph handles pytree flattening, so generated FX code
# should not try to flatten the original structured arguments again.
class _PyTreeCodeGenOutputsOnly(_PyTreeCodeGen):
    # pyre-ignore[3]
    def process_inputs(self, *args: Any) -> Any:
        return args

    # pyre-ignore[2, 3]
    def gen_fn_def(self, free_vars, maybe_return_annotation):
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: fx.GraphModule) -> fx.GraphModule:
    """Move the responsibility of flattening the input arguments from the
    graph module to the caller.

    Example:

        output = gm(my_struct)

        gm = gm(to_caller_flattened_graph_module)

        output = gm(*pytree.flatten(my_struct)[0])
    """
    # pyre-ignore[16]
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            # pyre-ignore[6]
            orig_args=None,  # type: ignore[arg-type]
            # pyre-ignore[6]
            in_spec=None,  # type: ignore[arg-type]
            # pyre-ignore[16]
            out_spec=gm._graph._codegen.pytree_info.out_spec,
        )
    )
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


@contextmanager
def gradients_tagging(params: Dict[str, nn.Parameter]):
    """
    This is a helper function that tags the gradient of the parameters
    with a special tag, so that we can identify them during SPMD expansion.

    It's safe to trace those hooks and we would remove those nodes later.
    """

    tagging_hooks: List[RemovableHandle] = []
    try:
        # Each hook inserts a dummy identity op on gradients. The tracer can see
        # those nodes, then _compile removes them after they served as tags.
        for p in params.values():
            h = p.register_hook(lambda grad: torch.ops.dummy.tag_grad(grad))
            tagging_hooks.append(h)
        yield
    finally:
        # remove those hooks after tracing
        for h in tagging_hooks:
            h.remove()


@contextmanager
def _rematerialize_optimizer(
    opt: optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, nn.Parameter],
    param_name_by_id: Dict[int, str],
):
    assert opt is not None

    # During tracing, optimizer.state must be addressable by the proxy/fake
    # parameters used in the stateless module execution.
    # update opt.state with proxy tensors
    orig_states = copy(opt.state)
    for n in named_states:
        # opt.state's key type is string, but optimizer uses Parameter as keys
        opt.state[params[n]] = named_states[n]  # type: ignore[index]

    # Optimizer param groups normally hold the original Parameter objects. Swap
    # them temporarily so optimizer.step() operates on the traced parameters.
    orig_param_groups = [
        (param_group, list(param_group["params"])) for param_group in opt.param_groups
    ]
    for param_group, orig_params in orig_param_groups:
        rematerialized_params = []
        for param in orig_params:
            param_name = param_name_by_id.get(id(param))
            if param_name is None:
                raise RuntimeError("Optimizer parameter is not owned by the traced module.")
            rematerialized_params.append(params[param_name])
        param_group["params"] = rematerialized_params

    try:
        yield
    finally:
        for param_group, orig_params in orig_param_groups:
            param_group["params"] = orig_params
        opt.state = orig_states


@contextmanager
def _enable_compile():
    # The return value of torch._utils.is_compiling changes optimizer behavior.
    # We need that function to return True to include optimizer in the graph.
    # See: https://github.com/pytorch/pytorch/blob/a524123c91ab399c9dd6882c1189596dd77e7734/torch/optim/optimizer.py#L41
    def f_true():
        return True

    orig_is_compiling_code = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code


# Bundle the traced graph with the live objects needed to execute it later.
@dataclass
class _CompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]


# Metadata attached to graph placeholders so the profiler can categorize memory
# as parameter, buffer, optimizer state, gradient, or activation-related.
@dataclass
class PlaceholderMetadata:
    name: str
    role: str
    shape: Optional[torch.Size]
    dtype: Optional[torch.dtype]
    requires_grad: bool = False


def _compile(func: Callable, *args: Any, **kwargs: Any):
    # Find the module and optimizer from the user's training-step arguments.
    # This prototype supports one model and one optimizer per compiled function.
    # 1. Extract nn.Module and Optimizer from args and kwargs
    mod, opt = None, None
    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, nn.Module):
            assert mod is None, "Only support single nn.Module for now"
            mod = arg
        if isinstance(arg, optim.Optimizer):
            assert opt is None, "Only support single Optimizer for now"
            opt = arg
    assert mod is not None, "Couldn't find nn.Module instances from the arguments."

    # Lift parameters and buffers out of the module so make_fx sees them as
    # explicit graph inputs instead of hidden Python object state.
    # 2. Trace the stateless version of the train_step
    params = dict(mod.named_parameters(remove_duplicate=False))
    param_name_by_id: Dict[int, str] = {}
    for name, param in params.items():
        param_name_by_id.setdefault(id(param), name)
    buffers = dict(mod.named_buffers(remove_duplicate=False))

    named_states: Dict[str, nn.Parameter] = {}
    # Adam state exists only after the optimizer has stepped at least once. The
    # starter/benchmark code initializes it before calling compile.
    # Pass named_states instead of opt.state to stateless_func, because
    # the later uses nn.Parameter as key. During tracing, we need to
    # make sure optimizers can find the states using proxy tensors.
    for n, p in params.items():
        if p in opt.state:
            # opt.state's key type is string, but optimizer uses
            # Parameter as keys
            named_states[n] = opt.state[p]

    # Lift states and parameters as function arguments so that make_fx
    # can trace operations applied to them

    def stateless_func(
        func: Callable,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, nn.Parameter],
        args: Any,
        kwargs: Any,
    ):
        # Rebind the module and optimizer to the lifted tensors, run the user's
        # training step, and return both normal output and mutated state values.
        with stateless._reparametrize_module(
            mod, {**params, **buffers}
        ), _rematerialize_optimizer(
            opt, named_states, params, param_name_by_id
        ) if opt else nullcontext():
            # Installing hooks onto gradients to identify the gradients.
            with gradients_tagging(params):
                ret = func(*args, **kwargs)

            # the return value of the function must be the original return value
            # updated paramaters and updated optimizer states
            return ret, list(mod.parameters()), list(named_states.values())

    # Fake tensors let make_fx trace shapes/dtypes/devices without forcing every
    # operation to allocate real outputs during the capture phase.
    tracing_mode = "fake"
    fake_mode = FakeTensorMode()

    def _get_fake_args(arg: torch.Tensor) -> torch.Tensor:
        fake_arg = fake_mode.from_tensor(arg)
        return fake_arg

    args = pytree.tree_map_only(torch.Tensor, _get_fake_args, args)
    kwargs = pytree.tree_map_only(torch.Tensor, _get_fake_args, kwargs)

    # make_fx records the full training iteration, using the decomposition table
    # to make optimizer mutations traceable.
    with _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        gm = make_fx(
            partial(stateless_func, func),
            tracing_mode=tracing_mode,
            decomposition_table=SPMD_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(params, buffers, named_states, args, kwargs)

    params_and_buffers: Dict[str, Union[torch.Tensor, nn.Parameter]] = {
        **params,
        **buffers,
    }

    # The wrapper executes the GraphModule with a flat list:
    # [parameters, buffers, optimizer_state, original_args, original_kwargs].
    flat_state, _ = pytree.tree_flatten([params_and_buffers, named_states])

    placeholder_metadata: List[PlaceholderMetadata] = []
    placeholder_metadata.extend(
        PlaceholderMetadata(
            name=name,
            role="parameter" if isinstance(tensor, nn.Parameter) else "buffer",
            shape=getattr(tensor, "shape", None),
            dtype=getattr(tensor, "dtype", None),
            requires_grad=bool(getattr(tensor, "requires_grad", False)),
        )
        for name, tensor in params_and_buffers.items()
    )
    placeholder_metadata.extend(
        PlaceholderMetadata(
            name=name,
            role="optimizer_state",
            shape=getattr(state, "shape", None),
            dtype=getattr(state, "dtype", None),
            requires_grad=bool(getattr(state, "requires_grad", False)),
        )
        for name, state in named_states.items()
    )

    # Remove helper-only graph artifacts. detach nodes do not affect this
    # analysis, and dummy tag_grad nodes were only needed during tracing.
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
        if node.target == torch.ops.dummy.tag_grad.default:
            grad_node = node.all_input_nodes[0]
            node.replace_all_uses_with(grad_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)

    # Store enough side metadata on the graph for the profiler and execution
    # wrapper to understand the flattened placeholder layout.
    gm = _to_caller_flattened_graph_module(gm)
    gm._ac_placeholder_metadata = placeholder_metadata
    gm._ac_num_flat_state = len(flat_state)
    gm._ac_param_buffer_count = len(params_and_buffers)
    gm._ac_optimizer_state_count = len(named_states)

    return _CompiledResult(gm, mod, opt, flat_state)


# Note that the Python convention of __dict__ requires the key to be str.
# TODO: ensure the key is unique.
COMPILED_OBJECT_KEY = "_compiled_obj"


def compile(func: Callable, gm_transformation: Callable):
    # Public decorator-like wrapper. It traces once, optionally transforms the
    # graph once, then reuses the resulting GraphModule for later calls.
    @wraps(func)
    def wrapper(*args, **kwargs):
        first_iter = False
        compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
        if compiled_obj is None:
            first_iter = True
            compiled_obj = _compile(func, *args, **kwargs)
            wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
        # Runtime inputs are the captured state tensors plus the current call's
        # original arguments flattened in the same pytree order used at trace time.
        flat_inps = compiled_obj.flat_state + pytree.tree_flatten([args, kwargs])[0]
        if first_iter and gm_transformation:
            compiled_obj.gm = gm_transformation(compiled_obj.gm, flat_inps)
        with torch.no_grad():
            # Some hand-rewritten graphs are easier to execute through the FX interpreter.
            if getattr(compiled_obj.gm, "_ac_run_with_interpreter", False):
                interpreter = fx.Interpreter(compiled_obj.gm, garbage_collect_values=True)
                output = interpreter.run(*flat_inps)[0]
            else:
                output = compiled_obj.gm(*flat_inps)[0]

        return output

    return wrapper
