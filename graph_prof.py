from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.fx as fx


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(str, Enum):
    PARAM = "parameter"
    GRAD = "gradient"
    ACT = "activation"
    OPT_STATE = "optimizer_state"
    BUFFER = "buffer"
    OTHER = "other"


@dataclass
class NodeRuntimeStat:
    name: str
    op: str
    target: str
    category: str
    phase: str
    elapsed_ms_total: float = 0.0
    elapsed_ms_avg: float = 0.0
    memory_before_bytes_total: int = 0
    memory_after_bytes_total: int = 0
    memory_peak_bytes_total: int = 0
    output_bytes_total: int = 0
    samples: int = 0


@dataclass
class ActivationInfo:
    name: str
    create_index: int
    last_forward_use_index: int
    first_backward_use_index: int
    first_backward_user_name: str
    size_bytes: int
    shape: Optional[List[int]]
    dtype: Optional[str]
    source_target: str
    required_input_names: List[str] = field(default_factory=list)
    recompute_cost_ms: float = 0.0
    retained: bool = True


@dataclass
class ProfilerSummary:
    node_stats: List[NodeRuntimeStat]
    activations: List[ActivationInfo]
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    activation_peak_bytes: int
    total_peak_bytes: int
    peak_breakdown_bytes: Dict[str, int]
    timeline_breakdown: List[Dict[str, int]]
    boundary: Dict[str, int]
    metadata: Dict[str, Any]


def _target_name(target: Any) -> str:
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        return f"{target.__module__}.{target.__name__}"
    return str(target)


def _flatten_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tensors(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten_tensors(item)


def _tensor_bytes_from_runtime(value: Any) -> int:
    return sum(t.numel() * t.element_size() for t in _flatten_tensors(value))


def _tensor_bytes_from_meta(node: fx.Node) -> int:
    val = node.meta.get("val")
    return sum(t.numel() * t.element_size() for t in _flatten_tensors(val))


def _tensor_shape_from_meta(node: fx.Node) -> Optional[List[int]]:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return list(val.shape)
    return None


def _tensor_dtype_from_meta(node: fx.Node) -> Optional[str]:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return str(val.dtype)
    return None


def _sum_placeholder_bytes(metadata: List[Any], role: str) -> int:
    total = 0
    for item in metadata:
        if getattr(item, "role", None) != role or getattr(item, "shape", None) is None:
            continue
        dtype = getattr(item, "dtype", None)
        if dtype is None:
            continue
        numel = 1
        for dim in torch.Size(item.shape):
            numel *= dim
        total += int(numel * torch.tensor([], dtype=dtype).element_size())
    return total


def _role_to_node_type(role: str) -> NodeType:
    mapping = {
        "parameter": NodeType.PARAM,
        "buffer": NodeType.BUFFER,
        "optimizer_state": NodeType.OPT_STATE,
        "gradient": NodeType.GRAD,
        "activation": NodeType.ACT,
    }
    return mapping.get(role, NodeType.OTHER)


class GraphProfiler(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        garbage_collect_values: bool = True,
        verbose: bool = False,
    ):
        super().__init__(module, garbage_collect_values)
        self.verbose = verbose
        self.nodes = list(self.module.graph.nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.placeholder_metadata = list(
            getattr(self.module, "_ac_placeholder_metadata", [])
        )
        self.runtime_stats: Dict[str, NodeRuntimeStat] = {}
        self.latest_summary: Optional[ProfilerSummary] = None
        self._runtime_tensors: Dict[str, Any] = {}
        self._forward_sep_index = self._find_boundary_index(torch.ops.separator.sep.default)
        self._backward_sep_index = self._find_boundary_index(
            torch.ops.separator.sep_backward.default
        )
        self._optimizer_start_index = self._find_optimizer_start_index()
        self.node_categories = self._infer_node_categories()
        self.activations = self._analyze_activations()
        self.parameter_bytes = _sum_placeholder_bytes(self.placeholder_metadata, "parameter")
        self.optimizer_state_bytes = _sum_placeholder_bytes(
            self.placeholder_metadata, "optimizer_state"
        )
        self.gradient_bytes = self._estimate_gradient_bytes()

    def _find_boundary_index(self, target: Any) -> int:
        for idx, node in enumerate(self.nodes):
            if node.target == target:
                return idx
        raise RuntimeError(f"Unable to find graph boundary node for target {target}.")

    def _find_optimizer_start_index(self) -> int:
        optimizer_markers = {
            torch.ops.aten._fused_adam.default,
            torch.ops.aten._foreach_add.Scalar,
            torch.ops.aten._foreach_add.List,
            torch.ops.aten._foreach_addcdiv.Scalar,
            torch.ops.aten._foreach_addcmul.Scalar,
            torch.ops.aten._foreach_div.List,
            torch.ops.aten._foreach_div.Scalar,
            torch.ops.aten._foreach_mul.Scalar,
            torch.ops.aten._foreach_neg.default,
            torch.ops.aten._foreach_reciprocal.default,
            torch.ops.aten._foreach_sqrt.default,
            torch.ops.aten._foreach_sub.Scalar,
        }
        for idx, node in enumerate(self.nodes):
            if node.target in optimizer_markers:
                return idx
        return len(self.nodes)

    def _infer_node_categories(self) -> Dict[str, NodeType]:
        categories: Dict[str, NodeType] = {}
        placeholders = [node for node in self.nodes if node.op == OP.PLACEHOLDER.value]
        for node, metadata in zip(placeholders, self.placeholder_metadata):
            categories[node.name] = _role_to_node_type(metadata.role)

        for node in self.nodes:
            if node.name in categories:
                continue
            idx = self.node_to_index[node]
            if idx < self._backward_sep_index:
                categories[node.name] = NodeType.ACT
            elif idx < self._optimizer_start_index:
                categories[node.name] = NodeType.GRAD
            else:
                categories[node.name] = NodeType.OTHER
        return categories

    def _estimate_gradient_bytes(self) -> int:
        total = 0
        for item in self.placeholder_metadata:
            if getattr(item, "role", None) != "parameter" or getattr(item, "shape", None) is None:
                continue
            dtype = getattr(item, "dtype", None)
            if dtype is None:
                continue
            numel = 1
            for dim in torch.Size(item.shape):
                numel *= dim
            total += int(numel * torch.tensor([], dtype=dtype).element_size())
        return total

    def _phase_for_index(self, idx: int) -> str:
        if idx <= self._forward_sep_index:
            return "forward"
        if idx < self._optimizer_start_index:
            return "backward"
        return "optimizer"

    def _activation_candidates(self) -> List[fx.Node]:
        candidates: List[fx.Node] = []
        for node in self.nodes:
            idx = self.node_to_index[node]
            if idx >= self._backward_sep_index:
                continue
            if node.op == OP.PLACEHOLDER.value:
                continue
            backward_users = [
                self.node_to_index[user]
                for user in node.users
                if self.node_to_index[user] >= self._backward_sep_index
            ]
            if backward_users:
                candidates.append(node)
        return candidates

    def _activation_frontier(self, node: fx.Node) -> List[str]:
        frontier: List[str] = []
        visited: set[str] = set()

        def visit(cur: fx.Node):
            for input_node in cur.all_input_nodes:
                input_idx = self.node_to_index[input_node]
                if input_node.op == OP.PLACEHOLDER.value:
                    if input_node.name not in visited:
                        frontier.append(input_node.name)
                        visited.add(input_node.name)
                    continue
                if input_idx >= self._backward_sep_index:
                    continue
                if self.node_categories.get(input_node.name) == NodeType.ACT and input_node.name != node.name:
                    if any(
                        self.node_to_index[user] >= self._backward_sep_index
                        for user in input_node.users
                    ):
                        if input_node.name not in visited:
                            frontier.append(input_node.name)
                            visited.add(input_node.name)
                        continue
                visit(input_node)

        visit(node)
        return sorted(frontier)

    def _analyze_activations(self) -> List[ActivationInfo]:
        activations: List[ActivationInfo] = []
        for node in self._activation_candidates():
            idx = self.node_to_index[node]
            forward_users = [
                self.node_to_index[user]
                for user in node.users
                if self.node_to_index[user] <= self._forward_sep_index
            ]
            backward_users = sorted(
                self.node_to_index[user]
                for user in node.users
                if self.node_to_index[user] >= self._backward_sep_index
            )
            if not backward_users:
                continue
            activations.append(
                ActivationInfo(
                    name=node.name,
                    create_index=idx,
                    last_forward_use_index=max(forward_users) if forward_users else idx,
                    first_backward_use_index=backward_users[0],
                    first_backward_user_name=self.nodes[backward_users[0]].name,
                    size_bytes=_tensor_bytes_from_meta(node),
                    shape=_tensor_shape_from_meta(node),
                    dtype=_tensor_dtype_from_meta(node),
                    source_target=_target_name(node.target),
                    required_input_names=self._activation_frontier(node),
                )
            )
        return sorted(activations, key=lambda item: item.create_index)

    def _ensure_stat(self, node: fx.Node) -> NodeRuntimeStat:
        existing = self.runtime_stats.get(node.name)
        if existing is not None:
            return existing
        idx = self.node_to_index[node]
        stat = NodeRuntimeStat(
            name=node.name,
            op=node.op,
            target=_target_name(node.target),
            category=self.node_categories.get(node.name, NodeType.OTHER).value,
            phase=self._phase_for_index(idx),
        )
        self.runtime_stats[node.name] = stat
        return stat

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        self._runtime_tensors = {}
        try:
            return super().run(
                *args, initial_env=initial_env, enable_io_processing=enable_io_processing
            )
        finally:
            self._runtime_tensors = {}

    def run_node(self, n: fx.Node) -> Any:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            memory_before = torch.cuda.memory_allocated()
            peak_before = torch.cuda.max_memory_allocated()
            start.record()
        else:
            start = end = None
            memory_before = 0
            peak_before = 0

        result = super().run_node(n)

        if use_cuda:
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
            memory_after = torch.cuda.memory_allocated()
            peak_after = torch.cuda.max_memory_allocated()
        else:
            elapsed_ms = 0.0
            memory_after = 0
            peak_after = 0

        stat = self._ensure_stat(n)
        stat.elapsed_ms_total += elapsed_ms
        stat.memory_before_bytes_total += int(memory_before)
        stat.memory_after_bytes_total += int(memory_after)
        stat.memory_peak_bytes_total += int(max(peak_before, peak_after))
        stat.output_bytes_total += _tensor_bytes_from_runtime(result)
        stat.samples += 1

        if self.verbose:
            print(
                f"[{stat.phase:>9}] {stat.name:<32} {stat.target:<50} "
                f"{elapsed_ms:8.3f} ms"
            )

        self._runtime_tensors[n.name] = result
        self._refresh_activation_sizes(n.name, result)
        return result

    def _refresh_activation_sizes(self, node_name: str, value: Any) -> None:
        output_bytes = _tensor_bytes_from_runtime(value)
        if output_bytes == 0:
            return
        for activation in self.activations:
            if activation.name == node_name and activation.size_bytes == 0:
                activation.size_bytes = output_bytes
                if activation.shape is None:
                    for tensor in _flatten_tensors(value):
                        activation.shape = list(tensor.shape)
                        activation.dtype = str(tensor.dtype)
                        break

    def reset_stats(self) -> None:
        self.runtime_stats = {}

    def aggregate_stats(self) -> None:
        for stat in self.runtime_stats.values():
            if stat.samples == 0:
                continue
            stat.elapsed_ms_avg = stat.elapsed_ms_total / stat.samples

        for activation in self.activations:
            activation.recompute_cost_ms = self._estimate_recompute_cost(activation)

        self.latest_summary = self.build_summary()

    def _estimate_recompute_cost(self, activation: ActivationInfo) -> float:
        name_to_node = {node.name: node for node in self.nodes}
        stop_names = set(activation.required_input_names)
        visited: set[str] = set()

        def visit(node: fx.Node) -> float:
            if node.name in visited or node.name in stop_names or node.op == OP.PLACEHOLDER.value:
                return 0.0
            if self.node_to_index[node] >= self._backward_sep_index:
                return 0.0
            visited.add(node.name)
            subtotal = self.runtime_stats.get(node.name, NodeRuntimeStat(
                name=node.name,
                op=node.op,
                target=_target_name(node.target),
                category=self.node_categories.get(node.name, NodeType.OTHER).value,
                phase=self._phase_for_index(self.node_to_index[node]),
            )).elapsed_ms_avg
            for input_node in node.all_input_nodes:
                subtotal += visit(input_node)
            return subtotal

        return visit(name_to_node[activation.name])

    def node_to_index_by_name(self, name: str) -> int:
        for node in self.nodes:
            if node.name == name:
                return self.node_to_index[node]
        raise KeyError(name)

    def _activation_timeline(self) -> List[Dict[str, int]]:
        timeline: List[Dict[str, int]] = []
        for idx, _node in enumerate(self.nodes):
            activation_live = 0
            for activation in self.activations:
                if activation.create_index <= idx <= activation.last_forward_use_index:
                    activation_live += activation.size_bytes
            total = (
                self.parameter_bytes
                + self.gradient_bytes
                + self.optimizer_state_bytes
                + activation_live
            )
            timeline.append(
                {
                    "index": idx,
                    "activation_bytes": activation_live,
                    "parameter_bytes": self.parameter_bytes,
                    "gradient_bytes": self.gradient_bytes,
                    "optimizer_state_bytes": self.optimizer_state_bytes,
                    "total_bytes": total,
                }
            )
        return timeline

    def build_summary(self) -> ProfilerSummary:
        timeline = self._activation_timeline()
        peak_entry = max(timeline, key=lambda item: item["total_bytes"])
        peak_breakdown = {
            "parameter_bytes": peak_entry["parameter_bytes"],
            "gradient_bytes": peak_entry["gradient_bytes"],
            "optimizer_state_bytes": peak_entry["optimizer_state_bytes"],
            "activation_bytes": peak_entry["activation_bytes"],
        }
        return ProfilerSummary(
            node_stats=sorted(
                self.runtime_stats.values(),
                key=lambda item: self.node_to_index_by_name(item.name),
            ),
            activations=self.activations,
            parameter_bytes=self.parameter_bytes,
            gradient_bytes=self.gradient_bytes,
            optimizer_state_bytes=self.optimizer_state_bytes,
            activation_peak_bytes=peak_entry["activation_bytes"],
            total_peak_bytes=peak_entry["total_bytes"],
            peak_breakdown_bytes=peak_breakdown,
            timeline_breakdown=timeline,
            boundary={
                "forward_end_index": self._forward_sep_index,
                "backward_begin_index": self._backward_sep_index,
                "optimizer_begin_index": self._optimizer_start_index,
            },
            metadata={
                "total_nodes": len(self.nodes),
                "activation_candidates": len(self.activations),
            },
        )

    def print_stats(self, limit: int = 20) -> None:
        summary = self.latest_summary or self.build_summary()
        print("Graph profiler summary")
        print(
            json.dumps(
                {
                    "boundary": summary.boundary,
                    "peak_breakdown_mb": {
                        key: round(value / (1024**2), 3)
                        for key, value in summary.peak_breakdown_bytes.items()
                    },
                    "activation_count": len(summary.activations),
                    "total_peak_mb": round(summary.total_peak_bytes / (1024**2), 3),
                },
                indent=2,
            )
        )
        print("Top activation candidates by size")
        for activation in sorted(
            summary.activations, key=lambda item: item.size_bytes, reverse=True
        )[:limit]:
            print(
                f"{activation.name:<30} size={activation.size_bytes / (1024**2):8.3f} MB "
                f"create={activation.create_index:4d} "
                f"last_fwd={activation.last_forward_use_index:4d} "
                f"first_bwd={activation.first_backward_use_index:4d} "
                f"recompute_cost={activation.recompute_cost_ms:8.3f} ms"
            )

    def export_summary(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.latest_summary or self.build_summary()
        output_path.write_text(
            json.dumps(asdict(summary), indent=2),
            encoding="utf-8",
        )
        return output_path
