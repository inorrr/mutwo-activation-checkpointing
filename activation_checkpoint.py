from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.fx as fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch.fx.experimental.proxy_tensor import make_fx

from graph_prof import ActivationInfo, GraphProfiler
from graph_tracer import SEPFunction


# User-tunable policy knobs for the simplified mu-TWO-style recompute planner.
@dataclass
class ActivationCheckpointConfig:
    memory_budget_mb: Optional[float] = None
    min_savings_mb: float = 0.25
    max_recompute_ratio: float = 1.0
    min_recompute_budget_ms: float = 1.0
    max_candidates: Optional[int] = 16
    prefer_peak_overlap: bool = True
    exclude_view_like_ops: bool = True


# Serializable plan: which activation node names to recompute, which to retain,
# why some candidates were skipped, and the modeled memory impact.
@dataclass
class ActivationCheckpointPlan:
    recompute: List[str] = field(default_factory=list)
    retain: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    estimated_saved_bytes: int = 0
    estimated_peak_bytes: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)


# Internal ranking object for one possible recomputation target.
@dataclass(frozen=True)
class _MutwoRecomputeCandidate:
    activation: ActivationInfo
    inactive_time_ms: float
    recompute_ratio: float
    recompute_overhead_ms: float


def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    # Replace only uses that occur after the recompute node. Forward-region uses
    # must keep the original value; later backward-region uses get the clone.
    old_node_users = dict(old_node.users)
    for node in reversed(list(graph.nodes)):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    # detach nodes do not change the tensor value needed for this analysis, but
    # they can make recompute subgraph extraction noisier.
    for node in list(gm.graph.nodes):
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def get_name_to_node_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    # FX node names are the stable IDs used by profiler summaries and plans.
    return {node.name: node for node in gm.graph.nodes}


def _get_forward_nodes(gm: fx.GraphModule) -> Tuple[List[fx.Node], int, int]:
    # Locate the forward and backward separator nodes inserted by SEPFunction.
    nodes = list(gm.graph.nodes)
    sep_idx = next(
        idx for idx, node in enumerate(nodes) if node.target == torch.ops.separator.sep.default
    )
    sep_bwd_idx = next(
        idx
        for idx, node in enumerate(nodes)
        if node.target == torch.ops.separator.sep_backward.default
    )
    return nodes, sep_idx, sep_bwd_idx


def _collect_required_inputs(
    gm: fx.GraphModule,
    target_name: str,
    allowed_roots: Set[str],
) -> Tuple[List[fx.Node], List[fx.Node]]:
    # Walk backward from a target activation to find the minimal producer chain
    # and the roots that must already be retained at recompute time.
    nodes, _sep_idx, sep_bwd_idx = _get_forward_nodes(gm)
    name_to_node = get_name_to_node_map(gm)
    target_node = name_to_node[target_name]
    visited: Set[str] = set()
    roots: List[fx.Node] = []
    subgraph_nodes: List[fx.Node] = []

    def visit(node: fx.Node) -> None:
        if node.name in visited:
            return
        visited.add(node.name)
        idx = nodes.index(node)
        # Placeholders and explicitly retained activations are valid roots for
        # the recompute subgraph.
        if node.op == "placeholder" or node.name in allowed_roots:
            roots.append(node)
            return
        # Recompute subgraphs should only clone forward-region producer nodes.
        if idx >= sep_bwd_idx:
            return
        for input_node in node.all_input_nodes:
            visit(input_node)
        subgraph_nodes.append(node)

    visit(target_node)
    roots = list(dict.fromkeys(roots))
    subgraph_nodes = [node for node in subgraph_nodes if node.name != target_name] + [target_node]
    return roots, subgraph_nodes


def _extract_recompute_subgraph(
    gm: fx.GraphModule, target_name: str, allowed_roots: Set[str]
) -> Tuple[fx.Graph, List[fx.Node]]:
    # Delegate the actual subgraph extraction to PyTorch's partitioner utility
    # once the desired roots and target output are known.
    name_to_node = get_name_to_node_map(gm)
    roots, _subgraph_nodes = _collect_required_inputs(gm, target_name, allowed_roots)
    recompute_subgraph = _extract_graph_with_inputs_outputs(
        joint_graph=gm.graph,
        inputs=roots,
        outputs=[name_to_node[target_name]],
    )
    return recompute_subgraph, roots


def _eligible_activation(
    activation: ActivationInfo, config: ActivationCheckpointConfig
) -> bool:
    # Ignore tiny values that cannot plausibly reduce peak memory.
    return activation.size_bytes >= int(config.min_savings_mb * 1024 * 1024)


_VIEW_LIKE_TARGET_MARKERS = (
    "aten.alias.",
    "aten.as_strided.",
    "aten.detach.",
    "aten.expand.",
    "aten.permute.",
    "aten.reshape.",
    "aten.select.",
    "aten.slice.",
    "aten.squeeze.",
    "aten.t.",
    "aten.transpose.",
    "aten.unsqueeze.",
    "aten.view.",
    "_operator.getitem",
)


def _is_view_like_activation(activation: ActivationInfo) -> bool:
    # View/alias ops can look large by shape while owning little or no storage.
    # Recomputing them often adds work without lowering allocator pressure.
    return any(marker in activation.source_target for marker in _VIEW_LIKE_TARGET_MARKERS)


def _activation_overlaps_peak(profiler: GraphProfiler, activation: ActivationInfo) -> bool:
    # A candidate only reduces modeled peak if it is live at the activation peak.
    summary = profiler.latest_summary or profiler.build_summary()
    if not summary.timeline_breakdown:
        return True
    peak_index = max(summary.timeline_breakdown, key=lambda item: item["activation_bytes"])[
        "index"
    ]
    return activation.create_index <= peak_index <= activation.first_backward_use_index


def _activation_inactive_time_ms(
    profiler: GraphProfiler, activation: ActivationInfo
) -> float:
    # Approximate how long the activation sits unused between its last forward
    # use and first backward use. Longer inactive intervals are better targets.
    nodes = getattr(profiler, "nodes", [])
    node_to_index = getattr(profiler, "node_to_index", {})
    elapsed = 0.0
    for node in nodes:
        idx = node_to_index.get(node)
        if idx is None:
            continue
        if activation.last_forward_use_index < idx < activation.first_backward_use_index:
            stat = profiler.runtime_stats.get(node.name)
            if stat is not None:
                elapsed += stat.elapsed_ms_avg
    if elapsed > 0:
        return elapsed
    return float(max(0, activation.first_backward_use_index - activation.last_forward_use_index))


def _build_mutwo_candidates(
    profiler: GraphProfiler,
    config: ActivationCheckpointConfig,
) -> Tuple[List[_MutwoRecomputeCandidate], Dict[str, str]]:
    # Convert profiler activation records into ranked planner candidates while
    # recording explicit skip reasons for later inspection.
    candidates: List[_MutwoRecomputeCandidate] = []
    skipped: Dict[str, str] = {}
    for activation in profiler.activations:
        if not _eligible_activation(activation, config):
            skipped[activation.name] = "below_min_savings"
            continue
        if config.exclude_view_like_ops and _is_view_like_activation(activation):
            skipped[activation.name] = "view_like_or_alias"
            continue

        # Bytes-per-millisecond is the practical "memory saved per recompute
        # cost" score used by the simplified policy.
        recompute_overhead = max(activation.recompute_cost_ms, 0.0)
        recompute_ratio = (
            float("inf")
            if recompute_overhead <= 0
            else activation.size_bytes / max(recompute_overhead, 1e-6)
        )
        candidates.append(
            _MutwoRecomputeCandidate(
                activation=activation,
                inactive_time_ms=_activation_inactive_time_ms(profiler, activation),
                recompute_ratio=recompute_ratio,
                recompute_overhead_ms=recompute_overhead,
            )
        )
    return candidates, skipped


def _simulated_activation_peak_bytes(
    profiler: GraphProfiler,
    recomputed_names: Set[str],
) -> int:
    # Model checkpointing by shortening recomputed activations to end at their
    # last forward use instead of first backward use.
    peak = 0
    nodes = getattr(profiler, "nodes", [])
    if nodes:
        indices = range(len(nodes))
    else:
        summary = profiler.latest_summary or profiler.build_summary()
        indices = (item["index"] for item in summary.timeline_breakdown)
    for idx in indices:
        live_bytes = 0
        for activation in profiler.activations:
            last_live_index = (
                activation.last_forward_use_index
                if activation.name in recomputed_names
                else activation.first_backward_use_index
            )
            if activation.create_index <= idx <= last_live_index:
                live_bytes += activation.size_bytes
        peak = max(peak, live_bytes)
    return peak


def _simulated_total_peak_bytes(
    profiler: GraphProfiler,
    recomputed_names: Set[str],
) -> int:
    # Total modeled peak is fixed non-activation memory plus simulated
    # activation peak.
    summary = profiler.latest_summary or profiler.build_summary()
    non_activation_bytes = (
        summary.parameter_bytes + summary.gradient_bytes + summary.optimizer_state_bytes
    )
    return non_activation_bytes + _simulated_activation_peak_bytes(
        profiler, recomputed_names
    )


def _mutwo_recompute_sort_key(
    candidate: _MutwoRecomputeCandidate,
) -> Tuple[float, float, int, int]:
    # Higher is better: save more bytes per recompute ms, prefer longer inactive
    # windows, then larger tensors, then earlier producers.
    return (
        candidate.recompute_ratio,
        candidate.inactive_time_ms,
        candidate.activation.size_bytes,
        -candidate.activation.create_index,
    )


def build_checkpoint_plan(
    profiler: GraphProfiler,
    config: Optional[ActivationCheckpointConfig] = None,
) -> ActivationCheckpointPlan:
    # Main planning loop. It is recompute-only: no offload/swapping and no
    # multi-model scheduler from the full mu-TWO system.
    config = config or ActivationCheckpointConfig()
    summary = profiler.latest_summary or profiler.build_summary()
    memory_limit_bytes = (
        int(config.memory_budget_mb * 1024 * 1024)
        if config.memory_budget_mb is not None
        else None
    )
    candidates, initial_skipped = _build_mutwo_candidates(profiler, config)
    plan = ActivationCheckpointPlan()
    plan.skipped.update(initial_skipped)
    current_peak_bytes = summary.total_peak_bytes
    plan.estimated_peak_bytes = current_peak_bytes

    # Prefer candidates that overlap the modeled activation peak because those
    # are the candidates capable of lowering peak memory.
    peak_overlaps = {
        candidate.activation.name: _activation_overlaps_peak(
            profiler, candidate.activation
        )
        for candidate in candidates
    }
    peak_candidates = [
        candidate
        for candidate in candidates
        if not config.prefer_peak_overlap or peak_overlaps[candidate.activation.name]
    ]
    if peak_candidates:
        for candidate in candidates:
            if candidate not in peak_candidates:
                plan.skipped[candidate.activation.name] = "outside_peak_live_set"
        candidates = peak_candidates
    elif config.prefer_peak_overlap:
        for candidate in candidates:
            plan.skipped[candidate.activation.name] = "outside_peak_live_set"
        candidates = []

    # Sort once by policy score; the loop below still simulates each candidate
    # because interactions between selected activations affect the peak.
    remaining = sorted(
        candidates,
        key=_mutwo_recompute_sort_key,
        reverse=True,
    )
    accumulated_saved = 0
    accumulated_recompute_ms = 0.0
    total_forward_ms = sum(
        stat.elapsed_ms_avg for stat in profiler.runtime_stats.values() if stat.phase == "forward"
    )
    allowed_recompute_ms = (
        float("inf")
        if total_forward_ms <= 0
        else max(
            total_forward_ms * config.max_recompute_ratio,
            config.min_recompute_budget_ms,
        )
    )

    # Greedily select candidates until the memory target or policy limits stop
    # the process.
    while remaining:
        if memory_limit_bytes is not None and current_peak_bytes <= memory_limit_bytes:
            break
        if config.max_candidates is not None and len(plan.recompute) >= config.max_candidates:
            for candidate in remaining:
                plan.skipped[candidate.activation.name] = "candidate_limit"
            break

        selected_candidate: Optional[_MutwoRecomputeCandidate] = None
        selected_peak = current_peak_bytes
        selected_cost = accumulated_recompute_ms
        for candidate in remaining:
            activation = candidate.activation
            projected_cost = accumulated_recompute_ms + candidate.recompute_overhead_ms
            # Do not exceed the configured recompute budget.
            if candidate.recompute_overhead_ms > 0 and projected_cost > allowed_recompute_ms:
                plan.skipped[activation.name] = "recompute_budget"
                continue

            # Only accept the candidate if simulated peak memory improves, unless
            # there is no explicit memory budget and the candidate limit is the
            # controlling stop condition.
            projected_recompute = set(plan.recompute) | {activation.name}
            projected_peak = _simulated_total_peak_bytes(profiler, projected_recompute)
            if projected_peak >= current_peak_bytes and memory_limit_bytes is not None:
                plan.skipped[activation.name] = "no_peak_reduction"
                continue
            selected_candidate = candidate
            selected_peak = projected_peak
            selected_cost = projected_cost
            break

        if selected_candidate is None:
            break

        # Commit the selected candidate and update the modeled peak.
        activation = selected_candidate.activation
        accumulated_saved += activation.size_bytes
        accumulated_recompute_ms = selected_cost
        plan.skipped.pop(activation.name, None)
        plan.recompute.append(activation.name)
        current_peak_bytes = selected_peak
        plan.estimated_peak_bytes = current_peak_bytes
        remaining = [
            candidate
            for candidate in remaining
            if candidate.activation.name != activation.name
        ]

        if memory_limit_bytes is None and config.max_candidates is None:
            break

    # Any activation not selected for recompute remains retained.
    plan.retain = [
        activation.name
        for activation in profiler.activations
        if activation.name not in set(plan.recompute)
    ]
    plan.estimated_saved_bytes = accumulated_saved
    plan.metadata = {
        "algorithm": "mutwo_simplified_recompute_only",
        "initial_peak_bytes": summary.total_peak_bytes,
        "memory_limit_bytes": memory_limit_bytes,
        "min_savings_mb": config.min_savings_mb,
        "max_candidates": config.max_candidates,
        "max_recompute_ratio": config.max_recompute_ratio,
        "min_recompute_budget_ms": config.min_recompute_budget_ms,
        "prefer_peak_overlap": config.prefer_peak_overlap,
        "exclude_view_like_ops": config.exclude_view_like_ops,
        "allowed_recompute_ms": allowed_recompute_ms,
        "estimated_recompute_ms": accumulated_recompute_ms,
    }
    return plan


def apply_activation_checkpointing(
    gm: fx.GraphModule,
    profiler: GraphProfiler,
    plan: ActivationCheckpointPlan,
) -> fx.GraphModule:
    # Mutate the FX graph according to the plan by cloning recompute subgraphs
    # into the backward region.
    name_to_node = get_name_to_node_map(gm)
    retained_names = set(plan.retain)
    rewritten: Set[str] = set()

    # Insert earlier-needed recomputations first so replacements happen in a
    # stable graph order.
    for activation in sorted(
        (item for item in profiler.activations if item.name in plan.recompute),
        key=lambda item: item.first_backward_use_index,
    ):
        if activation.name in rewritten:
            continue
        original_node = name_to_node[activation.name]
        # A recompute subgraph may depend on placeholders and values we decided
        # to retain, but not on activations that are being discarded.
        allowed_roots = retained_names | {
            node.name for node in gm.graph.nodes if node.op == "placeholder"
        }
        recompute_subgraph, _roots = _extract_recompute_subgraph(
            gm, activation.name, allowed_roots
        )
        insertion_point = next(
            node for node in gm.graph.nodes if node.name == activation.first_backward_user_name
        )

        # local_name_to_old maps nodes in the extracted recompute graph back to
        # existing or newly cloned nodes in the main graph.
        local_name_to_old: Dict[str, fx.Node] = dict(name_to_node)
        replacement_node: Optional[fx.Node] = None
        with gm.graph.inserting_before(insertion_point):
            for node in recompute_subgraph.nodes:
                if node.op in {"placeholder", "output"}:
                    continue
                copied = gm.graph.node_copy(
                    node, arg_transform=lambda arg: local_name_to_old[arg.name]
                )
                local_name_to_old[node.name] = copied
                if node.name == activation.name:
                    replacement_node = copied

        if replacement_node is None:
            raise RuntimeError(f"Failed to create recompute node for {activation.name}.")
        # Redirect only backward/later uses, preserving original forward uses.
        replace_subsequent_uses_of(gm.graph, original_node, replacement_node)
        rewritten.add(activation.name)
        retained_names.add(activation.name)
        name_to_node = get_name_to_node_map(gm)

    gm.graph.lint()
    gm.recompile()
    return gm


def activation_checkpointing(
    gm: fx.GraphModule,
    profiler: GraphProfiler,
    config: Optional[ActivationCheckpointConfig] = None,
) -> Tuple[fx.GraphModule, ActivationCheckpointPlan]:
    # Public API: normalize graph, plan selected activations, and apply the graph
    # rewrite in one call.
    gm = remove_detach_nodes(gm)
    plan = build_checkpoint_plan(profiler, config=config)
    rewritten = apply_activation_checkpointing(gm, profiler, plan)
    return rewritten, plan


def verify_graph_equivalence(
    baseline_gm: fx.GraphModule,
    rewritten_gm: fx.GraphModule,
    args: Sequence[torch.Tensor],
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    # Lightweight correctness check for toy/test paths: compare tensor outputs
    # of the original and rewritten GraphModules.
    with torch.no_grad():
        baseline_output = baseline_gm(*args)
        rewritten_output = rewritten_gm(*args)

    def flatten(value: object) -> Iterable[torch.Tensor]:
        # Output structures can be nested; compare tensor leaves only.
        if isinstance(value, torch.Tensor):
            yield value
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                yield from flatten(item)

    baseline_tensors = list(flatten(baseline_output))
    rewritten_tensors = list(flatten(rewritten_output))
    if len(baseline_tensors) != len(rewritten_tensors):
        return False
    return all(
        torch.allclose(lhs, rhs, atol=atol, rtol=rtol)
        for lhs, rhs in zip(baseline_tensors, rewritten_tensors)
    )


def custom_fn(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Tiny hand-written function used by tests and the module-level demo.
    z = torch.mm(w1, x)
    z = torch.nn.functional.relu(z)
    z = torch.mm(z, w2)
    z = torch.nn.functional.relu(z)
    z = z.sum()
    z = SEPFunction.apply(z)
    z.backward()
    return w1.grad, w2.grad


if __name__ == "__main__":
    w1 = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    w2 = torch.randn(2048, 512, device="cuda", requires_grad=True)
    x = torch.randn(1024, 2048, device="cuda")

    graph_module = make_fx(custom_fn)(w1, w2, x)
    graph_module = remove_detach_nodes(graph_module)
    profiler = GraphProfiler(graph_module)
    with torch.no_grad():
        profiler.run(w1, w2, x)
    profiler.aggregate_stats()
    new_graph_module, plan = activation_checkpointing(graph_module, profiler)
    print("Recompute plan:", plan.recompute)
    print(
        "Equivalent:",
        verify_graph_equivalence(graph_module, new_graph_module, (w1, w2, x)),
    )
