from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.fx as fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch.fx.experimental.proxy_tensor import make_fx

from graph_prof import ActivationInfo, GraphProfiler
from graph_tracer import SEPFunction


@dataclass
class ActivationCheckpointConfig:
    memory_budget_mb: Optional[float] = None
    min_savings_mb: float = 1.0
    max_recompute_ratio: float = 0.35
    min_recompute_budget_ms: float = 1.0
    max_candidates: Optional[int] = 4
    prefer_peak_overlap: bool = True
    exclude_view_like_ops: bool = True


@dataclass
class ActivationCheckpointPlan:
    recompute: List[str] = field(default_factory=list)
    retain: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    estimated_saved_bytes: int = 0
    estimated_peak_bytes: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _MutwoRecomputeCandidate:
    activation: ActivationInfo
    inactive_time_ms: float
    recompute_ratio: float
    recompute_overhead_ms: float


def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    old_node_users = dict(old_node.users)
    for node in reversed(list(graph.nodes)):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
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
    return {node.name: node for node in gm.graph.nodes}


def _get_forward_nodes(gm: fx.GraphModule) -> Tuple[List[fx.Node], int, int]:
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
        if node.op == "placeholder" or node.name in allowed_roots:
            roots.append(node)
            return
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
    return any(marker in activation.source_target for marker in _VIEW_LIKE_TARGET_MARKERS)


def _activation_overlaps_peak(profiler: GraphProfiler, activation: ActivationInfo) -> bool:
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
    candidates: List[_MutwoRecomputeCandidate] = []
    skipped: Dict[str, str] = {}
    for activation in profiler.activations:
        if not _eligible_activation(activation, config):
            skipped[activation.name] = "below_min_savings"
            continue
        if config.exclude_view_like_ops and _is_view_like_activation(activation):
            skipped[activation.name] = "view_like_or_alias"
            continue

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
            if candidate.recompute_overhead_ms > 0 and projected_cost > allowed_recompute_ms:
                plan.skipped[activation.name] = "recompute_budget"
                continue

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
        "allowed_recompute_ms": allowed_recompute_ms,
        "estimated_recompute_ms": accumulated_recompute_ms,
    }
    return plan


def apply_activation_checkpointing(
    gm: fx.GraphModule,
    profiler: GraphProfiler,
    plan: ActivationCheckpointPlan,
) -> fx.GraphModule:
    name_to_node = get_name_to_node_map(gm)
    retained_names = set(plan.retain)
    rewritten: Set[str] = set()

    for activation in sorted(
        (item for item in profiler.activations if item.name in plan.recompute),
        key=lambda item: item.first_backward_use_index,
    ):
        if activation.name in rewritten:
            continue
        original_node = name_to_node[activation.name]
        allowed_roots = retained_names | {
            node.name for node in gm.graph.nodes if node.op == "placeholder"
        }
        recompute_subgraph, _roots = _extract_recompute_subgraph(
            gm, activation.name, allowed_roots
        )
        insertion_point = next(
            node for node in gm.graph.nodes if node.name == activation.first_backward_user_name
        )

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
    with torch.no_grad():
        baseline_output = baseline_gm(*args)
        rewritten_output = rewritten_gm(*args)

    def flatten(value: object) -> Iterable[torch.Tensor]:
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
