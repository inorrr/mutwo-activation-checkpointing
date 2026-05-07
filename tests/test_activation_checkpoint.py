import copy
from types import SimpleNamespace
import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from activation_checkpoint import (
    ActivationCheckpointConfig,
    build_checkpoint_plan,
    activation_checkpointing,
    custom_fn,
    remove_detach_nodes,
    verify_graph_equivalence,
)
from graph_prof import ActivationInfo, GraphProfiler, ProfilerSummary


class ActivationCheckpointRewriteTest(unittest.TestCase):
    def test_rewrite_preserves_outputs(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w1 = torch.randn(16, 16, device=device, requires_grad=True)
        w2 = torch.randn(16, 16, device=device, requires_grad=True)
        x = torch.randn(16, 16, device=device)

        gm = make_fx(custom_fn)(w1, w2, x)
        gm = remove_detach_nodes(gm)
        baseline = copy.deepcopy(gm)
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            profiler.run(w1, w2, x)
        profiler.aggregate_stats()

        rewritten, plan = activation_checkpointing(
            gm,
            profiler,
            ActivationCheckpointConfig(min_savings_mb=0.0, max_candidates=1),
        )

        self.assertGreaterEqual(len(plan.recompute), 1)
        self.assertTrue(verify_graph_equivalence(baseline, rewritten, (w1, w2, x)))

    def test_mutwo_planner_uses_recompute_ratio_then_memory_simulator(self) -> None:
        fast_small = ActivationInfo(
            name="fast_small",
            create_index=1,
            last_forward_use_index=1,
            first_backward_use_index=10,
            first_backward_user_name="bw_fast",
            size_bytes=60,
            shape=[15],
            dtype="torch.float32",
            source_target="aten.mm.default",
            recompute_cost_ms=2.0,
        )
        slow_large = ActivationInfo(
            name="slow_large",
            create_index=2,
            last_forward_use_index=5,
            first_backward_use_index=10,
            first_backward_user_name="bw_slow",
            size_bytes=80,
            shape=[20],
            dtype="torch.float32",
            source_target="aten.mm.default",
            recompute_cost_ms=20.0,
        )
        summary = ProfilerSummary(
            node_stats=[],
            activations=[fast_small, slow_large],
            parameter_bytes=0,
            gradient_bytes=0,
            optimizer_state_bytes=0,
            activation_peak_bytes=140,
            total_peak_bytes=140,
            peak_breakdown_bytes={
                "parameter_bytes": 0,
                "gradient_bytes": 0,
                "optimizer_state_bytes": 0,
                "activation_bytes": 140,
            },
            timeline_breakdown=[
                {
                    "index": idx,
                    "activation_bytes": 140 if 2 <= idx <= 10 else 0,
                    "parameter_bytes": 0,
                    "gradient_bytes": 0,
                    "optimizer_state_bytes": 0,
                    "total_bytes": 140 if 2 <= idx <= 10 else 0,
                }
                for idx in range(12)
            ],
            boundary={
                "forward_end_index": 6,
                "backward_begin_index": 8,
                "optimizer_begin_index": 12,
            },
            metadata={},
        )
        profiler = SimpleNamespace(
            activations=[fast_small, slow_large],
            runtime_stats={},
            latest_summary=summary,
            build_summary=lambda: summary,
        )

        plan = build_checkpoint_plan(
            profiler,
            ActivationCheckpointConfig(
                memory_budget_mb=80 / (1024**2),
                min_savings_mb=0,
                max_candidates=None,
            ),
        )

        self.assertEqual(plan.recompute, ["fast_small"])
        self.assertEqual(plan.estimated_peak_bytes, 80)
        self.assertEqual(plan.metadata["algorithm"], "mutwo_simplified_recompute_only")

    def test_mutwo_planner_keeps_recomputing_until_budget_is_met(self) -> None:
        first = ActivationInfo(
            name="first",
            create_index=1,
            last_forward_use_index=1,
            first_backward_use_index=10,
            first_backward_user_name="bw_first",
            size_bytes=40,
            shape=[10],
            dtype="torch.float32",
            source_target="aten.mm.default",
            recompute_cost_ms=2.0,
        )
        second = ActivationInfo(
            name="second",
            create_index=2,
            last_forward_use_index=2,
            first_backward_use_index=10,
            first_backward_user_name="bw_second",
            size_bytes=40,
            shape=[10],
            dtype="torch.float32",
            source_target="aten.mm.default",
            recompute_cost_ms=4.0,
        )
        third = ActivationInfo(
            name="third",
            create_index=3,
            last_forward_use_index=3,
            first_backward_use_index=10,
            first_backward_user_name="bw_third",
            size_bytes=40,
            shape=[10],
            dtype="torch.float32",
            source_target="aten.mm.default",
            recompute_cost_ms=8.0,
        )
        summary = ProfilerSummary(
            node_stats=[],
            activations=[first, second, third],
            parameter_bytes=0,
            gradient_bytes=0,
            optimizer_state_bytes=0,
            activation_peak_bytes=120,
            total_peak_bytes=120,
            peak_breakdown_bytes={
                "parameter_bytes": 0,
                "gradient_bytes": 0,
                "optimizer_state_bytes": 0,
                "activation_bytes": 120,
            },
            timeline_breakdown=[
                {
                    "index": idx,
                    "activation_bytes": 120 if 3 <= idx <= 10 else 0,
                    "parameter_bytes": 0,
                    "gradient_bytes": 0,
                    "optimizer_state_bytes": 0,
                    "total_bytes": 120 if 3 <= idx <= 10 else 0,
                }
                for idx in range(12)
            ],
            boundary={
                "forward_end_index": 6,
                "backward_begin_index": 8,
                "optimizer_begin_index": 12,
            },
            metadata={},
        )
        profiler = SimpleNamespace(
            activations=[first, second, third],
            runtime_stats={},
            latest_summary=summary,
            build_summary=lambda: summary,
        )

        plan = build_checkpoint_plan(
            profiler,
            ActivationCheckpointConfig(
                memory_budget_mb=40 / (1024**2),
                min_savings_mb=0,
                max_candidates=None,
            ),
        )

        self.assertEqual(plan.recompute, ["first", "second"])
        self.assertEqual(plan.estimated_peak_bytes, 40)


if __name__ == "__main__":
    unittest.main()
