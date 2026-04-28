import copy
import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from activation_checkpoint import (
    ActivationCheckpointConfig,
    activation_checkpointing,
    custom_fn,
    remove_detach_nodes,
    verify_graph_equivalence,
)
from graph_prof import GraphProfiler


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


if __name__ == "__main__":
    unittest.main()
