import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from activation_checkpoint import custom_fn, remove_detach_nodes
from graph_prof import GraphProfiler


class GraphProfilerTest(unittest.TestCase):
    def test_profiler_finds_activation_lifetimes(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w1 = torch.randn(16, 16, device=device, requires_grad=True)
        w2 = torch.randn(16, 16, device=device, requires_grad=True)
        x = torch.randn(16, 16, device=device)

        gm = make_fx(custom_fn)(w1, w2, x)
        gm = remove_detach_nodes(gm)
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            profiler.run(w1, w2, x)
        profiler.aggregate_stats()
        summary = profiler.latest_summary

        self.assertIsNotNone(summary)
        self.assertGreaterEqual(len(summary.activations), 1)
        self.assertGreater(summary.total_peak_bytes, 0)
        self.assertLess(summary.boundary["forward_end_index"], summary.boundary["backward_begin_index"])


if __name__ == "__main__":
    unittest.main()
