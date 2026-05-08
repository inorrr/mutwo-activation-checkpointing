import unittest

import torch
import torch.nn as nn

from graph_tracer import SEPFunction, compile


class TwoGroupModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
) -> None:
    loss = SEPFunction.apply(model(batch).sum())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


class GraphTracerTest(unittest.TestCase):
    def test_compile_supports_multiple_optimizer_parameter_groups(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TwoGroupModel().to(device)
        optimizer = torch.optim.Adam(
            [
                {"params": model.first.parameters(), "lr": 1e-3},
                {"params": model.second.parameters(), "lr": 5e-4},
            ],
            foreach=True,
            **({"capturable": True} if device.type == "cuda" else {}),
        )
        batch = torch.randn(3, 4, device=device)

        for param in model.parameters():
            param.grad = torch.rand_like(param)
        optimizer.step()
        optimizer.zero_grad()

        seen = {}

        def transform(gm, _args):
            seen["param_group_count"] = len(optimizer.param_groups)
            seen["group_sizes"] = [len(group["params"]) for group in optimizer.param_groups]
            return gm

        compiled = compile(train_step, transform)
        compiled(model, optimizer, batch)

        self.assertEqual(seen["param_group_count"], 2)
        self.assertEqual(seen["group_sizes"], [2, 2])


if __name__ == "__main__":
    unittest.main()
