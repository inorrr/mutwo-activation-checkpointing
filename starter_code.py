import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn

from activation_checkpoint import ActivationCheckpointConfig, activation_checkpointing
from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile


# Small toy network used to exercise the full tracing/profiling/rewrite path
class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


# training iteration
def train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor
) -> None:
    loss = model(batch).sum() #run
    loss = SEPFunction.apply(loss) #seperator
    loss.backward()
    optim.step() # update parameters with grad
    optim.zero_grad() # clear!


# Build the callback that is invoked once the training step has been traced into
# an FX GraphModule. The callback profiles the graph and optionally rewrites it.
def graph_transformation_factory(
    output_dir: Path,
    use_activation_checkpointing: bool,
    checkpoint_config: ActivationCheckpointConfig,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    def transform(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        # run traced graph through the profiler
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                profiler.run(*args)
        profiler.aggregate_stats()
        profiler.export_summary(output_dir / "profiler_summary.json")
        profiler.print_stats(limit=10)
        # The rewritten graphs produced here are safest to execute through the
        # FX interpreter because they may contain cloned nodes inserted by hand.
        gm._ac_run_with_interpreter = True

        if not use_activation_checkpointing:
            return gm

        # Build a recompute plan
        rewritten_gm, plan = activation_checkpointing(gm, profiler, checkpoint_config)
        (output_dir / "checkpoint_plan.json").write_text(
            str(asdict(plan)), encoding="utf-8"
        )
        rewritten_gm._ac_run_with_interpreter = True
        return rewritten_gm

    return transform


def experiment(
    use_activation_checkpointing: bool,
    output_dir: Path,
    memory_budget_mb: float | None = None,
):
    logging.getLogger().setLevel(logging.INFO)

    torch.manual_seed(20)
    batch_size = 1000
    layers = 10
    dim = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyModel(dim=dim, layers=layers).to(device)
    batch = torch.randn(batch_size, dim, device=device)
    # Adam on CUDA needs capturable=True when the optimizer step is traced.
    optimizer_kwargs = {"capturable": True} if device.type == "cuda" else {}
    optim = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True, **optimizer_kwargs)

    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param, device=device)
    optim.step()
    optim.zero_grad()

    compiled_fn = compile(
        train_step,
        graph_transformation_factory(
            output_dir=output_dir,
            use_activation_checkpointing=use_activation_checkpointing,
            checkpoint_config=ActivationCheckpointConfig(memory_budget_mb=memory_budget_mb),
        ),
    )
    compiled_fn(model, optim, batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the starter activation-checkpointing example.")
    parser.add_argument("--use-ac", action="store_true", help="Enable activation checkpointing.")
    parser.add_argument("--output-dir", default="outputs/starter")
    parser.add_argument("--memory-budget-mb", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment(
        use_activation_checkpointing=args.use_ac,
        output_dir=Path(args.output_dir),
        memory_budget_mb=args.memory_budget_mb,
    )
