from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet152
from transformers import BertConfig, BertForMaskedLM

from activation_checkpoint import (
    ActivationCheckpointConfig,
    ActivationCheckpointPlan,
    activation_checkpointing,
)
from graph_prof import GraphProfiler, ProfilerSummary
from graph_tracer import SEPFunction, compile


DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass
class RunResult:
    model_name: str
    batch_size: int
    use_activation_checkpointing: bool
    latency_ms_avg: float
    latency_ms_std: float
    peak_memory_bytes_avg: float
    peak_memory_bytes_max: float
    correctness_ok: Optional[bool]
    output_dir: str
    profiler_summary_path: str
    profiler_breakdown_plot: str
    plan_path: Optional[str] = None
    rewritten_profiler_summary_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepResult:
    model_name: str
    runs: List[RunResult]
    output_dir: str
    memory_plot: str
    latency_plot: str


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _optimizer_kwargs(device: torch.device) -> Dict[str, Any]:
    if device.type == "cuda":
        return {"capturable": True, "foreach": True}
    return {"foreach": True}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value {type(value)}")


def _cleanup_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class Experiment:
    model_names = ["ResNet-152", "BERT"]
    default_batch_sizes: Dict[str, List[int]] = {
        "ResNet-152": [1, 2, 4],
        "BERT": [1, 2, 4],
    }

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        output_root: Path = DEFAULT_OUTPUT_DIR,
        seq_len: int = 128,
        vocab_size: int = 30_522,
        image_size: int = 224,
        debug_bert: bool = False,
    ):
        assert model_name in self.model_names, (
            f"Model {model_name} not supported. Expected one of {self.model_names}."
        )
        self.device = _device()
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.debug_bert = debug_bert
        self.output_root = Path(output_root)
        self.output_dir = self.output_root / model_name.replace(" ", "_") / f"bs_{batch_size}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.model_name == "ResNet-152":
            self.model = resnet152(weights=None).to(self.device)
            self.example_inputs = self._make_resnet_batch(batch_size)
            self.train_step = self._resnet_train_step
        else:
            self.model = self._build_bert().to(self.device)
            self.example_inputs = self._make_bert_batch(batch_size)
            self.train_step = self._bert_train_step

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            **_optimizer_kwargs(self.device),
        )

    def _build_bert(self) -> BertForMaskedLM:
        if self.debug_bert:
            config = BertConfig(
                vocab_size=self.vocab_size,
                hidden_size=384,
                intermediate_size=1536,
                num_hidden_layers=6,
                num_attention_heads=6,
                max_position_embeddings=max(self.seq_len, 128),
            )
        else:
            config = BertConfig(
                vocab_size=self.vocab_size,
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                max_position_embeddings=max(self.seq_len, 512),
            )
        return BertForMaskedLM(config)

    def _make_resnet_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.randn(batch_size, 3, self.image_size, self.image_size, device=self.device)
        targets = torch.randint(0, 1000, (batch_size,), device=self.device)
        return images, targets

    def _make_bert_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.randint(
            0, self.vocab_size, (batch_size, self.seq_len), device=self.device
        )
        attention_mask = torch.ones_like(input_ids, device=self.device)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _resnet_train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        logits = model(batch[0])
        loss = F.cross_entropy(logits, batch[1])
        loss = SEPFunction.apply(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _bert_train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        batch: Dict[str, torch.Tensor],
    ) -> None:
        outputs = model(**batch)
        loss = SEPFunction.apply(outputs.loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def init_opt_states(self) -> None:
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _new_batch_like(self) -> Any:
        if self.model_name == "ResNet-152":
            return self._make_resnet_batch(self.batch_size)
        return self._make_bert_batch(self.batch_size)

    def _plot_peak_memory_breakdown(self, summary: ProfilerSummary, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labels = ["Parameters", "Gradients", "Optimizer", "Activations"]
        values = [
            summary.peak_breakdown_bytes["parameter_bytes"] / (1024**2),
            summary.peak_breakdown_bytes["gradient_bytes"] / (1024**2),
            summary.peak_breakdown_bytes["optimizer_state_bytes"] / (1024**2),
            summary.peak_breakdown_bytes["activation_bytes"] / (1024**2),
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color=["#365c8d", "#4ac16d", "#d08c60", "#b33f62"])
        ax.set_ylabel("Memory (MB)")
        ax.set_title(f"Peak Memory Breakdown: {self.model_name} (bs={self.batch_size})")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    def _write_json(self, payload: Dict[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path

    def _measure_compiled_fn(
        self,
        compiled_fn,
        iterations: int = 3,
        warmup_iterations: int = 1,
    ) -> Tuple[List[float], List[int]]:
        latencies: List[float] = []
        peaks: List[int] = []
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup_iterations):
            compiled_fn(self.model, self.optimizer, self._new_batch_like())
        for _ in range(iterations):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            compiled_fn(self.model, self.optimizer, self._new_batch_like())
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                peaks.append(torch.cuda.max_memory_allocated())
            else:
                peaks.append(0)
            latencies.append((time.perf_counter() - start) * 1000.0)
        return latencies, peaks

    def _validate_correctness(
        self,
        transformed_gm: fx.GraphModule,
        original_gm: fx.GraphModule,
        flat_args: List[torch.Tensor],
    ) -> bool:
        with torch.no_grad():
            baseline = original_gm(*flat_args)
            transformed = transformed_gm(*flat_args)

        def flatten(value: Any) -> Iterable[torch.Tensor]:
            if isinstance(value, torch.Tensor):
                yield value
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    yield from flatten(item)

        base_tensors = list(flatten(baseline))
        transformed_tensors = list(flatten(transformed))
        if len(base_tensors) != len(transformed_tensors):
            return False
        return all(torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(base_tensors, transformed_tensors))

    def run_once(
        self,
        use_activation_checkpointing: bool,
        checkpoint_config: Optional[ActivationCheckpointConfig] = None,
        profiler_iters: int = 2,
    ) -> RunResult:
        self.init_opt_states()
        metadata: Dict[str, Any] = {
            "device": str(self.device),
            "seq_len": self.seq_len if self.model_name == "BERT" else None,
            "vocab_size": self.vocab_size if self.model_name == "BERT" else None,
            "image_size": self.image_size if self.model_name == "ResNet-152" else None,
            "debug_bert": self.debug_bert,
            "measurement_mode": "custom_fx_graph_rewrite",
        }
        artifacts_dir = self.output_dir / ("ac" if use_activation_checkpointing else "baseline")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        capture: Dict[str, Any] = {}

        def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
            profiler = GraphProfiler(gm)
            with torch.no_grad():
                for _ in range(profiler_iters):
                    profiler.run(*args)
            profiler.aggregate_stats()
            summary = profiler.latest_summary or profiler.build_summary()
            summary_path = profiler.export_summary(artifacts_dir / "profiler_summary.json")
            breakdown_path = self._plot_peak_memory_breakdown(
                summary, artifacts_dir / "peak_memory_breakdown.png"
            )
            capture["profiler"] = profiler
            capture["summary"] = summary
            capture["summary_path"] = summary_path
            capture["breakdown_path"] = breakdown_path
            capture["flat_args"] = args
            gm._ac_run_with_interpreter = True

            if not use_activation_checkpointing:
                capture["correctness_ok"] = None
                return gm

            original_gm = copy.deepcopy(gm)
            rewritten_gm, plan = activation_checkpointing(
                gm, profiler, checkpoint_config or ActivationCheckpointConfig()
            )
            plan_path = self._write_json(
                {
                    "plan": asdict(plan),
                    "activations": [asdict(item) for item in profiler.activations],
                },
                artifacts_dir / "checkpoint_plan.json",
            )
            capture["plan_path"] = plan_path
            capture["plan"] = plan
            capture["correctness_ok"] = self._validate_correctness(
                rewritten_gm, original_gm, args
            )
            _cleanup_memory(self.device)
            try:
                rewritten_profiler = GraphProfiler(rewritten_gm)
                with torch.no_grad():
                    for _ in range(profiler_iters):
                        rewritten_profiler.run(*args)
                rewritten_profiler.aggregate_stats()
                capture["rewritten_summary_path"] = rewritten_profiler.export_summary(
                    artifacts_dir / "rewritten_profiler_summary.json"
                )
            except torch.OutOfMemoryError as exc:
                _cleanup_memory(self.device)
                capture["rewritten_summary_path"] = None
                self._write_json(
                    {
                        "error": "rewritten_profiler_oom",
                        "message": str(exc).splitlines()[0],
                        "plan_recompute_count": len(plan.recompute),
                    },
                    artifacts_dir / "rewritten_profiler_summary_error.json",
                )
            rewritten_gm._ac_run_with_interpreter = True
            return rewritten_gm

        compiled_fn = compile(self.train_step, graph_transformation)
        compiled_fn(self.model, self.optimizer, self.example_inputs)
        _cleanup_memory(self.device)
        latencies, peaks = self._measure_compiled_fn(compiled_fn)

        return RunResult(
            model_name=self.model_name,
            batch_size=self.batch_size,
            use_activation_checkpointing=use_activation_checkpointing,
            latency_ms_avg=float(sum(latencies) / len(latencies)),
            latency_ms_std=float(torch.tensor(latencies).std(unbiased=False).item()),
            peak_memory_bytes_avg=float(sum(peaks) / len(peaks)),
            peak_memory_bytes_max=float(max(peaks)),
            correctness_ok=capture.get("correctness_ok"),
            output_dir=str(artifacts_dir),
            profiler_summary_path=str(capture["summary_path"]),
            profiler_breakdown_plot=str(capture["breakdown_path"]),
            plan_path=str(capture["plan_path"]) if "plan_path" in capture else None,
            rewritten_profiler_summary_path=(
                str(capture["rewritten_summary_path"])
                if "rewritten_summary_path" in capture
                else None
            ),
            metadata=metadata,
        )


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_sweep(
    runs: List[RunResult],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = sorted(
        [run for run in runs if not run.use_activation_checkpointing],
        key=lambda item: item.batch_size,
    )
    checkpointed = sorted(
        [run for run in runs if run.use_activation_checkpointing],
        key=lambda item: item.batch_size,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    if "memory" in metric:
        batch_sizes = sorted({run.batch_size for run in baseline + checkpointed})
        x_positions = list(range(len(batch_sizes)))
        width = 0.36
        baseline_by_batch = {run.batch_size: run for run in baseline}
        checkpointed_by_batch = {run.batch_size: run for run in checkpointed}
        if baseline:
            ax.bar(
                [pos - width / 2 for pos in x_positions],
                [
                    getattr(baseline_by_batch[batch_size], metric) / (1024**2)
                    if batch_size in baseline_by_batch
                    else 0
                    for batch_size in batch_sizes
                ],
                width=width,
                label="Baseline",
            )
        if checkpointed:
            ax.bar(
                [pos + width / 2 for pos in x_positions],
                [
                    getattr(checkpointed_by_batch[batch_size], metric) / (1024**2)
                    if batch_size in checkpointed_by_batch
                    else 0
                    for batch_size in batch_sizes
                ],
                width=width,
                label="Activation checkpointing",
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    else:
        if baseline:
            ax.plot(
                [run.batch_size for run in baseline],
                [getattr(run, metric) for run in baseline],
                marker="o",
                label="Baseline",
            )
        if checkpointed:
            ax.plot(
                [run.batch_size for run in checkpointed],
                [getattr(run, metric) for run in checkpointed],
                marker="o",
                label="Activation checkpointing",
            )
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def run_sweep(
    model_name: str,
    batch_sizes: List[int],
    output_root: Path = DEFAULT_OUTPUT_DIR,
    checkpoint_config: Optional[ActivationCheckpointConfig] = None,
    seq_len: int = 128,
    vocab_size: int = 30_522,
    image_size: int = 224,
    debug_bert: bool = False,
) -> SweepResult:
    output_dir = Path(output_root) / model_name.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: List[RunResult] = []

    for batch_size in batch_sizes:
        baseline = Experiment(
            model_name=model_name,
            batch_size=batch_size,
            output_root=output_root,
            seq_len=seq_len,
            vocab_size=vocab_size,
            image_size=image_size,
            debug_bert=debug_bert,
        ).run_once(use_activation_checkpointing=False)
        runs.append(baseline)

        checkpointed = Experiment(
            model_name=model_name,
            batch_size=batch_size,
            output_root=output_root,
            seq_len=seq_len,
            vocab_size=vocab_size,
            image_size=image_size,
            debug_bert=debug_bert,
        ).run_once(
            use_activation_checkpointing=True,
            checkpoint_config=checkpoint_config,
        )
        runs.append(checkpointed)

    rows = [
        {
            "model_name": run.model_name,
            "batch_size": run.batch_size,
            "use_activation_checkpointing": run.use_activation_checkpointing,
            "latency_ms_avg": run.latency_ms_avg,
            "latency_ms_std": run.latency_ms_std,
            "peak_memory_bytes_avg": run.peak_memory_bytes_avg,
            "peak_memory_bytes_max": run.peak_memory_bytes_max,
            "correctness_ok": run.correctness_ok,
            "output_dir": run.output_dir,
            "profiler_summary_path": run.profiler_summary_path,
            "plan_path": run.plan_path,
            "rewritten_profiler_summary_path": run.rewritten_profiler_summary_path,
        }
        for run in runs
    ]
    _write_csv(rows, output_dir / "sweep_results.csv")

    memory_plot = _plot_sweep(
        runs,
        metric="peak_memory_bytes_max",
        ylabel="Peak GPU memory (MB)",
        title=f"Peak Memory vs Batch Size: {model_name}",
        output_path=output_dir / "peak_memory_vs_batch_size.png",
    )
    latency_plot = _plot_sweep(
        runs,
        metric="latency_ms_avg",
        ylabel="Iteration latency (ms)",
        title=f"Iteration Latency vs Batch Size: {model_name}",
        output_path=output_dir / "latency_vs_batch_size.png",
    )

    manifest = {
        "model_name": model_name,
        "batch_sizes": batch_sizes,
        "runs": [asdict(run) for run in runs],
        "memory_plot": str(memory_plot),
        "latency_plot": str(latency_plot),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )

    return SweepResult(
        model_name=model_name,
        runs=runs,
        output_dir=str(output_dir),
        memory_plot=str(memory_plot),
        latency_plot=str(latency_plot),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activation checkpointing benchmarks.")
    parser.add_argument("--model", choices=Experiment.model_names, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=30_522)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--debug-bert", action="store_true")
    parser.add_argument("--memory-budget-mb", type=float, default=None)
    parser.add_argument("--min-savings-mb", type=float, default=0.25)
    parser.add_argument("--max-recompute-ratio", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_sizes = args.batch_sizes or Experiment.default_batch_sizes[args.model]
    config = ActivationCheckpointConfig(
        memory_budget_mb=args.memory_budget_mb,
        min_savings_mb=args.min_savings_mb,
        max_recompute_ratio=args.max_recompute_ratio,
        max_candidates=args.max_candidates,
    )
    result = run_sweep(
        model_name=args.model,
        batch_sizes=batch_sizes,
        output_root=Path(args.output_dir),
        checkpoint_config=config,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        image_size=args.image_size,
        debug_bert=args.debug_bert,
    )
    print(json.dumps(asdict(result), indent=2))
