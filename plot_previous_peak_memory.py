from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from benchmarks import RunResult, _plot_sweep


OUTPUT_ROOT = Path("outputs/final_runs_multi")
MODELS = ("ResNet-152", "BERT")


def _bool_from_csv(value: str) -> bool:
    return value == "True"


def _optional_bool_from_csv(value: str) -> bool | None:
    if not value:
        return None
    return _bool_from_csv(value)


def _load_runs(model_name: str) -> list[RunResult]:
    csv_path = OUTPUT_ROOT / model_name / "sweep_results.csv"
    runs: list[RunResult] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            runs.append(
                RunResult(
                    model_name=row["model_name"],
                    batch_size=int(row["batch_size"]),
                    use_activation_checkpointing=_bool_from_csv(
                        row["use_activation_checkpointing"]
                    ),
                    latency_ms_avg=float(row["latency_ms_avg"]),
                    latency_ms_std=float(row["latency_ms_std"]),
                    peak_memory_bytes_avg=float(row["peak_memory_bytes_avg"]),
                    peak_memory_bytes_max=float(row["peak_memory_bytes_max"]),
                    correctness_ok=_optional_bool_from_csv(row["correctness_ok"]),
                    output_dir=row["output_dir"],
                    profiler_summary_path=row["profiler_summary_path"] or None,
                    profiler_breakdown_plot=None,
                    plan_path=row["plan_path"] or None,
                    rewritten_profiler_summary_path=(
                        row["rewritten_profiler_summary_path"] or None
                    ),
                    metadata={},
                )
            )
    return runs


def _plot_combined(all_runs: dict[str, list[RunResult]]) -> Path:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 4))
    for ax, model_name in zip(axes, MODELS):
        runs = all_runs[model_name]
        batch_sizes = sorted({run.batch_size for run in runs})
        baseline = {
            run.batch_size: run.peak_memory_bytes_max / (1024**2)
            for run in runs
            if not run.use_activation_checkpointing
        }
        checkpointed = {
            run.batch_size: run.peak_memory_bytes_max / (1024**2)
            for run in runs
            if run.use_activation_checkpointing
        }
        ax.plot(
            batch_sizes,
            [baseline[batch_size] for batch_size in batch_sizes],
            marker="o",
            linewidth=2,
            label="Baseline",
        )
        ax.plot(
            batch_sizes,
            [checkpointed[batch_size] for batch_size in batch_sizes],
            marker="o",
            linewidth=2,
            label="Activation checkpointing",
        )
        ax.set_title(model_name)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Peak GPU memory (MB)")
        ax.set_xticks(batch_sizes)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("Peak Memory vs Batch Size from Previous Experiment Run")
    fig.tight_layout()
    output_path = OUTPUT_ROOT / "peak_memory_vs_batch_size_combined_from_csv.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    all_runs = {model_name: _load_runs(model_name) for model_name in MODELS}
    output_paths = []
    for model_name, runs in all_runs.items():
        output_paths.append(
            _plot_sweep(
                runs,
                metric="peak_memory_bytes_max",
                ylabel="Peak GPU memory (MB)",
                title=f"Peak Memory vs Batch Size: {model_name}",
                output_path=OUTPUT_ROOT
                / model_name
                / "peak_memory_vs_batch_size_from_csv.png",
            )
        )
    output_paths.append(_plot_combined(all_runs))
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
