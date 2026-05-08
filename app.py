from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st


OUTPUT_ROOT = Path("outputs/final_runs_multi")
MODEL_ROOTS = {
    "ResNet-152": OUTPUT_ROOT / "ResNet-152",
    "BERT": OUTPUT_ROOT / "BERT",
}


st.set_page_config(page_title="Activation Checkpointing Artifact Inspector", layout="wide")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mb(value: float | int | str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value) / (1024**2)


def _fmt_mb(value: float | int | str | None) -> str:
    return f"{_mb(value):.2f} MB"


def _batch_sizes(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(row["batch_size"]) for row in rows})


def _row_for(rows: list[dict[str, str]], batch_size: int, use_ac: bool) -> dict[str, str]:
    wanted = "True" if use_ac else "False"
    return next(
        row
        for row in rows
        if int(row["batch_size"]) == batch_size
        and row["use_activation_checkpointing"] == wanted
    )


def _top_activations(summary: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    activations = sorted(
        summary["activations"], key=lambda item: item["size_bytes"], reverse=True
    )
    return [
        {
            "name": item["name"],
            "shape": item["shape"],
            "size_mb": round(_mb(item["size_bytes"]), 2),
            "create": item["create_index"],
            "last_fwd": item["last_forward_use_index"],
            "first_bwd": item["first_backward_use_index"],
            "first_backward_user": item["first_backward_user_name"],
            "target": item["source_target"],
        }
        for item in activations[:limit]
    ]


def _activation_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in summary["activations"]}


def _plan_rows(plan_payload: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = _activation_by_name(summary)
    rows = []
    for name in plan_payload["plan"]["recompute"]:
        item = by_name.get(name, {})
        rows.append(
            {
                "activation": name,
                "size_mb": round(_mb(item.get("size_bytes")), 2),
                "target": item.get("source_target"),
                "create": item.get("create_index"),
                "first_backward_user": item.get("first_backward_user_name"),
                "inserted_before": item.get("first_backward_user_name"),
                "recompute_cost_ms": round(float(item.get("recompute_cost_ms") or 0.0), 4),
            }
        )
    return rows


def _sweep_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    table = []
    for batch_size in _batch_sizes(rows):
        baseline = _row_for(rows, batch_size, False)
        ac = _row_for(rows, batch_size, True)
        base_peak = _mb(baseline["peak_memory_bytes_max"])
        ac_peak = _mb(ac["peak_memory_bytes_max"])
        reduction = ((base_peak - ac_peak) / base_peak * 100.0) if base_peak else 0.0
        table.append(
            {
                "batch_size": batch_size,
                "baseline_peak_mb": round(base_peak, 2),
                "ac_peak_mb": round(ac_peak, 2),
                "memory_reduction_pct": round(reduction, 2),
                "baseline_latency_ms": round(float(baseline["latency_ms_avg"]), 2),
                "ac_latency_ms": round(float(ac["latency_ms_avg"]), 2),
                "correctness": ac["correctness_ok"] or "N/A",
            }
        )
    return table


def _timeline_plot(summary: dict[str, Any], plan_payload: dict[str, Any]) -> plt.Figure:
    timeline = summary["timeline_breakdown"]
    indices = [item["index"] for item in timeline]
    activation_mb = [_mb(item["activation_bytes"]) for item in timeline]
    total_mb = [_mb(item["total_bytes"]) for item in timeline]
    boundary = summary["boundary"]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(indices, total_mb, label="Total traced memory", linewidth=2)
    ax.plot(indices, activation_mb, label="Activation memory", linewidth=2)
    ax.axvline(boundary["forward_end_index"], color="#8a8a8a", linestyle="--", linewidth=1)
    ax.axvline(boundary["backward_begin_index"], color="#b35a1f", linestyle="--", linewidth=1)
    ax.set_xlabel("Graph node index")
    ax.set_ylabel("Memory (MB)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    by_name = _activation_by_name(summary)
    for name in plan_payload["plan"]["recompute"][:8]:
        item = by_name.get(name)
        if item:
            ax.scatter(item["create_index"], _mb(item["size_bytes"]), s=18, color="#b33f62")
    fig.tight_layout()
    return fig


def _bar_plot(sweep: list[dict[str, Any]], metric_a: str, metric_b: str, ylabel: str) -> plt.Figure:
    labels = [str(row["batch_size"]) for row in sweep]
    base = [row[metric_a] for row in sweep]
    ac = [row[metric_b] for row in sweep]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar([pos - 0.18 for pos in x], base, width=0.36, label="Baseline", color="#365c8d")
    ax.bar([pos + 0.18 for pos in x], ac, width=0.36, label="AC", color="#b33f62")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def _require_outputs(root: Path) -> tuple[list[dict[str, str]], list[int]]:
    csv_path = root / "sweep_results.csv"
    if not csv_path.exists():
        st.error(f"Missing {csv_path}")
        st.stop()
    rows = _load_csv(csv_path)
    return rows, _batch_sizes(rows)


available_models = {
    name: root for name, root in MODEL_ROOTS.items() if (root / "sweep_results.csv").exists()
}
if not available_models:
    st.error(f"No model sweep artifacts found under {OUTPUT_ROOT}")
    st.stop()

st.title("Activation Checkpointing Artifacts")

with st.sidebar:
    selected_model = st.selectbox("Model", list(available_models))
    root = available_models[selected_model]
    rows, batch_sizes = _require_outputs(root)
    selected_batch = st.selectbox("Batch size", batch_sizes, index=min(2, len(batch_sizes) - 1))
    baseline_row = _row_for(rows, selected_batch, False)
    ac_row = _row_for(rows, selected_batch, True)

st.caption(f"{selected_model} artifacts from `{root}`")

baseline_summary = _load_json(Path(baseline_row["profiler_summary_path"]))
ac_summary = _load_json(Path(ac_row["profiler_summary_path"]))
plan_payload = _load_json(Path(ac_row["plan_path"]))
rewritten_path = Path(ac_row["rewritten_profiler_summary_path"])
rewritten_summary = _load_json(rewritten_path) if rewritten_path.exists() else None
plan = plan_payload["plan"]
sweep = _sweep_rows(rows)

tabs = st.tabs(["Profiler", "Checkpoint Plan", "Rewrite Results"])

with tabs[0]:
    cols = st.columns(5)
    cols[0].metric("Graph nodes", baseline_summary["metadata"]["total_nodes"])
    cols[1].metric("Activation candidates", len(baseline_summary["activations"]))
    cols[2].metric("Traced peak", _fmt_mb(baseline_summary["total_peak_bytes"]))
    cols[3].metric("Activation peak", _fmt_mb(baseline_summary["activation_peak_bytes"]))
    cols[4].metric("CUDA peak", _fmt_mb(baseline_row["peak_memory_bytes_max"]))

    left, right = st.columns([1, 1])
    with left:
        breakdown = Path(baseline_row["output_dir"]) / "peak_memory_breakdown.png"
        if breakdown.exists():
            st.image(str(breakdown), use_container_width=True)
    with right:
        st.dataframe(_top_activations(baseline_summary, 15), use_container_width=True, hide_index=True)

with tabs[1]:
    cols = st.columns(5)
    cols[0].metric("Recomputed", len(plan["recompute"]))
    cols[1].metric("Retained", len(plan["retain"]))
    cols[2].metric("Estimated saved", _fmt_mb(plan["estimated_saved_bytes"]))
    cols[3].metric("Estimated peak", _fmt_mb(plan["estimated_peak_bytes"]))
    cols[4].metric("Candidate limit", plan["metadata"].get("max_candidates", "N/A"))

    st.pyplot(_timeline_plot(baseline_summary, plan_payload), use_container_width=True)
    st.dataframe(_plan_rows(plan_payload, baseline_summary), use_container_width=True, hide_index=True)

with tabs[2]:
    base_peak = _mb(baseline_row["peak_memory_bytes_max"])
    ac_peak = _mb(ac_row["peak_memory_bytes_max"])
    reduction = ((base_peak - ac_peak) / base_peak * 100.0) if base_peak else 0.0

    cols = st.columns(5)
    cols[0].metric("Correctness", ac_row["correctness_ok"] or "N/A")
    cols[1].metric("Baseline peak", f"{base_peak:.2f} MB")
    cols[2].metric("AC peak", f"{ac_peak:.2f} MB")
    cols[3].metric("Reduction", f"{reduction:.2f}%")
    cols[4].metric("AC latency", f"{float(ac_row['latency_ms_avg']):.2f} ms")

    left, right = st.columns([1, 1])
    with left:
        st.pyplot(
            _bar_plot(sweep, "baseline_peak_mb", "ac_peak_mb", "Peak memory (MB)"),
            use_container_width=True,
        )
    with right:
        st.pyplot(
            _bar_plot(sweep, "baseline_latency_ms", "ac_latency_ms", "Latency (ms)"),
            use_container_width=True,
        )

    stats = [
        {
            "graph": "baseline",
            "nodes": baseline_summary["metadata"]["total_nodes"],
            "activation_candidates": len(baseline_summary["activations"]),
            "traced_peak_mb": round(_mb(baseline_summary["total_peak_bytes"]), 2),
        },
        {
            "graph": "rewritten",
            "nodes": rewritten_summary["metadata"]["total_nodes"] if rewritten_summary else None,
            "activation_candidates": len(rewritten_summary["activations"]) if rewritten_summary else None,
            "traced_peak_mb": round(_mb(rewritten_summary["total_peak_bytes"]), 2)
            if rewritten_summary
            else None,
        },
    ]
    st.dataframe(stats, use_container_width=True, hide_index=True)
