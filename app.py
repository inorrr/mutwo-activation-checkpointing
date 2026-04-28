from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from benchmarks import DEFAULT_OUTPUT_DIR, Experiment, run_sweep
from activation_checkpoint import ActivationCheckpointConfig


st.set_page_config(page_title="CS265 Activation Checkpointing Demo", layout="wide")


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _render_profiler_summary(summary_path: str) -> None:
    summary = _load_json(summary_path)
    st.subheader("Profiler Summary")
    cols = st.columns(4)
    cols[0].metric("Peak total memory (MB)", round(summary["total_peak_bytes"] / (1024**2), 2))
    cols[1].metric(
        "Peak activation memory (MB)",
        round(summary["activation_peak_bytes"] / (1024**2), 2),
    )
    cols[2].metric("Activation candidates", len(summary["activations"]))
    cols[3].metric("Graph nodes", summary["metadata"]["total_nodes"])

    top_activations = sorted(
        summary["activations"], key=lambda item: item["size_bytes"], reverse=True
    )[:15]
    st.write("Largest activation lifetimes")
    st.dataframe(top_activations, use_container_width=True)


def _render_plan(plan_path: str | None) -> None:
    if not plan_path:
        return
    payload = _load_json(plan_path)
    plan = payload["plan"]
    st.subheader("Checkpoint Plan")
    st.write(
        {
            "recompute_count": len(plan["recompute"]),
            "retain_count": len(plan["retain"]),
            "estimated_saved_mb": round(plan["estimated_saved_bytes"] / (1024**2), 2),
        }
    )
    st.write("Recomputed activations")
    st.dataframe([{"name": name} for name in plan["recompute"]], use_container_width=True)


def _render_images(paths: list[str]) -> None:
    for path in paths:
        if path and Path(path).exists():
            st.image(str(path), use_container_width=True)


st.title("CS265 Activation Checkpointing Demo")
st.caption(
    "Run baseline and activation-checkpointed experiments on ResNet-152 or BERT and inspect profiler outputs."
)

with st.sidebar:
    model_name = st.selectbox("Model", Experiment.model_names)
    batch_size = st.slider("Batch size", min_value=1, max_value=8, value=2)
    seq_len = st.slider("BERT sequence length", min_value=32, max_value=256, value=128, step=32)
    use_ac = st.toggle("Enable activation checkpointing", value=True)
    debug_bert = st.toggle("Use reduced BERT for quick demos", value=True)
    memory_budget_mb = st.number_input(
        "Memory budget (MB, optional)", min_value=0.0, value=0.0, step=128.0
    )
    min_savings_mb = st.number_input("Minimum activation savings (MB)", min_value=0.0, value=1.0)
    max_recompute_ratio = st.slider("Max recompute ratio", 0.05, 1.0, 0.35, 0.05)
    run_button = st.button("Run experiment", type="primary")


if run_button:
    config = ActivationCheckpointConfig(
        memory_budget_mb=memory_budget_mb if memory_budget_mb > 0 else None,
        min_savings_mb=min_savings_mb,
        max_recompute_ratio=max_recompute_ratio,
    )
    experiment = Experiment(
        model_name=model_name,
        batch_size=batch_size,
        output_root=DEFAULT_OUTPUT_DIR,
        seq_len=seq_len,
        debug_bert=debug_bert,
    )
    with st.spinner("Running experiment..."):
        result = experiment.run_once(
            use_activation_checkpointing=use_ac,
            checkpoint_config=config,
        )

    cols = st.columns(4)
    cols[0].metric("Latency (ms)", round(result.latency_ms_avg, 2))
    cols[1].metric("Peak memory (MB)", round(result.peak_memory_bytes_max / (1024**2), 2))
    cols[2].metric("Mode", "AC" if result.use_activation_checkpointing else "Baseline")
    cols[3].metric(
        "Correctness",
        "OK" if result.correctness_ok in (None, True) else "Mismatch",
    )

    _render_profiler_summary(result.profiler_summary_path)
    _render_plan(result.plan_path)
    _render_images(
        [
            result.profiler_breakdown_plot,
        ]
    )

st.divider()
st.subheader("Saved Sweeps")
manifest_paths = sorted(DEFAULT_OUTPUT_DIR.glob("*/manifest.json"))
if not manifest_paths:
    st.info("No sweep manifests found yet. Run `python benchmarks.py ...` to generate them.")
else:
    selected_manifest = st.selectbox(
        "Available benchmark manifests",
        manifest_paths,
        format_func=lambda path: str(path.parent.name),
    )
    manifest = _load_json(selected_manifest)
    st.write({"model": manifest["model_name"], "batch_sizes": manifest["batch_sizes"]})
    st.dataframe(manifest["runs"], use_container_width=True)
    _render_images([manifest["memory_plot"], manifest["latency_plot"]])
