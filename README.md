# CS265 Activation Checkpointing Project

This repository extends the original Harvard CS265 starter scaffold into an end-to-end activation checkpointing prototype built on top of the provided FX tracing flow.

The implementation keeps the skeleton architecture intact:

- [graph_tracer.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/graph_tracer.py) still captures a single-iteration graph spanning forward, backward, and optimizer work.
- [graph_prof.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/graph_prof.py) now implements static analysis, node-level profiling, and peak-memory breakdown summaries.
- [activation_checkpoint.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/activation_checkpoint.py) now implements a profiler-driven checkpoint selection policy and a generalized subgraph rewrite pass.
- [benchmarks.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/benchmarks.py) runs baseline vs activation-checkpointed experiments for `ResNet-152` and `BERT`.
- [app.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/app.py) provides a Streamlit demo frontend.

## What Was Implemented

### Phase 1: Graph profiler

- Forward/backward/optimizer boundary detection using the existing separator ops.
- Static activation lifetime analysis:
  - creation point,
  - last use in forward,
  - first use in backward,
  - tensor size estimate.
- Placeholder categorization for:
  - parameters,
  - buffers,
  - optimizer state,
  - gradients,
  - activations,
  - other graph values.
- Runtime per-node measurements:
  - average node latency,
  - output tensor sizes,
  - CUDA memory snapshots when running on GPU.
- JSON export of profiler summaries and peak-memory breakdown data.

### Phase 2: Activation checkpointing policy

- A practical heuristic inspired by `μ-TWO`:
  - ranks activations by memory-saved / recompute-cost,
  - supports a memory budget,
  - supports a recompute budget ratio,
  - supports a minimum activation-size threshold.
- Produces a retained-vs-recomputed activation plan for inspection and demo.

### Phase 3: Graph extraction and rewrite

- Generalized recomputation rewrite using `_extract_graph_with_inputs_outputs`.
- Clones forward subgraphs and reinserts them immediately before the first backward-time use of a discarded activation.
- Replaces backward-region uses with the recomputed value while preserving forward correctness.
- Includes numerical equivalence checks for the toy validation path.

### Benchmarking and demo

- Baseline vs activation-checkpointing benchmark runs for:
  - `ResNet-152`
  - `BERT`
- Saves:
  - profiler summaries,
  - checkpoint plans,
  - peak memory breakdown plots,
  - sweep CSVs,
  - peak memory vs batch size plots,
  - latency vs batch size plots.
- Streamlit frontend for interactive demo runs and visualization.

## Environment Setup

The project assumes Python 3.12 and a CUDA-capable PyTorch installation for the full benchmark/demo flow.

### Conda environment

```powershell
conda create -n cs265 python=3.12
conda activate cs265
pip install numpy==2.2.2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers matplotlib streamlit expecttest
```

If you already used the setup work in this repository session, the `cs265` environment may already exist and already have the GPU-enabled PyTorch stack installed.

## Quick Start

### Starter flow

Run the small dummy-model example:

```powershell
conda activate cs265
python starter_code.py
```

Run the same starter flow with activation checkpointing enabled:

```powershell
python starter_code.py --use-ac --output-dir outputs/starter_ac
```

### Validation tests

```powershell
python -m unittest tests.test_graph_profiler tests.test_activation_checkpoint
```

## Running Benchmarks

### ResNet-152

```powershell
python benchmarks.py --model "ResNet-152" --batch-sizes 1 2 4
```

### BERT

```powershell
python benchmarks.py --model "BERT" --batch-sizes 1 2 4 --seq-len 128
```

For faster dry runs on a smaller BERT variant while keeping the benchmark path and UI aligned to the assignment:

```powershell
python benchmarks.py --model "BERT" --batch-sizes 1 2 --seq-len 128 --debug-bert
```

### Tuning the checkpoint policy

```powershell
python benchmarks.py --model "ResNet-152" --batch-sizes 1 2 4 --memory-budget-mb 6000 --max-recompute-ratio 0.25
```

## Launching the Frontend Demo

```powershell
streamlit run app.py
```

The app supports:

- model selection,
- batch size selection,
- activation checkpointing toggle,
- checkpoint policy tuning,
- viewing profiler summaries,
- viewing retained vs recomputed activations,
- viewing saved peak-memory plots and sweep plots.

## Outputs

Benchmark and demo artifacts are saved under [outputs](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/outputs).

Typical saved files include:

- `profiler_summary.json`
- `rewritten_profiler_summary.json`
- `checkpoint_plan.json`
- `peak_memory_breakdown.png`
- `sweep_results.csv`
- `peak_memory_vs_batch_size.png`
- `latency_vs_batch_size.png`
- `manifest.json`

These are intended to support:

- the written report,
- demo walkthroughs,
- code review,
- reproducibility.

## Assumptions and Limitations

- Full benchmark and demo flows assume CUDA for meaningful memory measurements.
- The `BERT` benchmark path uses a randomly initialized BERT-style masked language model created from configuration, not a downloaded pretrained checkpoint.
- The `--debug-bert` option uses a reduced BERT configuration for quicker validation and demo iteration.
- The checkpoint policy is heuristic and profiler-driven rather than a paper-perfect reproduction of `μ-TWO`.
- The profiler’s memory breakdown focuses on parameter, gradient, optimizer-state, and activation memory visible through the traced iteration and static lifetime analysis.
- On CPU, the profiler still runs, but timing and memory results are not representative of the intended project target.

## Repository Guide

- [graph_tracer.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/graph_tracer.py): FX capture of one training iteration.
- [graph_prof.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/graph_prof.py): graph profiling, activation lifetime analysis, summary export.
- [activation_checkpoint.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/activation_checkpoint.py): checkpoint selection and graph rewrite.
- [benchmarks.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/benchmarks.py): benchmark runner and plot generation.
- [starter_code.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/starter_code.py): lightweight starter example.
- [app.py](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/app.py): Streamlit demo frontend.
- [tests](C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/tests): validation tests for profiler and rewrite correctness.
