# Experimental Analysis

This document summarizes the current experimental artifacts saved under [outputs/final_runs_multi](/C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/outputs/final_runs_multi).

The final saved sweeps use:

- `ResNet-152`, mini-batch sizes `1`, `2`, `4`, `6`, and `8`, input image size `320 x 320`
- `BERT`, mini-batch sizes `1`, `2`, and `4`, sequence length `512`, vocabulary size `4096`, full non-debug BERT configuration

The plots, CSV files, profiler summaries, checkpoint plans, and rewritten-profiler summaries were generated for these settings. Activation-checkpointed measurements execute the custom FX graph rewrite path.

## Design Choices

The project uses the PyTorch FX workflow for graph capture, static activation-lifetime analysis, checkpoint-plan export, graph rewriting, and benchmark measurement. Activation-checkpointed benchmark runs execute the custom rewritten FX graph produced by the project extractor/rewriter rather than delegating to framework-level checkpointing helpers.

The final implementation uses:

- FX profiling and checkpoint planning for inspection artifacts.
- Custom FX subgraph extraction and rewriting for activation-checkpointed execution.
- The same compiled measurement path for baseline and activation-checkpointed runs.
- A checkpoint policy with `0.25 MB` minimum estimated savings per activation, a recompute budget up to `1.0x` measured forward-pass time, and a candidate limit tuned per model in the final runs.
- CUDA cache/peak-stat cleanup before measurement.
- Profiler cleanup so `GraphProfiler` does not retain runtime tensors after profiling.

The custom checkpoint planner uses a simplified, recompute-only version of the Mutwo policy from the MLSys 2023 paper. The full Mutwo system combines multi-model scheduling, activation swapping, and activation recomputation. This project only rewrites recomputation candidates, so the implemented subset does the following:

- Builds recomputation candidates from profiled activation size, lifetime, and estimated recompute cost.
- Models the inactive interval between an activation's last forward use and its first backward use.
- Scores recomputation candidates by memory saved per recompute cost, with inactive time and size as tie breakers.
- Prioritizes activations that overlap the modeled peak live set.
- Iteratively simulates total peak memory after each recompute decision and stops when the configured memory budget, candidate limit, or recompute-cost limit is reached.

The planner avoids spending checkpoint slots on view-like or alias-like candidates such as `t`, `view`, `transpose`, `expand`, and `getitem`. Those tensors can appear large from shape alone, but often do not own new CUDA storage, so recomputing them rarely reduces allocator peak memory.

For BERT, the benchmark uses `vocab_size=4096`. This keeps the language-modeling head from dominating memory and makes transformer activations easier to inspect. The benchmark also uses `seq_len=512`, which makes transformer activations large enough for checkpointing effects to be visible at larger batch sizes.

### Implementation Limitations

The implementation intentionally stays close to the course project requirement: it profiles an FX graph, chooses activation tensors, extracts the subgraphs that produce those tensors, and inserts recomputation nodes into the backward region. It checkpoints selected graph values rather than wrapping whole modules.

The main limitations are:

- The planner checkpoints individual FX activation values, not whole ResNet residual blocks or BERT transformer layers. This gives fine-grained validation of the graph rewrite, but it saves less allocator-level memory than block-level checkpointing.
- The peak-memory model is static and tensor-lifetime based. Actual CUDA peak memory also includes parameters, gradients, optimizer state, temporary kernel workspaces, cloned recomputation outputs, allocator fragmentation, and PyTorch caching behavior.
- The implementation is recompute-only. It does not implement Mutwo's broader activation swapping/offloading or multi-job scheduling components.
- The rewrite preserves correctness but may duplicate small producer chains many times. More aggressive candidate counts can improve savings, but they can also increase temporary memory and execution overhead.
- BERT batch sizes `6` and `8` were not included in the final BERT sweep because aggressive custom-FX rewrites at those settings can exceed the available 8 GB GPU during profiling of the rewritten graph. The saved BERT sweep therefore uses batch sizes `1`, `2`, and `4`.
- The benchmark uses only a few timed iterations, so latency should be treated as a coarse signal rather than a stable microbenchmark.

## A. Computation And Memory Profiling Statistics And Static Analysis

### ResNet-152

Runtime summary at batch size `1`:

| Mode | Avg latency (ms) | Peak GPU memory (MB) | Correctness check |
| --- | ---: | ---: | --- |
| Baseline | 275.97 | 1196.37 | N/A |
| With AC | 318.15 | 1206.87 | Passed |

Static memory breakdown from the batch-size-1 baseline profiler summary:

| Component | Peak memory (MB) |
| --- | ---: |
| Parameters | 229.62 |
| Gradients | 229.62 |
| Optimizer state | 0.00 |
| Activations | 352.54 |
| Total traced peak | 811.78 |

Static-analysis summary:

- Activation candidates identified: `778`
- The largest apparent candidate was `t` with shape `[2048, 1000]` and size `7.81 MB`
- The largest materialized feature-map candidates include tensors with shape `[1, 256, 80, 80]` and size `6.25 MB`
- The simplified Mutwo recompute policy selected `16` activations for recomputation in the batch-size-1 saved plan.
- Estimated saved activation bytes from the batch-size-1 plan: approximately `75.00 MB`
- Estimated static traced peak after the recompute plan: approximately `736.78 MB`

Largest baseline activation candidates:

| Activation | Shape | Size (MB) | First backward user |
| --- | --- | ---: | --- |
| `t` | `[2048, 1000]` | 7.81 | `t_1` |
| `relu__9` | `[1, 256, 80, 80]` | 6.25 | `convolution_backward_140` |
| `convolution_3` | `[1, 256, 80, 80]` | 6.25 | `cudnn_batch_norm_backward_151` |
| `convolution_7` | `[1, 256, 80, 80]` | 6.25 | `cudnn_batch_norm_backward_147` |
| `convolution_4` | `[1, 256, 80, 80]` | 6.25 | `cudnn_batch_norm_backward_150` |

Peak-memory breakdown plot:

![ResNet-152 baseline peak memory breakdown](../outputs/final_runs_multi/ResNet-152/bs_1/baseline/peak_memory_breakdown.png)

### BERT

Runtime summary at batch size `1`:

| Mode | Avg latency (ms) | Peak GPU memory (MB) | Correctness check |
| --- | ---: | ---: | --- |
| Baseline | 177.23 | 1718.10 | N/A |
| With AC | 164.43 | 1718.60 | Passed |

Static memory breakdown from the batch-size-1 baseline profiler summary:

| Component | Peak memory (MB) |
| --- | ---: |
| Parameters | 352.26 |
| Gradients | 352.26 |
| Optimizer state | 0.00 |
| Activations | 849.52 |
| Total traced peak | 1554.04 |

Static-analysis summary:

- Activation candidates identified: `366`
- The largest apparent candidates are expanded attention masks with shape `[1, 12, 512, 512]` and size `12.00 MB`
- Several projection matrices with shapes `[768, 4096]`, `[768, 3072]`, and `[3072, 768]` also appear as large recomputable candidates
- The batch-size-1 plan selected `16` activations for recomputation.
- Estimated saved activation bytes from the batch-size-1 plan: approximately `30.50 MB`
- Estimated static traced peak after the recompute plan: approximately `1531.54 MB`

Largest baseline activation candidates:

| Activation | Shape | Size (MB) | First backward user |
| --- | --- | ---: | --- |
| `expand_9` | `[1, 12, 512, 512]` | 12.00 | `_scaled_dot_product_efficient_attention_backward_5` |
| `expand_12` | `[1, 12, 512, 512]` | 12.00 | `_scaled_dot_product_efficient_attention_backward_2` |
| `expand_11` | `[1, 12, 512, 512]` | 12.00 | `_scaled_dot_product_efficient_attention_backward_3` |
| `expand_5` | `[1, 12, 512, 512]` | 12.00 | `_scaled_dot_product_efficient_attention_backward_9` |
| `expand_3` | `[1, 12, 512, 512]` | 12.00 | `_scaled_dot_product_efficient_attention_backward_11` |

Peak-memory breakdown plot:

![BERT baseline peak memory breakdown](../outputs/final_runs_multi/BERT/bs_1/baseline/peak_memory_breakdown.png)

### Interpretation

- The FX profiler continues to separate forward, backward, and optimizer regions and produce activation-lifetime data for checkpoint planning.
- Runtime measurements execute the custom rewritten FX graph for AC runs.
- ResNet shows visible peak-memory reductions for batch sizes `2`, `4`, `6`, and `8`; batch size `1` is slightly above baseline due to recompute-node overhead and allocator effects.
- BERT shows a clear reduction at batch size `4`, but batch sizes `1` and `2` are close to flat or slightly above baseline.
- Correctness checks passed for all AC-enabled saved runs.

### Why Peak-Memory Reduction Is Modest

The peak-memory reductions are real but modest because this project checkpoints selected FX activation tensors rather than entire high-level modules.

For ResNet, the selected tensors are mostly convolution and ReLU feature maps. Those feature maps grow with image and batch size, so the memory savings become visible once batch size reaches `2`. The batch-size-4 plan, for example, recomputes `16` activations and estimates about `206.25 MB` of activation savings. However, the measured CUDA peak only falls by about `9.92%` because parameters, gradients, optimizer state, temporary convolution workspaces, and allocator behavior still contribute to the peak.

For BERT, the reduction is smaller for several reasons:

- BERT has large fixed model, gradient, optimizer, and language-model-head memory. At batch sizes `1` and `2`, that fixed memory dominates the allocator peak, so removing selected activations barely changes total peak memory.
- Many apparent large tensors in BERT are view-like or alias-like attention-mask tensors such as `expand`. The planner intentionally skips those because recomputing them usually does not release real CUDA storage.
- The selected BERT activations are often elementwise outputs such as `add`, `gelu`, and `_log_softmax`. They are correct recomputation targets, but they do not capture the whole transformer-layer activation footprint the way module-level checkpointing would.
- Recomputed subgraphs can introduce temporary tensors during backward. At small batch sizes, those recomputation temporaries and allocator effects can offset the saved retained activations.

This is why BERT batch size `4` shows a meaningful reduction (`4.44%`), while batch sizes `1` and `2` do not.

## B. Peak Memory Consumption Vs Mini-Batch Size Bar Graph

ResNet-152 sweep bar graph:

![ResNet-152 peak memory vs batch size](../outputs/final_runs_multi/ResNet-152/peak_memory_vs_batch_size.png)

BERT sweep bar graph:

![BERT peak memory vs batch size](../outputs/final_runs_multi/BERT/peak_memory_vs_batch_size.png)

The peak-memory plots were also generated directly from the saved CSV files:

![Combined peak memory vs batch size from CSV](../outputs/final_runs_multi/peak_memory_vs_batch_size_combined_from_csv.png)

Observed values from the saved multi-batch runs:

| Model | Batch size | Baseline peak memory (MB) | AC peak memory (MB) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| ResNet-152 | 1 | 1196.37 | 1206.87 | -0.88% |
| ResNet-152 | 2 | 1470.85 | 1357.47 | 7.71% |
| ResNet-152 | 4 | 2173.58 | 1957.96 | 9.92% |
| ResNet-152 | 6 | 2853.26 | 2660.66 | 6.75% |
| ResNet-152 | 8 | 3546.12 | 3252.21 | 8.29% |
| BERT | 1 | 1718.10 | 1718.60 | -0.03% |
| BERT | 2 | 1733.62 | 1745.62 | -0.69% |
| BERT | 4 | 2419.96 | 2312.46 | 4.44% |

Interpretation:

- ResNet shows meaningful memory savings once batch size is at least `2`, with reductions between `6.75%` and `9.92%` in the custom-FX runs.
- BERT remains close to flat at smaller batch sizes: batch size `4` shows a clear reduction, while `1` and `2` are slightly above baseline.
- The small negative reductions are within the range of fixed overhead and allocator behavior for this custom graph-execution path; the profiler still records recomputation plans and all AC correctness checks pass.

## C. Iteration Latency Vs Mini-Batch Size

ResNet-152 latency plot:

![ResNet-152 latency vs batch size](../outputs/final_runs_multi/ResNet-152/latency_vs_batch_size.png)

BERT latency plot:

![BERT latency vs batch size](../outputs/final_runs_multi/BERT/latency_vs_batch_size.png)

Observed values from the saved multi-batch runs:

| Model | Batch size | Baseline latency (ms) | AC latency (ms) |
| --- | ---: | ---: | ---: |
| ResNet-152 | 1 | 275.97 | 318.15 |
| ResNet-152 | 2 | 368.47 | 474.47 |
| ResNet-152 | 4 | 350.66 | 322.27 |
| ResNet-152 | 6 | 450.58 | 410.08 |
| ResNet-152 | 8 | 439.32 | 439.33 |
| BERT | 1 | 177.23 | 164.43 |
| BERT | 2 | 251.02 | 265.41 |
| BERT | 4 | 388.00 | 375.07 |

Interpretation:

- Activation checkpointing trades compute for memory by recomputing selected activations during backward.
- ResNet pays visible overhead at batch sizes `1` and `2`, is roughly tied at `8`, and is faster at `4` and `6` in this run; those faster AC points should be treated as measurement noise or allocator/cache effects rather than a general checkpointing speedup.
- BERT latency is also mixed: AC is faster at batch sizes `1` and `4`, but slower at `2`.
- Longer runs with more iterations would give smoother latency estimates, but the plots are sufficient to show the measured behavior of the custom FX rewrite path.

### Why AC Latency Is Not Consistently Higher

In principle, activation checkpointing should add compute because discarded activations must be recomputed during backward. The measured latency is not consistently higher because the benchmark is not a controlled kernel-level timing study:

- Only three timed iterations are used per run, so run-to-run variance can be comparable to the recompute overhead.
- CUDA kernel selection, warmup behavior, allocator cache state, and temporary workspace reuse can shift timings by tens of milliseconds.
- The baseline and rewritten graphs can have different memory pressure. In some runs, lower memory pressure or different allocation order may offset some recompute cost.
- The FX interpreter path and cloned graph structure introduce overhead that is not identical across baseline and AC graphs.
- GPU background activity and asynchronous CUDA execution can add noise even with explicit synchronization around measured iterations.

Therefore, the latency results should be read as showing that the custom AC path remains in the same broad performance regime, not as evidence that checkpointing generally accelerates training.

## Reproduction

The plots and summaries were generated from:

```powershell
conda run -n cs265 python benchmarks.py --model "ResNet-152" --batch-sizes 1 2 4 6 8 --image-size 320 --output-dir outputs/final_runs_multi
conda run -n cs265 python benchmarks.py --model "BERT" --batch-sizes 1 2 4 --seq-len 512 --vocab-size 4096 --max-candidates 16 --output-dir outputs/final_runs_multi
conda run -n cs265 python plot_previous_peak_memory.py
```

The validation tests were run with:

```powershell
conda run -n cs265 python -m unittest tests.test_graph_profiler tests.test_activation_checkpoint tests.test_graph_tracer
```
