# Experimental Analysis

This document summarizes the experimental artifacts currently saved under [outputs/final_runs_multi](/C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/outputs/final_runs_multi).

The current saved runs are multi-batch experiments for:

- `ResNet-152`, batch sizes `1` and `2`
- `BERT`, batch sizes `1`, `2`, and `4`, sequence length `64`, `--debug-bert`

These artifacts include the requested profiler summaries and plots, with more than one batch size in the batch-sweep graphs.

## A. Computation And Memory Profiling Statistics And Static Analysis

### ResNet-152

Runtime summary:

| Mode | Avg latency (ms) | Peak GPU memory (MB) | Correctness check |
| --- | ---: | ---: | --- |
| Baseline | 200.36 | 3502.15 | N/A |
| With AC | 186.46 | 3507.19 | Passed |

Static memory breakdown from baseline profiler summary:

| Component | Peak memory (MB) |
| --- | ---: |
| Parameters | 229.62 |
| Gradients | 229.62 |
| Optimizer state | 0.00 |
| Activations | 7.82 |
| Total traced peak | 467.05 |

Static-analysis summary:

- Activation candidates identified: `778`
- The largest single candidate was `t` with shape `[2048, 1000]` and size `7.81 MB`
- Several early convolution / ReLU tensors at shape `[1, 256, 56, 56]` were among the largest reusable activations
- The AC policy selected `4` activations for recomputation in the saved run:
  - `t`
  - `relu_`
  - `convolution_4`
  - `convolution_7`
- Estimated saved activation bytes from the plan: approximately `7.72 MB`

Largest baseline activation candidates:

| Activation | Shape | Size (MB) | First backward user |
| --- | --- | ---: | --- |
| `t` | `[2048, 1000]` | 7.81 | `t_1` |
| `convolution` | `[1, 64, 112, 112]` | 3.06 | `cudnn_batch_norm_backward_154` |
| `relu_` | `[1, 64, 112, 112]` | 3.06 | `max_pool2d_with_indices_backward` |
| `convolution_3` | `[1, 256, 56, 56]` | 3.06 | `cudnn_batch_norm_backward_151` |
| `convolution_4` | `[1, 256, 56, 56]` | 3.06 | `cudnn_batch_norm_backward_150` |

Peak-memory breakdown plot:

![ResNet-152 baseline peak memory breakdown](/C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/outputs/final_runs_multi/ResNet-152/bs_1/baseline/peak_memory_breakdown.png)

### BERT

Runtime summary:

| Mode | Avg latency (ms) | Peak GPU memory (MB) | Correctness check |
| --- | ---: | ---: | --- |
| Baseline | 61.01 | 1692.41 | N/A |
| With AC | 52.73 | 1692.81 | Passed |

Static memory breakdown from baseline profiler summary:

| Component | Peak memory (MB) |
| --- | ---: |
| Parameters | 131.03 |
| Gradients | 131.03 |
| Optimizer state | 0.00 |
| Activations | 44.80 |
| Total traced peak | 306.86 |

Static-analysis summary:

- Activation candidates identified: `192`
- The largest single candidate was `t_37` with shape `[384, 30522]` and size `44.71 MB`
- Several intermediate projection matrices with shapes `[384, 1536]` and `[1536, 384]` appeared repeatedly as large recomputable tensors
- The AC policy selected `4` activations for recomputation in the saved run:
  - `t_37`
  - `t_34`
  - `t_35`
  - `t_29`
- Estimated saved activation bytes from the plan: approximately `51.46 MB`

Largest baseline activation candidates:

| Activation | Shape | Size (MB) | First backward user |
| --- | --- | ---: | --- |
| `t_37` | `[384, 30522]` | 44.71 | `t_38` |
| `_log_softmax` | `[64, 30522]` | 7.45 | `nll_loss_backward` |
| `t_4` | `[384, 1536]` | 2.25 | `t_170` |
| `t_5` | `[1536, 384]` | 2.25 | `t_166` |
| `t_10` | `[384, 1536]` | 2.25 | `t_146` |

Peak-memory breakdown plot:

![BERT baseline peak memory breakdown](/C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/outputs/final_runs_multi/BERT/bs_1/baseline/peak_memory_breakdown.png)

### Interpretation

- In both saved smoke runs, the profiler correctly separated forward, backward, and optimizer regions and produced activation lifetime data suitable for checkpoint planning.
- The BERT path showed a much larger single activation target than ResNet-152, which is consistent with the large output-projection matrix and log-softmax path near the language-modeling head.
- The AC planner behaved conservatively because it was capped to a small number of recomputation candidates for demo stability on this device.
- The correctness checks passed for both AC-enabled runs, so the graph rewrite is functioning as intended on the saved smoke experiments.

## B. Peak Memory Consumption Vs Mini-Batch Size Bar Graph (With And Without AC)

ResNet-152 sweep plot:

![ResNet-152 peak memory vs batch size](/C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/outputs/final_runs_multi/ResNet-152/peak_memory_vs_batch_size.png)

BERT sweep plot:

![BERT peak memory vs batch size](/C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/outputs/final_runs_multi/BERT/peak_memory_vs_batch_size.png)

Observed values from the current saved multi-batch runs:

| Model | Batch size | Baseline peak memory (MB) | AC peak memory (MB) |
| --- | ---: | ---: | ---: |
| ResNet-152 | 1 | 3502.15 | 3507.19 |
| ResNet-152 | 2 | 4014.65 | 4013.93 |
| BERT | 1 | 1692.41 | 1692.81 |
| BERT | 2 | 1761.91 | 1762.41 |
| BERT | 4 | 1899.30 | 1899.19 |

Interpretation:

- The plots now contain multiple measured batch sizes for both model families.
- On ResNet-152, the saved AC run is slightly lower than baseline at batch size `2`, but slightly higher at batch size `1`.
- On BERT, the memory deltas remain very small across the saved runs.
- Allocator-level `torch.cuda.max_memory_allocated()` still does not show a strong reduction from AC yet, even though the static activation analysis and rewrite plan were produced successfully.
- The static profiler output is still useful for identifying the largest activation candidates and demonstrating the selection/recompute pipeline.

## C. Iteration Latency Vs Mini-Batch Size Performance Graph (With And Without AC)

ResNet-152 latency plot:

![ResNet-152 latency vs batch size](/C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/outputs/final_runs_multi/ResNet-152/latency_vs_batch_size.png)

BERT latency plot:

![BERT latency vs batch size](/C:/Users/inorz/OneDrive/Documents/Harvard/mutwo-activation-checkpointing/outputs/final_runs_multi/BERT/latency_vs_batch_size.png)

Observed values from the current saved multi-batch runs:

| Model | Batch size | Baseline latency (ms) | AC latency (ms) |
| --- | ---: | ---: | ---: |
| ResNet-152 | 1 | 389.43 | 365.10 |
| ResNet-152 | 2 | 354.64 | 321.12 |
| BERT | 1 | 81.83 | 128.84 |
| BERT | 2 | 94.50 | 126.40 |
| BERT | 4 | 98.99 | 91.56 |

Interpretation:

- On ResNet-152, the saved AC runs were faster than baseline at both measured batch sizes.
- On BERT, AC was slower at batch sizes `1` and `2`, but faster at batch size `4`.
- These are still short demo-oriented sweeps, so the latency differences should be interpreted cautiously.
- For final report-quality conclusions, the same scripts should be rerun across multiple batch sizes with longer averaging windows.

## Reproduction

The plots and summaries above were generated from:

```powershell
python benchmarks.py --model "ResNet-152" --batch-sizes 1 2 --output-dir outputs/final_runs_multi
python benchmarks.py --model "BERT" --batch-sizes 1 2 4 --seq-len 64 --debug-bert --output-dir outputs/final_runs_multi
```

To generate even fuller plots for the same report structure, rerun with additional batch sizes:

```powershell
python benchmarks.py --model "ResNet-152" --batch-sizes 1 2 3 --output-dir outputs/final_runs_larger
python benchmarks.py --model "BERT" --batch-sizes 1 2 4 8 --seq-len 64 --debug-bert --output-dir outputs/final_runs_larger
```
