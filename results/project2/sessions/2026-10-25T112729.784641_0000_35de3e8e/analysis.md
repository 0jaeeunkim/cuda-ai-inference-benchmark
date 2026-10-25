# Project 2 session reconstruction

Session: `2026-10-25T112729.784641_0000_35de3e8e`; mode: `correctness`; raw rows: 12.
Protocol SHA-256: `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.

Metric: prepared-buffer warm E2E total time. Each run ends after the common device synchronization
with all FP32 CPU logits in original order. Setup, warm-up and correctness are outside timing.
Quartiles: type-7 linear interpolation. Speedup: ratio of within-session medians; no paired-run estimator.
All raw observations are retained, including slower variants.

| Batch | N | Variant | Runs | Median ms | IQR ms | Samples/s | A / variant |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 872 | A | 1 | 920.848969 | 0.000000 | 946.95 | 1.0000 |
| 1 | 872 | B | 1 | 919.833285 | 0.000000 | 948.00 | 1.0011 |
| 1 | 872 | C | 1 | 837.204544 | 0.000000 | 1041.56 | 1.0999 |
| 1 | 872 | D | 1 | 838.799630 | 0.000000 | 1039.58 | 1.0978 |
| 8 | 872 | A | 1 | 333.194140 | 0.000000 | 2617.09 | 1.0000 |
| 8 | 872 | B | 1 | 333.218715 | 0.000000 | 2616.90 | 0.9999 |
| 8 | 872 | C | 1 | 322.248146 | 0.000000 | 2705.99 | 1.0340 |
| 8 | 872 | D | 1 | 322.173636 | 0.000000 | 2706.62 | 1.0342 |
| 64 | 872 | A | 1 | 232.463890 | 0.000000 | 3751.12 | 1.0000 |
| 64 | 872 | B | 1 | 232.239103 | 0.000000 | 3754.75 | 1.0010 |
| 64 | 872 | C | 1 | 237.298727 | 0.000000 | 3674.69 | 0.9796 |
| 64 | 872 | D | 1 | 231.808174 | 0.000000 | 3761.73 | 1.0028 |

Smoke timings describe only their bounded N; they are not full 872-sample performance results.
No trace is included in this raw session. These measurements alone do not establish copy/compute overlap,
GPU saturation, PCIe bandwidth limits, or isolated transfer cost. Project 1 measurements are not denominators.
