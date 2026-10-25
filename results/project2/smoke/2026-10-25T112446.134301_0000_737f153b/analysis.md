# Project 2 session reconstruction

Session: `2026-10-25T112446.134301_0000_737f153b`; mode: `smoke`; raw rows: 18.
Protocol SHA-256: `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.

Metric: prepared-buffer warm E2E total time. Each run ends after the common device synchronization
with all FP32 CPU logits in original order. Setup, warm-up and correctness are outside timing.
Quartiles: type-7 linear interpolation. Speedup: ratio of within-session medians; no paired-run estimator.
All raw observations are retained, including slower variants.

| Batch | N | Variant | Runs | Median ms | IQR ms | Samples/s | A / variant |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | A | 2 | 5.608586 | 0.131775 | 891.49 | 1.0000 |
| 1 | 5 | B | 2 | 5.588827 | 0.085096 | 894.64 | 1.0035 |
| 1 | 5 | C | 2 | 5.160525 | 0.028300 | 968.89 | 1.0868 |
| 8 | 19 | A | 2 | 8.327353 | 0.028985 | 2281.64 | 1.0000 |
| 8 | 19 | B | 2 | 8.382152 | 0.216706 | 2266.72 | 0.9935 |
| 8 | 19 | C | 2 | 8.039578 | 0.051609 | 2363.31 | 1.0358 |
| 64 | 137 | A | 2 | 37.385599 | 0.073905 | 3664.51 | 1.0000 |
| 64 | 137 | B | 2 | 37.446419 | 0.145787 | 3658.56 | 0.9984 |
| 64 | 137 | C | 2 | 37.345121 | 0.518188 | 3668.48 | 1.0011 |

Smoke timings describe only their bounded N; they are not full 872-sample performance results.
No trace is included in this raw session. These measurements alone do not establish copy/compute overlap,
GPU saturation, PCIe bandwidth limits, or isolated transfer cost. Project 1 measurements are not denominators.
