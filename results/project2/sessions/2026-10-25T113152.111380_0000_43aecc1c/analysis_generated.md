# Project 2 session reconstruction

Session: `2026-10-25T113152.111380_0000_43aecc1c`; mode: `full`; raw rows: 384.
Protocol SHA-256: `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.

Metric: prepared-buffer warm E2E total time. Each run ends after the common device synchronization
with all FP32 CPU logits in original order. Setup, warm-up and correctness are outside timing.
Quartiles: type-7 linear interpolation. Speedup: ratio of within-session medians; no paired-run estimator.
All raw observations are retained, including slower variants.

| Batch | N | Variant | Runs | Median ms | IQR ms | Samples/s | A / variant |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 872 | A | 32 | 926.596983 | 3.841310 | 941.08 | 1.0000 |
| 1 | 872 | B | 32 | 925.341379 | 3.538267 | 942.35 | 1.0014 |
| 1 | 872 | C | 32 | 840.821155 | 2.998726 | 1037.08 | 1.1020 |
| 1 | 872 | D | 32 | 842.930695 | 2.262917 | 1034.49 | 1.0993 |
| 8 | 872 | A | 32 | 334.065311 | 0.437879 | 2610.27 | 1.0000 |
| 8 | 872 | B | 32 | 333.977351 | 0.445773 | 2610.96 | 1.0003 |
| 8 | 872 | C | 32 | 323.624145 | 0.198693 | 2694.48 | 1.0323 |
| 8 | 872 | D | 32 | 323.014217 | 0.580206 | 2699.57 | 1.0342 |
| 64 | 872 | A | 32 | 234.661066 | 0.438710 | 3716.00 | 1.0000 |
| 64 | 872 | B | 32 | 234.665266 | 0.812631 | 3715.93 | 1.0000 |
| 64 | 872 | C | 32 | 233.814589 | 0.625930 | 3729.45 | 1.0036 |
| 64 | 872 | D | 32 | 233.549758 | 0.569956 | 3733.68 | 1.0048 |

Smoke timings describe only their bounded N; they are not full 872-sample performance results.
No trace is included in this raw session. These measurements alone do not establish copy/compute overlap,
GPU saturation, PCIe bandwidth limits, or isolated transfer cost. Project 1 measurements are not denominators.
