# Project 2 session reconstruction

Session: `2026-10-25T112508.818449_0000_40f9eefb`; mode: `smoke`; raw rows: 24.
Protocol SHA-256: `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.

Metric: prepared-buffer warm E2E total time. Each run ends after the common device synchronization
with all FP32 CPU logits in original order. Setup, warm-up and correctness are outside timing.
Quartiles: type-7 linear interpolation. Speedup: ratio of within-session medians; no paired-run estimator.
All raw observations are retained, including slower variants.

| Batch | N | Variant | Runs | Median ms | IQR ms | Samples/s | A / variant |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | A | 2 | 5.504729 | 0.002219 | 908.31 | 1.0000 |
| 1 | 5 | B | 2 | 5.603827 | 0.124822 | 892.25 | 0.9823 |
| 1 | 5 | C | 2 | 5.134721 | 0.092096 | 973.76 | 1.0721 |
| 1 | 5 | D | 2 | 5.198904 | 0.035608 | 961.74 | 1.0588 |
| 8 | 19 | A | 2 | 8.244654 | 0.053287 | 2304.52 | 1.0000 |
| 8 | 19 | B | 2 | 8.255799 | 0.080513 | 2301.41 | 0.9986 |
| 8 | 19 | C | 2 | 8.076699 | 0.032161 | 2352.45 | 1.0208 |
| 8 | 19 | D | 2 | 8.028333 | 0.068876 | 2366.62 | 1.0269 |
| 64 | 137 | A | 2 | 37.169181 | 0.147911 | 3685.85 | 1.0000 |
| 64 | 137 | B | 2 | 37.046808 | 0.135918 | 3698.02 | 1.0033 |
| 64 | 137 | C | 2 | 37.431144 | 0.485888 | 3660.05 | 0.9930 |
| 64 | 137 | D | 2 | 37.235045 | 0.367188 | 3679.33 | 0.9982 |

Smoke timings describe only their bounded N; they are not full 872-sample performance results.
No trace is included in this raw session. These measurements alone do not establish copy/compute overlap,
GPU saturation, PCIe bandwidth limits, or isolated transfer cost. Project 1 measurements are not denominators.
