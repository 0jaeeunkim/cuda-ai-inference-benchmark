# Process-cold model/GPU startup experiment — v2 coarse analysis

This exploratory experiment is separate from the warm primary and M6.5
application-validation results. It measures **process-cold model/GPU startup**
with a warmed model file cache, not a full machine or application cold start.
Python interpreter/library import, pretokenized input loading, tokenization,
and network download are outside the timed job. The timed job begins before
local model load and ends when its final output is available on CPU. No warm-up
occurs inside a measured worker.

Source: v2 coarse session `61d2fd112e534e75b85acc7a08aee527`,
`cold_start_v2_raw.csv` SHA-256
`06d9200b6cab93ea8ed4560b068ee42063230dd78e89fbc9b419199f705184c4`.
Every reported median uses five fresh processes per device and N. CPU-first and
GPU-first launch order was alternated and recorded in raw rows; the majority
first-launch device reverses across adjacent N values. All 50 workers had
distinct PIDs, CPU/GPU prediction checksums agreed at each N, and the session
metadata records untimed cache priming and preserved evidence hashes.

| Samples/job N | CPU median total job (ms) | GPU median total job (ms) | CPU/GPU winner | CPU median throughput (samples/s) | GPU median throughput (samples/s) | GPU speedup vs CPU |
|---:|---:|---:|:---|---:|---:|---:|
| 1 | 114.224 | 279.144 | CPU | 8.75 | 3.58 | 0.41× |
| 8 | 183.929 | 271.900 | CPU | 43.50 | 29.42 | 0.68× |
| 32 | 452.759 | 318.829 | GPU | 70.68 | 100.37 | 1.42× |
| 128 | 1513.451 | 416.397 | GPU | 84.57 | 307.40 | 3.63× |
| 872 | 9685.965 | 1373.288 | GPU | 90.03 | 634.97 | 7.05× |

The observed coarse **CPU/GPU winner changes between N=8 and N=32**. These are
bracket endpoints, not an interpolated exact crossover. At N=8 the CPU/GPU
median gap is 87.971 ms; at N=32 it is 133.930 ms. Both exceed the larger
within-device IQR at their endpoint (23.133 and 20.443 ms respectively).

Stage medians support the fixed-startup-cost interpretation but are diagnostic,
not the crossover criterion. CPU and GPU median local model loads are about
79 ms across the scan. GPU device preparation is about 86–113 ms and its
first forward about 90 ms; the CPU first forward is about 27–35 ms. At larger
N, GPU remaining-inference time grows much more slowly than CPU remaining
inference. Stage medians need not sum to the median total job because their
rankings can come from different runs.

Launch-order stratification remains diagnostic evidence: GPU-first
processes were roughly 20–25 ms faster than CPU-first GPU processes at
N=1, 8, and 32. Five repeats give a 3:2 split at each N, with the majority
order reversed across N. The N=16 refinement uses ten repeats per device,
producing five launches in each order. Neither order stratum is substituted
for the pooled coarse median.

## N=16 refinement

N=16 was measured strictly inside the observed 8–32 bracket after a separate
two-process smoke. The full refinement session
`6ddc295a876f455a9069ba0ede788644` has ten fresh CPU and ten fresh GPU
workers, five launches in each order. Its raw source is
`cold_start_v2_refine_raw.csv` (SHA-256
`dd0bb7c278e15721479fdbe89ddb02150edc5749d6fb547ecfa7eefd4eaca96f`).
The smoke timings are excluded from the full refinement summary.

At N=16, median total-job time is **277.370 ms CPU** and **285.093 ms GPU**;
GPU speedup versus CPU is **0.973×**. CPU and GPU median-derived throughput
are 57.68 and 56.12 samples/s. CPU had the lower total time in nine of ten
paired launch repetitions. The observed median gap is only 7.722 ms, smaller
than the CPU IQR of 20.618 ms and GPU IQR of 26.723 ms. Thus the measured
winner at N=16 is CPU, but the small difference should not be treated as a
precise or universally stable threshold. The ten-run refinement fulfills the
predefined increased-repetition policy for a close candidate.

Together with the v2 coarse N=32 GPU win, the **observed cold CPU/GPU winner
change is bracketed by N=16 and N=32**. No exact crossing sample count is
claimed or interpolated. Additional points would be a separate, narrower
follow-up; this experiment stops after the planned one-point refinement.
All 20 refinement workers had distinct PIDs, matching CPU/GPU prediction
checksums, positive timings, and complete phase data. The summary was
independently reconstructed from raw rows, and v1, v2 coarse, warm-primary,
and application-validation evidence hashes remained unchanged.

The earlier v1 coarse session `b06201eef29f4c8abac0891f5cf0b08a` remains
preserved in its original files. It used the pre-update methodology and is not
pooled with v2 measurements. OS page-cache state is not forcibly reset or
measured, so "file-cache-warm" is a controlled intended condition rather than
a guarantee of identical physical cache residency for every worker. A single
server and ten total measured processes per coarse N do not establish behavior
on other systems or days.
