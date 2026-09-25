# M6 primary analysis

Source: full M5 session `efb3e675224644b4be20f2875c37e605` in
`primary_raw_runs.csv` (SHA-256
`c1433d8a93319f2c7f6330b3a3f962c9b23c23014e3ad8eadec0bf15b7e82ded`).
All 30 measurements per path and batch are included. The GPU values pool all
15 A→B and 15 B→A pairs; order-specific data remain in the raw CSV for audit.
No measurements were excluded. Fixed sequence length 64 and FP32 policy follow
`PROJECT_SPEC.md`. The original hypothesis in that file is unchanged.

## Definitions

- Median: conventional midpoint of 30 latency measurements.
- P95: nearest rank, the 29th sorted value of 30.
- Per-sample latency: median (or p95) batch latency divided by batch size.
- Throughput: batch size × 1000 / median batch latency in milliseconds.
- Speedup: CPU median batch latency / GPU median batch latency.
- E2E excess latency: GPU E2E median − GPU inference-only median.
- E2E relative overhead: 100 × E2E excess latency / GPU E2E median.
- Observed crossover: smallest tested batch with GPU median below CPU median;
  no interpolation. A win at batch 1 indicates no transition observed in range.

E2E excess includes the distinct timing boundaries and invocation paths. It is
**not** an isolated PCIe transfer-time measurement. The difference of medians
is not the median paired difference.

## Latency, throughput, and speedup

Latency cells are median / p95 in milliseconds. Throughput uses median latency
and is in samples per second.

| Batch | CPU latency | GPU inference-only latency | GPU E2E latency | CPU throughput | GPU inference-only throughput | GPU E2E throughput | GPU inference-only speedup | GPU E2E speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.950 / 11.135 | 1.255 / 1.260 | 1.287 / 1.297 | 91.3 | 797.1 | 777.1 | 8.73× | 8.51× |
| 2 | 17.191 / 17.724 | 1.568 / 1.580 | 1.601 / 1.614 | 116.3 | 1275.6 | 1249.4 | 10.96× | 10.74× |
| 4 | 30.249 / 30.637 | 2.025 / 2.035 | 2.058 / 2.070 | 132.2 | 1975.8 | 1943.9 | 14.94× | 14.70× |
| 8 | 56.000 / 56.878 | 2.747 / 2.776 | 2.777 / 2.807 | 142.9 | 2912.3 | 2881.0 | 20.39× | 20.17× |
| 16 | 111.643 / 114.155 | 4.362 / 4.382 | 4.399 / 4.417 | 143.3 | 3667.8 | 3636.9 | 25.59× | 25.38× |
| 32 | 229.113 / 230.044 | 7.640 / 7.666 | 7.683 / 7.722 | 139.7 | 4188.6 | 4165.2 | 29.99× | 29.82× |
| 64 | 460.802 / 463.597 | 14.862 / 14.912 | 14.912 / 14.954 | 138.9 | 4306.4 | 4291.9 | 31.01× | 30.90× |

The full precision values and per-sample latencies for every batch are in
`summary.csv`. Median per-sample latency from batch 1 to 64 falls from 10.950
to 7.200 ms on CPU, 1.255 to 0.232 ms on GPU inference-only, and 1.287 to
0.233 ms on GPU E2E.

| Batch | E2E excess latency (ms) | E2E relative overhead |
|---:|---:|---:|
| 1 | 0.0323 | 2.51% |
| 2 | 0.0328 | 2.05% |
| 4 | 0.0332 | 1.61% |
| 8 | 0.0298 | 1.07% |
| 16 | 0.0370 | 0.84% |
| 32 | 0.0430 | 0.56% |
| 64 | 0.0501 | 0.34% |

## Hypothesis assessment

1. **Small-batch fixed overhead limits GPU benefit: partially supported.**
   A measurable E2E excess of 0.0323 ms (2.51%) exists at batch 1, which
   supports the presence of additional cost in the E2E path. The predicted
   limited CPU/GPU benefit is not supported: GPU inference-only and E2E are
   already 8.73× and 8.51× faster than CPU at batch 1.
2. **GPU benefit grows with batch size: supported over the tested range, with
   diminishing gains near the upper end.** GPU inference-only speedup rises
   from 8.73× to 31.01×; E2E speedup rises from 8.51× to 30.90×. From batch
   16→32, inference-only throughput grows 14.20% and speedup 17.18%; from
   32→64 the gains are only 2.81% and 3.39%. E2E throughput/speedup gains
   similarly shrink from 14.52%/17.51% to 3.04%/3.62%. CPU throughput
   remains around 139–143 samples/s for batches 8–64. This is an observed
   flattening, not proof of a particular hardware bottleneck or a limit beyond
   batch 64.
3. **Inference-only becomes favorable earlier than E2E: partially supported.**
   Inference-only has lower median latency than E2E at every tested batch,
   supporting the expected direction of the gap. The predicted earlier
   favorable batch is not supported: both GPU paths beat CPU from batch 1, so
   **no CPU-GPU crossover was observed within the tested batch range** for
   either path.

The 30 pooled GPU observations account for the balanced order design. M3.5
found an order effect; order-stratified measurements are diagnostic evidence,
not the primary estimand. No crossover was constructed by interpolation.

## Limitations

- This is one server configuration and one full benchmark session. Thirty
  repeated invocations characterize within-session timing, not across-day or
  across-device uncertainty.
- CPU host timing, GPU inference-only CUDA Event timing, and GPU E2E synchronized
  host timing have intentionally different boundaries. The E2E excess includes
  invocation and timer-boundary differences as well as copies; it cannot be
  labeled pure PCIe transfer time.
- The dataset policy uses nested prefixes of the SST-2 validation split. All
  tensors are padded to 64 tokens, but actual token content and attention masks
  vary with batch size. The workload is fixed by protocol, not a random sample
  of all possible batches.
- CPU uses eight threads, GPU uses FP32 with TF32 disabled, and E2E uses
  pageable memory and preallocated buffers. Other policies can change the
  performance relationship.
- The sweep only tests powers of two from 1 to 64. It cannot identify behavior
  below 1, between tested sizes, or beyond 64.

## Regeneration

From the project root:

```bash
.venv/bin/python src/analysis.py
.venv/bin/python scripts/plot_results.py
```

`analysis_metadata.json` records raw and summary hashes, the session, formulas,
and crossover status. The SVG files in `plots/` are presentation candidates.
