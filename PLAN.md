# CUDA AI Crossover Plan

`AGENTS.md` and `PROJECT_SPEC.md` govern this work. The initial workload is
FP32 DistilBERT SST-2 inference. Tokenization and model loading are excluded
from the primary timing intervals. CPU and GPU receive the same token IDs and
attention masks. Individual measurements are retained before aggregation.

## Fixed inputs and selected sequence length

- Dataset: `nyu-mll/glue` SST-2 `validation` at revision
  `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`, in published row order (872 rows). Its
  fingerprint and selected row indices are recorded. The candidate pool for
  every batch size is this split; each condition uses its first `batch_size`
  rows. This deterministic prefix policy will remain fixed in the sweep.
- Token lengths include special tokens and are measured without truncation.
  M0 reports median, p90, p95, maximum, and the truncation rate for each of
  32, 64, and 128 over all 872 rows. Observed lengths: median 24.5, p90 40,
  p95 44, maximum 55. Truncation rates are 208/872 (23.85%) at 32 and 0/872
  (0%) at both 64 and 128. The smallest candidate covering every sample is 64.
- **M0 gate passed (2026-09-24):** the user selected primary fixed sequence
  length **64** from the recorded M0 distribution. Use `max_length=64`,
  truncation, and `padding="max_length"` for M1-M7; no provisional M3 length
  or sequence-length sweep is planned.
- M3.5 diagnostic passed. Its raw results remain separate; no further GPU
  clock or cache investigation is planned in the current project.
- CPU threads are fixed at 8 for all primary CPU measurements. GPU
  inference-only uses CUDA Events around forward on resident input. GPU
  end-to-end uses a synchronized host timer around copies from **pageable**
  CPU input tensors into preallocated GPU buffers, forward, and required logits
  D2H copy into a preallocated pageable CPU buffer. Copies are blocking. This
  is steady-state E2E: tokenization, model loading, and buffer allocation are
  outside the measured interval. GPU warm-up is at least 5 iterations per path.
- FP32 policy: set `torch.set_float32_matmul_precision("highest")`,
  `torch.backends.cuda.matmul.allow_tf32 = False`, and
  `torch.backends.cudnn.allow_tf32 = False`; record the PyTorch float32 matmul
  precision setting and both TF32 flags in environment metadata.
- Each tested condition has at least 30 individually preserved runs. Median is
  primary and p95 secondary. No runs are discarded for hypothesis fit.
- Codex runs short smoke tests. For the full sweep or other heavy runs, provide
  an exact command for the user to run in the server terminal after smoke tests
  pass; inspect the preserved output before moving to the next milestone.

## M0 — Environment inspection and input-length decision

- Implement: inspect GPU, CPU, OS, driver, CUDA Toolkit/runtime, Python and
  package versions, CUDA availability, device capability, and actual CUDA
  arithmetic. Load the fixed SST-2 pool and DistilBERT tokenizer, calculate
  untruncated token lengths and recommend a fixed primary length.
- Files: `src/data.py`, `src/benchmark.py`, `results/environment.json`,
  `results/token_lengths.csv`, `PLAN.md`.
- Validate: CUDA FP32 operation succeeds; 872 validation rows load; tokenizer
  covers every row; statistics recompute from preserved lengths. Report any
  missing dependency or access failure before changing the environment.
- Complete when the actual environment and length distribution are recorded
  with all three truncation rates and a length recommendation. The user
  selected 64, satisfying the gate before M1.

## M1 — CPU baseline

- Implement: load the specified model in FP32 on CPU, set eval mode, run
  no-gradient inference for batch sizes 1 and 8 using the fixed pool and
  selected primary length. Fix and record CPU thread count.
- Files: `src/benchmark.py`, `results/validation.json`.
- Validate: one small CPU forward succeeds before broader work; output shape,
  finite logits, input IDs, attention masks, and predictions are recorded.
- Complete when both batches produce valid CPU predictions.

## M2 — GPU correctness

- Implement: run the same model weights and tokenized inputs on RTX 5080 in
  FP32, then compare predicted labels with CPU; record maximum absolute logits
  difference and input equality checks.
- Files: `src/benchmark.py`, `results/validation.json`.
- Validate: a single GPU forward succeeds first. Pass for each batch requires
  identical selected sample indices, input IDs, and attention masks on CPU and
  GPU; finite logits with shape `[batch_size, 2]`; and exact predicted-label
  agreement for every sample. Any failed condition is a correctness failure.
  Record maximum absolute logits difference as a diagnostic only; small FP32
  rounding differences do not fail correctness when these criteria pass.
- Complete when both batch sizes pass. No performance result is accepted if
  correctness fails.

## M3 — Timing prototype, batches 1 and 8 only

- Implement: warm up CPU and both GPU paths, then collect 30 CPU, 30 GPU
  inference-only, and 30 GPU end-to-end runs per batch. Preserve every run,
  timing boundary, session metadata, and correctness status.
- Files: `src/timing.py`, `src/benchmark.py`, `results/raw_runs.csv`,
  `results/validation.json`, `results/environment.json`.
- Validate: first run one warm-up and one measured iteration as a smoke test;
  check all latencies are positive and finite, each condition has 30 runs,
  calculations reconstruct from latency and batch size, CUDA Events are
  synchronized before reading, and the end-to-end host timer synchronizes
  before start and after D2H completion.
- Complete when batches 1 and 8 have valid raw CPU/GPU data and median/p95
  can be reported. Stop here; do not run the full sweep or plot results.

## M4 — Reusable benchmark harness

- Implement: a reusable runner with CPU measurements in a separate block and
  GPU pairs in balanced order. For every primary batch, use 15 A→B pairs and
  15 B→A pairs, yielding 30 runs per GPU path. Store pair ID, execution order,
  and position in append-only raw CSV. A is resident-input CUDA Event forward;
  B is the original synchronized host E2E boundary without internal events.
  Fix sequence length 64, FP32/highest, TF32 off, CPU threads 8, pageable host
  memory, and exclusion of tokenization and model loading.
- Files: `src/benchmark_harness.py`, `README.md`, `requirements.txt`,
  `results/m4_smoke_raw.csv`, `results/m4_smoke_validation.json`.
- Keep `requirements.txt` to minimal direct dependencies, not a full
  environment freeze; preserve observed package versions in metadata.
- Validate: correctness gate plus batch 1 and 8 smoke only, with short measured
  runs; check the raw schema, balanced pair order, arithmetic, synchronization
  boundaries, and unchanged M3/M3.5 raw evidence.
- Complete when the smoke passes and an exact user-terminal M5 command is
  ready. Codex does not run the full sweep.

## M5 — Full sweep

- **Completed (2026-09-25):** User ran session
  `efb3e675224644b4be20f2875c37e605`. The 630 raw rows cover all seven
  batch sizes with 30 CPU, 30 GPU inference-only, and 30 GPU end-to-end runs
  per batch; each batch has 15 pairs per GPU order. All correctness checks
  passed. Independent CSV checks confirmed complete pairs, positive finite
  latencies, and derived metrics within CSV rounding precision. M3 and M3.5
  raw-file hashes remained unchanged. Primary CSV SHA-256:
  `c1433d8a93319f2c7f6330b3a3f962c9b23c23014e3ad8eadec0bf15b7e82ded`.
- Implement: measure batch sizes 1, 2, 4, 8, 16, 32, and 64 at the fixed
  primary sequence length, FP32, with all three timing paths. After short smoke
  tests pass, Codex supplies the exact full-sweep command; the user runs that
  command in the server terminal and Codex audits the resulting files.
- Files: `results/primary_raw_runs.csv`, per-session metadata, and the
  unchanged `results/environment.json`.
- Validate: correctness for every batch; 30 raw runs per path and batch;
  positive finite latency and complete metadata; no silent exclusions.
- Complete when the complete sweep is preserved and internally consistent.

## M6 — Analysis

- **Completed (2026-09-25):** M6 summary was reconstructed from all 630
  rows of the audited primary session. Seven batch rows, median/p95 and
  derived metrics, five SVG plot candidates, provenance hashes, and an
  interpretation are saved under `results/`. An independent calculation
  checked every summary metric against raw latency values. Original M3,
  M3.5, and M5 raw hashes remained unchanged. M7 has not started.
- Implement: aggregate medians and p95, per-sample latency, throughput, and
  CPU/GPU speedup; produce latency and throughput plots and describe observed
  crossover separately for inference-only and end-to-end. No crossover within
  the tested batch sizes is a valid result and must be reported as such.
- Files: `src/analysis.py`, `scripts/plot_results.py`, `results/summary.csv`,
  `results/plots/`, `README.md`.
- Validate: recompute summary from raw rows; speedup equals CPU/GPU latency;
  observed crossover uses the smallest tested batch with lower GPU median, or
  explicitly reports none when no tested batch meets that criterion.
- Complete when results and plots are traceable to raw runs and hypotheses are
  evaluated without editing evidence.

## M6.5 — Application-level validation

- **Completed (2026-09-25):** Smoke confirmed exact 872-row coverage,
  final batch sizes 1/8/8/40, and matching CPU/GPU predictions on first and
  last batches. The user ran full session
  `832f8e758ad143dfb9f89fe9c9d36b77`. Independent audit reconstructed
  four summary rows from 40 raw measurements, checked CPU/GPU predictions on
  all 872 rows in every condition, verified forward counts 872/109/28/14,
  and confirmed M3–M6 evidence hashes unchanged. Application raw SHA-256:
  `d87f9dfcaa5ba8b531835da59adedb27c51da472738aef6c5641c9955f903e95`.
  M7 has not started.
- Implement: process all 872 SST-2 validation rows exactly once per run at
  batch sizes 1, 8, 32, and 64. Compare CPU and GPU E2E total execution time
  with pretokenized inputs and preloaded FP32 models. Include final partial
  batches (8 rows at batch 32; 40 rows at batch 64), per-batch H2D and D2H
  copies for GPU E2E, and loop overhead. Keep allocations outside timing.
  Use five repetitions per path/size, with five warm-ups for each distinct
  full/partial shape immediately before that path's measured runs.
- Files: `src/application_validation.py`,
  `results/application_validation_smoke.json`,
  `results/application_validation_raw.csv`,
  `results/application_validation_summary.csv`, and a full-session JSON.
  Existing M3–M6 raw, summary, plots, and analysis remain unchanged.
- Validate: short smoke verifies exact 872-row coverage and executes first and
  final batches on CPU/GPU; full run validates CPU/GPU predictions for all
  872 rows per condition, expected forward counts 872/109/28/14, 5 total-time
  runs per path/size, finite positive times, throughput arithmetic, and source
  evidence hashes. Nearest-rank p95 of five runs is the maximum.
- Complete when the user runs the full command in the server terminal and
  Codex audits separate raw/summary files. Do not begin M7 before this gate.

## M6.6 — Process-cold model/GPU startup experiment (exploratory)

- **Completed (2026-09-25):** Audited v2 coarse session
  `61d2fd112e534e75b85acc7a08aee527` and N=16 refinement session
  `6ddc295a876f455a9069ba0ede788644`. The latter preserved 20 fresh runs,
  balanced 5:5 order, matching CPU/GPU predictions, and unchanged earlier
  evidence. The observed winner-change bracket narrowed from N=8–32 to
  N=16–32. Results, uncertainty, and limitations are in
  `results/cold_start_analysis.md`; M7 has not started.
- Implement: use local cached pretrained DistilBERT and a pretokenized artifact
  from the pinned SST-2 validation pool. Keep batch size 1, sequence length
  64, FP32/highest, TF32 off, and CPU threads 8 fixed. Each measurement runs
  in a fresh Python process. Time cached model load, CUDA initialization and
  model transfer, first forward, remaining inference, and total job from model
  load start through final CPU-visible output. Exclude network, tokenization,
  pretokenized artifact load, and interpreter/library imports from timing.
- Sequence: **smoke test → coarse scan** at N=1,8,32,128,872 with five fresh
  CPU and five fresh GPU processes per N → identify any change in median
  total-job winner between adjacent tested N values → **refine only an
  observed bracket** → write a separate summary and interpretation. If the
  median CPU/GPU gap at a bracket endpoint is small relative to run-to-run
  variation (for example, no larger than the larger device IQR), use **10–15
  fresh processes per device for each refinement candidate N**. Otherwise
  choose a justified repetition count before handoff. Do not interpolate a
  crossover or perform an exhaustive scan.
- Files: `src/cold_start.py`, `results/cold_start_inputs.pt` and its metadata,
  `results/cold_start_smoke.json`, separate v2 exploratory
  `results/cold_start_v2_raw.csv` and `results/cold_start_v2_summary.csv`,
  and later `results/cold_start_analysis.md`. Existing M0–M6 and M6.5 files
  remain unchanged. Keep results interpretation out of this plan.
- Validate: offline/local model loading with one untimed cache-prime before
  measurements; fresh Python interpreter via `subprocess`/`exec`, no inherited
  CUDA context, and no CUDA-initializing calls before the worker total timer;
  balanced launch order recorded in raw data; fixed input checksum and
  CPU/GPU prediction agreement; positive
  finite synchronized phase/total timings; exact N forward invocations;
  independently reproducible medians and bracket detection. Codex runs only
  the short smoke; the user runs heavy coarse/refinement commands.
- Complete when the coarse result, and refinement if a bracket exists, are
  audited and interpreted in exploratory result files. M7 starts afterward.

## M7 — Audit

- Implement: inspect workload equivalence, sequence length, timing boundaries,
  synchronization, warm-up, raw-run preservation, arithmetic, confounders,
  and claim support.
- Files: `results/audit.md`, revisions to code or documentation if findings
  require them; any remeasurements append new sessions to raw evidence.
- Validate: independent raw-to-summary reconstruction and checklist against
  every success criterion in `PROJECT_SPEC.md`.
- Complete when findings, limitations, and any remaining uncertainty are
  documented and all material defects are fixed or explicitly reported.

- **M7 provenance documentation resolved:** see `results/cold_start_metadata_corrigendum.json` and `results/cold_start_metadata_corrigendum.sha256`. M7 conditional pass resolved for provenance documentation; original session JSON and audit verdict remain preserved.
