# CUDA Individual Project 1 — Experiment Specification

## 1. Objective

Evaluate how CPU and GPU inference performance changes with batch size for a real AI application.

The purpose is not merely to demonstrate that GPUs can be faster than CPUs.

The experiment should determine under what workload conditions GPU acceleration becomes beneficial and whether CUDA fixed overhead and CPU–GPU data-transfer overhead remain visible in a real Transformer inference workload.

---

## 2. Research Question

How do CPU and GPU latency and throughput change with batch size during DistilBERT inference, and under what conditions does GPU execution become beneficial?

A secondary question is:

How different are the CPU–GPU crossover characteristics when GPU inference-only latency and GPU end-to-end latency are considered separately?

---

## 3. Hypothesis

For small batch sizes, GPU acceleration may provide limited benefit because of fixed execution overhead, synchronization, and CPU-to-GPU / GPU-to-CPU transfer costs.

As batch size increases, GPU parallel execution is expected to provide greater performance benefits.

GPU inference-only performance is expected to become favorable at a smaller batch size than GPU end-to-end performance because end-to-end execution includes data-transfer overhead.

The experiment must not force the results to match this hypothesis.

Unexpected or contradictory results must be preserved and analyzed.

---

## 3A. Pilot Observation / Analysis Focus

M3 observed lower GPU inference-only and GPU end-to-end median latency than CPU
at batch size 1. M3.5 confirmed equivalent GPU forward workloads and exposed
an execution-order effect; it did not remeasure CPU. The initial hypothesis
above remains unchanged. A crossover is not assumed to exist in the tested
batch-size range.

The full experiment primarily analyzes batch-size scaling, latency,
throughput, CPU/GPU speedup, and the difference between GPU inference-only and
GPU end-to-end measurements. Report a crossover if observed under Section 11;
otherwise report **no observed crossover** for the tested range.

---

## 4. Workload

### Model

Hugging Face:

`distilbert-base-uncased-finetuned-sst-2-english`

### Dataset

GLUE SST-2

Use the same fixed sample pool for CPU and GPU measurements.

### Task

Sentiment-classification inference.

Training is not part of this project.

### Framework

* Python
* PyTorch
* Hugging Face Transformers / Datasets

### Precision

FP32 for the primary experiment.

Do not enable FP16, BF16, AMP, or quantization in the baseline experiment.

For the primary experiment, set PyTorch float32 matmul precision to `highest`
and disable CUDA matmul TF32 and cuDNN TF32. Record these settings.

---

## 5. Input Policy

The primary independent variable is batch size.

Initial batch-size sweep:

`1, 2, 4, 8, 16, 32, 64`

Batch size 128 may be tested only after the primary sweep is working correctly and GPU memory capacity is confirmed.

### Sequence-Length Policy

Batch size should be the primary experimental variable.

Therefore, sequence length must not unintentionally vary across batch sizes in a way that changes the computational workload.

Before fixing the benchmark configuration:

1. Inspect the token-length distribution of the selected SST-2 sample pool using the DistilBERT tokenizer.
2. Report basic statistics such as:
   - median
   - p90
   - p95
   - maximum
3. Based on the observed distribution, select a reasonable fixed sequence length for the primary benchmark.

Prefer a practical value such as:

`32`, `64`, or `128`

The selected value should represent the real SST-2 workload reasonably well without unnecessarily padding most samples.

After the sequence length is selected, keep it fixed for the entire primary batch-size experiment using:

- truncation enabled
- fixed `max_length`
- `padding = "max_length"`

Record the selected sequence length and the reason for choosing it in `PLAN.md` and the experiment metadata.

**M0 decision (2026-09-24):** The primary fixed sequence length is **64**. For the
872 SST-2 validation samples, DistilBERT token lengths including special tokens
had median 24.5, p90 40, p95 44, and maximum 55. Length 32 would truncate
208/872 samples (23.85%); lengths 64 and 128 truncate none. Length 64
preserves every selected sample while avoiding the extra padding of 128.
Use `max_length=64`, `truncation=True`, and `padding="max_length"` for every
primary CPU and GPU condition.

Do not change the sequence length between CPU and GPU runs or between different batch sizes.

Tokenization must be performed before the primary inference benchmark.

CPU and GPU must receive logically identical tokenized inputs.

Use a deterministic fixed sample pool and record how samples are selected.

---

## 6. CPU Baseline

Run the same DistilBERT model in FP32 on CPU.

Requirements:

* `model.eval()`
* inference without gradient calculation
* same samples
* same tokenized tensors
* same sequence length
* same output semantics

Use an appropriate monotonic high-resolution host timer.

Record the CPU model and CPU thread configuration.

Keep the CPU thread configuration fixed throughout the experiment. Use
`torch.get_num_threads() == 8` for primary CPU runs and record the interop
thread count as well.

---

## 7. GPU Inference-Only Measurement

The input tensors must already reside in GPU memory before the measured interval.

The measured interval should contain the model forward execution only.

CUDA execution is asynchronous.

Therefore:

* use CUDA Events for GPU execution timing where appropriate;
* synchronize correctly before consuming measured values;
* never use an unsynchronized Python host timer as GPU execution latency.

Model loading and initial device transfer of model parameters are excluded.

---

## 8. GPU End-to-End Measurement

The end-to-end measurement begins with already-tokenized CPU tensors.

The intended timing boundary is:

CPU tensor
→ H2D input transfer
→ GPU model forward
→ required output D2H transfer
→ completion

Tokenization is excluded.

Model loading is excluded.

Memory allocation or other one-time initialization should not be repeatedly included unless intentionally defined otherwise.

The primary E2E path uses pageable CPU input and output tensors, blocking
copies, and GPU input and CPU output buffers allocated before measurement.
It is steady-state E2E. The CUDA Events placed inside E2E forward for M3.5
were diagnostic instrumentation and are excluded from the primary E2E path.

Document the actual implementation of the timing boundary in the code and experimental report.

---

## 9. Benchmark Protocol

Before collecting performance measurements:

* verify CPU inference;
* verify GPU inference;
* verify CPU/GPU prediction consistency;
* execute GPU warm-up iterations.

Initial protocol:

* GPU warm-up: 5 iterations minimum
* measured iterations: 30 per experimental condition
* primary statistic: median
* secondary statistic: p95

For the M4/M5 primary protocol, measure CPU in a separate block. For each
batch, use 30 GPU pairs with alternating execution order: 15 pairs
inference-only then E2E, and 15 pairs E2E then inference-only. This yields 30
runs per GPU path. Perform GPU warm-up after the CPU block and before measured
GPU pairs. Preserve pair ID, execution order, and path position for each run.

Preserve every measured run.

Do not retain only aggregated statistics.

---

## 10. Metrics

For every tested batch size, collect at least:

### CPU

* batch latency
* latency per sample
* throughput in samples/second

### GPU inference-only

* batch latency
* latency per sample
* throughput in samples/second
* CPU/GPU speedup

### GPU end-to-end

* batch latency
* latency per sample
* throughput in samples/second
* CPU/GPU speedup

Define:

`speedup = CPU latency / GPU latency`

for equivalent workloads.

---

## 11. Crossover Definition

For the tested batch sizes, define the observed crossover as the smallest tested batch size where:

`GPU median latency < CPU median latency`

Report inference-only and end-to-end crossover separately.

Do not interpolate an exact crossover between tested batch sizes unless interpolation is explicitly labeled as an estimate.

If the measurements are non-monotonic, report the observed behavior rather than forcing a crossover interpretation.

---

## 12. Correctness

CPU execution serves as the reference.

Before accepting GPU performance results:

* compare predicted labels;
* confirm equivalent input samples;
* confirm equivalent tokenized inputs.

Optionally record maximum absolute logits difference.

Performance measurements associated with an invalid computation must not be used as valid benchmark results.

---

## 13. Raw Results

Preserve the M3 pilot raw measurements in `results/raw_runs.csv` and the
M3.5 diagnostic raw measurements in `results/m35_diagnostic_raw.csv`. Store
M4 smoke measurements separately in `results/m4_smoke_raw.csv`. Store M5
primary-sweep raw measurements in `results/primary_raw_runs.csv` so the
existing evidence is never rewritten to fit the new paired schema.

Each file should preserve individual runs.

Suggested columns include:

* batch_size
* run
* measurement_type
* latency_ms
* latency_per_sample_ms
* throughput_samples_per_sec
* prediction_match
* pair_id, execution_order, and position for GPU runs

Aggregated results should be stored separately, for example:

`results/summary.csv`

Do not overwrite raw evidence during aggregation.

---

## 14. Environment Metadata

Record at least:

* GPU model
* CPU model
* OS
* NVIDIA driver
* CUDA runtime
* PyTorch version
* CUDA version reported by PyTorch
* Transformers version
* Datasets version
* Python version
* CPU thread and interop thread configuration
* float32 matmul precision and TF32 flags
* pageable host-memory policy
* dtype
* model identifier
* dataset identifier
* sequence length
* batch-size sweep
* warm-up iterations
* measured iterations

The primary GPU environment is expected to contain an NVIDIA GeForce RTX 5080.

The actual detected environment is authoritative and must be recorded rather than assumed.

---

## 15. Expected Repository Structure

A reasonable target structure is:

```text
.
├── AGENTS.md
├── PROJECT_SPEC.md
├── PLAN.md
├── README.md
├── requirements.txt
├── src/
│   ├── benchmark.py
│   ├── data.py
│   └── timing.py
├── scripts/
│   └── plot_results.py
└── results/
    ├── environment.json
    ├── raw_runs.csv
    └── summary.csv
```

The exact implementation structure may change when justified.

---

## 16. Initial Milestones

### M0 — Environment Inspection

Verify the actual RTX 5080 CUDA environment and software stack.

### M1 — CPU Baseline

Load the workload and execute correct CPU inference.

### M2 — GPU Correctness

Run identical inference on GPU and validate outputs against CPU.

### M3 — Timing Prototype

For batch sizes 1 and 8 only:

* measure CPU latency;
* measure GPU inference-only latency;
* measure GPU end-to-end latency;
* verify timing methodology;
* preserve raw measurements.

Do not perform the complete sweep before M3 passes validation.

### M4 — Benchmark Harness

Create a reusable benchmark runner and CSV output.

### M5 — Full Sweep

Run batch sizes:

`1, 2, 4, 8, 16, 32, 64`

### M6 — Analysis

Aggregate results and generate plots.

### M6.5 — Application-level Validation

Before M7, process the entire fixed SST-2 validation pool of 872 samples at
batch sizes `1, 8, 32, 64`. One run covers every sample exactly once in
published order and includes the final incomplete batch. Compare CPU total
execution time with GPU end-to-end total execution time; both include the
per-batch forward loop, while GPU E2E also includes pageable H2D input copies,
logits D2H copies, and final synchronization. Exclude tokenization, model
loading, and buffer allocation from timing. Keep sequence length 64, FP32,
eight CPU threads, highest float32 matmul precision, and TF32 disabled.

Validate CPU/GPU predictions for every sample and batch condition before full
timing. Warm up each distinct full/partial batch shape before measuring its
path. Repeat the full 872-sample run five times per path and batch size; report
median and nearest-rank p95 total time, throughput derived from median total
time, CPU/GPU E2E speedup, and forward invocation count. With five repeats,
nearest-rank p95 equals the maximum and is only a coarse tail indicator.
Preserve application raw and summary files separately from M3–M6 evidence.
Run a short coverage/partial-batch smoke test first. The user runs the full
application validation in the server terminal under `AGENTS.md`.

### M6.6 — Cold-Start Crossover Experiment

Run the separate exploratory experiment defined in Section 16A after M6.5 and
before M7. Preserve primary and application-validation evidence unchanged.

### M7 — Audit

Review the complete experiment for:

* unfair CPU/GPU comparison
* incorrect CUDA timing
* inconsistent timing boundaries
* input inconsistencies
* warm-up artifacts
* incorrect throughput calculations
* unsupported causal claims

---

## 16A. Exploratory Experiment — Cold-Start Crossover

This **process-cold model/GPU startup experiment** is separate from the
primary warm benchmark and M6.5 application-level validation. It does not
measure a full machine or application cold start. Do not combine its raw
observations with either warm experiment.
The original Initial Hypothesis in Section 3 remains unchanged.

### Purpose and hypothesis

Test whether fresh-process model startup changes the CPU/GPU relationship for
small inference jobs. **Hypothesis:** GPU device initialization and model
transfer add a fixed cold-start cost, so CPU may be competitive for small jobs;
as samples per job increase, GPU inference may amortize that cost and a
CPU–GPU crossover may appear. Preserve results even if no crossover appears.

### Fixed conditions and independent variable

- Independent variable: sample count `N` per job, using the first `N` rows of
  the same pinned SST-2 validation split in published order.
- Initial coarse scan: `N = 1, 8, 32, 128, 872`; no exhaustive scan.
- Batch size `1` for both devices, fixed sequence length `64`, FP32 model,
  eight CPU intra-op threads, float32 matmul precision `highest`, CUDA matmul
  TF32 disabled, and cuDNN TF32 disabled.
- CPU and GPU process logically identical pretokenized inputs. The GPU path
  copies each input to the device and returns each logit output to pageable CPU
  memory. No warm-up occurs within a cold measurement.
- Every CPU and GPU cold measurement starts in a new Python process. Neither
  `torch.cuda.empty_cache()` nor reusing a previously initialized CUDA context
  qualifies as a cold start.
- This is process-cold, not a machine reboot or OS page-cache flush. Before
  the coarse scan, perform one **untimed local CPU model load** to prime the
  model file cache. All measured processes then use the same intended
  process-cold / file-cache-warm policy. OS cache state is not forcibly reset
  or guaranteed identical at the hardware level.
- Launch every measurement with `subprocess`/`exec` or `multiprocessing spawn`
  into a fresh Python interpreter. Do not inherit an initialized CUDA context
  by forking a parent. GPU workers must start their total-job timer before
  any `torch.cuda.*` call that could initialize CUDA; GPU environment inspection
  belongs in the orchestrator, outside measured workers.
- Use the pinned model weights from the local cache with offline loading.
  Network download and tokenization occur outside timing. Loading the
  pretokenized input artifact and Python interpreter/library import time are
  also outside the defined job interval; record this boundary explicitly.

### Timing and recorded fields

Start the monotonic host timer immediately before loading the cached pretrained
model into CPU memory. End after the final required inference output is
available on CPU and, for GPU, CUDA has synchronized. Record per fresh process:

- `model_load_ms`: cached model weight load, FP32 conversion, and eval setup;
- `device_prepare_ms`: GPU CUDA initialization, model-to-device transfer,
  preallocated device input buffer, and synchronization (zero for CPU);
- `first_forward_ms`: first batch-1 forward, including H2D and D2H for GPU;
- `remaining_inference_ms`: all subsequent batch-1 forwards and required GPU
  transfers/completion (zero when `N=1`);
- `total_job_ms`: complete defined interval, including small host work between
  recorded stages.

Use synchronized host timing for GPU stages. Preserve model identity, fixed
inputs, process ID, predicted-output checksum, and each raw measurement.
Repeat each CPU/GPU coarse condition in five independent fresh processes and
compare medians of **`total_job_ms`**. At each `N`, alternate CPU-first and
GPU-first launches; reverse the majority first-launch device on alternate
coarse `N` values. Record order and position with every raw measurement. Stage
medians explain where time is spent; they do not replace total-job medians for
crossover decisions.

### Crossover and refinement

At adjacent coarse `N` values, look for a change in which device has the lower
median total-job time. A bracket exists only where that observed ordering
changes. Report the bracket as tested endpoints without interpolating an exact
crossover. Measure additional sample counts only inside an observed bracket;
if no bracket exists, report **no observed cold-start crossover in the coarse
scan** and do not refine. Keep cold raw, summary, and interpretation in
separate exploratory result files. Run a short fresh-process smoke test first;
under `AGENTS.md`, the user executes the heavy coarse/refinement runs in the
server terminal.

---

## 17. Non-Goals

The initial experiment does not include:

* training
* multiple Transformer models
* FP16 / BF16
* quantization
* sequence-length sweep
* custom CUDA kernels
* CUDA stream optimization
* kernel-level optimization
* Nsight Compute profiling
* multi-GPU execution

These may only be considered after the primary experiment is complete.

---

## 18. Success Criteria

The experiment is considered complete when:

1. CPU and GPU execute equivalent DistilBERT SST-2 inference correctly.
2. GPU timing correctly handles CUDA asynchronous execution.
3. Inference-only and end-to-end measurements have explicitly defined boundaries.
4. Every tested batch size has reproducible raw measurements.
5. Environment metadata is preserved.
6. CPU/GPU latency and throughput can be reconstructed from raw results.
7. Observed crossover behavior can be reported without relying on fabricated or selectively filtered measurements.
8. Limitations and alternative explanations are documented.
