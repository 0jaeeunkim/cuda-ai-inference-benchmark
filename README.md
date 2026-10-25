# CPU–GPU Performance Analysis of DistilBERT Inference with CUDA

This project measures how batch size and startup state affect CPU and GPU inference latency for a fixed DistilBERT sentiment-classification workload. The specification and final audit are in [`PROJECT_SPEC.md`](PROJECT_SPEC.md) and [`results/audit.md`](results/audit.md).

## Setup and workload

- Hardware: NVIDIA GeForce RTX 5080 and AMD Ryzen 7 9700X; Linux, PyTorch 2.12.0+cu130. Full versions and settings are in [`results/environment.json`](results/environment.json).
- Model: pinned `distilbert-base-uncased-finetuned-sst-2-english`; dataset: pinned SST-2 validation split (872 samples).
- Fixed sequence length 64, FP32 with TF32 disabled, eight CPU threads, and pageable host memory for GPU E2E. Tokenization is excluded from all timed paths.

## Three experiments

| Experiment | Measured work | Main observation |
|---|---|---|
| Warm batch-size scaling | CPU, GPU inference-only, and GPU E2E at batches 1–64; 30 runs per condition | GPU E2E speedup over CPU was 8.51× at batch 1 and 30.90× at batch 64. No CPU–GPU crossover was observed in this tested warm range. |
| Full application validation | All 872 samples exactly once per run at batches 1, 8, 32, 64; five runs per path | At batch 64, median total time was 6.286 s CPU versus 0.203 s GPU E2E (30.97×). |
| Process-cold startup | Fresh interpreter per run, warmed model file cache, batch 1; varying samples/job | CPU led at N=1 and 8; GPU led at N=32 and above. The observed-median transition bracket after N=16 refinement is 16–32, with uncertainty at N=16. |

The cold experiment times local model load, GPU device preparation, and inference through the final CPU-visible output. It **does not** measure full machine or application cold start; interpreter/import and input-artifact loading are excluded. The cold session JSON has two inherited timing-policy flags corrected by the append-only [`corrigendum`](results/cold_start_metadata_corrigendum.json); original evidence is preserved.

## Repository structure

- `src/`: benchmark, application-validation, cold-start, and analysis code.
- `scripts/`: plot generation.
- `results/`: raw CSVs, summaries, plots, metadata, analyses, M7 audit, and the small pretokenized cold input artifact.
- [`PLAN.md`](PLAN.md), [`FINAL_HANDOFF.md`](FINAL_HANDOFF.md): milestones and concise result handoff.

## Reproduction

From this directory, create a Python 3.10 environment and install the direct dependencies:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use a compatible NVIDIA driver/CUDA runtime and the pinned model/dataset revisions in `results/environment.json`. These commands run substantial experiments and may download the pinned artifacts if they are not cached:

```bash
.venv/bin/python src/benchmark_harness.py --mode smoke
.venv/bin/python src/benchmark_harness.py --mode full
.venv/bin/python src/application_validation.py --mode smoke
.venv/bin/python src/application_validation.py --mode full
.venv/bin/python src/cold_start.py --mode smoke
.venv/bin/python src/cold_start.py --mode coarse
.venv/bin/python src/cold_start_refine.py --mode run --count 16 --coarse-session 61d2fd112e534e75b85acc7a08aee527
```

The cold commands require the pinned model already cached locally and run offline. The N=16 refinement applies to the recorded coarse session; a new run requires its own qualifying coarse session ID. Existing results can be inspected without rerunning the benchmarks in [`results/analysis.md`](results/analysis.md), [`results/cold_start_analysis.md`](results/cold_start_analysis.md), and [`results/audit.md`](results/audit.md).

## Interpretation limits

Results come from one server, one model/task, FP32, and sequence length 64. Warm, whole-application, and process-cold timing boundaries differ and must not be pooled. The process-cold bracket is not an exact crossover point. No profiler evidence identifies GPU utilization, saturation, or the isolated PCIe transfer cost.

## CUDA Project 2

Asynchronous DistilBERT inference continues in the [Project 2 specification](docs/project2/PROJECT_SPEC.md), [progress log](docs/project2/PLAN.md), and [terminal handoff](docs/project2/RUNBOOK.md). Project 1 source and evidence remain preserved; new runs write to `results/project2/`.

The [audited Project 2 results](results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/analysis.md) include the 384-run comparison, short CUDA trace interpretation, setup costs and standalone figures.
