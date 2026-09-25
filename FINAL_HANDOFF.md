### Goal

Explain how batch size affects FP32 DistilBERT SST-2 inference on one RTX 5080 system, then separate warm workload scaling from process-cold model/GPU startup. M7 verdict: **conditional pass** because cold session JSON has two incorrect timing-policy flags; measured raw totals and conclusions reconstruct.

### Experimental Setup

RTX 5080, Ryzen 7 9700X, fixed SST-2 validation pool (872 samples), pinned DistilBERT revision, sequence length 64, FP32/highest, TF32 disabled, eight CPU threads. Warm E2E uses pageable memory and excludes model loading/tokenization/allocation. Cold totals start before local cached model load but after interpreter/import/input load; its model file cache is primed.

### Experiment 1 — Warm Batch Scaling

Batches 1, 2, 4, 8, 16, 32, 64; 30 CPU, 30 GPU inference-only and 30 GPU E2E measurements per batch. GPU paths use 15 pairs in each execution order. GPU E2E wins at batch 1 (CPU **10.950 ms**, GPU **1.287 ms**, **8.51×**) and batch 64 (CPU **460.802 ms**, GPU **14.912 ms**, **30.90×**). No CPU–GPU crossover observed in the tested warm range. GPU E2E throughput rises only **3.04%** from batch 32 to 64. E2E relative excess falls **2.51%→0.34%**.

### Experiment 2 — Full SST-2 Application Validation

Every one of 872 samples is processed once per run; the last partial batch is included. Five runs per device/batch. At batch 1, CPU/GPU E2E median total is **9.582/1.094 s** (**8.76×**); at batch 64, **6.286/0.203 s** (**30.97×**). The 872-sample scaling aligns with the warm benchmark under its separate whole-loop timing definition.

### Experiment 3 — Process-Cold Startup

Fresh Python interpreter per measurement; batch 1; N=1, 8, 32, 128, 872 (five processes/device), then N=16 (ten/device). CPU/GPU median total job: **114.224/279.144 ms** at N=1, **452.759/318.829 ms** at N=32. At N=16, **277.370/285.093 ms**; the **7.722 ms** gap is below CPU/GPU IQRs (**20.618/26.723 ms**). Observed-median winner-change bracket: **16–32 samples/job**, not an exact threshold. This is process-cold/model-file-cache-warm, not full machine cold start.

### Hypotheses vs Results

Original H1: partially supported for measured E2E excess, but limited batch-1 GPU benefit is contradicted by 8.51× warm E2E speedup. H2: supported within batches 1–64; gain diminishes near 32–64. H3: inference-only is slightly faster, but “earlier E2E crossover” is untestable because both GPU paths win at batch 1. Exploratory cold hypothesis: observed median ordering changes with N, with uncertainty at N=16.

### Strongest Evidence

Primary raw has **630/630** expected valid rows; application **40/40**; cold v2 coarse **50/50** and refinement **20/20**. All summaries independently reconstruct; CPU/GPU correctness checks pass. All 70 cold workers have distinct PIDs. Stored provenance hashes match current evidence files. No outliers were removed.

### Claims Safe to Use

1. Warm GPU E2E is **8.51×** faster at batch 1 and **30.90×** at batch 64 on this configuration; no crossover is observed across tested warm batches.
2. GPU E2E throughput gain is **3.04%** from batch 32 to 64, an observed diminishing gain.
3. Across 872 samples, batch-64 GPU E2E median total is **0.203 s** versus CPU **6.286 s** (**30.97×**).
4. Warm E2E relative excess declines from **2.51%** at batch 1 to **0.34%** at batch 64; this is not pure transfer time.
5. Process-cold observed-median winner changes within **N=16–32**; N=16's **7.722 ms** median gap is smaller than both IQRs.

### Claims Not Supported

“GPU utilization increased”; “GPU became saturated”; “compute-bound”; “PCIe transfer time”; “exact crossover is N=16”; “all AI applications behave this way”; “GPU is always faster”; “batch 32 is optimal.” Use measured throughput, diminishing gain, E2E excess, tested-range wording, and observed-median bracket instead.

### Limitations

One system/model/task; fixed FP32 and sequence length 64. Warm and cold timing boundaries differ. Application p95 equals the maximum of five runs. Cold coarse has five fresh runs/device/N, and N=16 remains uncertain. Process-cold excludes interpreter/import/input loading and does not guarantee identical OS cache residency. No profiler identifies hardware bottlenecks. Cold session JSON incorrectly says model load and buffer allocation are outside timing: actual code times model load and GPU input-buffer allocation; preserve original JSON and add a linked corrigendum before machine-readable reuse.

### Candidate Figures

Existing warm candidates: `results/plots/figure1_median_latency.svg` (and `_log.svg`), `figure2_throughput.svg`, `figure3_speedup.svg`, `figure4_e2e_relative_overhead.svg`. A presentation may derive separate application total-time and process-cold total-job charts from their respective summary CSVs; label timing boundaries and N=16 uncertainty. Do not merge warm and cold curves on one quantitative axis without a boundary warning.

### Recommended Presentation Story

Start with the original hypotheses, show warm GPU advantage from batch 1 and diminishing gain near 32–64, validate the scaling across all 872 samples, then explain why the distinct process-cold job definition reverses the small-job ordering. End with the 16–32 observed-median bracket, limitations, and the metadata corrigendum note. Avoid a causal saturation/transfer narrative.

### Relevant Result Files

- `results/audit.md`; `PROJECT_SPEC.md`; `PLAN.md`
- Warm: `results/primary_raw_runs.csv`, `results/summary.csv`, `results/analysis.md`, `results/analysis_metadata.json`
- Full workload: `results/application_validation_raw.csv`, `results/application_validation_summary.csv`, `results/application_validation_full_832f8e758ad143dfb9f89fe9c9d36b77.json`
- Cold: `results/cold_start_v2_raw.csv`, `results/cold_start_v2_summary.csv`, `results/cold_start_v2_refine_raw.csv`, `results/cold_start_v2_refine_summary.csv`, `results/cold_start_analysis.md`
- Environment: `results/environment.json`; figures: `results/plots/`
