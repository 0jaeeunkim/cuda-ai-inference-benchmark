# Project 2 — server terminal handoff

All commands below run from:

```bash
cd /home/jekim/cuda/cuda-ai-inference-benchmark
```

## Current gate and environment choice

2026-10-25 (Asia/Seoul): user installation and pinned cache preparation completed.
The `cuda-project2` Conda environment passed the CUDA FP32 and one-row model-forward
checks. ABC smoke (18 rows) and ABCD smoke (24 rows) passed; the all-variant smoke
has a valid gate. Evidence: `results/project2/m2_runtime_validation.json`.
The actual GPU is GeForce RTX 5070 Ti, compute capability 12.0, driver 595.99.02.
Project 1 used a different GPU; its environment and results remain historical evidence.
Full correctness (section 6) also passed all 12 conditions ×872 rows, with maximum
absolute logits difference 0.0; its CPU references, raw data and gate were audited.
Evidence: `results/project2/sessions/2026-10-25T112729.784641_0000_35de3e8e/correctness_audit.json`.
Sections 7–8 are now completed and audited. Full session: `2026-10-25T113152.111380_0000_43aecc1c`.
Trace: `aaaea78b1d90460c91df9e18816974a5_trace`. All 384 runs passed correctness; maximum logits error was 0.0.
Results and figures: `results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/analysis.md`
and `report.html`. D showed 1.408 μs of H2D/kernel overlap in the single B=8,N=19 trace;
its additional performance over C was small, and D was slower at B=1.

Use user-local Miniforge 26.9.4-0 at `/home/jekim/miniforge3`, with the separate
Conda environment `cuda-project2`. No `conda init`, base activation or base package
installation is required. Use `conda run` explicitly in every command.

| Component | Selected version | Verification |
| --- | --- | --- |
| Python | 3.10.20 | conda-forge linux-64 builds exist |
| PyTorch | 2.12.0+cu130 | official CUDA 13.0 cp310 linux x86_64 wheel exists |
| Transformers | 5.16.1 | official PyPI distribution; Python >=3.10 |
| Datasets | 5.0.1 | official PyPI distribution; Python >=3.10 |
| Tokenizers | 0.23.2 | satisfies Transformers >=0.23.1,<0.24.0 |
| Hugging Face Hub | 1.32.0 | satisfies Transformers >=1.5.0,<2.0 |

These retain Project 1's Python and five recorded packages. No replacement is needed.
PyTorch's 2.12 release guidance recommends CUDA 13.0+ wheels for Blackwell and requires
Linux driver >=580.65.06; the observed 595.99.02 meets that requirement. This is a
distribution/driver compatibility check, not a completed CUDA runtime/model test.
The wheel supplies its CUDA runtime dependencies; this workload needs no separate
CUDA Toolkit installation or compiler.

Official sources checked on 2026-10-25:

- [PyTorch 2.12 release and Blackwell/driver guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/)
- [Official cu130 wheel index](https://download.pytorch.org/whl/cu130/torch/)
- [Transformers distribution metadata](https://pypi.org/pypi/transformers/5.16.1/json)
- [Datasets distribution metadata](https://pypi.org/pypi/datasets/5.0.1/json)
- [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus)
- [CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [Miniforge release](https://github.com/conda-forge/miniforge/releases/tag/26.9.4-0)
- [conda-forge Python 3.10.20](https://api.anaconda.org/release/conda-forge/python/3.10.20)

Machine-readable distribution checks are saved in
`results/project2/distribution_check.json` and `conda_distribution_check.json`.
Only small release metadata/checksum files were fetched by Codex, with no installation
or model/weight download. The installer script checks the publisher's SHA256 before execution.

## Recommended setup / retry command

```bash
bash scripts/project2_setup.sh all
```

This runs stages 1–4 in order and stops immediately if any stage fails. Downloads
use curl when available, then wget, then Python's standard-library downloader.
The observed server has wget and Python; curl is not required. Downloads use a
temporary `.partial` file and replace the destination only after success.
Conda and environment prerequisites are checked before dependent stages. A
previously created `cuda-project2` with Python 3.10.20 is reused on retry; an
existing incompatible or incomplete environment is preserved for inspection.

## 1. Install Conda in the user account

```bash
bash scripts/project2_setup.sh install-conda
```

Expected: `/home/jekim/miniforge3/bin/conda`; installer/checksum under
`.cache/project2-installers/`. An existing prefix is not overwritten.

## 2. Create the separate environment

```bash
bash scripts/project2_setup.sh create-env
```

Expected: Conda environment `cuda-project2`, Python 3.10.20. Base stays separate.
An existing environment is not deleted or forcibly recreated.

## 3. Install pinned packages

```bash
bash scripts/project2_setup.sh install-packages
```

Expected: successful `pip check`; `results/project2/setup/pip-freeze.txt` and
`results/project2/setup/conda-explicit.txt`. The selected top-level versions are fixed;
resolved transitive packages are recorded for reproduction. No vision/audio packages are requested.

## 4. Download only the pinned workload into the explicit cache

```bash
bash scripts/project2_setup.sh download-cache
```

The implemented `--mode prepare` loads the exact revisions below and validates one tokenized row.
This is the sole runner mode that permits online Hub access. It does not run the full
correctness experiment or benchmark.

- Model/tokenizer: `distilbert-base-uncased-finetuned-sst-2-english`, revision
  `714eb0fa89d2f80546fda750413ed43d93601a13`.
- Data: `nyu-mll/glue`, SST-2 validation, revision
  `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`.
- `HF_HOME=/home/jekim/cuda/cuda-ai-inference-benchmark/.cache/huggingface`.
- Model/tokenizer cache: `$HF_HOME/hub`; dataset cache: `$HF_HOME/datasets`.
- Expected evidence: `results/project2/cache_preparation.json`.

The setup script exports these locations for `conda run`; the runner uses the same
defaults afterward. Explicit `HF_HOME`, `HF_HUB_CACHE`, or `HF_DATASETS_CACHE` values
can override the runner defaults and become part of the environment/gate identity.

**After completion:** send Codex `cache_preparation.json`, `setup/pip-freeze.txt`, or
the failing stage and its error. Codex can then run the short checks in section 5.

## 5. M0–M2 short checks after installation (Codex may execute these)

```bash
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode env
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode smoke --variants ABC
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode smoke
```

Expected:

- `results/project2/environment.json` with `status: ready`, the 32×32 CUDA FP32 test
  and one-row cached model forward passed. Prior snapshots stay in `environments/`.
- New `results/project2/smoke/<session_id>/` directories containing `metadata.json`,
  `raw.csv`, `correctness.json`, three `reference_b*_n*.pt` files, `summary.csv`,
  `analysis.md`; the all-variant smoke writes `gate.json` plus `smoke_gate.json`.
- ABC smoke has 18 timed rows; A–D smoke has 24. Cases: B/N = 1/5, 8/19, 64/137,
  one short warm-up per variant and two timed runs per condition. ABC timings are
  developmental only and do not publish the all-variant smoke gate.
- Full/tail forward shapes, two-slot reuse, original output order, finite FP32
  logits, exact predicted classes and `atol=1e-5,rtol=1e-4` agreement with a
  separate blocking A pass. Every timed output is checked outside its timer.
- The all-variant smoke also checks device/stream/event synchronization and
  stream wait-event with tiny pinned scalar transfers.

All runtime modes except `prepare` are offline. A missing cache is an error, never
an automatic download. A changed dataset fingerprint is a blocker to investigate
and record, not a reason to relax correctness or copy the old metadata.

## 6. M3 full correctness (user only, after smoke succeeds)

```bash
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode correctness
```

Expected: new `results/project2/sessions/<session_id>/`, 12 raw rows, all 872 logits
in each B×variant condition, B=64 tail 40, coverage/label/reference artifacts,
`correctness_gate.json`. Despite its synchronized timing rows, this one-run mode
is a correctness gate rather than the 32-run performance experiment.

## 7. Separate short CUDA timeline (user only)

```bash
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode trace
```

Expected: `results/project2/diagnostics/<id>_trace/metadata.json` and
`traces/{A,C,D}_b8_n19.json` (Chrome/Perfetto JSON). Trace files are gitignored.
The command checks CUDA activity support and rejects exports containing no GPU
kernel/copy events. Profiling failure is reported separately and does not imply
that correctness failed. Inspect actual stream/copy/kernel intervals before any
claim of observed overlap. No trace timing is added to benchmark raw data.

## 8. M4 full benchmark (user only, after M3 correctness passes)

```bash
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode full
```

Expected: new `results/project2/sessions/<session_id>/raw.csv` with exactly 384 rows
(B=1/8/64 × A–D ×32), metadata, correctness, CPU reference artifacts, summary and
analysis. Five short warm-up sequences per variant cover full/partial shape and
slot reuse. The four balanced orders rotate their starting row by batch; each
variant occurs eight times at each position. A is remeasured under the new output
contract, on this new environment, in the same session as B–D.

Raw rows are flushed/fsynced individually. Failure/interruption leaves a failed
session and preserves completed raw rows. Failed/incomplete sessions cannot be
summarized as normal results. Missing/stale smoke or correctness gates stop full
execution. Source, protocol, environment, cache/model files, input identity and
gate evidence are checked; source changes require new smoke and correctness.

**After completion:** share the printed session directory's `metadata.json`,
`correctness.json`, `raw.csv`, `summary.csv`, `analysis.md`, and trace metadata if
available. Codex will audit and interpret them without merging sessions.

## Implementation decisions and limitations

- `src/project2_protocol.py` defines a separate Project 2 policy. No Project 1
  source, pinned-memory guard or recorded result is edited.
- D uses two persistent input slots and two streams. H2D waits on the prior
  slot's input-consumed event; compute waits on upload-ready; input-consumed is
  recorded after forward. Each batch has precreated events, with CUDA handles
  initialized before timing to avoid lazy event-allocation cost in the interval.
- C and D use pinned, immutable full CPU input arrays prepared outside timing.
  Per-variant host allocation/copy setup costs and bytes are recorded separately.
  No per-batch pinning or pageable-to-pinned staging occurs in timed loops.
- GPU logits and their D2H are on the same compute stream, so allocator reuse is
  ordered after their copy. Persistent cross-stream inputs live through the final
  device sync. An error path also synchronizes before releasing those buffers.
- The primary timer includes Python dispatch, copies, model forward, event
  record/wait, and the same final device-wide sync for every variant. Output
  comparisons, sentinel fill, all preparation, traces and file I/O are outside.
- Pinned allocation/setup is excluded by definition; these totals are not input
  preparation or application startup E2E. No request-latency claim follows.
- Official API references used: [pinned/non-blocking guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
  [Event API, 2.12](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.Event.html),
  [allocator stream lifetime, 2.12](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.record_stream.html).
- Lecture review and optimization selection were completed in the project-management
  conversation. Codex does not need to add or read lecture PDFs; missing PDFs are
  not a blocker. Work follows the fixed A–D experiment specification, existing
  source and relevant API documentation. Lecture citations in the specification
  are background, not an additional review gate.
- Bounded GPU correctness and synchronized timing smoke have passed under the
  installed environment. The maximum A–D logits difference was 0.0 on the smoke
  cases and every full-run output. The 384-run benchmark and separate trace were
  audited. Actual overlap was observed only in the profiled D B=8,N=19 sequence;
  broader overlap or saturation claims do not follow. Smoke and one-run correctness
  timings are not pooled with the full performance experiment.

## Saved-result audits and report reproduction (no experiment rerun)

Run from the same repository root:

```bash
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_results_audit.py --session results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c
python3 src/project2_trace_audit.py --trace-session results/project2/diagnostics/aaaea78b1d90460c91df9e18816974a5_trace --benchmark-session results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c
python3 scripts/project2_report.py --session results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c --trace-session results/project2/diagnostics/aaaea78b1d90460c91df9e18816974a5_trace --node /home/jekim/.vscode-server/cli/servers/Stable-2a59476c9bfcb90b3ddc372c36762471b7dfad1c/server/node --vega .cache/project2-report/vega.min.js
```

The last command uses the already present VS Code Node binary and a pinned Vega
5.30.0 bundle (514,536 bytes), isolated under the ignored report cache. No Python
package was installed or upgraded for plotting. The bundle URL/hash, postprocessing
source hashes and output hashes are recorded in `report_metadata.json`; the bundle
URL is `https://cdn.jsdelivr.net/npm/vega@5.30.0/build/vega.min.js`. If it is missing,
fetch that exact file and check the recorded hash before rerendering. The SVG and
`report.html` files themselves are standalone and require no Node or network to view.
The figures use the [Vega View SVG export API](https://vega.github.io/vega/docs/api/view/).

These commands read stored data and regenerate only derived audit/report artifacts.
The benchmark source/protocol and original raw, metadata, correctness and summary
remain unchanged. The initial machine-generated analysis was preserved separately
as `analysis_generated.md` before writing the interpreted `analysis.md`.
