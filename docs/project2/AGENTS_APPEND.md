# 기존 루트 AGENTS.md에 추가할 블록

아래 BEGIN/END 사이 내용을 루트 `AGENTS.md` 끝에 한 번 추가한다. 아래 제목을 포함한 기존 문서 전체를 루트 파일로 교체하지 않는다.

<!-- BEGIN CUDA PROJECT 2 SCOPE -->
## Project 2 scope and source of truth

- The user has authorized Project 2 as a continuation of Project 1: asynchronous DistilBERT inference pipeline and synchronization optimization.
- Keep all existing general working rules, including Korean communication, short smoke tests, environment reuse, raw evidence preservation, and user-executed heavy operations.
- Root `PROJECT_SPEC.md`, `PLAN.md`, and `FINAL_HANDOFF.md` describe Project 1 and remain historical evidence.
- For Project 2 experimental policy, use `docs/project2/PROJECT_SPEC.md`. Track current progress and decisions in `docs/project2/PLAN.md`.
- Lecture review and optimization selection were completed in the project-management conversation. Adding or reading lecture PDFs is not required, and their absence is not a blocker. Lecture/page citations in PROJECT_SPEC.md are background only. Codex implements the fixed A–D specification, validates correctness/timing, prepares experiment commands, and analyzes results.
- Project 1 non-goals such as "no stream optimization" apply to Project 1. They do not prohibit the explicitly authorized Project 2 stream/event work.
- Preserve existing Project 1 source behavior, validation guards, and recorded results. Add Project 2 code and write new evidence under `results/project2/`.
- Do not reuse a Project 1 pageable-only policy check as the Project 2 policy. Do not remove that original check to permit pinned memory.
- Implement and validate autonomously within this scope. Do not ask for confirmation for routine reversible implementation decisions.
- Codex may execute short development checks and bounded smoke tests; the user runs full correctness passes, full benchmark sweeps, and substantial profiling/download/install operations under the existing heavy-operation rule.
- A checkpoint is an evidence gate, not a request for permission to continue routine work. Report blockers and hand over heavy commands when appropriate.
- Never claim a measurement, overlap, or speedup before observing evidence. A slower optimized variant is a valid result.
<!-- END CUDA PROJECT 2 SCOPE -->
