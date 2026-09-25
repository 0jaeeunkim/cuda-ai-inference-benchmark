# Project Instructions

## 1. Communication

- Communicate with the user in Korean.
- Repository documentation and code may be written in English.
- Keep chat responses concise and decision-oriented.
- Do not repeat information that is already established in `PROJECT_SPEC.md`, `PLAN.md`, or previous confirmed decisions.
- Report only:
  - important findings;
  - decisions;
  - blockers;
  - validation results;
  - necessary next actions.

Do not provide long summaries unless explicitly requested.

---

## 2. Source of Truth

Use the repository files as persistent context.

- `PROJECT_SPEC.md`: authoritative experimental specification.
- `PLAN.md`: current milestones and progress.
- `AGENTS.md`: working rules.

Do not repeatedly restate these files in chat.

When a decision becomes stable and is needed later, update the appropriate repository file instead of repeatedly carrying it in conversation.

Do not duplicate the same information across multiple documents unless necessary.

---

## 3. Context and Token Efficiency

Minimize unnecessary context consumption.

- Read only files relevant to the current task.
- For large files, inspect only relevant sections or search for relevant symbols/text first.
- Do not repeatedly reread entire files when a targeted read is sufficient.
- Do not scan the entire repository unless the task genuinely requires it.
- Prefer targeted search, file ranges, diffs, and specific functions over full-file inspection.
- Avoid reopening unchanged files without a reason.

When modifying code:

- make the smallest reasonable diff;
- do not rewrite unrelated sections;
- do not output entire files in chat;
- summarize the changed files and important behavior only.

Do not generate multiple alternative implementations unless there is a meaningful design decision that requires comparison.

---

## 4. Chat Output Efficiency

Avoid dumping large outputs into chat.

Do not paste:

- long terminal logs;
- full CSV files;
- full benchmark results;
- full package lists;
- large diffs;
- complete source files;
- repeated environment metadata.

Instead:

- save raw outputs to files;
- inspect them locally;
- report only the relevant summary, errors, or representative lines.

For benchmark results, keep raw evidence under `results/` and report only concise statistics in chat.

If detailed output may be useful later, save it to a file rather than consuming chat context.

---

## 5. Heavy Operations Must Be Run by the User

Codex may directly execute short, bounded development commands such as:

- file inspection;
- syntax checks;
- small unit tests;
- small smoke tests;
- tiny benchmark prototypes;
- environment queries.

Do NOT directly execute operations involving substantial time, compute, storage, network transfer, or potentially long blocking execution.

Examples include:

- large package installations;
- CUDA / PyTorch reinstallation;
- large model or dataset downloads;
- full dataset preprocessing;
- long builds;
- full benchmark sweeps;
- hundreds of benchmark repetitions;
- large profiling runs;
- exhaustive test suites;
- long-running training or inference jobs.

For such operations:

1. Prepare and validate the code using a small smoke test if possible.
2. Give the user the exact terminal command to run.
3. State the required working directory.
4. State briefly what output files should be produced.
5. Let the user execute the command in their own terminal.
6. Continue after the user reports completion or the generated files become available.

Do not keep a Codex task occupied waiting for a long-running experiment when the same experiment can be run independently in the user's terminal.

---

## 6. Development Before Full Execution

Use the smallest useful workload while developing.

Examples:

- batch sizes `1` and `8` instead of the full sweep;
- 1–3 repetitions instead of 30;
- a small sample subset instead of the full dataset;
- one correctness case before all cases.

Development sequence:

1. Inspect.
2. Implement minimum functionality.
3. Run a small smoke test.
4. Verify correctness.
5. Verify timing methodology.
6. Prepare the full-run command.
7. Ask the user to execute the full run in their terminal.

Do not run a full experiment merely to discover basic implementation errors.

---

## 7. Environment Discipline

Inspect the existing environment before installing anything.

Reuse:

- existing Python environments;
- installed PyTorch;
- installed CUDA runtime;
- cached Hugging Face models and datasets;
- existing package caches.

Do not reinstall or upgrade packages when the current environment already satisfies the experiment.

Do not repeatedly query environment metadata after it has been recorded unless the environment changes or a compatibility issue appears.

Store reproducibility metadata once in an appropriate file such as:

`results/environment.json`

and reuse it.

---

## 8. Experimental Integrity

- Never fabricate benchmark results.
- Never discard inconvenient measurements simply because they contradict the hypothesis.
- Validate correctness before interpreting performance.
- Preserve raw measurements.
- Respect CUDA asynchronous execution when measuring latency.
- Keep CPU and GPU workloads equivalent.
- Do not silently change timing boundaries or experimental variables.

If the experiment specification needs to change, explain the reason briefly and update `PROJECT_SPEC.md`.

---

## 9. Scope Discipline

Do not proactively expand the project.

Avoid adding unrelated:

- models;
- precision modes;
- profilers;
- optimizations;
- datasets;
- benchmarks;
- visualizations;
- dependencies.

Complete the current milestone using the simplest adequate implementation.

Suggest extensions only after the core experiment is working or when an extension resolves a specific experimental problem.

---

## 10. Progress Reporting

Do not narrate every command.

During work, report only when:

- an important result is found;
- an assumption is invalidated;
- a blocker appears;
- user action is required;
- a milestone is complete.

At milestone completion, use a compact report:

- Completed
- Validation
- Issues
- Files changed
- Next step

Avoid repeating the full project background.

---

## 11. Terminal Handoff Format

When the user needs to run a heavy command, provide only:

**Run in terminal**

```bash
<exact command>
```

**Working directory**

`<directory>`

**Expected output**

`<file or concise expected result>`

**After completion**

Tell the user exactly what result or file Codex needs next.

Do not include unnecessary explanation unless the command has an important risk or experimental implication.

---

## 12. Default Principle

Optimize for:

**minimum context + minimum computation + maximum information gain**

Prefer one targeted inspection over broad exploration.

Prefer one small validation over a full experiment.

Prefer persistent repository files over repeated chat explanations.

Prefer raw results stored on disk over large results pasted into conversation.

Prefer concise progress reports over narration.