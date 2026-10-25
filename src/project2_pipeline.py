"""Project 2 A-D runners. Heavy modes are for the user's server terminal."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
import uuid
from pathlib import Path

import project2_protocol as p


def cache_policy(online=False):
    # Separate, explicit cache. Existing environment variables may override it.
    os.environ.setdefault("HF_HOME", str(p.ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    os.environ["HF_HUB_OFFLINE"] = "0" if online else "1"
    os.environ["HF_DATASETS_OFFLINE"] = "0" if online else "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return {k: os.environ[k] for k in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
                                     "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}


def packages():
    result = {}
    for name in ("torch", "transformers", "datasets", "tokenizers", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def configure():
    import torch
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment")
    # Independent Project 2 policy; Project 1 pageable guards stay intact.
    if (torch.get_float32_matmul_precision() != "highest"
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise RuntimeError("Project 2 FP32 policy mismatch")
    return torch


def tensor_hash(tensor):
    return p.digest_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def token_hash(tokens):
    return p.json_hash({k: tensor_hash(v) for k, v in sorted(tokens.items())})


def load_workload(count, online=False):
    cache_policy(online)
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pool = load_dataset(p.DATASET_ID, "sst2", split="validation", revision=p.DATASET_REVISION)
    if len(pool) != 872 or list(pool["idx"]) != list(range(872)):
        raise RuntimeError("SST-2 published sample coverage/order changed")
    dataset_hash = p.json_hash([{k: pool[i][k] for k in ("idx", "sentence", "label")}
                               for i in range(len(pool))])
    tokenizer = AutoTokenizer.from_pretrained(p.MODEL_ID, revision=p.MODEL_REVISION,
                                              use_fast=True, local_files_only=not online)
    model = AutoModelForSequenceClassification.from_pretrained(
        p.MODEL_ID, revision=p.MODEL_REVISION, local_files_only=not online).float().eval()
    if (getattr(model.config, "_commit_hash", None) != p.MODEL_REVISION
            or any(x.dtype != torch.float32 for x in model.parameters()) or model.training):
        raise RuntimeError("Pinned model/eval/FP32 policy mismatch")
    encoded = tokenizer(list(pool["sentence"][:count]), padding="max_length", truncation=True,
                        max_length=64, return_tensors="pt")
    tokens = {k: encoded[k].contiguous() for k in ("input_ids", "attention_mask")}
    if any(v.shape != (count, 64) or v.dtype != torch.int64 or v.is_pinned()
           for v in tokens.values()):
        raise RuntimeError("Project 2 token policy mismatch")
    if not bool(((tokens["attention_mask"] == 0) | (tokens["attention_mask"] == 1)).all()):
        raise RuntimeError("Invalid attention mask")
    snapshot = Path(snapshot_download(p.MODEL_ID, revision=p.MODEL_REVISION, local_files_only=True))
    # Full cached model/tokenizer file hashes, computed once per process outside timing.
    model_files = {f.name: p.sha256(f) for f in sorted(snapshot.iterdir()) if f.is_file()}
    old = p.read_json(p.ROOT / "results/environment.json")
    fingerprint = pool._fingerprint
    if fingerprint != p.OLD_FINGERPRINT:
        raise RuntimeError(f"Dataset fingerprint changed ({fingerprint}); inspect library/cache differences "
                           "before approving a new environment snapshot")
    identity = {
        "model_id": p.MODEL_ID, "model_revision": p.MODEL_REVISION,
        "dataset_id": p.DATASET_ID, "dataset_revision": p.DATASET_REVISION,
        "dataset_fingerprint": fingerprint, "dataset_content_sha256": dataset_hash,
        "model_files_sha256": model_files, "model_snapshot": str(snapshot),
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_is_fast": tokenizer.is_fast,
        "tokenizer_backend": str(getattr(tokenizer, "backend", None)),
        "tokenizer_requested_revision": p.MODEL_REVISION,
        "tokenizer_reported_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "prefix_137_inputs_sha256": token_hash({k: v[:137] for k, v in tokens.items()})
            if count >= 137 else None,
        "legacy_datasets_version": old.get("packages", {}).get("datasets"),
    }
    return model, tokens, list(pool["label"][:count]), identity


def runtime_identity(torch):
    gpu = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "packages": packages(), "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu.name, "gpu_uuid": str(getattr(gpu, "uuid", "unknown")),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_bytes": gpu.total_memory, "cuda_arch_list": torch.cuda.get_arch_list(),
        "driver": p.command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_version": torch.backends.cudnn.version(), "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cache": cache_policy(),
    }


def save_environment(meta):
    meta["snapshot_id"] = p.json_hash({k: v for k, v in meta.items()
                                      if k not in ("recorded_at_utc", "snapshot_id")})
    archived = p.RESULTS / "environments" / f"{meta['snapshot_id']}.json"
    if not archived.exists():
        p.write_json(archived, meta)
    p.write_json(p.RESULTS / "environment.json", p.read_json(archived))
    return meta


def environment():
    meta = {"recorded_at_utc": p.now(), "status": "blocked", "python": sys.version,
            "python_executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages(), "cache": cache_policy(),
            "logical_cpus": os.cpu_count(),
            "cpu_model": next((line.split(":", 1)[1].strip()
                               for line in Path("/proc/cpuinfo").read_text().splitlines()
                               if line.startswith("model name")), None),
            "cuda_runtime": None, "cuda_fp32_smoke": "not_run", "model_forward_smoke": "not_run",
            "gpu_driver_query": p.command(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                                           "--format=csv,noheader"]),
            "conda_executable": p.command(["bash", "-c", "command -v conda"]),
            "project1": p.verify_project1(), "blockers": []}
    missing = [name for name, version in meta["packages"].items() if version is None]
    if missing:
        meta["blockers"].append("Missing packages: " + ", ".join(missing))
    else:
        try:
            torch = configure()
            meta["runtime"] = runtime_identity(torch)
            meta["cuda_runtime"] = torch.version.cuda
            with torch.inference_mode():
                x = torch.ones((32, 32), dtype=torch.float32, device="cuda")
                y = x @ x
                torch.cuda.synchronize()
                if not torch.equal(y.cpu(), torch.full((32, 32), 32.0)):
                    raise RuntimeError("FP32 CUDA matrix multiplication failed")
                meta["cuda_fp32_smoke"] = "passed"
                model, tokens, _, workload = load_workload(1)
                model = model.to("cuda").eval()
                logits = model(**{k: v.to("cuda") for k, v in tokens.items()}).logits.cpu()
                if logits.shape != (1, 2) or not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("Cached model forward failed")
                meta["model_forward_smoke"] = {"passed": True, "logits_sha256": tensor_hash(logits)}
                # Prefix hash is supplied by smoke and compared through its gate.
                meta["workload"] = workload
        except Exception as exc:
            meta["blockers"].append(f"{type(exc).__name__}: {exc}")
    if not meta["blockers"]:
        meta["status"] = "ready"
    save_environment(meta)
    print(json.dumps({"status": meta["status"], "blockers": meta["blockers"],
                      "snapshot": "results/project2/environment.json"}))
    return meta["status"] == "ready"


class Prepared:
    """Persistent buffers. CPU inputs stay immutable until the final device sync.

    Copy writes slot i%2; compute reads it. In D, consumed[i-2] protects reuse.
    Compute writes each disjoint CPU output span by D2H; CPU reads after sync.
    GPU logits are allocated and consumed on the SAME compute stream; CUDA's
    stream-aware allocator cannot recycle their storage before the queued D2H.
    Persistent device inputs are created before setup sync and live past all use.
    """

    def __init__(self, torch, model, tokens, batch_size, variant):
        prepare_started = time.perf_counter_ns()
        self.torch, self.model, self.variant = torch, model, variant
        self.count, self.batch = len(tokens["input_ids"]), batch_size
        self.compute = torch.cuda.Stream()
        self.copy = torch.cuda.Stream() if variant == "D" else self.compute
        pinned = variant != "A"
        started = time.perf_counter_ns()
        self.inputs = {k: torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=pinned)
                       for k, v in tokens.items()}
        for k, v in tokens.items():
            self.inputs[k].copy_(v)
        self.output = torch.empty((self.count, 2), dtype=torch.float32, device="cpu", pin_memory=pinned)
        self.host_setup_ms = (time.perf_counter_ns() - started) / 1e6
        self.host_bytes = sum(v.numel() * v.element_size() for v in self.inputs.values())
        self.host_bytes += self.output.numel() * self.output.element_size()
        self.slots = [{k: torch.empty((batch_size, 64), dtype=torch.int64, device="cuda")
                       for k in self.inputs} for _ in range(2)]
        self.plan = self.make_plan(p.spans(batch_size, self.count))
        warm_spans = p.spans(batch_size, self.count)[:3]
        if self.plan[-1]["end"] - self.plan[-1]["start"] != batch_size:
            warm_spans = p.spans(batch_size, self.count)[:2] + [p.spans(batch_size, self.count)[-1]]
        self.warm_plan = self.make_plan(warm_spans)
        # Event objects are lazy in PyTorch: materialize CUDA handles before timing.
        for item in self.plan + self.warm_plan:
            for key in ("ready", "consumed"):
                if item[key] is not None:
                    item[key].record(self.compute)
        torch.cuda.synchronize()  # joins model/default-stream allocations and event setup
        self.check_policy(tokens)
        self.layout["buffer_stream_event_setup_ms"] = (time.perf_counter_ns() - prepare_started) / 1e6

    def make_plan(self, ranges):
        plan = []
        for i, (start, end) in enumerate(ranges):
            size, slot = end - start, i % 2
            plan.append({"start": start, "end": end, "slot": slot,
                         "host": {k: v[start:end] for k, v in self.inputs.items()},
                         "device": {k: v[:size] for k, v in self.slots[slot].items()},
                         "output": self.output[start:end],
                         "ready": self.torch.cuda.Event() if self.variant == "D" else None,
                         "consumed": self.torch.cuda.Event() if self.variant == "D" else None})
        return plan

    def check_policy(self, tokens):
        torch = self.torch
        pinned = self.variant != "A"
        if self.output.is_pinned() != pinned or any(v.is_pinned() != pinned for v in self.inputs.values()):
            raise RuntimeError("Project 2 variant memory policy mismatch")
        if any(not torch.equal(tokens[k], self.inputs[k]) for k in tokens):
            raise RuntimeError("Variant token values changed")
        regions = [(v.data_ptr(), v.data_ptr() + v.numel() * v.element_size())
                   for slot in self.slots for v in slot.values()]
        for i, (a, b) in enumerate(regions):
            if any(max(a, c) < min(b, d) for c, d in regions[i + 1:]):
                raise RuntimeError("Device input storage aliases")
        outputs = [(x["output"].data_ptr(), x["output"].data_ptr() + x["output"].numel() * 4)
                   for x in self.plan]
        if any(b != c for (_, b), (c, _) in zip(outputs, outputs[1:])):
            raise RuntimeError("CPU output spans overlap or have gaps")
        self.layout = {"variant": self.variant, "host_pinned": pinned,
                       "host_bytes": self.host_bytes, "pinned_bytes": self.host_bytes if pinned else 0,
                       "host_allocation_and_copy_setup_ms": self.host_setup_ms,
                       "device_input_slots": 2, "device_input_bytes": 2 * 2 * self.batch * 64 * 8,
                       "device_storage_ranges": regions, "output_storage_ranges": outputs,
                       "stream_ids": {"compute": self.compute.cuda_stream, "copy": self.copy.cuda_stream},
                       "actual_batch_sizes": [x["end"] - x["start"] for x in self.plan]}

    def enqueue(self, plan=None, inspect=False, trace=False):
        torch = self.torch
        plan = self.plan if plan is None else plan
        observed = []

        def marker(name):
            return torch.profiler.record_function(name) if trace else contextlib.nullcontext()

        for i, item in enumerate(plan):
            with torch.cuda.stream(self.copy), marker(f"{self.variant}/H2D/{i}/slot{item['slot']}"):
                if self.variant == "D" and i >= 2:
                    self.copy.wait_event(plan[i - 2]["consumed"])
                for key in self.inputs:
                    item["device"][key].copy_(item["host"][key], non_blocking=self.variant in "CD")
                if self.variant == "D":
                    item["ready"].record(self.copy)
            with torch.cuda.stream(self.compute):
                if self.variant == "D":
                    self.compute.wait_event(item["ready"])
                with marker(f"{self.variant}/forward/{i}/slot{item['slot']}"):
                    logits = self.model(**item["device"]).logits
                if self.variant == "D":
                    # Last input use is forward completion, not its Python return.
                    item["consumed"].record(self.compute)
                with marker(f"{self.variant}/D2H/{i}/rows{item['start']}:{item['end']}"):
                    item["output"].copy_(logits, non_blocking=self.variant in "CD")
                if inspect:
                    observed.append({"start": item["start"], "end": item["end"], "slot": item["slot"],
                                     "input_shape": list(item["device"]["input_ids"].shape),
                                     "logits_shape": list(logits.shape),
                                     "output_address": item["output"].data_ptr()})
        return observed

    def warm(self, repetitions):
        for _ in range(repetitions):
            self.enqueue(self.warm_plan)
            self.torch.cuda.synchronize()

    def run(self, inspect=False, trace=False):
        self.torch.cuda.synchronize()
        self.output.fill_(float("nan"))  # sentinel/coverage initialization outside timer
        started = time.perf_counter_ns()
        observed = self.enqueue(inspect=inspect, trace=trace)
        self.torch.cuda.synchronize()  # same completion boundary for A, B, C, D
        total_ms = (time.perf_counter_ns() - started) / 1e6
        if not math.isfinite(total_ms) or total_ms <= 0:
            raise RuntimeError("Invalid synchronized host time")
        return total_ms, observed


def compare(torch, output, reference):
    shape_ok = tuple(output.shape) == tuple(reference.shape) and output.dtype == torch.float32
    finite = bool(torch.isfinite(output).all()) and bool(torch.isfinite(reference).all())
    prediction = bool(torch.equal(output.argmax(-1), reference.argmax(-1)))
    close = bool(torch.allclose(output, reference, atol=1e-5, rtol=1e-4))
    result = {"passed": shape_ok and finite and prediction and close,
              "shape": list(output.shape), "finite": finite, "prediction_match": prediction,
              "logits_allclose": close, "atol": 1e-5, "rtol": 1e-4,
              "max_abs_error": float((output - reference).abs().max()),
              "output_sha256": tensor_hash(output)}
    return result


def sync_probe(torch):
    """Tiny behavior checks, not speed comparisons and never part of main time."""
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    host = torch.tensor([7.0], pin_memory=True)
    output = torch.empty(1, pin_memory=True)
    device = torch.empty(1, device="cuda")
    ready, done = torch.cuda.Event(), torch.cuda.Event()
    torch.cuda.synchronize()
    with torch.cuda.stream(a):
        device.copy_(host, non_blocking=True)
        ready.record(a)
    with torch.cuda.stream(b):
        b.wait_event(ready)  # GPU-side wait; does not establish CPU-visible output
        output.copy_(device, non_blocking=True)
        done.record(b)
    done.synchronize()
    if output.item() != 7:
        raise RuntimeError("event.synchronize/stream.wait_event check failed")
    with torch.cuda.stream(b):
        output.copy_(device, non_blocking=True)
    b.synchronize()
    if output.item() != 7:
        raise RuntimeError("stream.synchronize check failed")
    with torch.cuda.stream(b):
        output.copy_(device, non_blocking=True)
    torch.cuda.synchronize()
    if output.item() != 7:
        raise RuntimeError("device synchronize check failed")
    return {"passed": True, "device_synchronize": "all device streams; main completion",
            "stream_synchronize": "selected stream; CPU waits",
            "event_synchronize": "recorded event; CPU waits before reading",
            "stream_wait_event": "GPU dependency; producer record submitted before wait"}


def prepare():
    cache_policy(online=True)
    model, _, _, workload = load_workload(1, online=True)
    p.write_json(p.RESULTS / "cache_preparation.json", {
        "completed_at_utc": p.now(), "workload": workload, "cache": cache_policy(online=True),
        "packages": packages(), "model_parameters": sum(v.numel() for v in model.parameters())})
    print(json.dumps({"cache": "prepared", "output": "results/project2/cache_preparation.json"}))


def identity_for(runtime, workload, environment_sha256):
    return {"protocol_sha256": p.protocol_hash(), "runtime_sha256": p.json_hash(runtime),
            "workload_sha256": p.json_hash(workload), "environment_sha256": environment_sha256,
            "project1_manifest_sha256": p.sha256(p.RESULTS / "project1_manifest.json")}


def verify_environment(runtime, workload):
    path = p.RESULTS / "environment.json"
    if not path.exists():
        raise RuntimeError("Run --mode env after cache preparation")
    env = p.read_json(path)
    if env["status"] != "ready" or env["runtime"] != runtime:
        raise RuntimeError("Environment snapshot missing/stale; run --mode env")
    # env uses one input row; the independently recorded full dataset/model identity must agree.
    left = {k: v for k, v in workload.items() if k != "prefix_137_inputs_sha256"}
    right = {k: v for k, v in env["workload"].items() if k != "prefix_137_inputs_sha256"}
    if left != right:
        raise RuntimeError("Cached workload differs from environment verification")
    return p.sha256(path)


def gpu_state():
    return p.command(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu",
                      "--format=csv,noheader"])


def run(mode, variants):
    p.verify_project1()
    cache_policy()
    session_id = p.now().replace(":", "").replace("+", "_") + "_" + uuid.uuid4().hex[:8]
    session = p.RESULTS / ("smoke" if mode == "smoke" else "sessions") / session_id
    session.mkdir(parents=True, exist_ok=False)
    meta = {"session_id": session_id, "mode": mode, "variants": variants, "status": "running",
            "started_at_utc": p.now(), "command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
            "source_commit": p.command(["git", "rev-parse", "HEAD"]), "sources": p.sources(),
            "git_status": p.command(["git", "status", "--porcelain"]),
            "protocol_id": p.PROTOCOL_ID, "protocol_sha256": p.protocol_hash(), "policy": p.POLICY,
            "gpu_state_start": gpu_state(), "setups": {}, "references": {}}
    correctness = {"runs": [], "coverage": [], "references": {}}
    p.write_json(session / "metadata.json", meta)
    try:
        torch = configure()
        max_count = 137 if mode == "smoke" else 872
        model, tokens, labels, workload = load_workload(max_count)
        runtime = runtime_identity(torch)
        environment_hash = verify_environment(runtime, workload)
        identity = identity_for(runtime, workload, environment_hash)
        env = p.read_json(p.RESULTS / "environment.json")
        meta.update({"identity": identity, "runtime": runtime, "workload": workload,
                     "environment_sha256": environment_hash, "inputs_sha256": token_hash(tokens),
                     "environment_snapshot": f"results/project2/environments/{env['snapshot_id']}.json",
                     "labels_sha256": p.json_hash(labels), "row_indices": list(range(max_count))})
        if mode != "smoke":
            meta["smoke_gate"] = p.require_gate("smoke", identity)
        if mode == "full":
            meta["correctness_gate"] = p.require_gate("correctness", identity)
        with torch.inference_mode():
            model = model.to("cuda").eval()
            if mode == "smoke":
                correctness["synchronization_api_probe"] = sync_probe(torch)
            cases = p.SMOKE_CASES if mode == "smoke" else tuple((b, 872) for b in p.BATCHES)
            for batch, count in cases:
                selected = {k: v[:count] for k, v in tokens.items()}
                runners = {v: Prepared(torch, model, selected, batch, v) for v in variants}
                # Reference A uses a separate blocking pass and full CPU output snapshot.
                baseline = runners.get("A") or Prepared(torch, model, selected, batch, "A")
                _, observed = baseline.run(inspect=True)
                reference = baseline.output.clone()
                if reference.shape != (count, 2) or not bool(torch.isfinite(reference).all()):
                    raise RuntimeError("Invalid A reference logits")
                ref_name = f"reference_b{batch}_n{count}.pt"
                torch.save({"logits": reference, "labels": labels[:count], "row_indices": list(range(count))},
                           session / ref_name)
                reference_info = {"logits_sha256": tensor_hash(reference), "file": ref_name,
                                  "file_sha256": p.sha256(session / ref_name),
                                  "input_sha256": token_hash(selected),
                                  "accuracy_vs_dataset_labels": float((reference.argmax(-1) == torch.tensor(labels[:count])).float().mean())}
                correctness["references"][str(batch)] = reference_info
                meta["references"][str(batch)] = reference_info
                meta["setups"][str(batch)] = {v: r.layout for v, r in runners.items()}
                for variant, runner in runners.items():
                    runner.warm(1 if mode == "smoke" else 5)
                    _, records = runner.run(inspect=True)
                    check = compare(torch, runner.output, reference)
                    expected = p.spans(batch, count)
                    if ([(r["start"], r["end"]) for r in records] != expected
                            or any(r["slot"] != i % 2 or r["input_shape"] != [b - a, 64]
                                   or r["logits_shape"] != [b - a, 2]
                                   for i, ((a, b), r) in enumerate(zip(expected, records)))):
                        raise RuntimeError("Observed forward/tail/slot/output coverage mismatch")
                    correctness["coverage"].append({"batch_size": batch, "variant": variant,
                                                     "records": records, "check": check})
                    if not check["passed"]:
                        raise RuntimeError(f"Untimed correctness failed: {batch}/{variant}")
                p.write_json(session / "metadata.json", meta)
                for b, n, ri, eo, position, variant in p.expected_runs(mode, variants):
                    if b != batch:
                        continue
                    runner = runners[variant]
                    total, _ = runner.run()
                    check = compare(torch, runner.output, reference)  # outside timer, every run
                    record = {"batch_size": batch, "sample_count": count, "variant": variant,
                              "round": ri, "position": position, **check}
                    correctness["runs"].append(record)
                    p.append_row(session / "raw.csv", {
                        "session_id": session_id, "protocol_id": p.PROTOCOL_ID,
                        "protocol_sha256": meta["protocol_sha256"], "source_commit": meta["source_commit"],
                        "timestamp_utc": p.now(), "batch_size": batch, "sample_count": count,
                        "variant": variant, "round": ri, "execution_order": eo, "position": position,
                        "forward_count": len(runner.plan), "final_batch_size": runner.plan[-1]["end"] - runner.plan[-1]["start"],
                        "total_ms": total, "throughput_samples_per_sec": count * 1000 / total,
                        "correctness_pass": check["passed"], "max_abs_error": check["max_abs_error"],
                        "output_sha256": check["output_sha256"]})
                    p.write_json(session / "correctness.json", correctness)
                    if not check["passed"]:
                        raise RuntimeError(f"Timed correctness failed: {batch}/{variant}/{ri}")
                del runners, baseline
        p.write_json(session / "correctness.json", correctness)
        meta.update({"status": "completed", "finished_at_utc": p.now(),
                     "gpu_state_end": gpu_state(), "project1_after": p.verify_project1(),
                     "raw_sha256": p.sha256(session / "raw.csv"),
                     "correctness_sha256": p.sha256(session / "correctness.json")})
        p.audit_raw(session / "raw.csv", meta)
        p.write_json(session / "metadata.json", meta)
        if mode in ("smoke", "correctness") and variants == p.VARIANTS:
            p.publish_gate(session, identity, mode)
        from project2_analysis import summarize
        summarize(session)
        print(json.dumps({"status": "completed", "session": str(session.relative_to(p.ROOT)),
                          "mode": mode, "protocol_sha256": meta["protocol_sha256"]}))
    except BaseException as exc:
        # Persistent cross-stream buffers must outlive in-flight work on errors too.
        if "torch" in locals():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        meta.update({"status": "failed", "finished_at_utc": p.now(),
                     "error": f"{type(exc).__name__}: {exc}"})
        p.write_json(session / "metadata.json", meta)
        p.write_json(session / "correctness.json", correctness)
        (session / "error.txt").write_text(traceback.format_exc())
        print(f"Failed session preserved: {session}", file=sys.stderr)
        raise


def trace_run():
    cache_policy()
    torch = configure()
    model, tokens, _, workload = load_workload(137)
    runtime = runtime_identity(torch)
    identity = identity_for(runtime, workload, verify_environment(runtime, workload))
    smoke = p.require_gate("smoke", identity)
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("PyTorch profiler CUDA activity unavailable; no overlap claim possible")
    path = p.RESULTS / "diagnostics" / (uuid.uuid4().hex + "_trace")
    path.mkdir(parents=True, exist_ok=False)
    meta = {"status": "running", "identity": identity, "smoke_gate": smoke,
            "timing_used_in_benchmark": False, "overlap": "uninspected"}
    try:
        with torch.inference_mode():
            model = model.to("cuda").eval()
            for variant in "ACD":
                runner = Prepared(torch, model, {k: v[:19] for k, v in tokens.items()}, 8, variant)
                runner.warm(1)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    runner.run(trace=True)
                trace_dir = path / "traces"
                trace_dir.mkdir(exist_ok=True)
                trace_path = trace_dir / f"{variant}_b8_n19.json"
                prof.export_chrome_trace(str(trace_path))
                # A CPU-only fallback trace must not pass as CUDA activity evidence.
                events = json.loads(trace_path.read_text()).get("traceEvents", [])
                cuda_events = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy")]
                if not cuda_events:
                    raise RuntimeError("Exported trace has no GPU kernel/copy activities")
        meta["status"] = "completed"
    except BaseException as exc:
        meta.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        torch.cuda.synchronize()
        p.write_json(path / "metadata.json", meta)
    print(json.dumps({"trace_session": str(path.relative_to(p.ROOT)), "overlap": "requires trace inspection"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("check", "env", "prepare", "smoke", "correctness", "full", "trace"))
    parser.add_argument("--variants", default=p.VARIANTS, help="Smoke only: ABC or ABCD (default)")
    args = parser.parse_args()
    if args.mode == "check":
        result = {**p.static_check(), "protocol_sha256": p.protocol_hash(), "checked_at_utc": p.now()}
        p.write_json(p.RESULTS / "static_validation.json", result)
        print(json.dumps(result))
    elif args.mode == "env":
        if not environment():
            raise SystemExit(2)
    elif args.mode == "prepare":
        prepare()
    elif args.mode == "trace":
        trace_run()
    else:
        if args.variants not in ("ABC", "ABCD") or (args.mode != "smoke" and args.variants != "ABCD"):
            parser.error("Only --mode smoke permits --variants ABC")
        run(args.mode, args.variants)


if __name__ == "__main__":
    main()
