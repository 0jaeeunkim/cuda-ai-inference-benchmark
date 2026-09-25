"""M4/M5 benchmark harness with separate CPU runs and balanced GPU pairs.

The primary E2E path reuses timing.gpu_end_to_end_ms without the internal
forward Events used only by the M3.5 diagnostic. No tokenization, model load,
or buffer allocation occurs in a measured interval.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from benchmark import (CPU_THREADS, configure, cpu_check, gpu_check,
                       read_json, write_json)  # noqa: E402
from data import (BATCH_SIZES_PRIMARY, BATCH_SIZES_PROTOTYPE, DATASET_CONFIG,
                  DATASET_ID, DATASET_REVISION, MODEL_ID,
                  PRIMARY_SEQUENCE_LENGTH, load_pool, make_batch)  # noqa: E402
from timing import (cpu_forward_ms, gpu_end_to_end_ms,
                    gpu_forward_event_ms)  # noqa: E402

RESULTS = ROOT / "results"
PROTOCOL = "m4_cpu_separate_gpu_balanced_v1"
CPU_WARMUP = 5
GPU_WARMUP_PAIRS = 10  # five pairs per order; ten warm-ups per GPU path
CPU_RUNS_FULL = 30
GPU_PAIRS_FULL = 30  # fifteen per order; thirty measurements per GPU path
CPU_RUNS_SMOKE = 2
GPU_PAIRS_SMOKE = 2  # one per order; two measurements per GPU path
COLUMNS = (
    "session_id", "timestamp_utc", "protocol", "batch_size", "sequence_length",
    "run", "measurement_type", "pair_id", "execution_order", "position",
    "latency_ms", "latency_per_sample_ms", "throughput_samples_per_sec",
    "prediction_match",
)
MEASUREMENT_TYPES = ("cpu", "gpu_inference_only", "gpu_end_to_end")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("benchmark_harness.py", "benchmark.py", "data.py", "timing.py"):
        path = ROOT / "src" / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def legacy_hashes() -> dict[str, str]:
    paths = {
        "M3": RESULTS / "raw_runs.csv",
        "M3.5": RESULTS / "m35_diagnostic_raw.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"Missing {name} raw evidence: {path}")
    return {name: file_sha256(path) for name, path in paths.items()}


def check_policy() -> dict:
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    current = {
        "sequence_length": PRIMARY_SEQUENCE_LENGTH,
        "model_dtype": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "torch_get_num_threads": torch.get_num_threads(),
        "torch_get_num_interop_threads": torch.get_num_interop_threads(),
        "host_memory": "pageable",
        "tokenization_in_timed_interval": False,
        "model_load_in_timed_interval": False,
        "buffer_allocation_in_timed_interval": False,
    }
    if (current["sequence_length"] != 64
            or current["float32_matmul_precision"] != "highest"
            or current["cuda_matmul_allow_tf32"]
            or current["cudnn_allow_tf32"]
            or current["torch_get_num_threads"] != CPU_THREADS):
        raise RuntimeError(f"Primary policy mismatch: {current}")
    return current


def pair_order(pair_id: int, batch_index: int) -> str:
    """Alternate orders; reverse the first order on alternate batch sizes."""
    if pair_id < 1:
        raise ValueError("pair_id must be positive")
    ab_first = batch_index % 2 == 0
    is_odd_pair = pair_id % 2 == 1
    return "A->B" if ab_first == is_odd_pair else "B->A"


def prepare_raw_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as handle:
            existing = next(csv.reader(handle))
        if existing != list(COLUMNS):
            raise RuntimeError(f"Existing raw CSV schema mismatch: {path}")


def check_inputs(cpu_inputs, gpu_inputs, gpu_buffers, cpu_output) -> None:
    size = cpu_inputs["input_ids"].shape[0]
    if cpu_output.is_pinned() or cpu_output.dtype != torch.float32:
        raise RuntimeError("E2E output buffer must be pageable FP32 CPU memory")
    for key in ("input_ids", "attention_mask"):
        source = cpu_inputs[key]
        resident = gpu_inputs[key]
        buffer = gpu_buffers[key]
        if (tuple(source.shape) != (size, PRIMARY_SEQUENCE_LENGTH)
                or source.dtype != torch.int64 or source.is_pinned()
                or source.device.type != "cpu"
                or resident.device.type != "cuda"
                or buffer.device.type != "cuda"
                or not torch.equal(source, resident.cpu())):
            raise RuntimeError(f"Primary input policy mismatch: {key}")
        buffer.copy_(source, non_blocking=False)
        if not torch.equal(resident, buffer):
            raise RuntimeError(f"A/B tokenized input mismatch: {key}")


def audit_session(raw_path: Path, session_id: str, batch_sizes: tuple[int, ...],
                  cpu_runs: int, gpu_pairs: int) -> int:
    with raw_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise RuntimeError("Raw CSV header changed")
        rows = [row for row in reader if row["session_id"] == session_id]
    expected = len(batch_sizes) * (cpu_runs + 2 * gpu_pairs)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} session rows, got {len(rows)}")
    groups = defaultdict(list)
    pairs = defaultdict(list)
    for row in rows:
        size = int(row["batch_size"])
        run = int(row["run"])
        kind = row["measurement_type"]
        latency = float(row["latency_ms"])
        if (size not in batch_sizes or int(row["sequence_length"]) != PRIMARY_SEQUENCE_LENGTH
                or row["protocol"] != PROTOCOL or kind not in MEASUREMENT_TYPES
                or row["prediction_match"] != "True"
                or not math.isfinite(latency) or latency <= 0):
            raise RuntimeError(f"Invalid raw row: {row}")
        if not math.isclose(float(row["latency_per_sample_ms"]), latency / size,
                            rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError("Per-sample latency mismatch")
        if not math.isclose(float(row["throughput_samples_per_sec"]),
                            size * 1000 / latency, rel_tol=1e-7, abs_tol=1e-6):
            raise RuntimeError("Throughput mismatch")
        groups[(size, kind)].append(run)
        if kind == "cpu":
            if row["pair_id"] or row["execution_order"] or row["position"]:
                raise RuntimeError("CPU row unexpectedly belongs to a GPU pair")
        else:
            pair_id = int(row["pair_id"])
            position = int(row["position"])
            if run != pair_id or position not in (1, 2):
                raise RuntimeError("GPU run/pair/position mismatch")
            batch_index = batch_sizes.index(size)
            if row["execution_order"] != pair_order(pair_id, batch_index):
                raise RuntimeError("GPU order mismatch")
            pairs[(size, pair_id)].append((position, kind))
    for size in batch_sizes:
        if sorted(groups[(size, "cpu")]) != list(range(1, cpu_runs + 1)):
            raise RuntimeError(f"Incomplete CPU runs: {size}")
        for kind in MEASUREMENT_TYPES[1:]:
            if sorted(groups[(size, kind)]) != list(range(1, gpu_pairs + 1)):
                raise RuntimeError(f"Incomplete GPU runs: {size}, {kind}")
        orders = Counter(pair_order(pair_id, batch_sizes.index(size))
                         for pair_id in range(1, gpu_pairs + 1))
        if orders != {"A->B": gpu_pairs // 2, "B->A": gpu_pairs // 2}:
            raise RuntimeError(f"Unbalanced GPU order: {size}")
        for pair_id in range(1, gpu_pairs + 1):
            expected_kinds = (["gpu_inference_only", "gpu_end_to_end"]
                              if pair_order(pair_id, batch_sizes.index(size)) == "A->B"
                              else ["gpu_end_to_end", "gpu_inference_only"])
            if sorted(pairs[(size, pair_id)]) != list(enumerate(expected_kinds, start=1)):
                raise RuntimeError(f"Incomplete GPU pair: {size}, {pair_id}")
    return len(rows)


def run(mode: str) -> dict:
    if mode not in ("smoke", "full"):
        raise ValueError(mode)
    smoke = mode == "smoke"
    batch_sizes = BATCH_SIZES_PROTOTYPE if smoke else BATCH_SIZES_PRIMARY
    cpu_runs = CPU_RUNS_SMOKE if smoke else CPU_RUNS_FULL
    gpu_pairs = GPU_PAIRS_SMOKE if smoke else GPU_PAIRS_FULL
    if gpu_pairs % 2:
        raise RuntimeError("GPU pair count must be even")
    policy = check_policy()
    protocol_hash = protocol_sha256()
    evidence_before = legacy_hashes()
    environment = read_json(RESULTS / "environment.json")
    if (environment.get("primary_sequence_length") != PRIMARY_SEQUENCE_LENGTH
            or environment.get("model_id") != MODEL_ID
            or environment.get("dataset_id") != DATASET_ID
            or environment.get("dataset_revision") != DATASET_REVISION):
        raise RuntimeError("M0/M3 environment metadata mismatch")
    if not smoke:
        gate = read_json(RESULTS / "m4_smoke_validation.json")
        if (not gate.get("passed") or gate.get("protocol_sha256") != protocol_hash
                or gate.get("legacy_raw_sha256") != evidence_before
                or gate.get("batch_sizes") != list(BATCH_SIZES_PROTOTYPE)):
            raise RuntimeError("M4 smoke gate missing or stale; run --mode smoke")

    pool = load_pool()
    if pool._fingerprint != environment.get("pool_fingerprint"):
        raise RuntimeError("Fixed SST-2 pool fingerprint changed")
    model_revision = environment.get("model_revision")
    if not model_revision:
        raise RuntimeError("Model revision missing from M3 metadata")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True,
                                               revision=model_revision)
    inputs = {size: make_batch(pool, tokenizer, size, PRIMARY_SEQUENCE_LENGTH)
              for size in batch_sizes}
    cpu_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=model_revision).float().eval().cpu()
    if (getattr(cpu_model.config, "_commit_hash", None) != model_revision
            or any(p.dtype != torch.float32 for p in cpu_model.parameters())):
        raise RuntimeError("Model revision or FP32 policy changed")
    gpu_model = copy.deepcopy(cpu_model).to("cuda").eval()
    cpu_correct = cpu_check(inputs, cpu_model)
    gpu_correct = gpu_check(inputs, cpu_model, gpu_model)
    if not all(item["passed"] for item in cpu_correct.values()) or not all(
            item["passed"] for item in gpu_correct.values()):
        raise RuntimeError("Correctness gate failed")

    session_id = uuid.uuid4().hex
    raw_path = RESULTS / ("m4_smoke_raw.csv" if smoke else "primary_raw_runs.csv")
    prepare_raw_file(raw_path)
    with raw_path.open("a", newline="") as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()
        def record(size: int, run_index: int, kind: str, latency: float,
                   pair_id: int | None = None, order: str = "",
                   position: int | None = None) -> None:
            if not math.isfinite(latency) or latency <= 0:
                raise RuntimeError("Invalid measured latency")
            writer.writerow({
                "session_id": session_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": PROTOCOL,
                "batch_size": size,
                "sequence_length": PRIMARY_SEQUENCE_LENGTH,
                "run": run_index,
                "measurement_type": kind,
                "pair_id": "" if pair_id is None else pair_id,
                "execution_order": order,
                "position": "" if position is None else position,
                "latency_ms": f"{latency:.9f}",
                "latency_per_sample_ms": f"{latency / size:.9f}",
                "throughput_samples_per_sec": f"{size * 1000 / latency:.9f}",
                "prediction_match": True,
            })
            handle.flush()

        # All CPU timing finishes before any measured GPU pair begins.
        for size in batch_sizes:
            for _ in range(CPU_WARMUP):
                cpu_forward_ms(cpu_model, inputs[size])
            for run_index in range(1, cpu_runs + 1):
                record(size, run_index, "cpu", cpu_forward_ms(cpu_model, inputs[size]))

        for batch_index, size in enumerate(batch_sizes):
            cpu_inputs = inputs[size]
            gpu_inputs = {key: value.to("cuda") for key, value in cpu_inputs.items()}
            gpu_buffers = {key: torch.empty_like(value, device="cuda")
                           for key, value in cpu_inputs.items()}
            cpu_output = torch.empty((size, 2), dtype=torch.float32)
            check_inputs(cpu_inputs, gpu_inputs, gpu_buffers, cpu_output)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            def A() -> float:
                return gpu_forward_event_ms(gpu_model, gpu_inputs, start, end)

            def B() -> float:
                return gpu_end_to_end_ms(gpu_model, cpu_inputs, gpu_buffers,
                                         cpu_output)

            # One untimed correctness check of B's actual transfer buffers.
            with torch.no_grad():
                for key in ("input_ids", "attention_mask"):
                    gpu_buffers[key].copy_(cpu_inputs[key], non_blocking=False)
                cpu_output.copy_(gpu_model(**gpu_buffers).logits, non_blocking=False)
                torch.cuda.synchronize()
                expected = cpu_correct[str(size)]["predicted_labels"]
                if cpu_output.argmax(-1).tolist() != expected:
                    raise RuntimeError(f"E2E buffer output mismatch: batch {size}")

            for warmup_pair in range(1, GPU_WARMUP_PAIRS + 1):
                if pair_order(warmup_pair, batch_index) == "A->B":
                    A(); B()
                else:
                    B(); A()
            for pair_id in range(1, gpu_pairs + 1):
                order = pair_order(pair_id, batch_index)
                if order == "A->B":
                    a_ms = A()
                    record(size, pair_id, "gpu_inference_only", a_ms,
                           pair_id, order, 1)
                    b_ms = B()
                    record(size, pair_id, "gpu_end_to_end", b_ms,
                           pair_id, order, 2)
                else:
                    b_ms = B()
                    record(size, pair_id, "gpu_end_to_end", b_ms,
                           pair_id, order, 1)
                    a_ms = A()
                    record(size, pair_id, "gpu_inference_only", a_ms,
                           pair_id, order, 2)
            for key in ("input_ids", "attention_mask"):
                if not torch.equal(gpu_inputs[key], gpu_buffers[key]):
                    raise RuntimeError(f"A/B inputs diverged after batch {size}")
    row_count = audit_session(raw_path, session_id, batch_sizes, cpu_runs, gpu_pairs)
    evidence_after = legacy_hashes()
    if evidence_before != evidence_after:
        raise RuntimeError("Legacy M3/M3.5 raw evidence changed")
    report = {
        "passed": True,
        "mode": mode,
        "session_id": session_id,
        "protocol": PROTOCOL,
        "protocol_sha256": protocol_hash,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_sizes": list(batch_sizes),
        "cpu_runs_per_batch": cpu_runs,
        "cpu_warmup_per_batch": CPU_WARMUP,
        "gpu_pairs_per_batch": gpu_pairs,
        "gpu_pairs_per_order": gpu_pairs // 2,
        "gpu_warmup_pairs_per_batch": GPU_WARMUP_PAIRS,
        "gpu_measurements_per_path_per_batch": gpu_pairs,
        "raw_rows_in_session": row_count,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "legacy_raw_sha256": evidence_after,
        "policy": policy,
        "model_revision": model_revision,
        "dataset_revision": DATASET_REVISION,
        "pool_fingerprint": pool._fingerprint,
        "cpu_correctness": cpu_correct,
        "gpu_correctness": gpu_correct,
        "timing_boundaries": {
            "cpu": "perf_counter_ns around CPU forward; separate block",
            "gpu_inference_only": "CUDA Events around forward on resident GPU inputs; end event synchronized",
            "gpu_end_to_end": "synchronized host clock: pageable CPU input H2D, forward, logits D2H; no internal forward Events",
        },
    }
    if smoke:
        write_json(RESULTS / "m4_smoke_validation.json", report)
    write_json(RESULTS / f"{mode}_session_{session_id}.json", report)
    print(json.dumps({"passed": True, "mode": mode, "session_id": session_id,
                      "batch_sizes": list(batch_sizes), "raw_rows_in_session": row_count,
                      "raw_path": report["raw_path"],
                      "gpu_pairs_per_order": gpu_pairs // 2,
                      "legacy_raw_preserved": True}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
