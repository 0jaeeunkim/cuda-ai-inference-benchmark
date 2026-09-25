"""Whole-SST-2 application validation, separate from the M5 primary sweep.

One timed run processes each of the 872 validation rows exactly once. CPU and
GPU E2E total host time includes the forward loop and, on GPU, each H2D input
copy and D2H logits copy. Model load, tokenization, and buffer allocation are
outside timing. Run --mode smoke before handing --mode full to the user.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from benchmark import read_json, write_json  # noqa: E402
from benchmark_harness import check_policy  # noqa: E402
from data import (DATASET_REVISION, MODEL_ID, PRIMARY_SEQUENCE_LENGTH,
                  load_pool)  # noqa: E402

RESULTS = ROOT / "results"
BATCHES = (1, 8, 32, 64)
SAMPLE_COUNT = 872
REPEATS = 5
WARMUP_PER_SHAPE = 5
PROTOCOL = "application_validation_whole_sst2_v1"
RAW = RESULTS / "application_validation_raw.csv"
SUMMARY = RESULTS / "application_validation_summary.csv"
SMOKE_GATE = RESULTS / "application_validation_smoke.json"
RAW_COLUMNS = (
    "session_id", "timestamp_utc", "protocol", "batch_size", "run",
    "measurement_type", "sample_count", "forward_invocations",
    "final_batch_size", "total_execution_ms", "throughput_samples_per_sec",
    "prediction_match",
)
SUMMARY_COLUMNS = (
    "session_id", "batch_size", "sample_count", "forward_invocations",
    "final_batch_size", "runs_per_path", "p95_method",
    "cpu_median_total_ms", "cpu_p95_total_ms",
    "gpu_e2e_median_total_ms", "gpu_e2e_p95_total_ms",
    "cpu_median_derived_throughput_samples_per_sec",
    "gpu_e2e_median_derived_throughput_samples_per_sec",
    "gpu_e2e_speedup_vs_cpu",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_sha256() -> str:
    h = hashlib.sha256()
    for name in ("application_validation.py", "benchmark.py",
                 "benchmark_harness.py", "data.py", "timing.py"):
        path = ROOT / "src" / name
        h.update(name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def evidence_hashes() -> dict[str, str]:
    paths = [
        RESULTS / "raw_runs.csv", RESULTS / "m35_diagnostic_raw.csv",
        RESULTS / "primary_raw_runs.csv", RESULTS / "summary.csv",
        RESULTS / "analysis_metadata.json", RESULTS / "analysis.md",
        *sorted((RESULTS / "plots").glob("*.svg")),
    ]
    if len(paths) != 11 or any(not path.is_file() for path in paths):
        raise RuntimeError("M3–M6 evidence is incomplete")
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def chunks(size: int, count: int = SAMPLE_COUNT) -> list[tuple[int, int]]:
    spans = [(start, min(start + size, count)) for start in range(0, count, size)]
    covered = [i for start, end in spans for i in range(start, end)]
    if len(covered) != count or covered != list(range(count)):
        raise RuntimeError(f"Missing or duplicate sample at batch size {size}")
    if len(spans) != math.ceil(count / size) or spans[-1][1] != count:
        raise RuntimeError("Incorrect final batch")
    return spans


def check_model(cpu_model, revision: str) -> None:
    if (getattr(cpu_model.config, "_commit_hash", None) != revision
            or not all(p.dtype == torch.float32 for p in cpu_model.parameters())):
        raise RuntimeError("Model revision or FP32 policy mismatch")


def make_buffers(spans: list[tuple[int, int]]) -> dict[int, tuple[dict, torch.Tensor]]:
    buffers = {}
    for size in sorted({end - start for start, end in spans}):
        inputs = {key: torch.empty((size, PRIMARY_SEQUENCE_LENGTH), dtype=torch.int64,
                                   device="cuda") for key in ("input_ids", "attention_mask")}
        output = torch.empty((size, 2), dtype=torch.float32, device="cpu")
        if output.is_pinned():
            raise RuntimeError("GPU E2E output buffer must be pageable")
        buffers[size] = (inputs, output)
    return buffers


def gpu_forward_e2e(model, tokens: dict[str, torch.Tensor],
                    span: tuple[int, int], buffers: dict) -> torch.Tensor:
    start, end = span
    gpu_inputs, output = buffers[end - start]
    for key in ("input_ids", "attention_mask"):
        gpu_inputs[key].copy_(tokens[key][start:end], non_blocking=False)
    output.copy_(model(**gpu_inputs).logits, non_blocking=False)
    return output


def check_correctness(cpu_model, gpu_model, tokens: dict, spans: list,
                      buffers: dict, representative_only: bool) -> dict:
    selected = [spans[0], spans[-1]] if representative_only else spans
    if selected[0] == selected[-1]:
        selected = selected[:1]
    cpu_labels, gpu_labels = [], []
    for start, end in selected:
        inputs = {key: value[start:end] for key, value in tokens.items()}
        cpu_logits = cpu_model(**inputs).logits
        gpu_logits = gpu_forward_e2e(gpu_model, tokens, (start, end), buffers)
        torch.cuda.synchronize()
        if (tuple(cpu_logits.shape) != (end - start, 2)
                or tuple(gpu_logits.shape) != (end - start, 2)
                or not torch.isfinite(cpu_logits).all().item()
                or not torch.isfinite(gpu_logits).all().item()):
            raise RuntimeError("Invalid CPU or GPU logits")
        cpu_labels.extend(cpu_logits.argmax(-1).tolist())
        gpu_labels.extend(gpu_logits.argmax(-1).tolist())
    if cpu_labels != gpu_labels:
        raise RuntimeError("CPU/GPU predicted labels differ")
    return {
        "passed": True,
        "checked_samples": len(cpu_labels),
        "checked_forward_invocations_per_path": len(selected),
        "cpu_gpu_prediction_match": True,
        "prediction_sha256": hashlib.sha256(bytes(cpu_labels)).hexdigest(),
        "scope": "first and final batch only" if representative_only else "all 872 samples",
    }


def warm_up(path: str, cpu_model, gpu_model, tokens: dict,
            spans: list, buffers: dict) -> None:
    distinct = [spans[0]]
    if spans[-1][1] - spans[-1][0] != spans[0][1] - spans[0][0]:
        distinct.append(spans[-1])
    for _ in range(WARMUP_PER_SHAPE):
        for start, end in distinct:
            if path == "cpu":
                inputs = {key: value[start:end] for key, value in tokens.items()}
                _ = cpu_model(**inputs).logits
            else:
                gpu_forward_e2e(gpu_model, tokens, (start, end), buffers)
    if path == "gpu_end_to_end":
        torch.cuda.synchronize()


def timed_cpu_total_ms(model, tokens: dict, spans: list) -> float:
    start_ns = time.perf_counter_ns()
    for start, end in spans:
        _ = model(**{key: value[start:end] for key, value in tokens.items()}).logits
    return (time.perf_counter_ns() - start_ns) / 1e6


def timed_gpu_e2e_total_ms(model, tokens: dict, spans: list, buffers: dict) -> float:
    torch.cuda.synchronize()
    start_ns = time.perf_counter_ns()
    for span in spans:
        gpu_forward_e2e(model, tokens, span, buffers)
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start_ns) / 1e6


def prepare_csv(path: Path, columns: tuple[str, ...]) -> None:
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as handle:
            if tuple(next(csv.reader(handle))) != columns:
                raise RuntimeError(f"Existing CSV schema mismatch: {path}")


def audit_raw(session_id: str) -> dict[tuple[int, str], list[float]]:
    with RAW.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["session_id"] == session_id]
    if len(rows) != len(BATCHES) * 2 * REPEATS:
        raise RuntimeError("Incomplete application raw session")
    grouped = defaultdict(list)
    runs = defaultdict(list)
    for row in rows:
        size = int(row["batch_size"])
        path = row["measurement_type"]
        latency = float(row["total_execution_ms"])
        if (row["protocol"] != PROTOCOL or size not in BATCHES
                or path not in ("cpu", "gpu_end_to_end")
                or int(row["sample_count"]) != SAMPLE_COUNT
                or int(row["forward_invocations"]) != len(chunks(size))
                or int(row["final_batch_size"]) != chunks(size)[-1][1] - chunks(size)[-1][0]
                or row["prediction_match"] != "True"
                or not math.isfinite(latency) or latency <= 0
                or not math.isclose(float(row["throughput_samples_per_sec"]),
                                    SAMPLE_COUNT * 1000 / latency, rel_tol=1e-8,
                                    abs_tol=1e-6)):
            raise RuntimeError(f"Invalid application raw row: {row}")
        grouped[size, path].append(latency)
        runs[size, path].append(int(row["run"]))
    for size in BATCHES:
        for path in ("cpu", "gpu_end_to_end"):
            if sorted(runs[size, path]) != list(range(1, REPEATS + 1)):
                raise RuntimeError(f"Incomplete application runs: {size}, {path}")
    return grouped


def write_summary(session_id: str, grouped: dict) -> None:
    prepare_csv(SUMMARY, SUMMARY_COLUMNS)
    with SUMMARY.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()
        for size in BATCHES:
            cpu = sorted(grouped[size, "cpu"])
            gpu = sorted(grouped[size, "gpu_end_to_end"])
            cpu_median, gpu_median = statistics.median(cpu), statistics.median(gpu)
            row = {
                "session_id": session_id,
                "batch_size": size,
                "sample_count": SAMPLE_COUNT,
                "forward_invocations": len(chunks(size)),
                "final_batch_size": chunks(size)[-1][1] - chunks(size)[-1][0],
                "runs_per_path": REPEATS,
                "p95_method": "nearest_rank",
                "cpu_median_total_ms": cpu_median,
                "cpu_p95_total_ms": cpu[math.ceil(.95 * REPEATS) - 1],
                "gpu_e2e_median_total_ms": gpu_median,
                "gpu_e2e_p95_total_ms": gpu[math.ceil(.95 * REPEATS) - 1],
                "cpu_median_derived_throughput_samples_per_sec": SAMPLE_COUNT * 1000 / cpu_median,
                "gpu_e2e_median_derived_throughput_samples_per_sec": SAMPLE_COUNT * 1000 / gpu_median,
                "gpu_e2e_speedup_vs_cpu": cpu_median / gpu_median,
            }
            writer.writerow({key: f"{value:.9f}" if isinstance(value, float) else value
                             for key, value in row.items()})


def run(mode: str) -> dict:
    policy = check_policy()
    before = evidence_hashes()
    proto_hash = protocol_sha256()
    if mode == "full":
        gate = read_json(SMOKE_GATE)
        if (not gate.get("passed") or gate.get("protocol_sha256") != proto_hash
                or gate.get("preserved_evidence_sha256") != before
                or gate.get("batch_sizes") != list(BATCHES)):
            raise RuntimeError("Application smoke gate is missing or stale")
    environment = read_json(RESULTS / "environment.json")
    pool = load_pool()
    if (len(pool) != SAMPLE_COUNT
            or pool._fingerprint != environment.get("pool_fingerprint")
            or environment.get("dataset_revision") != DATASET_REVISION
            or environment.get("primary_sequence_length") != PRIMARY_SEQUENCE_LENGTH):
        raise RuntimeError("Fixed SST-2 dataset or primary policy changed")
    revision = environment.get("model_revision")
    if not revision:
        raise RuntimeError("Pinned model revision missing")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, use_fast=True)
    encoded = tokenizer(list(pool["sentence"]), padding="max_length", truncation=True,
                        max_length=PRIMARY_SEQUENCE_LENGTH, return_tensors="pt")
    tokens = {key: encoded[key] for key in ("input_ids", "attention_mask")}
    if any(tuple(value.shape) != (SAMPLE_COUNT, PRIMARY_SEQUENCE_LENGTH)
           or value.dtype != torch.int64 or value.is_pinned() for value in tokens.values()):
        raise RuntimeError("Application tokenization policy mismatch")
    cpu_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=revision).float().eval().cpu()
    check_model(cpu_model, revision)
    gpu_model = copy.deepcopy(cpu_model).to("cuda").eval()
    session_id = uuid.uuid4().hex if mode == "full" else None
    correctness = {}
    coverage = {}
    with torch.inference_mode():
        for size in BATCHES:
            spans = chunks(size)
            buffers = make_buffers(spans)
            coverage[str(size)] = {
                "sample_count": SAMPLE_COUNT,
                "forward_invocations": len(spans),
                "final_batch_size": spans[-1][1] - spans[-1][0],
                "last_span": list(spans[-1]),
                "unique_rows_covered_once": True,
            }
            correctness[str(size)] = check_correctness(
                cpu_model, gpu_model, tokens, spans, buffers,
                representative_only=mode == "smoke")
            if mode == "smoke":
                continue
            prepare_csv(RAW, RAW_COLUMNS)
            with RAW.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
                if handle.tell() == 0:
                    writer.writeheader()
                for path, measured in (
                    ("cpu", lambda: timed_cpu_total_ms(cpu_model, tokens, spans)),
                    ("gpu_end_to_end", lambda: timed_gpu_e2e_total_ms(gpu_model, tokens, spans, buffers)),
                ):
                    warm_up(path, cpu_model, gpu_model, tokens, spans, buffers)
                    for run_index in range(1, REPEATS + 1):
                        latency = measured()
                        if not math.isfinite(latency) or latency <= 0:
                            raise RuntimeError("Invalid application total time")
                        writer.writerow({
                            "session_id": session_id,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "protocol": PROTOCOL,
                            "batch_size": size,
                            "run": run_index,
                            "measurement_type": path,
                            "sample_count": SAMPLE_COUNT,
                            "forward_invocations": len(spans),
                            "final_batch_size": spans[-1][1] - spans[-1][0],
                            "total_execution_ms": f"{latency:.9f}",
                            "throughput_samples_per_sec": f"{SAMPLE_COUNT * 1000 / latency:.9f}",
                            "prediction_match": True,
                        })
                        handle.flush()
    if mode == "full":
        grouped = audit_raw(session_id)
        write_summary(session_id, grouped)
    after = evidence_hashes()
    if before != after:
        raise RuntimeError("M3–M6 evidence changed")
    report = {
        "passed": True,
        "mode": mode,
        "session_id": session_id,
        "protocol": PROTOCOL,
        "protocol_sha256": proto_hash,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_sizes": list(BATCHES),
        "sample_count": SAMPLE_COUNT,
        "coverage": coverage,
        "correctness": correctness,
        "repeats_per_path_per_batch": REPEATS if mode == "full" else 0,
        "warmup_per_shape_per_path": WARMUP_PER_SHAPE if mode == "full" else 0,
        "policy": policy,
        "dataset_revision": DATASET_REVISION,
        "pool_fingerprint": pool._fingerprint,
        "model_revision": revision,
        "timing_boundaries": {
            "cpu": "perf_counter_ns around 872-sample CPU forward loop",
            "gpu_end_to_end": "synchronized perf_counter_ns around 872-sample loop; each batch copies pageable CPU tokens H2D, runs forward, copies logits D2H into preallocated pageable CPU buffer; final synchronize",
            "excluded": "tokenization, model loading, buffer allocation, correctness, warm-up",
        },
        "preserved_evidence_sha256": after,
        "raw_path": str(RAW.relative_to(ROOT)) if mode == "full" else None,
        "summary_path": str(SUMMARY.relative_to(ROOT)) if mode == "full" else None,
        "raw_rows_in_session": len(BATCHES) * 2 * REPEATS if mode == "full" else 0,
        "summary_rows_in_session": len(BATCHES) if mode == "full" else 0,
        "p95_method": "nearest rank; with 5 runs, p95 equals the maximum",
    }
    write_json(SMOKE_GATE if mode == "smoke" else RESULTS / f"application_validation_full_{session_id}.json", report)
    print(json.dumps({
        "passed": True, "mode": mode, "session_id": session_id,
        "sample_count": SAMPLE_COUNT, "batch_sizes": list(BATCHES),
        "forward_invocations": {k: v["forward_invocations"] for k, v in coverage.items()},
        "final_batch_sizes": {k: v["final_batch_size"] for k, v in coverage.items()},
        "correctness_checked_samples": {k: v["checked_samples"] for k, v in correctness.items()},
        "raw_rows_in_session": report["raw_rows_in_session"],
        "preserved_evidence_unchanged": before == after,
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
