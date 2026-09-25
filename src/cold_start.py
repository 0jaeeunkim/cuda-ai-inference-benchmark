"""Orchestrate separate fresh-process cold-start CPU/GPU jobs.

Smoke: N=2, one fresh CPU and GPU process. Coarse: N=1,8,32,128,872,
five fresh processes per device and N. Heavy coarse runs belong in the user's
server terminal. Raw and summary files are separate from warm experiments.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from benchmark import read_json, write_json  # noqa: E402
from benchmark_harness import check_policy  # noqa: E402
from data import (DATASET_REVISION, MODEL_ID, PRIMARY_SEQUENCE_LENGTH,
                  load_pool)  # noqa: E402

RESULTS = ROOT / "results"
WORKER = ROOT / "src" / "cold_start_worker.py"
INPUTS = RESULTS / "cold_start_inputs.pt"
INPUT_META = RESULTS / "cold_start_inputs.json"
SMOKE_RAW = RESULTS / "cold_start_smoke_raw.csv"
SMOKE_GATE = RESULTS / "cold_start_smoke.json"
COARSE_RAW = RESULTS / "cold_start_v2_raw.csv"
COARSE_SUMMARY = RESULTS / "cold_start_v2_summary.csv"
COARSE_N = (1, 8, 32, 128, 872)
REPEATS = 5
PROTOCOL = "process_cold_file_cache_warm_v2"
RAW_COLUMNS = (
    "session_id", "timestamp_utc", "protocol", "phase", "sample_count",
    "batch_size", "repeat", "device", "execution_order", "position",
    "worker_pid", "model_load_ms", "device_prepare_ms", "first_forward_ms",
    "remaining_inference_ms", "total_job_ms", "throughput_samples_per_sec",
    "prediction_sha256", "inputs_sha256", "model_revision",
    "gpu_uninitialized_before_job",
)
PHASES = ("model_load_ms", "device_prepare_ms", "first_forward_ms",
          "remaining_inference_ms", "total_job_ms")
SUMMARY_COLUMNS = (
    "session_id", "sample_count", "batch_size", "runs_per_device",
    *(f"{device}_median_{phase}" for device in ("cpu", "gpu") for phase in PHASES),
    "cpu_median_derived_throughput_samples_per_sec",
    "gpu_median_derived_throughput_samples_per_sec",
    "gpu_speedup_vs_cpu_total_job", "lower_median_total_job",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_sha256() -> str:
    h = hashlib.sha256()
    for name in ("cold_start.py", "cold_start_worker.py", "benchmark.py",
                 "benchmark_harness.py", "data.py"):
        path = ROOT / "src" / name
        h.update(name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def preserved_hashes() -> dict[str, str]:
    paths = [
        RESULTS / "environment.json", RESULTS / "raw_runs.csv",
        RESULTS / "m35_diagnostic_raw.csv", RESULTS / "primary_raw_runs.csv",
        RESULTS / "summary.csv", RESULTS / "analysis_metadata.json",
        RESULTS / "analysis.md", RESULTS / "application_validation_raw.csv",
        RESULTS / "application_validation_summary.csv",
        RESULTS / "application_validation_smoke.json",
        *sorted(RESULTS.glob("application_validation_full_*.json")),
        *sorted((RESULTS / "plots").glob("*.svg")),
    ]
    # Preserve the completed pre-update v1 exploratory scan as separate
    # evidence; v2 writes to new files and never appends to the v1 CSVs.
    old_raw = RESULTS / "cold_start_raw.csv"
    old_summary = RESULTS / "cold_start_summary.csv"
    if old_raw.exists() != old_summary.exists():
        raise RuntimeError("Incomplete earlier cold-start v1 evidence")
    if old_raw.exists():
        paths.extend((old_raw, old_summary))
        for report_path in sorted(RESULTS.glob("cold_start_coarse_*.json")):
            if read_json(report_path).get("protocol") == "cold_start_fresh_process_v1":
                paths.append(report_path)
    if (len(list(RESULTS.glob("application_validation_full_*.json"))) < 1
            or len(list((RESULTS / "plots").glob("*.svg"))) != 5
            or any(not path.is_file() for path in paths)):
        raise RuntimeError("Existing M0–M6.5 evidence is incomplete")
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def prepare_inputs(environment: dict) -> dict:
    if INPUTS.exists() != INPUT_META.exists():
        raise RuntimeError("Incomplete cold-start input artifact")
    if INPUTS.exists():
        metadata = read_json(INPUT_META)
        if (metadata.get("artifact_sha256") != sha256(INPUTS)
                or metadata.get("model_revision") != environment.get("model_revision")
                or metadata.get("pool_fingerprint") != environment.get("pool_fingerprint")
                or metadata.get("sample_count") != 872
                or metadata.get("sequence_length") != 64):
            raise RuntimeError("Cold-start input artifact or source changed")
        return metadata
    pool = load_pool()
    if (len(pool) != 872 or pool._fingerprint != environment.get("pool_fingerprint")
            or environment.get("dataset_revision") != DATASET_REVISION):
        raise RuntimeError("Pinned SST-2 validation pool changed")
    revision = environment.get("model_revision")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=revision, use_fast=True, local_files_only=True)
    encoded = tokenizer(list(pool["sentence"]), padding="max_length", truncation=True,
                        max_length=PRIMARY_SEQUENCE_LENGTH, return_tensors="pt")
    tokens = {key: encoded[key] for key in ("input_ids", "attention_mask")}
    if any(tuple(value.shape) != (872, 64) or value.dtype != torch.int64
           for value in tokens.values()):
        raise RuntimeError("Cold-start tokenization policy mismatch")
    temporary = INPUTS.with_suffix(".pt.tmp")
    torch.save(tokens, temporary)
    temporary.replace(INPUTS)
    metadata = {
        "artifact_sha256": sha256(INPUTS),
        "artifact_path": str(INPUTS.relative_to(ROOT)),
        "dataset_revision": DATASET_REVISION,
        "pool_fingerprint": pool._fingerprint,
        "model_revision": revision,
        "sample_count": 872,
        "sequence_length": 64,
        "sample_selection": "published validation order; each job uses prefix 0..N-1",
        "tokenization": "truncation=True, padding=max_length, max_length=64",
    }
    write_json(INPUT_META, metadata)
    return metadata


def prime_model_cache(revision: str) -> float:
    """Untimed CPU-only model load to warm the OS file cache before workers."""
    if torch.cuda.is_initialized():
        raise RuntimeError("Parent CUDA context must remain uninitialized")
    start_ns = time.perf_counter_ns()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=revision, local_files_only=True,
    ).float().eval().cpu()
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
    if (getattr(model.config, "_commit_hash", None) != revision
            or not all(p.dtype == torch.float32 for p in model.parameters())
            or torch.cuda.is_initialized()):
        raise RuntimeError("Cache priming changed model or initialized parent CUDA")
    del model
    gc.collect()
    return elapsed_ms


def cpu_first(count: int, repeat: int) -> bool:
    """Alternate launch order within N; reverse the majority on alternate N."""
    return (repeat + COARSE_N.index(count)) % 2 == 1 if count in COARSE_N else repeat % 2 == 1


def run_worker(device: str, count: int, artifact: dict) -> dict:
    command = [
        sys.executable, str(WORKER), "--device", device,
        "--count", str(count), "--inputs", str(INPUTS),
        "--inputs-sha256", artifact["artifact_sha256"],
        "--model-revision", artifact["model_revision"],
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=180, check=False)
    if result.returncode:
        raise RuntimeError(f"Fresh {device} N={count} worker failed: {result.stderr[-2000:]}")
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if (report["worker_parent_pid"] != os.getpid()
            or report["python_executable"] != sys.executable
            or report["worker_pid"] == os.getpid()
            or report["device"] != device or report["sample_count"] != count
            or report["batch_size"] != 1 or report["forward_invocations"] != count
            or report["inputs_sha256"] != artifact["artifact_sha256"]
            or report["model_revision"] != artifact["model_revision"]
            or report["cpu_threads"] != 8
            or report["float32_matmul_precision"] != "highest"
            or report["cuda_matmul_allow_tf32"] or report["cudnn_allow_tf32"]
            or not report["gpu_uninitialized_before_job"]):
        raise RuntimeError(f"Cold worker policy mismatch: {report}")
    for name in PHASES:
        value = report[name]
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"Invalid cold phase {name}: {value}")
    if (report["model_load_ms"] <= 0 or report["first_forward_ms"] <= 0
            or report["total_job_ms"] <= 0
            or (device == "cpu" and report["device_prepare_ms"] != 0)
            or (device == "gpu" and report["device_prepare_ms"] <= 0)
            or (count == 1 and report["remaining_inference_ms"] != 0)
            or report["total_job_ms"] + 0.1 < sum(report[name] for name in PHASES[:-1])
            or not math.isclose(report["throughput_samples_per_sec"],
                                count * 1000 / report["total_job_ms"], rel_tol=1e-9)):
        raise RuntimeError("Cold worker timing arithmetic mismatch")
    return report


def prepare_csv(path: Path, columns: tuple[str, ...]) -> None:
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as handle:
            if tuple(next(csv.reader(handle))) != columns:
                raise RuntimeError(f"Existing CSV schema mismatch: {path}")


def append_raw(writer, session_id: str, phase: str, count: int, repeat: int,
               device: str, order: str, position: int, report: dict) -> None:
    row = {
        "session_id": session_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "phase": phase,
        "sample_count": count,
        "batch_size": 1,
        "repeat": repeat,
        "device": device,
        "execution_order": order,
        "position": position,
        "worker_pid": report["worker_pid"],
        "prediction_sha256": report["prediction_sha256"],
        "inputs_sha256": report["inputs_sha256"],
        "model_revision": report["model_revision"],
        "gpu_uninitialized_before_job": report["gpu_uninitialized_before_job"],
        **{name: f"{report[name]:.9f}" for name in PHASES},
        "throughput_samples_per_sec": f"{report['throughput_samples_per_sec']:.9f}",
    }
    writer.writerow(row)


def audit_and_summarize(session_id: str, artifact: dict) -> tuple[list[dict], list[list[int]]]:
    with COARSE_RAW.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise RuntimeError("Cold raw schema changed")
        rows = [row for row in reader if row["session_id"] == session_id]
    if len(rows) != len(COARSE_N) * 2 * REPEATS:
        raise RuntimeError("Incomplete coarse session")
    groups = defaultdict(list)
    for row in rows:
        count, device, repeat = int(row["sample_count"]), row["device"], int(row["repeat"])
        if (row["protocol"] != PROTOCOL or row["phase"] != "coarse"
                or count not in COARSE_N or device not in ("cpu", "gpu")
                or int(row["batch_size"]) != 1
                or row["inputs_sha256"] != artifact["artifact_sha256"]
                or row["model_revision"] != artifact["model_revision"]
                or row["gpu_uninitialized_before_job"] != "True"):
            raise RuntimeError("Invalid cold raw row")
        groups[count, device].append(row)
    summary = []
    winners = []
    for count in COARSE_N:
        medians = {}
        checksums = set()
        for device in ("cpu", "gpu"):
            device_rows = groups[count, device]
            if sorted(int(row["repeat"]) for row in device_rows) != list(range(1, REPEATS + 1)):
                raise RuntimeError(f"Missing fresh runs for N={count}, {device}")
            for row in device_rows:
                expected_order = "CPU->GPU" if cpu_first(count, int(row["repeat"])) else "GPU->CPU"
                expected_position = 1 if (device == "cpu") == (expected_order == "CPU->GPU") else 2
                if (row["execution_order"] != expected_order
                        or int(row["position"]) != expected_position
                        or not math.isclose(float(row["throughput_samples_per_sec"]),
                                            count * 1000 / float(row["total_job_ms"]),
                                            rel_tol=1e-8, abs_tol=1e-6)):
                    raise RuntimeError("Cold run order or arithmetic mismatch")
                checksums.add(row["prediction_sha256"])
            medians[device] = {
                phase: statistics.median(float(row[phase]) for row in device_rows)
                for phase in PHASES
            }
        if len(checksums) != 1:
            raise RuntimeError(f"Cold CPU/GPU predicted labels differ at N={count}")
        cpu_total = medians["cpu"]["total_job_ms"]
        gpu_total = medians["gpu"]["total_job_ms"]
        winner = "cpu" if cpu_total < gpu_total else "gpu" if gpu_total < cpu_total else "tie"
        winners.append(winner)
        summary.append({
            "session_id": session_id,
            "sample_count": count,
            "batch_size": 1,
            "runs_per_device": REPEATS,
            **{f"{device}_median_{phase}": medians[device][phase]
               for device in ("cpu", "gpu") for phase in PHASES},
            "cpu_median_derived_throughput_samples_per_sec": count * 1000 / cpu_total,
            "gpu_median_derived_throughput_samples_per_sec": count * 1000 / gpu_total,
            "gpu_speedup_vs_cpu_total_job": cpu_total / gpu_total,
            "lower_median_total_job": winner,
        })
    brackets = [[COARSE_N[i], COARSE_N[i + 1]]
                for i in range(len(COARSE_N) - 1)
                if winners[i] in ("cpu", "gpu") and winners[i + 1] in ("cpu", "gpu")
                and winners[i] != winners[i + 1]]
    return summary, brackets


def write_summary(rows: list[dict]) -> None:
    prepare_csv(COARSE_SUMMARY, SUMMARY_COLUMNS)
    with COARSE_SUMMARY.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.9f}" if isinstance(value, float) else value
                             for key, value in row.items()})


def run(mode: str) -> None:
    policy = check_policy()
    before = preserved_hashes()
    environment = read_json(RESULTS / "environment.json")
    if (environment.get("model_id") != MODEL_ID
            or environment.get("dataset_revision") != DATASET_REVISION
            or environment.get("primary_sequence_length") != 64):
        raise RuntimeError("Pinned primary environment changed")
    artifact = prepare_inputs(environment)
    proto = protocol_sha256()
    if mode == "coarse":
        gate = read_json(SMOKE_GATE)
        if (not gate.get("passed") or gate.get("protocol_sha256") != proto
                or gate.get("preserved_evidence_sha256") != before
                or gate.get("inputs_sha256") != artifact["artifact_sha256"]):
            raise RuntimeError("Cold-start smoke gate missing or stale")
    # The parent may inspect GPU availability, but it never launches a worker
    # through fork inheritance. Each subprocess execs sys.executable afresh.
    cache_priming_ms = prime_model_cache(artifact["model_revision"])
    session_id = uuid.uuid4().hex
    path = SMOKE_RAW if mode == "smoke" else COARSE_RAW
    prepare_csv(path, RAW_COLUMNS)
    points = (2,) if mode == "smoke" else COARSE_N
    repetitions = 1 if mode == "smoke" else REPEATS
    outputs = defaultdict(list)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()
        for count in points:
            for repeat in range(1, repetitions + 1):
                devices = ("cpu", "gpu") if cpu_first(count, repeat) else ("gpu", "cpu")
                order = "CPU->GPU" if cpu_first(count, repeat) else "GPU->CPU"
                for position, device in enumerate(devices, 1):
                    report = run_worker(device, count, artifact)
                    append_raw(writer, session_id, mode, count, repeat, device,
                               order, position, report)
                    handle.flush()
                    outputs[count].append(report)
                    print(f"{mode}: N={count} repeat={repeat} {device} "
                          f"total_job_ms={report['total_job_ms']:.3f}", flush=True)
            if len({r["prediction_sha256"] for r in outputs[count]}) != 1:
                raise RuntimeError(f"CPU/GPU predictions differ at N={count}")
    summary, brackets = (audit_and_summarize(session_id, artifact)
                         if mode == "coarse" else ([], []))
    if mode == "coarse":
        write_summary(summary)
    after = preserved_hashes()
    if before != after:
        raise RuntimeError("M0–M6.5 evidence changed during cold-start run")
    report = {
        "passed": True,
        "mode": mode,
        "session_id": session_id,
        "protocol": PROTOCOL,
        "protocol_sha256": proto,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_counts": list(points),
        "repeats_per_device_per_count": repetitions,
        "fresh_processes": len(points) * repetitions * 2,
        "worker_launch_method": "subprocess.run([sys.executable, worker]) with fresh interpreter exec; no multiprocessing fork",
        "worker_parent_pid_and_executable_verified": True,
        "untimed_model_cache_priming_ms": cache_priming_ms,
        "file_cache_policy": "one untimed local CPU model load before measured worker launches; process-cold/file-cache-warm; OS cache not flushed",
        "launch_order_policy": "alternate CPU->GPU and GPU->CPU within each N; reverse first order on alternate coarse N",
        "raw_rows_in_session": len(points) * repetitions * 2,
        "raw_path": str(path.relative_to(ROOT)),
        "summary_path": str(COARSE_SUMMARY.relative_to(ROOT)) if mode == "coarse" else None,
        "summary_rows_in_session": len(summary),
        "crossover_brackets": brackets,
        "crossover_definition": "change in lower median total_job_ms at adjacent tested N; no interpolation",
        "model_revision": artifact["model_revision"],
        "dataset_revision": DATASET_REVISION,
        "pool_fingerprint": artifact["pool_fingerprint"],
        "inputs_sha256": artifact["artifact_sha256"],
        "preserved_evidence_sha256": after,
        "fixed_policy": policy,
        "timing_boundary": "after imports/config/input load; before cached model load through final CPU-visible output; GPU includes CUDA init, model transfer, H2D/D2H and synchronize",
        "interpreter_import_and_input_load_excluded": True,
        "network_and_tokenization_excluded": True,
    }
    if mode == "smoke" and SMOKE_GATE.exists():
        old = read_json(SMOKE_GATE)
        old_id = old.get("session_id")
        if old_id:
            archive = RESULTS / f"cold_start_smoke_{old_id}.json"
            if not archive.exists():
                write_json(archive, old)
    write_json(SMOKE_GATE if mode == "smoke" else RESULTS / f"cold_start_coarse_{session_id}.json", report)
    print(json.dumps({"passed": True, "mode": mode, "session_id": session_id,
                      "raw_rows_in_session": report["raw_rows_in_session"],
                      "summary_rows_in_session": report["summary_rows_in_session"],
                      "sample_counts": report["sample_counts"],
                      "crossover_brackets": brackets,
                      "preserved_evidence_unchanged": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "coarse"), required=True)
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
