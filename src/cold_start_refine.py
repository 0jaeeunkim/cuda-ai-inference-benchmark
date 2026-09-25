"""One-point process-cold refinement inside an observed v2 coarse bracket.

Uses cold_start.py's unchanged v2 worker launcher and protocol. Smoke runs one
fresh CPU and GPU process; --mode run starts ten fresh processes per device.
Neither mode modifies the v2 coarse raw/summary or any warm result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cold_start as c

ROOT = c.ROOT
RESULTS = c.RESULTS
SMOKE_RAW = RESULTS / "cold_start_v2_refine_smoke_raw.csv"
SMOKE_GATE = RESULTS / "cold_start_v2_refine_smoke.json"
REFINE_RAW = RESULTS / "cold_start_v2_refine_raw.csv"
REFINE_SUMMARY = RESULTS / "cold_start_v2_refine_summary.csv"
REFINE_PROTOCOL = "process_cold_refinement_v1"
RUN_REPEATS = 10


def protocol_hash() -> str:
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    h.update(c.protocol_sha256().encode())
    return h.hexdigest()


def coarse_context(session_id: str, count: int) -> tuple[dict, list[int]]:
    path = RESULTS / f"cold_start_coarse_{session_id}.json"
    report = c.read_json(path)
    if (not report.get("passed") or report.get("mode") != "coarse"
            or report.get("protocol") != c.PROTOCOL
            or report.get("protocol_sha256") != c.protocol_sha256()
            or report.get("raw_rows_in_session") != 50
            or report.get("summary_rows_in_session") != 5):
        raise RuntimeError("Audited v2 coarse session required")
    brackets = [bracket for bracket in report.get("crossover_brackets", [])
                if bracket[0] < count < bracket[1]]
    if len(brackets) != 1:
        raise RuntimeError(f"N={count} is not strictly inside one observed bracket")
    with c.COARSE_SUMMARY.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["session_id"] == session_id]
    by_count = {int(row["sample_count"]): row for row in rows}
    bracket = brackets[0]
    if (len(rows) != 5 or set(by_count) != set(c.COARSE_N)
            or by_count[bracket[0]]["lower_median_total_job"]
            == by_count[bracket[1]]["lower_median_total_job"]):
        raise RuntimeError("Coarse bracket does not have opposite endpoint winners")
    return report, bracket


def evidence_hashes(coarse_session_id: str) -> dict[str, str]:
    paths = [c.COARSE_RAW, c.COARSE_SUMMARY,
             RESULTS / f"cold_start_coarse_{coarse_session_id}.json",
             c.INPUTS, c.INPUT_META]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("Missing v2 coarse evidence or fixed inputs")
    return {**c.preserved_hashes(),
            **{str(path.relative_to(ROOT)): c.sha256(path) for path in paths}}


def audit_session(path: Path, session_id: str, count: int,
                  repetitions: int, phase: str, inputs_sha: str) -> dict:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != c.RAW_COLUMNS:
            raise RuntimeError("Refinement raw schema changed")
        rows = [row for row in reader if row["session_id"] == session_id]
    if len(rows) != 2 * repetitions or len({row["worker_pid"] for row in rows}) != 2 * repetitions:
        raise RuntimeError("Missing or reused fresh worker process")
    groups = defaultdict(list)
    checksums = set()
    for row in rows:
        device = row["device"]
        repeat = int(row["repeat"])
        order = "CPU->GPU" if c.cpu_first(count, repeat) else "GPU->CPU"
        position = 1 if (device == "cpu") == (order == "CPU->GPU") else 2
        total = float(row["total_job_ms"])
        if (row["phase"] != phase or row["protocol"] != c.PROTOCOL
                or row["sample_count"] != str(count) or row["batch_size"] != "1"
                or row["inputs_sha256"] != inputs_sha
                or row["gpu_uninitialized_before_job"] != "True"
                or device not in ("cpu", "gpu")
                or row["execution_order"] != order
                or int(row["position"]) != position
                or not math.isfinite(total) or total <= 0
                or not math.isclose(float(row["throughput_samples_per_sec"]),
                                    count * 1000 / total, rel_tol=1e-8,
                                    abs_tol=1e-6)):
            raise RuntimeError(f"Invalid refinement raw row: {row}")
        groups[device].append(row)
        checksums.add(row["prediction_sha256"])
    if len(checksums) != 1:
        raise RuntimeError("Cold CPU/GPU predictions differ")
    for device in ("cpu", "gpu"):
        if sorted(int(row["repeat"]) for row in groups[device]) != list(range(1, repetitions + 1)):
            raise RuntimeError(f"Incomplete {device} runs")
    return groups


def summarize(groups: dict, session_id: str, count: int) -> tuple[dict, dict]:
    medians = {device: {name: statistics.median(float(row[name]) for row in groups[device])
                        for name in c.PHASES} for device in ("cpu", "gpu")}
    cpu_total = medians["cpu"]["total_job_ms"]
    gpu_total = medians["gpu"]["total_job_ms"]
    winner = "cpu" if cpu_total < gpu_total else "gpu" if gpu_total < cpu_total else "tie"
    row = {
        "session_id": session_id,
        "sample_count": count,
        "batch_size": 1,
        "runs_per_device": RUN_REPEATS,
        **{f"{device}_median_{name}": medians[device][name]
           for device in ("cpu", "gpu") for name in c.PHASES},
        "cpu_median_derived_throughput_samples_per_sec": count * 1000 / cpu_total,
        "gpu_median_derived_throughput_samples_per_sec": count * 1000 / gpu_total,
        "gpu_speedup_vs_cpu_total_job": cpu_total / gpu_total,
        "lower_median_total_job": winner,
    }
    iqr = {}
    for device in ("cpu", "gpu"):
        values = sorted(float(item["total_job_ms"]) for item in groups[device])
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        iqr[device] = quartiles[2] - quartiles[0]
    diagnostic = {
        "cpu_median_total_job_ms": cpu_total,
        "gpu_median_total_job_ms": gpu_total,
        "median_gap_ms": abs(cpu_total - gpu_total),
        "cpu_iqr_ms": iqr["cpu"],
        "gpu_iqr_ms": iqr["gpu"],
        "gap_no_larger_than_max_iqr": abs(cpu_total - gpu_total) <= max(iqr.values()),
        "lower_median_total_job": winner,
    }
    return row, diagnostic


def run(mode: str, count: int, coarse_session_id: str) -> None:
    coarse, bracket = coarse_context(coarse_session_id, count)
    before = evidence_hashes(coarse_session_id)
    policy = c.check_policy()  # GPU environment inspection happens in parent only.
    artifact = c.prepare_inputs(c.read_json(RESULTS / "environment.json"))
    if artifact["artifact_sha256"] != coarse["inputs_sha256"]:
        raise RuntimeError("Refinement input artifact differs from coarse")
    code_hash = protocol_hash()
    if mode == "run":
        gate = c.read_json(SMOKE_GATE)
        if (not gate.get("passed") or gate.get("protocol_sha256") != code_hash
                or gate.get("coarse_session_id") != coarse_session_id
                or gate.get("sample_count") != count
                or gate.get("inputs_sha256") != artifact["artifact_sha256"]
                or gate.get("preserved_evidence_sha256") != before):
            raise RuntimeError("Refinement smoke gate is missing or stale")
    priming_ms = c.prime_model_cache(artifact["model_revision"])
    repetitions = 1 if mode == "smoke" else RUN_REPEATS
    path = SMOKE_RAW if mode == "smoke" else REFINE_RAW
    c.prepare_csv(path, c.RAW_COLUMNS)
    session_id = uuid.uuid4().hex
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=c.RAW_COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()
        for repeat in range(1, repetitions + 1):
            devices = ("cpu", "gpu") if c.cpu_first(count, repeat) else ("gpu", "cpu")
            order = "CPU->GPU" if devices[0] == "cpu" else "GPU->CPU"
            for position, device in enumerate(devices, 1):
                result = c.run_worker(device, count, artifact)
                c.append_raw(writer, session_id, "refine_smoke" if mode == "smoke" else "refine",
                             count, repeat, device, order, position, result)
                handle.flush()
                print(f"{mode}: N={count} repeat={repeat} {device} "
                      f"total_job_ms={result['total_job_ms']:.3f}", flush=True)
    groups = audit_session(path, session_id, count, repetitions,
                           "refine_smoke" if mode == "smoke" else "refine",
                           artifact["artifact_sha256"])
    row, diagnostic = summarize(groups, session_id, count) if mode == "run" else (None, None)
    if row:
        c.prepare_csv(REFINE_SUMMARY, c.SUMMARY_COLUMNS)
        with REFINE_SUMMARY.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=c.SUMMARY_COLUMNS)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({key: f"{value:.9f}" if isinstance(value, float) else value
                             for key, value in row.items()})
    after = evidence_hashes(coarse_session_id)
    if before != after:
        raise RuntimeError("Coarse or earlier evidence changed")
    new_bracket = (bracket if diagnostic is None or diagnostic["lower_median_total_job"] == "tie"
                   else [count, bracket[1]] if diagnostic["lower_median_total_job"]
                   == next(item["lower_median_total_job"] for item in
                           csv.DictReader(c.COARSE_SUMMARY.open(newline=""))
                           if item["session_id"] == coarse_session_id
                           and int(item["sample_count"]) == bracket[0])
                   else [bracket[0], count])
    report = {
        "passed": True,
        "mode": mode,
        "protocol": REFINE_PROTOCOL,
        "protocol_sha256": code_hash,
        "session_id": session_id,
        "coarse_session_id": coarse_session_id,
        "coarse_bracket": bracket,
        "sample_count": count,
        "repeats_per_device": repetitions,
        "fresh_processes": 2 * repetitions,
        "raw_rows_in_session": 2 * repetitions,
        "raw_path": str(path.relative_to(ROOT)),
        "summary_path": str(REFINE_SUMMARY.relative_to(ROOT)) if row else None,
        "new_observed_bracket": new_bracket if row else None,
        "diagnostic": diagnostic,
        "untimed_model_cache_priming_ms": priming_ms,
        "worker_launch_method": "subprocess.run([sys.executable, worker]) with fresh interpreter exec",
        "policy": policy,
        "inputs_sha256": artifact["artifact_sha256"],
        "preserved_evidence_sha256": after,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    c.write_json(SMOKE_GATE if mode == "smoke" else
                 RESULTS / f"cold_start_refine_{session_id}.json", report)
    print(json.dumps({"passed": True, "mode": mode, "session_id": session_id,
                      "sample_count": count, "coarse_bracket": bracket,
                      "new_observed_bracket": report["new_observed_bracket"],
                      "raw_rows_in_session": report["raw_rows_in_session"],
                      "preserved_evidence_unchanged": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "run"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--coarse-session", required=True)
    args = parser.parse_args()
    run(args.mode, args.count, args.coarse_session)


if __name__ == "__main__":
    main()
