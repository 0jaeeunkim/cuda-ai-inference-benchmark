"""Rebuild M6 statistics from one complete, unfiltered primary raw session.

P95 uses nearest rank (the 29th sorted value for 30 measurements). Throughput
and speedup are calculated from median batch latency, not averaged run ratios.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "primary_raw_runs.csv"
SUMMARY = ROOT / "results" / "summary.csv"
METADATA = ROOT / "results" / "analysis_metadata.json"
BATCHES = (1, 2, 4, 8, 16, 32, 64)
PATHS = ("cpu", "gpu_inference_only", "gpu_end_to_end")
RAW_COLUMNS = (
    "session_id", "timestamp_utc", "protocol", "batch_size", "sequence_length",
    "run", "measurement_type", "pair_id", "execution_order", "position",
    "latency_ms", "latency_per_sample_ms", "throughput_samples_per_sec",
    "prediction_match",
)
MEASURES = (
    "median_ms", "p95_ms", "median_per_sample_ms", "p95_per_sample_ms",
    "median_derived_throughput_samples_per_sec",
)
SUMMARY_COLUMNS = (
    "session_id", "batch_size", "sequence_length", "measurements_per_path",
    "p95_method",
    *(f"{path}_{measure}" for path in PATHS for measure in MEASURES),
    "gpu_inference_only_speedup_vs_cpu",
    "gpu_end_to_end_speedup_vs_cpu",
    "e2e_excess_latency_ms",
    "e2e_relative_overhead_pct",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float], batch_size: int) -> dict[str, float]:
    sorted_values = sorted(values)
    median = statistics.median(sorted_values)
    p95 = sorted_values[math.ceil(0.95 * len(sorted_values)) - 1]
    return {
        "median_ms": median,
        "p95_ms": p95,
        "median_per_sample_ms": median / batch_size,
        "p95_per_sample_ms": p95 / batch_size,
        "median_derived_throughput_samples_per_sec": batch_size * 1000 / median,
    }


def validate_and_group(rows: list[dict[str, str]], session_id: str) -> dict:
    if len(rows) != len(BATCHES) * len(PATHS) * 30:
        raise ValueError(f"Expected 630 rows in session, found {len(rows)}")
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    runs: dict[tuple[int, str], list[int]] = defaultdict(list)
    pairs: dict[tuple[int, int], list[tuple[int, str, str]]] = defaultdict(list)
    orders: dict[int, Counter[str]] = defaultdict(Counter)
    seen_gpu = False
    for row in rows:
        batch = int(row["batch_size"])
        path = row["measurement_type"]
        latency = float(row["latency_ms"])
        run = int(row["run"])
        if (row["session_id"] != session_id or batch not in BATCHES or path not in PATHS
                or row["sequence_length"] != "64"
                or row["protocol"] != "m4_cpu_separate_gpu_balanced_v1"
                or row["prediction_match"] != "True"
                or not math.isfinite(latency) or latency <= 0):
            raise ValueError(f"Invalid primary row: {row}")
        if not math.isclose(float(row["latency_per_sample_ms"]), latency / batch,
                            rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Per-sample latency mismatch")
        if not math.isclose(float(row["throughput_samples_per_sec"]),
                            1000 * batch / latency, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Throughput mismatch")
        grouped[batch, path].append(latency)
        runs[batch, path].append(run)
        if path == "cpu":
            if seen_gpu or row["pair_id"] or row["execution_order"] or row["position"]:
                raise ValueError("CPU measurements are not a separate first block")
        else:
            seen_gpu = True
            pair_id = int(row["pair_id"])
            position = int(row["position"])
            order = row["execution_order"]
            if pair_id != run or not 1 <= pair_id <= 30 or position not in (1, 2):
                raise ValueError("Invalid GPU pair ID or position")
            pairs[batch, pair_id].append((position, path, order))
            if path == "gpu_inference_only":
                orders[batch][order] += 1
    for batch in BATCHES:
        for path in PATHS:
            if sorted(runs[batch, path]) != list(range(1, 31)):
                raise ValueError(f"Incomplete runs: {batch}, {path}")
        if orders[batch] != {"A->B": 15, "B->A": 15}:
            raise ValueError(f"Unbalanced GPU order: {batch}")
        for pair_id in range(1, 31):
            pair = pairs[batch, pair_id]
            if len(pair) != 2 or pair[0][2] != pair[1][2]:
                raise ValueError(f"Incomplete GPU pair: {batch}, {pair_id}")
            expected = ((1, "gpu_inference_only"), (2, "gpu_end_to_end")) if pair[0][2] == "A->B" else ((1, "gpu_end_to_end"), (2, "gpu_inference_only"))
            if tuple(sorted((pos, path) for pos, path, _ in pair)) != expected:
                raise ValueError(f"Incorrect pair order: {batch}, {pair_id}")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", help="Required if raw CSV has multiple sessions")
    args = parser.parse_args()
    with RAW.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise ValueError("Unexpected raw CSV schema")
        all_rows = list(reader)
    sessions = {row["session_id"] for row in all_rows}
    if not sessions:
        raise ValueError("Empty raw CSV")
    if args.session_id is None and len(sessions) != 1:
        raise ValueError("Multiple sessions: specify --session-id explicitly")
    session_id = args.session_id or next(iter(sessions))
    if session_id not in sessions:
        raise ValueError(f"Session not found: {session_id}")
    rows = [row for row in all_rows if row["session_id"] == session_id]
    session_path = ROOT / "results" / f"full_session_{session_id}.json"
    session = json.loads(session_path.read_text())
    if (not session.get("passed") or session.get("mode") != "full"
            or session.get("raw_rows_in_session") != 630
            or session.get("batch_sizes") != list(BATCHES)
            or session.get("policy", {}).get("sequence_length") != 64):
        raise ValueError("Full-session metadata did not pass the primary gate")
    grouped = validate_and_group(rows, session_id)
    output = []
    for batch in BATCHES:
        by_path = {path: summarize(grouped[batch, path], batch) for path in PATHS}
        cpu = by_path["cpu"]["median_ms"]
        gpu = by_path["gpu_inference_only"]["median_ms"]
        e2e = by_path["gpu_end_to_end"]["median_ms"]
        entry = {
            "session_id": session_id,
            "batch_size": batch,
            "sequence_length": 64,
            "measurements_per_path": 30,
            "p95_method": "nearest_rank",
            **{f"{path}_{measure}": value for path in PATHS
               for measure, value in by_path[path].items()},
            "gpu_inference_only_speedup_vs_cpu": cpu / gpu,
            "gpu_end_to_end_speedup_vs_cpu": cpu / e2e,
            "e2e_excess_latency_ms": e2e - gpu,
            "e2e_relative_overhead_pct": 100 * (e2e - gpu) / e2e,
        }
        output.append(entry)
    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for entry in output:
            writer.writerow({key: f"{value:.9f}" if isinstance(value, float) else value
                             for key, value in entry.items()})
    crossovers = {}
    for path in PATHS[1:]:
        winning = [batch for batch in BATCHES if
                   next(row for row in output if row["batch_size"] == batch)
                   [f"{path}_median_ms"] <
                   next(row for row in output if row["batch_size"] == batch)
                   ["cpu_median_ms"]]
        crossovers[path] = {
            "first_tested_batch_with_gpu_below_cpu": winning[0] if winning else None,
            "observed_transition_within_tested_range": bool(winning and winning[0] > BATCHES[0]),
        }
    metadata = {
        "session_id": session_id,
        "source_raw_path": str(RAW.relative_to(ROOT)),
        "source_raw_sha256": sha256(RAW),
        "source_session_metadata_path": str(session_path.relative_to(ROOT)),
        "summary_path": str(SUMMARY.relative_to(ROOT)),
        "summary_sha256": sha256(SUMMARY),
        "source_rows_in_session": len(rows),
        "all_gpu_orders_pooled": True,
        "measurements_per_batch_path": 30,
        "p95_method": "nearest rank; 29th sorted measurement of 30",
        "throughput_definition": "batch_size * 1000 / median_batch_latency_ms",
        "speedup_definition": "CPU median batch latency / GPU median batch latency",
        "e2e_excess_definition": "E2E median - inference-only median; not pure transfer time",
        "e2e_relative_overhead_definition": "100 * (E2E median - inference-only median) / E2E median",
        "crossover_definition": "smallest tested batch with GPU median below CPU median; no interpolation",
        "crossovers": crossovers,
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"passed": True, "session_id": session_id,
                      "summary": str(SUMMARY.relative_to(ROOT)),
                      "rows": len(output), "crossovers": crossovers}, indent=2))


if __name__ == "__main__":
    main()
