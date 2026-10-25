"""Project 2 policy, coverage, provenance and gates (standard library only)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/project2"
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
DATASET_ID = "nyu-mll/glue"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
OLD_FINGERPRINT = "0e2b00033b48693f"
BATCHES = (1, 8, 64)
SMOKE_CASES = ((1, 5), (8, 19), (64, 137))
ORDERS = ("ABDC", "BCAD", "CDBA", "DACB")
VARIANTS = "ABCD"
PROTOCOL_ID = "project2_prepared_buffer_full_logits_v1"
POLICY = {
    "protocol_id": PROTOCOL_ID, "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION, "dataset_id": DATASET_ID,
    "dataset_revision": DATASET_REVISION, "dataset_config": "sst2",
    "dataset_split": "validation", "sample_count": 872,
    "sequence_length": 64, "dtype": "float32", "input_dtype": "int64",
    "tf32": False, "matmul_precision": "highest", "cpu_threads": 8,
    "batch_sizes": BATCHES, "variants": VARIANTS, "device_input_slots": 2,
    "full_rounds": 32, "orders": ORDERS, "warmup_sequences": 5,
    "smoke_cases": SMOKE_CASES, "smoke_rounds": 2, "smoke_warmup": 1,
    "atol": 1e-5, "rtol": 1e-4,
    "metric": "prepared-buffer warm E2E total time",
    "timer": "device synchronize; perf_counter_ns; all H2D/forward/D2H; device synchronize; stop",
    "excluded": ["imports", "tokenization", "model load", "buffer/pinning/stream/event setup",
                 "warmup", "correctness", "file I/O", "profiling"],
    "host_memory": {"A": "pageable", "B": "pinned", "C": "pinned", "D": "pinned"},
    "streams": {"ABC": "one compute stream: H2D -> forward -> D2H",
                "D": "copy: H2D; compute: forward -> D2H"},
    "events": "per-batch upload-ready/input-consumed; precreated; no rerecord generations",
    "completion": "all CPU FP32 [N,2] logits in original order; same final device sync",
}
COLUMNS = (
    "session_id", "protocol_id", "protocol_sha256", "source_commit", "timestamp_utc",
    "batch_size", "sample_count", "variant", "round", "execution_order", "position",
    "forward_count", "final_batch_size", "total_ms", "throughput_samples_per_sec",
    "correctness_pass", "max_abs_error", "output_sha256",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def json_hash(value) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def command(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def sources():
    paths = ["src/project2_protocol.py", "src/project2_pipeline.py",
             "src/project2_analysis.py", "docs/project2/PROJECT_SPEC.md"]
    return {p: sha256(ROOT / p) for p in paths}


def protocol_hash():
    return json_hash({"policy": POLICY, "sources": sources()})


def spans(batch_size: int, count: int):
    if batch_size not in BATCHES or not 0 < count <= 872:
        raise ValueError("Invalid Project 2 batch size/sample count")
    result = [(i, min(i + batch_size, count)) for i in range(0, count, batch_size)]
    covered = [i for a, b in result for i in range(a, b)]
    if covered != list(range(count)) or len(result) != math.ceil(count / batch_size):
        raise RuntimeError("Coverage failure")
    return result


def order(batch_index: int, round_index: int):
    return ORDERS[(batch_index + round_index) % 4]


def expected_runs(mode: str, variants=VARIANTS):
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("Invalid variants")
    if mode != "smoke" and variants != VARIANTS:
        raise ValueError("Only smoke permits a variant subset")
    cases = SMOKE_CASES if mode == "smoke" else tuple((b, 872) for b in BATCHES)
    rounds = {"smoke": 2, "correctness": 1, "full": 32}[mode]
    for bi, (batch, count) in enumerate(cases):
        for ri in range(rounds):
            execution_order = "".join(v for v in order(bi, ri) if v in variants)
            for position, variant in enumerate(execution_order, 1):
                yield batch, count, ri + 1, execution_order, position, variant


def verify_project1():
    manifest = read_json(RESULTS / "project1_manifest.json")
    changed = [p for p, value in manifest["files"].items()
               if not (ROOT / p).is_file() or sha256(ROOT / p) != value]
    if changed:
        raise RuntimeError(f"Project 1 preservation failed: {changed}")
    return {"passed": True, "files_checked": len(manifest["files"]),
            "manifest_sha256": sha256(RESULTS / "project1_manifest.json")}


def audit_raw(path: Path, metadata: dict, allow_running=False):
    if metadata["status"] != "completed" and not allow_running:
        raise RuntimeError("Incomplete/failed session cannot be summarized")
    if metadata.get("raw_sha256") and sha256(path) != metadata["raw_sha256"]:
        raise RuntimeError("Raw evidence checksum mismatch")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise RuntimeError("Raw schema mismatch")
        rows = list(reader)
    expected = list(expected_runs(metadata["mode"], metadata["variants"]))
    if len(rows) != len(expected):
        raise RuntimeError(f"Expected {len(expected)} raw rows, got {len(rows)}")
    for row, (batch, count, ri, eo, position, variant) in zip(rows, expected):
        integer_fields = {"batch_size": batch, "sample_count": count, "round": ri,
                          "position": position, "forward_count": len(spans(batch, count)),
                          "final_batch_size": spans(batch, count)[-1][1] - spans(batch, count)[-1][0]}
        if any(int(row[k]) != v for k, v in integer_fields.items()):
            raise RuntimeError("Raw coverage/order integer mismatch")
        fields = {"session_id": metadata["session_id"], "protocol_id": PROTOCOL_ID,
                  "protocol_sha256": metadata["protocol_sha256"],
                  "source_commit": metadata["source_commit"], "variant": variant,
                  "execution_order": eo, "correctness_pass": "True"}
        if any(row[k] != v for k, v in fields.items()):
            raise RuntimeError("Raw identity/order/correctness mismatch")
        total, throughput = float(row["total_ms"]), float(row["throughput_samples_per_sec"])
        if not math.isfinite(total) or total <= 0 or not math.isclose(
                throughput, count * 1000 / total, rel_tol=1e-9):
            raise RuntimeError("Raw timing/throughput mismatch")
        if not math.isfinite(float(row["max_abs_error"])) or float(row["max_abs_error"]) < 0:
            raise RuntimeError("Invalid correctness diagnostic")
        if len(row["output_sha256"]) != 64:
            raise RuntimeError("Missing CPU-visible output checksum")
    return rows


def require_gate(kind: str, identity: dict):
    pointer_path = RESULTS / f"{kind}_gate.json"
    if not pointer_path.exists():
        raise RuntimeError(f"Missing {kind} gate; run --mode {kind} first")
    pointer = read_json(pointer_path)
    session = ROOT / pointer["session_path"]
    gate = read_json(session / "gate.json")
    if sha256(session / "gate.json") != pointer["gate_sha256"] or gate["identity"] != identity:
        raise RuntimeError(f"Stale {kind} gate: source/protocol/environment/input changed")
    meta = read_json(session / "metadata.json")
    if (meta["mode"] != kind or meta["variants"] != VARIANTS or meta["status"] != "completed"
            or meta["identity"] != identity):
        raise RuntimeError(f"Invalid {kind} gate session")
    for relative, value in gate["artifacts"].items():
        if sha256(session / relative) != value:
            raise RuntimeError(f"Changed {kind} gate evidence: {relative}")
    rows = audit_raw(session / "raw.csv", meta)
    correctness = read_json(session / "correctness.json")
    if (not gate["passed"] or len(correctness["runs"]) != len(rows)
            or len(correctness["coverage"]) != len(BATCHES) * len(VARIANTS)
            or not all(c["passed"] for c in correctness["runs"])
            or not all(c["check"]["passed"] for c in correctness["coverage"])):
        raise RuntimeError(f"Failed {kind} correctness")
    if kind == "smoke" and not correctness["synchronization_api_probe"]["passed"]:
        raise RuntimeError("Missing synchronization API behavior check")
    return {"session_id": meta["session_id"], "gate_sha256": pointer["gate_sha256"]}


def publish_gate(session: Path, identity: dict, kind: str):
    artifacts = {p: sha256(session / p) for p in ["metadata.json", "correctness.json", "raw.csv"]}
    for reference in read_json(session / "correctness.json")["references"].values():
        artifacts[reference["file"]] = sha256(session / reference["file"])
    write_json(session / "gate.json", {"passed": True, "identity": identity, "artifacts": artifacts})
    write_json(RESULTS / f"{kind}_gate.json", {
        "session_path": str(session.relative_to(ROOT)), "gate_sha256": sha256(session / "gate.json")})


def append_row(path: Path, row: dict):
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def static_check():
    full = list(expected_runs("full"))
    assert len(full) == 384
    for bi, batch in enumerate(BATCHES):
        distribution = Counter((v, p) for b, _, _, _, p, v in full if b == batch)
        assert set(distribution.values()) == {8} and len(distribution) == 16
        assert len(spans(batch, 872)) == {1: 872, 8: 109, 64: 14}[batch]
        assert spans(batch, 872)[-1][1] - spans(batch, 872)[-1][0] == {1: 1, 8: 8, 64: 40}[batch]
    for batch, count in SMOKE_CASES:
        assert len(spans(batch, count)) >= 3
    return {"passed": True, "full_rows": len(full), "smoke_rows": len(list(expected_runs("smoke"))),
            "balanced_positions_per_variant": 8, "project1": verify_project1()}
