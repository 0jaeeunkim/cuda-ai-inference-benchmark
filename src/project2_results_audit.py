"""Independent audit of saved full Project 2 results; CPU reads only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import statistics

import project2_protocol as p


def audit(session):
    import torch
    from project2_pipeline import tensor_hash
    meta = p.read_json(session / "metadata.json")
    correct = p.read_json(session / "correctness.json")
    assert meta["mode"] == "full" and meta["status"] == "completed"
    assert meta["sources"] == p.sources() and meta["protocol_sha256"] == p.protocol_hash()
    assert p.sha256(session / "correctness.json") == meta["correctness_sha256"]
    assert p.sha256(p.ROOT / meta["environment_snapshot"]) == meta["environment_sha256"]
    gates = {kind: p.require_gate(kind, meta["identity"]) for kind in ("smoke", "correctness")}
    rows = p.audit_raw(session / "raw.csv", meta)
    assert len(rows) == len(correct["runs"]) == 384
    assert len(correct["coverage"]) == 12 and meta["row_indices"] == list(range(872))
    for raw, check in zip(rows, correct["runs"]):
        assert all(check[k] for k in ("passed", "finite", "prediction_match", "logits_allclose"))
        assert check["shape"] == [872, 2]
        assert all(str(check[k]) == raw[k] for k in
                   ("batch_size", "sample_count", "variant", "round", "position", "output_sha256", "max_abs_error"))
    for item in correct["coverage"]:
        expected, observed = p.spans(item["batch_size"], 872), item["records"]
        assert item["check"]["passed"] and [(r["start"], r["end"]) for r in observed] == expected
        for i, (r, (a, b)) in enumerate(zip(observed, expected)):
            assert r["slot"] == i % 2 and r["input_shape"] == [b - a, 64] and r["logits_shape"] == [b - a, 2]
            if i:
                previous = observed[i - 1]
                assert previous["output_address"] + (previous["end"] - previous["start"]) * 8 == r["output_address"]
    for batch, reference in correct["references"].items():
        path = session / reference["file"]
        assert p.sha256(path) == reference["file_sha256"]
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        logits = artifact["logits"]
        assert logits.shape == (872, 2) and logits.dtype == torch.float32 and bool(torch.isfinite(logits).all())
        assert artifact["row_indices"] == list(range(872)) and len(artifact["labels"]) == 872
        assert p.json_hash(artifact["labels"]) == meta["labels_sha256"]
        assert tensor_hash(logits) == reference["logits_sha256"]
        for row in rows:
            if row["batch_size"] == batch:
                # Observed exact equality in this session, stronger than the fixed tolerance.
                assert row["output_sha256"] == reference["logits_sha256"]
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["batch_size"]), row["variant"])].append(float(row["total_ms"]))
    medians = {key: statistics.median(values) for key, values in groups.items()}
    with (session / "summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 12
    for output in summary:
        batch, variant = int(output["batch_size"]), output["variant"]
        values = groups[(batch, variant)]
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        expected = {"median_total_ms": medians[(batch, variant)],
                    "q1_total_ms": quartiles[0], "q3_total_ms": quartiles[2],
                    "iqr_total_ms": quartiles[2] - quartiles[0],
                    "throughput_from_median_samples_per_sec": 872000 / medians[(batch, variant)],
                    "speedup_vs_A_ratio_of_medians": medians[(batch, "A")] / medians[(batch, variant)]}
        assert int(output["runs"]) == 32 and int(output["sample_count"]) == 872
        assert all(math.isclose(float(output[k]), v, rel_tol=1e-12, abs_tol=1e-10) for k, v in expected.items())
    result = {"passed": True, "audited_at_utc": p.now(), "session_id": meta["session_id"],
              "protocol_sha256": p.protocol_hash(), "rows": len(rows), "runs_per_condition": 32,
              "all_correctness_pass": True, "max_abs_error": max(r["max_abs_error"] for r in correct["runs"]),
              "coverage_checks": 12, "summary_checks": 12, "actual_cpu_references_checked": 3,
              "gates": gates, "balanced_order": {str(b): dict(Counter(
                  f"{r['variant']}_pos{r['position']}" for r in rows if int(r["batch_size"]) == b)) for b in p.BATCHES},
              "project1": p.verify_project1(),
              "evidence_sha256": {name: p.sha256(session / name) for name in
                                  ("raw.csv", "summary.csv", "correctness.json", "metadata.json")},
              "auditor_sha256": p.sha256(Path(__file__).resolve())}
    p.write_json(session / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.session.resolve())
    print({"passed": report["passed"], "rows": report["rows"], "max_abs_error": report["max_abs_error"]})
