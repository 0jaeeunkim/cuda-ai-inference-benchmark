"""Integrity tests use temporary synthetic rows, never benchmark evidence."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project2_protocol as p
from project2_analysis import summarize


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.session = self.root / "results/project2/smoke/test"
        self.session.mkdir(parents=True)
        self.identity = {"protocol_sha256": "test-only"}

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, mode="smoke"):
        meta = {"session_id": "synthetic-test-only", "mode": mode, "variants": "ABCD",
                "status": "completed", "identity": self.identity,
                "source_commit": "test", "protocol_sha256": "test-only"}
        checks = []
        for b, n, ri, eo, position, v in p.expected_runs(mode):
            total = float((p.VARIANTS.index(v) + 1) * ri)
            p.append_row(self.session / "raw.csv", {
                "session_id": meta["session_id"], "protocol_id": p.PROTOCOL_ID,
                "protocol_sha256": meta["protocol_sha256"], "source_commit": "test",
                "timestamp_utc": p.now(), "batch_size": b, "sample_count": n,
                "variant": v, "round": ri, "execution_order": eo, "position": position,
                "forward_count": len(p.spans(b, n)),
                "final_batch_size": p.spans(b, n)[-1][1] - p.spans(b, n)[-1][0],
                "total_ms": total, "throughput_samples_per_sec": n * 1000 / total,
                "correctness_pass": True, "max_abs_error": 0, "output_sha256": "a" * 64})
            checks.append({"passed": True})
        coverage = [{"check": {"passed": True}} for _ in range(12)]
        correct = {"runs": checks, "coverage": coverage, "references": {},
                   "synchronization_api_probe": {"passed": True}}
        p.write_json(self.session / "metadata.json", meta)
        p.write_json(self.session / "correctness.json", correct)
        return meta

    def rewrite(self, rows):
        with (self.session / "raw.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=p.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_coverage_tail_and_balanced_full_order(self):
        runs = list(p.expected_runs("full"))
        self.assertEqual(len(runs), 384)
        for bi, b in enumerate(p.BATCHES):
            for v in p.VARIANTS:
                for pos in range(1, 5):
                    self.assertEqual(sum(x[0] == b and x[4] == pos and x[5] == v for x in runs), 8)
        self.assertEqual([b - a for a, b in p.spans(64, 872)], [64] * 13 + [40])
        self.assertEqual([b - a for a, b in p.spans(8, 19)], [8, 8, 3])
        self.assertEqual([b - a for a, b in p.spans(64, 137)], [64, 64, 9])

    def test_invalid_workload_rejected(self):
        for b, n in ((0, 5), (2, 19), (8, 0), (64, 873)):
            with self.assertRaises(ValueError):
                p.spans(b, n)
        with self.assertRaises(ValueError):
            list(p.expected_runs("full", "ABC"))

    def test_raw_reconstruction_and_ratio_of_medians(self):
        self.fixture("full")
        result = summarize(self.session)
        self.assertEqual(result, {"raw_rows": 384, "summary_rows": 12})
        with (self.session / "summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        d = next(x for x in rows if x["variant"] == "D" and x["batch_size"] == "64")
        self.assertEqual(float(d["median_total_ms"]), 66)
        self.assertEqual(float(d["iqr_total_ms"]), 62)
        self.assertEqual(float(d["speedup_vs_A_ratio_of_medians"]), .25)
        self.assertAlmostEqual(float(d["throughput_from_median_samples_per_sec"]), 872000 / 66)

    def test_missing_duplicate_and_reordered_rows_rejected(self):
        meta = self.fixture()
        rows = p.audit_raw(self.session / "raw.csv", meta)
        for damaged in (rows[:-1], [rows[0]] + rows[0:-1], list(reversed(rows))):
            with self.subTest(size=len(damaged)):
                self.rewrite(damaged)
                with self.assertRaises(RuntimeError):
                    p.audit_raw(self.session / "raw.csv", meta)

    def test_bad_timing_coverage_or_completion_rejected(self):
        meta = self.fixture()
        rows = p.audit_raw(self.session / "raw.csv", meta)
        for key, value in (("throughput_samples_per_sec", "1"), ("sample_count", "872"),
                           ("correctness_pass", "False"), ("total_ms", "nan"),
                           ("forward_count", "1"), ("final_batch_size", "64")):
            altered = [dict(row) for row in rows]
            altered[0][key] = value
            self.rewrite(altered)
            with self.assertRaises(RuntimeError):
                p.audit_raw(self.session / "raw.csv", meta)

    def test_failed_session_cannot_be_summarized(self):
        meta = self.fixture()
        meta["status"] = "failed"
        p.write_json(self.session / "metadata.json", meta)
        with self.assertRaisesRegex(RuntimeError, "Incomplete/failed"):
            summarize(self.session)
        self.assertFalse((self.session / "summary.csv").exists())

    def test_gate_missing_stale_and_tampered(self):
        self.fixture()
        with patch.object(p, "ROOT", self.root), patch.object(p, "RESULTS", self.root / "results/project2"):
            with self.assertRaisesRegex(RuntimeError, "Missing smoke"):
                p.require_gate("smoke", self.identity)
            p.publish_gate(self.session, self.identity, "smoke")
            self.assertEqual(p.require_gate("smoke", self.identity)["session_id"], "synthetic-test-only")
            with self.assertRaisesRegex(RuntimeError, "Stale smoke"):
                p.require_gate("smoke", {"protocol_sha256": "modified-source"})
            with (self.session / "raw.csv").open("a") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(RuntimeError, "Changed smoke"):
                p.require_gate("smoke", self.identity)

    def test_gate_requires_all_variants_and_sync_probe(self):
        meta = self.fixture()
        meta["variants"] = "ABC"
        p.write_json(self.session / "metadata.json", meta)
        with patch.object(p, "ROOT", self.root), patch.object(p, "RESULTS", self.root / "results/project2"):
            p.publish_gate(self.session, self.identity, "smoke")
            with self.assertRaisesRegex(RuntimeError, "Invalid smoke"):
                p.require_gate("smoke", self.identity)


if __name__ == "__main__":
    unittest.main()
