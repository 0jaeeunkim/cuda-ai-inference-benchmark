"""Reconstruct Project 2 summaries from complete raw sessions, without PyTorch."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import project2_protocol as p


def summarize(session: Path):
    meta = p.read_json(session / "metadata.json")
    rows = p.audit_raw(session / "raw.csv", meta)
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["batch_size"]), int(row["sample_count"]), row["variant"])].append(float(row["total_ms"]))
    medians = {key: statistics.median(values) for key, values in groups.items()}
    summary = []
    for (batch, count, variant), values in sorted(groups.items()):
        # Type-7 linear quartiles; same definition for 2-run smoke and 32-run full.
        ordered = sorted(values)

        def percentile(fraction):
            index = (len(ordered) - 1) * fraction
            lower = int(index)
            upper = min(lower + 1, len(ordered) - 1)
            return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])

        median = medians[(batch, count, variant)]
        baseline = medians.get((batch, count, "A"))
        summary.append({"session_id": meta["session_id"], "mode": meta["mode"],
                        "batch_size": batch, "sample_count": count, "variant": variant,
                        "runs": len(values), "median_total_ms": median,
                        "q1_total_ms": percentile(.25), "q3_total_ms": percentile(.75),
                        "iqr_total_ms": percentile(.75) - percentile(.25),
                        "throughput_from_median_samples_per_sec": count * 1000 / median,
                        "speedup_vs_A_ratio_of_medians": baseline / median if baseline else ""})
    path = session / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Project 2 session reconstruction", "",
             f"Session: `{meta['session_id']}`; mode: `{meta['mode']}`; raw rows: {len(rows)}.",
             f"Protocol SHA-256: `{meta['protocol_sha256']}`.", "",
             "Metric: prepared-buffer warm E2E total time. Each run ends after the common device synchronization",
             "with all FP32 CPU logits in original order. Setup, warm-up and correctness are outside timing.",
             "Quartiles: type-7 linear interpolation. Speedup: ratio of within-session medians; no paired-run estimator.",
             "All raw observations are retained, including slower variants.", "",
             "| Batch | N | Variant | Runs | Median ms | IQR ms | Samples/s | A / variant |",
             "| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        ratio = row["speedup_vs_A_ratio_of_medians"]
        lines.append(f"| {row['batch_size']} | {row['sample_count']} | {row['variant']} | {row['runs']} | "
                     f"{row['median_total_ms']:.6f} | {row['iqr_total_ms']:.6f} | "
                     f"{row['throughput_from_median_samples_per_sec']:.2f} | {ratio:.4f} |")
    lines += ["", "Smoke timings describe only their bounded N; they are not full 872-sample performance results.",
              "No trace is included in this raw session. These measurements alone do not establish copy/compute overlap,",
              "GPU saturation, PCIe bandwidth limits, or isolated transfer cost. Project 1 measurements are not denominators.", ""]
    (session / "analysis.md").write_text("\n".join(lines))
    return {"raw_rows": len(rows), "summary_rows": len(summary)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()
    print(summarize(args.session.resolve()))


if __name__ == "__main__":
    main()
