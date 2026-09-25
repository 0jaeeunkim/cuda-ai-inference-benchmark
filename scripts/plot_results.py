"""Render five presentation-ready SVG candidates from the M6 summary CSV.

Run src/analysis.py first. SVG output uses only the Python standard library.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary.csv"
META = ROOT / "results" / "analysis_metadata.json"
OUT = ROOT / "results" / "plots"
WIDTH, HEIGHT = 1200, 720
LEFT, RIGHT, TOP, BOTTOM = 120, 1135, 145, 580
COLORS = {"cpu": "#3B4453", "gpu_inference_only": "#0072B2",
          "gpu_end_to_end": "#D55E00"}
LABELS = {"cpu": "CPU", "gpu_inference_only": "GPU inference-only",
          "gpu_end_to_end": "GPU end-to-end"}
BATCHES = (1, 2, 4, 8, 16, 32, 64)


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def txt(x: float, y: float, content: str, size: int = 20, color: str = "#253044",
        anchor: str = "start", weight: int = 400, extra: str = "") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" {extra}>{esc(content)}</text>')


def plot(name: str, title: str, ylabel: str, lines: list[tuple[str, str, list[float]]],
         yticks: list[float], ymax: float, caption: str, ylog: bool = False,
         reference: float | None = None, percent: bool = False) -> None:
    def xcoord(batch: int) -> float:
        return LEFT + math.log2(batch) / 6 * (RIGHT - LEFT)

    def ycoord(value: float) -> float:
        if ylog:
            return BOTTOM - math.log10(value) / 3 * (BOTTOM - TOP)
        return BOTTOM - value / ymax * (BOTTOM - TOP)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{esc(title)}">',
             '<rect width="100%" height="100%" fill="#FFFFFF"/>',
             '<g font-family="Arial, Helvetica, sans-serif">',
             txt(LEFT, 53, title, 31, weight=700),
             txt(LEFT, 91, "Fixed sequence length 64  |  FP32  |  30 measurements per path and batch", 17, "#5A6575")]
    for idx, (label, color, _) in enumerate(lines):
        lx = LEFT + idx * 320
        parts.append(f'<line x1="{lx}" y1="116" x2="{lx+30}" y2="116" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<circle cx="{lx+15}" cy="116" r="5" fill="{color}"/>')
        parts.append(txt(lx + 40, 122, label, 17))
    for value in yticks:
        y = ycoord(value)
        parts.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{RIGHT}" y2="{y:.2f}" stroke="#E5E9EF" stroke-width="1"/>')
        tick = f"{value:g}%" if percent else f"{value:g}"
        parts.append(txt(LEFT - 14, y + 6, tick, 17, "#59677A", "end"))
    for batch in BATCHES:
        x = xcoord(batch)
        parts.append(f'<line x1="{x:.2f}" y1="{TOP}" x2="{x:.2f}" y2="{BOTTOM}" stroke="#F1F3F6" stroke-width="1"/>')
        parts.append(txt(x, BOTTOM + 29, str(batch), 18, "#59677A", "middle"))
    parts.append(f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="#6B7788" stroke-width="1.5"/>')
    parts.append(f'<line x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="#6B7788" stroke-width="1.5"/>')
    if reference is not None:
        y = ycoord(reference)
        parts.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{RIGHT}" y2="{y:.2f}" stroke="#8B96A5" stroke-width="2" stroke-dasharray="8 7"/>')
        parts.append(txt(RIGHT - 6, y - 9, "1× parity", 15, "#6B7788", "end"))
    for label, color, values in lines:
        if len(values) != len(BATCHES) or any(not math.isfinite(v) or (ylog and v <= 0) for v in values):
            raise ValueError(f"Invalid series {label}")
        points = " ".join(f"{xcoord(batch):.2f},{ycoord(value):.2f}"
                          for batch, value in zip(BATCHES, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
        for batch, value in zip(BATCHES, values):
            x, y = xcoord(batch), ycoord(value)
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{color}" stroke="white" stroke-width="1.5"/>')
    parts.append(txt((LEFT + RIGHT) / 2, 644, "Batch size (samples)", 20, anchor="middle", weight=600))
    parts.append(txt(32, (TOP + BOTTOM) / 2, ylabel, 20, anchor="middle", weight=600,
                     extra=f'transform="rotate(-90 32 {(TOP + BOTTOM) / 2:.1f})"'))
    parts.append(txt(LEFT, 688, caption, 16, "#59677A"))
    parts.extend(["</g>", "</svg>"])
    (OUT / name).write_text("\n".join(parts) + "\n")


def main() -> None:
    metadata = json.loads(META.read_text())
    if hashlib.sha256(SUMMARY.read_bytes()).hexdigest() != metadata["summary_sha256"]:
        raise ValueError("Summary differs from analyzed version; rerun analysis")
    raw = ROOT / metadata["source_raw_path"]
    if hashlib.sha256(raw.read_bytes()).hexdigest() != metadata["source_raw_sha256"]:
        raise ValueError("Raw data differs from analyzed version; rerun analysis")
    with SUMMARY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["batch_size"]) for row in rows] != list(BATCHES):
        raise ValueError("Unexpected batch sizes or row order")
    if any(row["session_id"] != metadata["session_id"] for row in rows):
        raise ValueError("Mixed session IDs")
    def values(column: str) -> list[float]:
        return [float(row[column]) for row in rows]

    OUT.mkdir(parents=True, exist_ok=True)
    latency = [(LABELS[path], COLORS[path], values(f"{path}_median_ms"))
               for path in COLORS]
    plot("figure1_median_latency.svg", "Batch size vs median batch latency",
         "Median batch latency (ms)", latency, [0, 100, 200, 300, 400, 500], 500,
         "Median of 30 runs; lower is better. CPU and GPU share the same workload.")
    plot("figure1_median_latency_log.svg", "Batch size vs median batch latency (log scale)",
         "Median batch latency (ms, log scale)", latency,
         [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], 1000,
         "Same medians as Figure 1; log scale makes both GPU curves visible.", ylog=True)
    throughput = [(LABELS[path], COLORS[path], values(f"{path}_median_derived_throughput_samples_per_sec"))
                  for path in COLORS]
    plot("figure2_throughput.svg", "Batch size vs throughput",
         "Throughput (samples/s)", throughput, [0, 1000, 2000, 3000, 4000, 5000], 5000,
         "Throughput = batch size / median batch latency; higher is better.")
    speedup = [("GPU inference-only / CPU", COLORS["gpu_inference_only"],
                values("gpu_inference_only_speedup_vs_cpu")),
               ("GPU end-to-end / CPU", COLORS["gpu_end_to_end"],
                values("gpu_end_to_end_speedup_vs_cpu"))]
    plot("figure3_speedup.svg", "Batch size vs CPU/GPU speedup",
         "CPU median / GPU median (×)", speedup, list(range(0, 36, 5)), 35,
         "Speedup uses equivalent batch size and median latency; dashed line = parity.",
         reference=1)
    overhead = [("E2E relative overhead", COLORS["gpu_end_to_end"],
                 values("e2e_relative_overhead_pct"))]
    plot("figure4_e2e_relative_overhead.svg", "Batch size vs E2E relative overhead",
         "E2E excess / E2E median (%)", overhead,
         [0, .5, 1, 1.5, 2, 2.5, 3], 3,
         "Excess = E2E median - inference-only median; includes path differences.",
         percent=True)
    print(json.dumps({"passed": True, "plot_files": [str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.svg"))]}, indent=2))


if __name__ == "__main__":
    main()
