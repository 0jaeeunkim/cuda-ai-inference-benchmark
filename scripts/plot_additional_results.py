"""Render cold-transition and full-application figures without running benchmarks.

Median series come only from the preserved summary CSVs. The N=16 IQR widths
are taken from the already-published M7 audit (the refinement summary has no
quartile columns). A symmetric, median-centered IQR-width indicator is shown;
its endpoints are not claimed to be the original Q1/Q3 values.

Requires ImageMagick `convert` for PNG previews. SVG generation uses only the
Python standard library. Existing warm figures are never written.
"""

from __future__ import annotations

import csv
import hashlib
import html
import math
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
COLD = RESULTS / "cold_start_v2_summary.csv"
REFINE = RESULTS / "cold_start_v2_refine_summary.csv"
APP = RESULTS / "application_validation_summary.csv"
AUDIT = RESULTS / "audit.md"
SOURCE_FILES = (COLD, REFINE, APP, AUDIT)
WIDTH, HEIGHT = 1200, 720
LEFT, RIGHT, TOP, BOTTOM = 120, 1135, 145, 580
CPU = "#3B4453"
GPU = "#D55E00"  # Matches the warm GPU end-to-end series.
INK = "#253044"
MUTED = "#59677A"


def checked_rows(path: Path, counts: set[int], key: str, session: str) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    found = {int(row[key]): row for row in rows}
    if len(found) != len(rows) or set(found) != counts:
        raise ValueError(f"Unexpected or duplicate conditions in {path}")
    if any(row["session_id"] != session for row in rows):
        raise ValueError(f"Mixed session IDs in {path}")
    return found


def number(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid {column}: {value}")
    return value


def audit_iqr() -> tuple[float, float]:
    match = re.search(r"CPU IQR ([0-9.]+) ms and GPU IQR ([0-9.]+) ms", AUDIT.read_text())
    if not match:
        raise ValueError("N=16 IQR widths missing from preserved M7 audit")
    values = tuple(map(float, match.groups()))
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Invalid audited IQR widths")
    return values  # Rounded to 0.001 ms in audit.md.


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, content: str, size: int = 18, color: str = INK,
         anchor: str = "start", weight: int = 400, extra: str = "") -> str:
    return (f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" {extra}>{esc(content)}</text>')


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0,
         extra: str = "") -> str:
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" {extra}/>')


def shell(aria: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{esc(aria)}">',
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            '<g font-family="Arial, Helvetica, sans-serif">']


def legend(parts: list[str], labels: tuple[str, str]) -> None:
    for x, label, color in ((LEFT, labels[0], CPU), (LEFT + 310, labels[1], GPU)):
        parts.extend((line(x, 116, x + 42, 116, color, 5),
                      f'<circle cx="{x+21}" cy="116" r="6.5" fill="{color}"/>',
                      text(x + 55, 124, label, 22)))


def axes(parts: list[str], xs: dict[int, float], ticks: list[float], yfn,
         xlabel: str, ylabel: str, caption: str, log_scale: bool = False) -> None:
    for tick in ticks:
        y = yfn(tick)
        parts.extend((line(LEFT, y, RIGHT, y, "#E5E9EF"),
                      text(LEFT - 14, y + 6, f"{tick:g}", 17, MUTED, "end")))
    for value, x in xs.items():
        parts.extend((line(x, TOP, x, BOTTOM, "#F1F3F6"),
                      text(x, BOTTOM + 29, str(value), 18, MUTED, "middle")))
    parts.extend((line(LEFT, TOP, LEFT, BOTTOM, "#6B7788", 1.5),
                  line(LEFT, BOTTOM, RIGHT, BOTTOM, "#6B7788", 1.5),
                  text((LEFT + RIGHT) / 2, 644, xlabel, 20, INK, "middle", 600),
                  text(0, 0, ylabel, 20, INK, "middle", 600,
                       extra=f'transform="translate(32 {(TOP + BOTTOM) / 2:.1f}) rotate(-90)"')))


def series(parts: list[str], xs: dict[int, float], values: dict[int, float], yfn,
           color: str, connect: list[list[int]]) -> None:
    for segment in connect:
        points = " ".join(f"{xs[n]:.2f},{yfn(values[n]):.2f}" for n in segment)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                     'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    for n, value in values.items():
        parts.append(f'<circle cx="{xs[n]:.2f}" cy="{yfn(value):.2f}" r="5.5" '
                     f'fill="{color}" stroke="white" stroke-width="1.5"/>')


def write_svg(name: str, parts: list[str]) -> Path:
    path = PLOTS / name
    parts.extend(("</g>", "</svg>"))
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def render_png(svg: Path) -> Path:
    convert = shutil.which("convert")
    if convert is None:
        raise RuntimeError("ImageMagick convert is needed for PNG previews")
    png = svg.with_suffix(".png")
    subprocess.run([convert, "-background", "white", "-density", "144",
                    str(svg), "-resize", f"{WIDTH}x{HEIGHT}!", str(png)],
                   check=True, capture_output=True, text=True)
    return png


def cold_figure(coarse: dict[int, dict[str, str]], refinement: dict[int, dict[str, str]]) -> tuple[Path, dict]:
    xs = {1: 170, 8: 470, 16: 770, 32: 1070}
    rows = {n: coarse[n] for n in (1, 8, 32)} | {16: refinement[16]}
    cpu = {n: number(row, "cpu_median_total_job_ms") for n, row in rows.items()}
    gpu = {n: number(row, "gpu_median_total_job_ms") for n, row in rows.items()}
    cpu_iqr, gpu_iqr = audit_iqr()
    ymax = 500.0
    yfn = lambda v: BOTTOM - v / ymax * (BOTTOM - TOP)
    parts = shell("Process-cold CPU and GPU median total job time at measured N 1, 8, 16, 32")
    parts.append(text(LEFT, 53, "Process-Cold CPU–GPU Transition", 31, INK, weight=700))
    parts.append(f'<rect x="{xs[16]}" y="{TOP}" width="{xs[32]-xs[16]}" '
                 f'height="{BOTTOM-TOP}" fill="#F6F8FB"/>')
    parts.append(text((xs[16] + xs[32]) / 2, TOP + 28, "Observed transition region",
                      17, MUTED, "middle"))
    legend(parts, ("CPU", "GPU"))
    # The summary has IQR widths, not Q1/Q3; identify centered whiskers in the legend.
    parts.extend((line(795, 103, 795, 129, MUTED, 2),
                  line(788, 103, 802, 103, MUTED, 2),
                  line(788, 129, 802, 129, MUTED, 2),
                  text(815, 124, "Centered IQR width at N=16", 20, MUTED)))
    axes(parts, xs, [0, 100, 200, 300, 400, 500], yfn,
         "Samples per Job (N)", "Median Total Job Time (ms)",
         "Process-cold, file-cache-warm; N=16 bars show audited IQR widths centered on medians.")
    # Do not connect N=16 to 32: a connecting pair would imply an interpolated crossing.
    for values, color in ((cpu, CPU), (gpu, GPU)):
        series(parts, xs, values, yfn, color, [[1, 8, 16]])
    for offset, median, width, color in ((-12, cpu[16], cpu_iqr, CPU),
                                          (12, gpu[16], gpu_iqr, GPU)):
        x = xs[16] + offset
        ytop, ybottom = yfn(median + width / 2), yfn(median - width / 2)
        parts.extend((line(x, ytop, x, ybottom, color, 2),
                      line(x - 6, ytop, x + 6, ytop, color, 2),
                      line(x - 6, ybottom, x + 6, ybottom, color, 2)))
    parts.append(text(715, 420, "Near tie", 18, INK, "end", 600))
    parts.append(line(723, 412, xs[16] - 16, yfn((cpu[16] + gpu[16]) / 2) + 10,
                      "#6B7788", 1.5))
    svg = write_svg("figure5_cold_transition.svg", parts)
    return svg, {"N": list(xs), "cpu_median_total_job_ms": cpu,
                 "gpu_median_total_job_ms": gpu,
                 "n16_audited_iqr_width_ms": {"cpu": cpu_iqr, "gpu": gpu_iqr}}


def application_figure(rows: dict[int, dict[str, str]], log_scale: bool) -> tuple[Path, dict]:
    xs = {1: 190, 8: 470, 32: 750, 64: 1030}
    cpu = {n: number(row, "cpu_median_total_ms") / 1000 for n, row in rows.items()}
    gpu = {n: number(row, "gpu_e2e_median_total_ms") / 1000 for n, row in rows.items()}
    if any(int(row["sample_count"]) != 872 for row in rows.values()):
        raise ValueError("Application summary is not the complete 872-sample workload")
    if [int(rows[n]["final_batch_size"]) for n in xs] != [1, 8, 8, 40]:
        raise ValueError("Unexpected final partial batch sizes")
    speedup = number(rows[64], "gpu_e2e_speedup_vs_cpu")
    if not math.isclose(speedup, cpu[64] / gpu[64], rel_tol=1e-8):
        raise ValueError("Batch-64 speedup does not reconstruct from medians")
    label = f"{speedup:.2f}× speedup"
    if log_scale:
        yfn = lambda v: BOTTOM - (math.log10(v) - math.log10(0.1)) / (math.log10(20) - math.log10(0.1)) * (BOTTOM - TOP)
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
        name = "figure6_application_total_log.svg"
    else:
        yfn = lambda v: BOTTOM - v / 10 * (BOTTOM - TOP)
        ticks = [0, 2, 4, 6, 8, 10]
        name = "figure6_application_total.svg"
    parts = shell("Full 872-sample SST-2 application CPU and GPU E2E total time")
    title = "Full 872-Sample Application Validation"
    if log_scale:
        title += " (Log Scale)"
    parts.append(text(LEFT, 53, title, 31, INK, weight=700))
    legend(parts, ("CPU", "GPU E2E"))
    axes(parts, xs, ticks, yfn, "Batch Size", "Total Execution Time (s)",
         "Complete 872-sample workload; includes final partial batch; median of five runs.",
         log_scale=log_scale)
    series(parts, xs, cpu, yfn, CPU, [list(xs)])
    series(parts, xs, gpu, yfn, GPU, [list(xs)])
    xbracket = 1080
    ycpu, ygpu = yfn(cpu[64]), yfn(gpu[64])
    parts.extend((line(xbracket, ycpu, xbracket, ygpu, "#6B7788", 1.5),
                  line(xbracket - 7, ycpu, xbracket + 7, ycpu, "#6B7788", 1.5),
                  line(xbracket - 7, ygpu, xbracket + 7, ygpu, "#6B7788", 1.5),
                  text(xbracket - 14, (ycpu + ygpu) / 2 + 6, label, 18, INK,
                       "end", 600)))
    svg = write_svg(name, parts)
    return svg, {"batch_size": list(xs), "cpu_median_total_s": cpu,
                 "gpu_e2e_median_total_s": gpu, "batch64_speedup": speedup,
                 "final_batch_sizes": {n: int(rows[n]["final_batch_size"]) for n in xs}}


def main() -> None:
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCE_FILES}
    warm_files = list(PLOTS.glob("figure[1-4]*"))
    warm_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in warm_files}
    coarse = checked_rows(COLD, {1, 8, 32, 128, 872}, "sample_count",
                          "61d2fd112e534e75b85acc7a08aee527")
    refine = checked_rows(REFINE, {16}, "sample_count",
                          "6ddc295a876f455a9069ba0ede788644")
    app = checked_rows(APP, {1, 8, 32, 64}, "batch_size",
                       "832f8e758ad143dfb9f89fe9c9d36b77")
    PLOTS.mkdir(parents=True, exist_ok=True)
    rendered = [cold_figure(coarse, refine), application_figure(app, False),
                application_figure(app, True)]
    pngs = [render_png(svg) for svg, _ in rendered]
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest
           for path, digest in source_hashes.items()):
        raise RuntimeError("Figure source changed during rendering")
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest
           for path, digest in warm_hashes.items()):
        raise RuntimeError("Existing warm figure changed")
    for svg, values in rendered:
        if not svg.exists() or svg.stat().st_size == 0:
            raise RuntimeError(f"Empty SVG: {svg}")
        print(f"{svg.relative_to(ROOT)}: {values}")
    for png in pngs:
        with png.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(f"Invalid PNG: {png}")
        print(f"{png.relative_to(ROOT)}: {png.stat().st_size} bytes")
    print("passed: summary medians, audited N=16 IQR widths, speedup, "
          "source preservation, and existing warm-figure preservation")


if __name__ == "__main__":
    main()
