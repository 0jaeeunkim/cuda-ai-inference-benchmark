"""Render publication-style v2 figures from preserved summary data.

This script does not run benchmarks or read raw measurements. Figure 5 uses the
N=16 IQR widths already recorded in results/audit.md; since quartiles were not
preserved in a summary CSV, whiskers show median +/- IQR/2, not empirical Q1/Q3.
Existing figures in results/plots/ are never overwritten.
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
OUT = RESULTS / "plots" / "v2"
W, H = 1200, 720
L, R, T, B = 120, 1100, 150, 570
DARK = "#27313D"
CPU = "#303A46"
BLUE = "#0072B2"
ORANGE = "#C65D12"
MUTED = "#566273"
GRID = "#E4E8EC"
PALE = "#F3F7FA"
FONT = "Liberation Sans, Arial, sans-serif"
WARM_BATCHES = (1, 2, 4, 8, 16, 32, 64)
APP_BATCHES = (1, 8, 32, 64)
COLD_COUNTS = (1, 8, 16, 32)
INPUTS = (RESULTS / "summary.csv", RESULTS / "application_validation_summary.csv",
          RESULTS / "cold_start_v2_summary.csv", RESULTS / "cold_start_v2_refine_summary.csv",
          RESULTS / "audit.md")


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def read_rows(path: Path, key: str, expected: set[int], session_id: str) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    out = {int(row[key]): row for row in rows}
    if len(out) != len(rows) or set(out) != expected:
        raise ValueError(f"Unexpected or duplicate conditions in {path}")
    if any(row["session_id"] != session_id for row in rows):
        raise ValueError(f"Mixed sessions in {path}")
    return out


def val(row: dict[str, str], field: str) -> float:
    x = float(row[field])
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"Invalid {field}: {x}")
    return x


def audited_iqr() -> tuple[float, float]:
    audit = (RESULTS / "audit.md").read_text(encoding="utf-8")
    match = re.search(r"CPU IQR ([0-9.]+) ms and GPU IQR ([0-9.]+) ms", audit)
    if not match:
        raise ValueError("N=16 IQR widths missing from M7 audit")
    result = tuple(map(float, match.groups()))
    if any(not math.isfinite(x) or x <= 0 for x in result):
        raise ValueError("Invalid IQR width")
    return result


def tag_text(x: float, y: float, content: str, size: int = 18, color: str = DARK,
             anchor: str = "start", weight: int = 400, extra: str = "") -> str:
    return (f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" {extra}>{esc(content)}</text>')


def tag_line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1,
             extra: str = "") -> str:
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width:.1f}" {extra}/>')


def marker(x: float, y: float, kind: str, color: str, condition: int, value: float,
           series_name: str) -> str:
    data = (f'data-series="{esc(series_name)}" data-condition="{condition}" '
            f'data-value="{value:.9f}"')
    if kind == "circle":
        return (f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.0" fill="{color}" '
                f'stroke="white" stroke-width="1.0" {data}/>')
    if kind == "square":
        return (f'<rect x="{x-5:.2f}" y="{y-5:.2f}" width="10" height="10" '
                f'fill="white" stroke="{color}" stroke-width="2.3" {data}/>')
    if kind == "diamond":
        points = f"{x:.2f},{y-6:.2f} {x+6:.2f},{y:.2f} {x:.2f},{y+6:.2f} {x-6:.2f},{y:.2f}"
        return (f'<polygon points="{points}" fill="{color}" stroke="white" '
                f'stroke-width="1.0" {data}/>')
    raise ValueError(kind)


class Chart:
    def __init__(self, title: str, xlabel: str, ylabel: str, xmap: dict[int, float],
                 yticks: list[float], ymin: float, ymax: float, *, logy: bool = False,
                 description: str = "") -> None:
        self.title = title
        self.xmap = xmap
        self.ymin, self.ymax, self.logy = ymin, ymax, logy
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" role="img">',
            f'<title>{esc(title)}</title>',
            f'<desc>{esc(description)}</desc>',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<g font-family="{FONT}">',
            tag_text(L, 57, title, 36, DARK, weight=600),
        ]
        self.yticks = yticks
        self.xlabel, self.ylabel = xlabel, ylabel

    def y(self, value: float) -> float:
        if not math.isfinite(value) or (self.logy and value <= 0):
            raise ValueError(value)
        if self.logy:
            fraction = ((math.log10(value) - math.log10(self.ymin)) /
                        (math.log10(self.ymax) - math.log10(self.ymin)))
        else:
            fraction = (value - self.ymin) / (self.ymax - self.ymin)
        return B - fraction * (B - T)

    def band(self, x1: float, x2: float, label: str) -> None:
        self.parts.append(f'<rect x="{x1:.2f}" y="{T}" width="{x2-x1:.2f}" '
                          f'height="{B-T}" fill="{PALE}"/>')
        self.parts.append(tag_text((x1+x2)/2, T+29, label, 23, MUTED, "middle"))

    def axes(self) -> None:
        for tick in self.yticks:
            y = self.y(tick)
            self.parts.append(tag_line(L, y, R, y, GRID, 0.9))
            self.parts.append(tag_line(L-6, y, L, y, DARK, 1.1))
            self.parts.append(tag_text(L-13, y+8, f"{tick:g}", 23, MUTED, "end"))
        for value, x in self.xmap.items():
            self.parts.append(tag_line(x, B, x, B+6, DARK, 1.1))
            self.parts.append(tag_text(x, B+36, str(value), 23, MUTED, "middle"))
        self.parts.extend((tag_line(L, T, L, B, DARK, 1.4),
                           tag_line(L, B, R, B, DARK, 1.4),
                           tag_text((L+R)/2, 660, self.xlabel, 29, DARK, "middle", 600),
                           tag_text(0, 0, self.ylabel, 29, DARK, "middle", 600,
                                    extra=f'transform="translate(37 {(T+B)/2:.2f}) rotate(-90)"')))

    def legend(self, entries: list[tuple[str, str, str, bool]]) -> None:
        starts = ([L] if len(entries) == 1 else
                  [L, 575] if len(entries) == 2 else [L, 445, 765])
        for x, (label, color, kind, dashed) in zip(starts, entries):
            dash = 'stroke-dasharray="7 5"' if dashed else ""
            self.parts.append(tag_line(x, 105, x+37, 105, color, 2.8, dash))
            self.parts.append(marker(x+18.5, 105, kind, color, 0, 0, "legend"))
            self.parts.append(tag_text(x+50, 114, label, 25, DARK))

    def series(self, name: str, values: dict[int, float], color: str,
               marker_kind: str, *, dashed: bool = False) -> None:
        if set(values) != set(self.xmap):
            raise ValueError(f"Incomplete series {name}")
        ordered = list(self.xmap)
        points = " ".join(f"{self.xmap[n]:.2f},{self.y(values[n]):.2f}" for n in ordered)
        dash = 'stroke-dasharray="7 5"' if dashed else ""
        self.parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                          f'stroke-width="2.8" stroke-linecap="round" '
                          f'stroke-linejoin="round" {dash}/>')
        for n in ordered:
            self.parts.append(marker(self.xmap[n], self.y(values[n]), marker_kind,
                                     color, n, values[n], name))

    def finish(self, name: str) -> Path:
        self.parts.extend(("</g>", "</svg>"))
        path = OUT / name
        path.write_text("\n".join(self.parts)+"\n", encoding="utf-8")
        return path


def warm_x() -> dict[int, float]:
    return {n: L + math.log2(n)/6*(R-L) for n in WARM_BATCHES}


def add_batch_one_gpu_inset(chart: Chart, inference_ms: float, e2e_ms: float) -> None:
    """Magnify the two measured GPU medians at batch size one."""
    x0, y0, width, height = 210, 205, 460, 215
    low, high = 1.24, 1.30
    if not (low < inference_ms < e2e_ms < high):
        raise ValueError("Batch-one medians no longer fit the declared inset range")

    def inset_y(value: float) -> float:
        return 390 - (value - low) / (high - low) * 120

    chart.parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
                       'fill="white" stroke="#8B96A3" stroke-width="1.4"/>')
    chart.parts.append(tag_text(x0 + 22, y0 + 34, "GPU detail · batch size 1", 25,
                                DARK, weight=600))
    chart.parts.append(tag_line(300, 270, 300, 390, DARK, 1.1))
    for tick in (1.24, 1.26, 1.28, 1.30):
        y = inset_y(tick)
        chart.parts.append(tag_line(300, y, 622, y, GRID, 0.8))
        chart.parts.append(tag_text(287, y + 6, f"{tick:.2f}", 19, MUTED, "end"))
    for label, value, color, kind in (
        ("GPU inference-only", inference_ms, BLUE, "square"),
        ("GPU E2E", e2e_ms, ORANGE, "diamond"),
    ):
        y = inset_y(value)
        chart.parts.append(tag_line(350, y, 445, y, color, 3.0,
                                    'stroke-dasharray="7 5"' if kind == "square" else ""))
        if kind == "square":
            chart.parts.append(
                f'<rect x="440.00" y="{y-5:.2f}" width="10" height="10" '
                f'fill="white" stroke="{color}" stroke-width="2.3" '
                f'data-inset-series="{esc(label)}" data-value="{value:.9f}"/>')
        else:
            points = f"445.00,{y-6:.2f} 451.00,{y:.2f} 445.00,{y+6:.2f} 439.00,{y:.2f}"
            chart.parts.append(
                f'<polygon points="{points}" fill="{color}" stroke="white" '
                f'stroke-width="1.0" data-inset-series="{esc(label)}" '
                f'data-value="{value:.9f}"/>')
        chart.parts.append(tag_text(467, y + 7, f"{value:.3f} ms", 22, DARK))


def warm_figures(rows: dict[int, dict[str, str]]) -> list[Path]:
    xs = warm_x()
    series = [
        ("CPU", {n: val(rows[n], "cpu_median_ms") for n in xs}, CPU, "circle", False),
        ("GPU inference-only", {n: val(rows[n], "gpu_inference_only_median_ms") for n in xs}, BLUE, "square", True),
        ("GPU E2E", {n: val(rows[n], "gpu_end_to_end_median_ms") for n in xs}, ORANGE, "diamond", False),
    ]
    outputs = []
    for logy, name, title, ticks, ymin, ymax in [
        (False, "figure1_median_latency.svg", "Median batch latency", [0,100,200,300,400,500], 0, 500),
        (True, "figure1_median_latency_log.svg", "Median batch latency · log scale", [1,2,5,10,20,50,100,200,500,1000], 1, 1000),
    ]:
        chart = Chart(title, "Batch size", "Median batch latency (ms)", xs,
                      ticks, ymin, ymax, logy=logy,
                      description="Thirty repeated measurements per path and batch; fixed sequence length 64.")
        chart.axes()
        chart.legend([(s[0],s[2],s[3],s[4]) for s in series])
        for label, values, color, kind, dashed in series:
            chart.series(label, values, color, kind, dashed=dashed)
        if not logy:
            add_batch_one_gpu_inset(chart, series[1][1][1], series[2][1][1])
        outputs.append(chart.finish(name))
    chart = Chart("Inference throughput", "Batch size", "Throughput (samples/s)", xs,
                  [0,1000,2000,3000,4000,5000], 0, 5000,
                  description="Throughput equals batch size divided by median batch latency.")
    chart.axes()
    chart.legend([(s[0],s[2],s[3],s[4]) for s in series])
    for label, _, color, kind, dashed in series:
        values = {n: val(rows[n], f'{"gpu_end_to_end" if label=="GPU E2E" else "gpu_inference_only" if label=="GPU inference-only" else "cpu"}_median_derived_throughput_samples_per_sec') for n in xs}
        chart.series(label, values, color, kind, dashed=dashed)
    outputs.append(chart.finish("figure2_throughput.svg"))
    chart = Chart("CPU-to-GPU speedup", "Batch size", "CPU median / GPU median (×)", xs,
                  [0,5,10,15,20,25,30,35], 0, 35,
                  description="Speedup is CPU median batch latency divided by GPU median batch latency.")
    chart.axes()
    yparity=chart.y(1)
    chart.parts.append(tag_line(L, yparity, R, yparity, "#8B949E", 1.6,
                                'stroke-dasharray="6 5"'))
    chart.parts.append(tag_text(R-5, yparity-11, "1× parity", 21, MUTED, "end"))
    chart.legend([("GPU inference-only",BLUE,"square",True),("GPU E2E",ORANGE,"diamond",False)])
    chart.series("GPU inference-only", {n:val(rows[n],"gpu_inference_only_speedup_vs_cpu") for n in xs},
                 BLUE,"square",dashed=True)
    chart.series("GPU E2E", {n:val(rows[n],"gpu_end_to_end_speedup_vs_cpu") for n in xs},
                 ORANGE,"diamond")
    outputs.append(chart.finish("figure3_speedup.svg"))
    chart=Chart("GPU E2E relative excess", "Batch size", "Relative E2E excess (%)", xs,
                [0,0.5,1,1.5,2,2.5,3],0,3,
                description="100 × (E2E median − inference-only median) / E2E median; not isolated transfer time.")
    chart.axes()
    chart.legend([("E2E relative excess",ORANGE,"diamond",False)])
    chart.series("E2E relative excess", {n:val(rows[n],"e2e_relative_overhead_pct") for n in xs},
                 ORANGE,"diamond")
    outputs.append(chart.finish("figure4_e2e_relative_overhead.svg"))
    return outputs


def cold_figure(coarse: dict[int, dict[str,str]], refine: dict[int,dict[str,str]]) -> Path:
    xs=dict(zip(COLD_COUNTS,(165.0,460.0,755.0,1050.0)))
    rows={n:coarse[n] for n in (1,8,32)} | {16:refine[16]}
    cpu={n:val(rows[n],"cpu_median_total_job_ms") for n in xs}
    gpu={n:val(rows[n],"gpu_median_total_job_ms") for n in xs}
    cpu_iqr,gpu_iqr=audited_iqr()
    chart=Chart("Process-cold CPU–GPU transition", "Samples per job (N)",
                "Median total job time (ms)",xs,[0,100,200,300,400,500],0,500,
                description="N=16 whiskers show median±IQR/2 using audited IQR widths, not empirical quartile endpoints. Lines connect measured N only; no exact crossover is inferred.")
    chart.band(xs[16],xs[32],"Observed transition region")
    chart.axes()
    chart.legend([("CPU",CPU,"circle",False),("GPU",ORANGE,"diamond",False)])
    # A third legend key describes the centered IQR-width indicator.
    xkey=775
    chart.parts.extend((tag_line(xkey,94,xkey,116,MUTED,1.8),
                        tag_line(xkey-6,94,xkey+6,94,MUTED,1.8),
                        tag_line(xkey-6,116,xkey+6,116,MUTED,1.8),
                        tag_text(xkey+19,114,"N=16 centered IQR width",22,MUTED)))
    chart.series("CPU",cpu,CPU,"circle")
    chart.series("GPU",gpu,ORANGE,"diamond")
    for offset,median,width,color in ((-12,cpu[16],cpu_iqr,CPU),
                                      (12,gpu[16],gpu_iqr,ORANGE)):
        x=xs[16]+offset
        high,low=chart.y(median+width/2),chart.y(median-width/2)
        chart.parts.extend((tag_line(x,high,x,low,color,2.0),
                            tag_line(x-6,high,x+6,high,color,2.0),
                            tag_line(x-6,low,x+6,low,color,2.0)))
    chart.parts.append(tag_text(695,418,"Near tie",23,DARK,"end",600))
    chart.parts.append(tag_line(703,407,xs[16]-16,chart.y((cpu[16]+gpu[16])/2)+8,
                                MUTED,1.2))
    return chart.finish("figure5_cold_transition.svg")


def app_figures(rows: dict[int,dict[str,str]]) -> list[Path]:
    xs=dict(zip(APP_BATCHES,(165.0,460.0,755.0,1050.0)))
    if any(int(rows[n]["sample_count"])!=872 for n in xs):
        raise ValueError("Not a complete 872-sample workload")
    if [int(rows[n]["final_batch_size"]) for n in xs]!=[1,8,8,40]:
        raise ValueError("Unexpected final partial batch")
    cpu={n:val(rows[n],"cpu_median_total_ms")/1000 for n in xs}
    gpu={n:val(rows[n],"gpu_e2e_median_total_ms")/1000 for n in xs}
    speedup=val(rows[64],"gpu_e2e_speedup_vs_cpu")
    if not math.isclose(speedup,cpu[64]/gpu[64],rel_tol=1e-8):
        raise ValueError("Batch-64 speedup does not reconstruct")
    outputs=[]
    for logy,name,title,ticks,ymin,ymax in [
        (False,"figure6_application_total.svg","Full 872-sample execution time",[0,2,4,6,8,10],0,10),
        (True,"figure6_application_total_log.svg","Full 872-sample execution time · log scale",[.1,.2,.5,1,2,5,10,20],.1,20),
    ]:
        chart=Chart(title,"Batch size","Total execution time (s)",xs,ticks,ymin,ymax,
                    logy=logy,
                    description="Complete 872-sample validation workload, including the final partial batch; median of five runs per path and batch.")
        chart.axes()
        chart.legend([("CPU",CPU,"circle",False),("GPU E2E",ORANGE,"diamond",False)])
        chart.series("CPU",cpu,CPU,"circle")
        chart.series("GPU E2E",gpu,ORANGE,"diamond")
        xb=1082
        ycpu,ygpu=chart.y(cpu[64]),chart.y(gpu[64])
        chart.parts.extend((tag_line(xb,ycpu,xb,ygpu,MUTED,1.4),
                            tag_line(xb-6,ycpu,xb+6,ycpu,MUTED,1.4),
                            tag_line(xb-6,ygpu,xb+6,ygpu,MUTED,1.4),
                            tag_text(xb-15,(ycpu+ygpu)/2+6,f"{speedup:.2f}× speedup",
                                     24,DARK,"end",600)))
        outputs.append(chart.finish(name))
    return outputs


def render_png(svg: Path) -> Path:
    convert=shutil.which("convert")
    if convert is None:
        raise RuntimeError("ImageMagick convert is required for PNG previews")
    png=svg.with_suffix(".png")
    command=[convert,"-background","white","-density","180",str(svg),
             "-resize","2400x1440!","-strip","-define","png:exclude-chunks=date,time",
             f"PNG24:{png}"]
    subprocess.run(command,check=True,capture_output=True,text=True)
    if png.read_bytes()[:8]!=b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Invalid PNG {png}")
    return png


def main() -> None:
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}
    v1={p:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (RESULTS/"plots").glob("figure*") if p.is_file()}
    warm=read_rows(RESULTS/"summary.csv","batch_size",set(WARM_BATCHES),
                   "efb3e675224644b4be20f2875c37e605")
    app=read_rows(RESULTS/"application_validation_summary.csv","batch_size",set(APP_BATCHES),
                  "832f8e758ad143dfb9f89fe9c9d36b77")
    cold=read_rows(RESULTS/"cold_start_v2_summary.csv","sample_count",{1,8,32,128,872},
                   "61d2fd112e534e75b85acc7a08aee527")
    refine=read_rows(RESULTS/"cold_start_v2_refine_summary.csv","sample_count",{16},
                     "6ddc295a876f455a9069ba0ede788644")
    OUT.mkdir(parents=True,exist_ok=True)
    svg_paths=warm_figures(warm)+[cold_figure(cold,refine)]+app_figures(app)
    png_paths=[render_png(svg) for svg in svg_paths]
    if len(svg_paths)!=8 or len(png_paths)!=8:
        raise RuntimeError("Missing v2 figure")
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in before.items()):
        raise RuntimeError("Source summary/audit changed")
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in v1.items()):
        raise RuntimeError("Existing v1 figure changed")
    print(f"v2 complete: {len(svg_paths)} SVG, {len(png_paths)} PNG; summaries and v1 figures unchanged")
    for p in svg_paths:
        print(p.relative_to(ROOT))


if __name__=="__main__":
    main()
