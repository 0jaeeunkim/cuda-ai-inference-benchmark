"""Postprocess audited Project 2 artifacts; never rerun an experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import html
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import project2_protocol as p


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def speedup_figure(summary):
    values = []
    for row in summary:
        batch, variant = int(row["batch_size"]), row["variant"]
        baseline = next(float(r["median_total_ms"]) for r in summary
                        if int(r["batch_size"]) == batch and r["variant"] == "A")
        median = float(row["median_total_ms"])
        values.append({"condition": f"B={batch}  {variant}", "variant": variant,
                       "speedup": baseline / median,
                       "low": baseline / float(row["q3_total_ms"]),
                       "high": baseline / float(row["q1_total_ms"]),
                       "label": f"{median:.3f} ms  |  {baseline / median:.4f}x"})
    return {"$schema": "https://vega.github.io/schema/vega/v5.json", "width": 700, "height": 460,
            "background": "white", "padding": {"left": 95, "right": 220, "top": 75, "bottom": 75},
            "title": {"text": "Prepared-buffer warm E2E: speedup over A", "anchor": "start",
                      "subtitle": ["872 rows per run; 32 runs per condition; RTX 5070 Ti; FP32; S=64",
                                   "Dots: ratios of medians. Whiskers: transformed run-time IQR, not confidence intervals."]},
            "data": [{"name": "results", "values": values}],
            "scales": [{"name": "x", "type": "linear", "domain": [.99, 1.115], "range": "width", "zero": False},
                       {"name": "y", "type": "band", "domain": [r["condition"] for r in values], "range": "height", "padding": .4},
                       {"name": "color", "type": "ordinal", "domain": list("ABCD"),
                        "range": ["#64748b", "#d18b20", "#15803d", "#2563eb"]}],
            "axes": [{"orient": "bottom", "scale": "x", "title": "Median(A) / median(variant)", "format": ".2f", "grid": True},
                     {"orient": "left", "scale": "y", "domain": False, "ticks": False, "labelPadding": 10}],
            "marks": [{"type": "rule", "encode": {"enter": {"x": {"scale": "x", "value": 1},
                        "y": {"value": 0}, "y2": {"signal": "height"}, "stroke": {"value": "#94a3b8"}, "strokeDash": {"value": [4, 4]}}}},
                      {"type": "rule", "from": {"data": "results"}, "encode": {"enter": {
                          "x": {"scale": "x", "field": "low"}, "x2": {"scale": "x", "field": "high"},
                          "y": {"scale": "y", "field": "condition", "band": .5},
                          "stroke": {"scale": "color", "field": "variant"}, "strokeWidth": {"value": 3}}}},
                      {"type": "symbol", "from": {"data": "results"}, "encode": {"enter": {
                          "x": {"scale": "x", "field": "speedup"}, "y": {"scale": "y", "field": "condition", "band": .5},
                          "size": {"value": 80}, "fill": {"scale": "color", "field": "variant"}}}},
                      {"type": "text", "from": {"data": "results"}, "encode": {"enter": {
                          "x": {"signal": "width + 12"}, "y": {"scale": "y", "field": "condition", "band": .5},
                          "baseline": {"value": "middle"}, "text": {"field": "label"}, "fontSize": {"value": 12}}}}]}


def timeline_figure(records, zoom=False):
    values = []
    for r in records:
        if zoom and r["variant"] != "D":
            continue
        stream = int(r["stream"])
        lane = f"{r['variant']} stream {stream}"
        values.append({"lane": lane, "kind": r["kind"], "batch": int(r["batch"]),
                       "start": float(r["start_us"]) / 1000, "end": float(r["end_us"]) / 1000})
    domain = [0, max(r["end"] for r in values) * 1.01]
    if zoom:
        selected = [r for r in values if r["kind"] == "H2D" and r["batch"] == 1]
        domain = [min(r["start"] for r in selected) - .005, max(r["end"] for r in selected) + .005]
    lanes = sorted({r["lane"] for r in values})
    return {"$schema": "https://vega.github.io/schema/vega/v5.json", "width": 900,
            "height": 160 if zoom else 280, "background": "white",
            "padding": {"left": 115, "right": 35, "top": 75, "bottom": 95},
            "title": {"text": "D: H2D of batch 1 overlaps batch 0 kernels" if zoom else "Observed CUDA activity: A / C / D",
                      "anchor": "start", "subtitle": ["B=8, N=19, batches [8,8,3]; one profiled sequence per variant",
                          "Actual intervals; each variant starts at its first GPU activity. Profiled timing is excluded from benchmark raw."]},
            "data": [{"name": "events", "values": values}],
            "scales": [{"name": "x", "type": "linear", "domain": domain, "range": "width", "zero": False},
                       {"name": "y", "type": "band", "domain": lanes, "range": "height", "padding": .3},
                       {"name": "color", "type": "ordinal", "domain": ["kernel", "H2D", "logits_D2H", "internal_D2H", "memset"],
                        "range": ["#475569", "#2563eb", "#16a34a", "#dc2626", "#cbd5e1"]}],
            "axes": [{"orient": "bottom", "scale": "x", "title": "GPU time from first activity (ms)", "format": ".3f" if zoom else ".1f", "grid": True},
                     {"orient": "left", "scale": "y", "domain": False, "ticks": False, "labelPadding": 10}],
            "legends": [{"fill": "color", "orient": "bottom", "direction": "horizontal"}],
            "marks": [{"type": "rect", "clip": True, "from": {"data": "events"}, "encode": {"enter": {
                "x": {"scale": "x", "field": "start"}, "x2": {"scale": "x", "field": "end"},
                "y": {"scale": "y", "field": "lane"}, "height": {"scale": "y", "band": 1},
                "fill": {"scale": "color", "field": "kind"}}}}]}


def report(session, trace_session, node, vega):
    meta = p.read_json(session / "metadata.json")
    audit = p.read_json(session / "audit.json")
    trace = p.read_json(trace_session / "trace_audit.json")
    assert audit["passed"] and audit["evidence_sha256"]["raw.csv"] == p.sha256(session / "raw.csv")
    assert meta["protocol_sha256"] == p.protocol_hash()
    with (session / "summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    with (trace_session / "gpu_intervals.csv").open() as handle:
        intervals = list(csv.DictReader(handle))
    lookup = {(int(r["batch_size"]), r["variant"]): r for r in summary}
    median = lambda b, v: float(lookup[(b, v)]["median_total_ms"])
    comparisons = []
    for batch in p.BATCHES:
        a, b, c, d = (median(batch, v) for v in "ABCD")
        comparisons.append({"batch_size": batch, "A_ms": a, "B_ms": b, "C_ms": c, "D_ms": d,
                            "A_over_B": a / b, "B_over_C": b / c, "C_over_D": c / d,
                            "A_over_C": a / c, "A_over_D": a / d,
                            "C_time_reduction_percent": 100 * (1 - c / a),
                            "D_time_reduction_percent": 100 * (1 - d / a)})
    write_csv(session / "comparisons.csv", comparisons)
    setup = []
    for batch in p.BATCHES:
        for variant in "ABCD":
            r = meta["setups"][str(batch)][variant]
            setup.append({"batch_size": batch, "variant": variant,
                          **{k: r[k] for k in ("host_pinned", "host_bytes", "pinned_bytes",
                             "host_allocation_and_copy_setup_ms", "buffer_stream_event_setup_ms", "device_input_bytes")}})
    write_csv(session / "setup_costs.csv", setup)
    plots = session / "plots"
    plots.mkdir(exist_ok=True)
    specs = {"speedup": speedup_figure(summary), "timeline": timeline_figure(intervals),
             "overlap_detail": timeline_figure(intervals, zoom=True)}
    spec_paths = []
    for name, spec in specs.items():
        path = plots / f"{name}.vega.json"
        p.write_json(path, spec)
        spec_paths.append(path)
    subprocess.run([str(node), str(ROOT / "scripts/project2_render.cjs"), str(vega),
                    *map(str, spec_paths)], check=True, timeout=30)
    lines = ["# Project 2 — 결과 분석", "", f"Session: `{meta['session_id']}`.",
             f"Protocol SHA256: `{meta['protocol_sha256']}`.", "",
             "## 핵심 결과", "",
             "384개 raw와 각 실행의 correctness, 실행 순서, 메타데이터 및 요약 산식을 감사해 모두 통과했다. "
             "모든 logits는 finite FP32 [872,2]이며 A reference와 최대 절대오차 0.0, prediction 일치다. "
             "전체 측정값과 더 느린 버전도 보존했다. Project 1 보호 파일 85개 해시가 유지됐다.", "",
             "주 지표는 prepared-buffer warm E2E total time이다. 각 수치는 CPU 입력에서 원래 순서의 "
             "872개 CPU-visible logits 완료까지이며, 공통 최종 device synchronize를 포함한다. "
             "모델 로딩·토큰화·pinning/버퍼/Stream/Event 준비·warm-up·correctness·파일 I/O·trace는 제외한다.", "",
             "| Batch | A ms | B ms | C ms | D ms | A/C | A/D | C→D 시간 변화 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in comparisons:
        change = 100 * (r["D_ms"] / r["C_ms"] - 1)
        lines.append(f"| {r['batch_size']} | {r['A_ms']:.3f} | {r['B_ms']:.3f} | {r['C_ms']:.3f} | {r['D_ms']:.3f} | "
                     f"{r['A_over_C']:.4f}× | {r['A_over_D']:.4f}× | {change:+.3f}% |")
    lines += ["", "수치는 각 조건 32회의 중앙값이다. Speedup은 같은 batch/session의 중앙값 비율이다. "
              "Throughput은 872×1000/median_ms이며 speedup 1.1020×는 throughput 약 10.20% 증가, "
              "처리시간 약 9.26% 감소를 뜻한다. paired-run speedup이나 요청별 latency 추정이 아니다.", "",
              "![A 대비 speedup 및 변환 IQR](plots/speedup.svg)", "",
              "Whisker는 각 조건 Q1–Q3 시간을 고정한 A 중앙값으로 나눈 변환 IQR이다. 신뢰구간이나 통계적 유의성 지표가 아니다. "
              "[summary.csv](summary.csv)에 원래 시간 IQR과 throughput을 보존했다.", "",
              "## A→B, B→C, C→D 해석", "",
              "A→B: blocking copy를 유지하고 pinned memory만 사용했을 때 변화가 매우 작았다. "
              "A/B 중앙값 비율은 batch 1/8/64에서 각각 " + ", ".join(f"{r['A_over_B']:.6f}×" for r in comparisons) + ". "
              "Pinned memory 단독의 큰 이득은 이 workload에서 관찰되지 않았다.", "",
              "B→C: 같은 pinned memory에서 명시적 blocking copy의 Host 대기를 줄인 C가 가장 큰 변화를 보였다. "
              "B/C 중앙값 비율은 각각 " + ", ".join(f"{r['B_over_C']:.6f}×" for r in comparisons) + ". "
              "작은 batch에서 개선이 컸다. 이 차이는 Python 제출, 복사 API와 동기화 경로가 함께 달라진 결과이며 순수 PCIe 전송시간이 아니다.", "",
              "C→D: 두 Stream과 Event 파이프라인을 추가해도 추가 효과는 작았고 batch 1에서는 D가 약 0.251% 느렸다. "
              "batch 8/64에서는 D의 중앙값 시간이 C보다 약 0.188%/0.113% 낮았다. "
              "이는 단일 session의 기술 통계이며 작은 차이의 일반적 우위나 유의성을 주장하지 않는다.", "",
              "## Trace에서 확인한 사실", "",
              "별도 B=8,N=19의 [8,8,3] 한 sequence를 A/C/D 각각 수집했다. "
              "각 trace에는 forward kernel 366개, 입력 H2D 6개, logits D2H 3개가 있다. "
              "CUDA correlation을 CPU batch marker에 연결해 GPU interval의 실제 교집합을 계산했다. "
              "전체 forward envelope 내부의 공백을 overlap으로 세지 않았다.", "",
              "| 버전 | Compute Stream | H2D Stream | H2D 합계 μs | 실제 kernel과 overlap μs | Host stream sync 호출 | forward 내부 sync |",
              "| :---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for v in "ACD":
        r = trace["variants"][v]
        lines.append(f"| {v} | {r['compute_streams'][0]} | {r['copy_streams'][0]} | {r['h2d_total_duration_us']:.3f} | "
                     f"{r['h2d_kernel_overlap_us']:.3f} | {r['cpu_stream_sync_count']} | {r['forward_internal_stream_sync_count']} |")
    lines += ["", "D에서 다음 batch의 H2D가 이전 forward kernel과 실제로 겹쳤다. "
              "batch 1 입력 두 복사의 overlap 약 0.832 μs, batch 2 입력 두 복사의 overlap 약 0.576 μs, 합계 약 1.408 μs다. "
              "A/C의 같은 종류 overlap은 0이다. D의 slot 0 재사용과 각 forward 시작 전 upload 완료도 trace에서 확인했다. "
              "이 관찰은 B=8,N=19 한 trace에 한정되며 full 872-row 실행이나 다른 batch의 overlap 양을 추정하지 않는다.", "",
              "![CUDA timeline](plots/timeline.svg)", "", "![D overlap 상세](plots/overlap_detail.svg)", "",
              "Trace에는 C/D에도 forward마다 1-byte D2H와 Host stream sync가 하나씩 남아 있다. "
              "CPU 연산은 aten::is_nonzero → aten::item → aten::_local_scalar_dense로 식별됐다. "
              "C/D는 runner의 명시적 blocking copy 대기를 제거하지만 모델 내부 동기화까지 제거하는 구현은 아니다. "
              "이를 바꾸는 모델/커널 최적화는 이번 범위에 추가하지 않았다.", "",
              "프로파일러의 cycle 경고는 각 새 profiler 인스턴스의 단일 capture 결과에 대해 "
              "모든 9개 batch marker와 CUDA 활동이 존재함을 확인한 뒤 판단했다. "
              "Trace 시간과 Host wait 합계는 instrumentation 영향을 받으며 benchmark raw와 합치지 않는다.", "",
              "## 준비 비용과 해석 한계", "",
              "Pinned variant는 각 899,904 B의 전체 입력+출력을 준비한다. 입력 892,928 B, 출력 6,976 B다. "
              "각 variant의 host allocation/pinning/copy 및 buffer/Stream/Event 준비 비용은 "
              "[setup_costs.csv](setup_costs.csv)에 별도 기록했다. 매 batch pinning/staging은 없고 이 비용은 주 timer 밖이다. "
              "따라서 결과를 입력 공급·준비나 시작 비용까지 포함한 application E2E라고 부르지 않는다.", "",
              "Batch 64의 한 full batch 입력 H2D는 65,536 B, logits D2H는 512 B로 작다. "
              "B=8 trace에서도 실제 copy 자체는 짧았다. 관측한 중첩 양이 작고 Event/제출 비용도 있으므로 "
              "overlap 달성이 큰 E2E 개선으로 이어지지 않을 수 있다는 해석과 일치한다. "
              "어느 요인이 정확히 얼마를 설명하는지 또는 전체 성능 상한은 이 증거로 단정하지 않는다.", "",
              "한 RTX 5070 Ti, 한 model/dataset, FP32/S=64, 한 session의 결과다. "
              "각 batch별 순서는 균형화되어 각 variant가 각 position에 8회씩 등장한다. "
              "Batch 순서는 1→8→64로 고정되어 있고 전체 session의 온도/clock 이력은 수집하지 않았다. "
              "다른 GPU 작업의 부재, GPU 포화·occupancy·대역폭 병목 또는 여러 날의 재현성을 주장하지 않는다. "
              "Project 1의 다른 GPU·출력 계약 수치는 speedup 분모로 사용하지 않았다.", "",
              "## 감사 및 재현", "",
              "- [전체 결과 감사](audit.json): 384 raw, 12 coverage, 실제 CPU reference 3개, 해시·gate·순서·모든 요약 산식 검증.",
              f"- [Trace 감사]({os.path.relpath(trace_session / 'trace_audit.json', session)}): 원본 trace hash, CUDA correlation, 실제 interval 교집합.",
              "- [비교 산식](comparisons.csv), [그래프 사양](plots/speedup.vega.json), [후처리 provenance](report_metadata.json).",
              "- 실행 코드는 변경하지 않았다. 추가한 감사/그림 코드만 별도로 hash를 기록했다. 전체 correctness/benchmark를 재실행하지 않았다.", "",
              "결론: 이 조건에서는 명시적 Host 대기 감소(C)의 이득이 주로 작은 batch에서 나타났다. "
              "D의 실제 overlap은 확인됐지만 C보다 큰 추가 성능 이득은 관찰되지 않았다. 더 느린 D의 batch 1 결과도 유효한 결과로 보존했다.", ""]
    old = session / "analysis.md"
    if old.exists() and not (session / "analysis_generated.md").exists():
        (session / "analysis_generated.md").write_bytes(old.read_bytes())
    old.write_text("\n".join(lines))
    svgs = [(plots / f"{name}.svg").read_text() for name in specs]
    (session / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Project 2 results</title>'
        '<style>body{font:16px system-ui;margin:32px;color:#172033}svg{max-width:100%;height:auto}section{margin:28px 0}</style>'
        '<h1>Project 2 — DistilBERT async pipeline</h1>'
        '<p>384 audited runs. FP32, S=64, 872 rows/run, RTX 5070 Ti. D overlaps H2D with kernels in the short trace; added performance over C is small.</p>'
        + ''.join('<section>' + svg + '</section>' for svg in svgs)
        + '<p>Figures show within-session measurements and a separate B=8,N=19 trace. See analysis.md for limitations and checksums.</p>')
    provenance = {"generated_at_utc": p.now(), "command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
                  "benchmark_session": meta["session_id"], "protocol_sha256": p.protocol_hash(),
                  "raw_sha256": p.sha256(session / "raw.csv"), "trace_audit_sha256": p.sha256(trace_session / "trace_audit.json"),
                  "postprocessing_sources": {str(path.relative_to(ROOT)): p.sha256(path) for path in
                       [Path(__file__), ROOT / "scripts/project2_render.cjs", ROOT / "src/project2_trace_audit.py", ROOT / "src/project2_results_audit.py"]},
                  "vega": {"version": "5.30.0", "bundle_sha256": p.sha256(vega),
                           "url": "https://cdn.jsdelivr.net/npm/vega@5.30.0/build/vega.min.js"},
                  "artifact_sha256": {str(path.relative_to(session)): p.sha256(path) for path in
                        [session / "analysis.md", session / "report.html", session / "comparisons.csv", session / "setup_costs.csv",
                         *sorted(plots.glob("*.svg")), *spec_paths]}, "project1": p.verify_project1()}
    p.write_json(session / "report_metadata.json", provenance)
    return {"report": str(session / "analysis.md"), "figures": 3, "benchmark_rerun": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--trace-session", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--vega", type=Path, required=True)
    args = parser.parse_args()
    print(report(args.session.resolve(), args.trace_session.resolve(), args.node.resolve(), args.vega.resolve()))
