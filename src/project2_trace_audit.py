"""Read-only Kineto trace audit; never runs inference or profiling."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import project2_protocol as p


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersection_duration(left, right):
    left, right = merged(left), merged(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0.0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def inspect(path, variant):
    events = json.loads(path.read_text())["traceEvents"]
    markers = [e for e in events if e.get("cat") == "user_annotation"
               and e.get("name", "").startswith(variant + "/")]
    assert len(markers) == 9, "Expected three H2D/forward/D2H batches"
    host = defaultdict(list)
    for event in events:
        if event.get("cat") in ("cuda_runtime", "cuda_driver"):
            host[event.get("args", {}).get("correlation")].append(event)

    def stage_for_host(event):
        stages = [m["name"] for m in markers if m["pid"] == event["pid"]
                  and m["tid"] == event["tid"] and m["ts"] <= event["ts"]
                  and event["ts"] < m["ts"] + m["dur"]]
        assert len(stages) <= 1
        return stages[0] if stages else None

    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    origin = min(e["ts"] for e in gpu)
    records = []
    for event in gpu:
        candidates = host[event.get("args", {}).get("correlation")]
        stages = {stage_for_host(h) for h in candidates} - {None}
        assert len(stages) == 1, "GPU activity must map uniquely to a batch marker"
        stage = stages.pop()
        _, operation, index, _ = stage.split("/")
        name, category = event["name"], event["cat"]
        kind = {"kernel": "kernel", "gpu_memset": "memset"}.get(category, "internal_D2H")
        if category == "gpu_memcpy":
            if "HtoD" in name:
                kind = "H2D"
                assert operation == "H2D"
            elif operation == "D2H":
                kind = "logits_D2H"
            else:
                assert operation == "forward" and "DtoH" in name
        records.append({"variant": variant, "batch": int(index), "operation": operation,
                        "kind": kind, "stream": event["args"]["stream"],
                        "start_us": event["ts"] - origin,
                        "end_us": event["ts"] - origin + event["dur"],
                        "duration_us": event["dur"], "bytes": event["args"].get("bytes", 0),
                        "name": name, "correlation": event["args"].get("correlation")})
    kernels = [r for r in records if r["kind"] == "kernel"]
    uploads = [r for r in records if r["kind"] == "H2D"]
    outputs = [r for r in records if r["kind"] == "logits_D2H"]
    scalar = [r for r in records if r["kind"] == "internal_D2H"]
    assert len(uploads) == 6 and len(outputs) == 3 and len(kernels) > 0
    assert [r["bytes"] for r in sorted(outputs, key=lambda r: r["batch"])] == [64, 64, 24]
    for batch, n in enumerate((8, 8, 3)):
        assert [r["bytes"] for r in uploads if r["batch"] == batch] == [n * 64 * 8] * 2
    compute_streams = {r["stream"] for r in kernels}
    copy_streams = {r["stream"] for r in uploads}
    assert len(compute_streams) == len(copy_streams) == 1
    assert all(r["stream"] in compute_streams for r in outputs)
    assert (compute_streams != copy_streams) == (variant == "D")
    intervals = lambda rs: [(r["start_us"], r["end_us"]) for r in rs]
    total_overlap = intersection_duration(intervals(uploads), intervals(kernels))
    per_batch = []
    for batch in range(3):
        ks = [r for r in kernels if r["batch"] == batch]
        hs = [r for r in uploads if r["batch"] == batch]
        assert ks
        # Own upload must finish before the first forward kernel.
        assert max(r["end_us"] for r in hs) <= min(r["start_us"] for r in ks)
        previous = [r for r in kernels if r["batch"] == batch - 1]
        overlap = intersection_duration(intervals(hs), intervals(previous))
        per_batch.append({"batch": batch, "input_bytes": sum(r["bytes"] for r in hs),
                          "h2d_duration_us": sum(r["duration_us"] for r in hs),
                          "previous_forward_kernel_overlap_us": overlap,
                          "kernel_count": len(ks), "first_kernel_us": min(r["start_us"] for r in ks),
                          "last_kernel_end_us": max(r["end_us"] for r in ks)})
    if variant == "D":
        # Third upload overwrites slot 0 only after the first forward finished.
        assert min(r["start_us"] for r in uploads if r["batch"] == 2) >= per_batch[0]["last_kernel_end_us"]
    syncs = [e for e in events if e.get("cat") == "cuda_runtime" and e["name"] == "cudaStreamSynchronize"]
    forward_sync = [e for e in syncs if "/forward/" in (stage_for_host(e) or "")]
    waits = [e for e in events if e.get("cat") == "cuda_runtime" and e["name"] == "cudaStreamWaitEvent"]
    internal_ops = Counter(e["name"] for e in events if e.get("cat") == "cpu_op"
                           and e["name"] in ("aten::is_nonzero", "aten::item", "aten::_local_scalar_dense"))
    result = {"trace_sha256": p.sha256(path), "batch_size": 8, "sample_count": 19,
              "kernel_count": len(kernels), "h2d_count": len(uploads), "logits_d2h_count": len(outputs),
              "compute_streams": sorted(compute_streams), "copy_streams": sorted(copy_streams),
              "h2d_total_duration_us": sum(r["duration_us"] for r in uploads),
              "h2d_kernel_overlap_us": total_overlap,
              "overlap_observed": total_overlap > 0,
              "per_batch": per_batch, "cpu_stream_sync_count": len(syncs),
              "forward_internal_stream_sync_count": len(forward_sync),
              "stream_wait_event_count": len(waits), "internal_cpu_scalar_ops": dict(internal_ops),
              "forward_internal_d2h_count": len(scalar),
              "forward_internal_d2h_bytes": [r["bytes"] for r in scalar],
              "gpu_activity_envelope_us": max(r["end_us"] for r in records),
              "gpu_kernel_union_us": sum(b - a for a, b in merged(intervals(kernels))),
              "limitations": "One profiled B=8,N=19 sequence; instrumentation changes timing. Event interval overlap, not hardware saturation evidence."}
    return result, records


def audit(trace_session, benchmark_session):
    trace_meta = p.read_json(trace_session / "metadata.json")
    bench = p.read_json(benchmark_session / "metadata.json")
    assert trace_meta["status"] == bench["status"] == "completed"
    assert trace_meta["identity"] == bench["identity"] and not trace_meta["timing_used_in_benchmark"]
    result = {"audited_at_utc": p.now(), "trace_session": str(trace_session.relative_to(p.ROOT)),
              "benchmark_session": bench["session_id"], "protocol_sha256": bench["protocol_sha256"],
              "trace_metadata_sha256": p.sha256(trace_session / "metadata.json"), "variants": {}}
    records = []
    for variant in "ACD":
        finding, events = inspect(trace_session / "traces" / f"{variant}_b8_n19.json", variant)
        result["variants"][variant] = finding
        records += events
    p.write_json(trace_session / "trace_audit.json", result)
    with (trace_session / "gpu_intervals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-session", type=Path, required=True)
    parser.add_argument("--benchmark-session", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.trace_session.resolve(), args.benchmark_session.resolve())
    print(json.dumps({v: {k: r[k] for k in ("h2d_kernel_overlap_us", "overlap_observed",
                                           "cpu_stream_sync_count", "forward_internal_stream_sync_count")}
                      for v, r in result["variants"].items()}, indent=2))
