"""M3.5 paired GPU timing diagnostic; does not change the M3 benchmark.

A: resident GPU input -> model forward timed by CUDA Events.
B: pageable CPU input -> blocking H2D -> model forward -> blocking logits D2H,
   timed by synchronized host clock; extra CUDA Events time B's forward only.
Every batch alternates A->B and B->A pairs. No CPU inference or nvidia-smi
call occurs inside the measured pair loop.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402

from benchmark import configure, read_json, write_json  # noqa: E402
from data import (BATCH_SIZES_PROTOTYPE, DATASET_REVISION, MODEL_ID,
                  PRIMARY_SEQUENCE_LENGTH, load_pool, load_tokenizer,
                  make_batch)  # noqa: E402
from timing import gpu_forward_event_ms  # noqa: E402

RESULTS = ROOT / "results"
MEASURED_PAIRS_PER_ORDER = 30
WARMUP_PAIRS_PER_ORDER = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def check_workload(cpu_model, gpu_model, cpu_inputs, gpu_inputs, gpu_buffers,
                   cpu_output, model_revision: str | None) -> dict:
    """Check equal weights and GPU inputs before any measured pair."""
    assert torch.is_inference_mode_enabled()
    assert not cpu_model.training and not gpu_model.training
    assert model_revision == getattr(gpu_model.config, "_commit_hash", None)
    assert all(p.dtype == torch.float32 for p in gpu_model.parameters())
    cpu_state = cpu_model.state_dict()
    gpu_state = gpu_model.state_dict()
    assert cpu_state.keys() == gpu_state.keys()
    weights_equal = all(torch.equal(cpu_state[key], gpu_state[key].cpu())
                        for key in cpu_state)
    assert weights_equal
    size = cpu_inputs["input_ids"].shape[0]
    assert not cpu_output.is_pinned() and cpu_output.shape == (size, 2)
    tensor_checks = {}
    for key in ("input_ids", "attention_mask"):
        source = cpu_inputs[key]
        resident = gpu_inputs[key]
        buffer = gpu_buffers[key]
        assert source.shape == resident.shape == buffer.shape == (size, PRIMARY_SEQUENCE_LENGTH)
        assert source.dtype == resident.dtype == buffer.dtype == torch.int64
        assert source.device.type == "cpu" and not source.is_pinned()
        assert resident.device.type == buffer.device.type == "cuda"
        buffer.copy_(source, non_blocking=False)
        assert torch.equal(resident, buffer)
        assert torch.equal(resident.cpu(), source)
        tensor_checks[key] = {
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "cpu_device": str(source.device),
            "gpu_device": str(resident.device),
            "cpu_pageable": not source.is_pinned(),
            "resident_equals_e2e_buffer": True,
            "source_sha256": tensor_sha256(source),
        }
    torch.cuda.synchronize()
    logits_a = gpu_model(**gpu_inputs).logits
    logits_b = gpu_model(**gpu_buffers).logits
    assert logits_a.shape == logits_b.shape == (size, 2)
    assert torch.isfinite(logits_a).all().item() and torch.isfinite(logits_b).all().item()
    predictions_equal = torch.equal(logits_a.argmax(-1), logits_b.argmax(-1))
    assert predictions_equal
    cpu_output.copy_(logits_b, non_blocking=False)
    assert torch.equal(cpu_output.argmax(-1), logits_a.argmax(-1).cpu())
    torch.cuda.synchronize()
    return {
        "passed": True,
        "same_gpu_model_object_for_A_and_B": True,
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "cpu_gpu_weights_exactly_equal": weights_equal,
        "parameter_count": sum(p.numel() for p in gpu_model.parameters()),
        "model_eval": not gpu_model.training,
        "inference_mode_enabled": torch.is_inference_mode_enabled(),
        "model_dtype": "torch.float32",
        "sequence_length": PRIMARY_SEQUENCE_LENGTH,
        "tensor_checks": tensor_checks,
        "A_B_predictions_equal": predictions_equal,
        "A_B_max_abs_logits_difference": (logits_a - logits_b).abs().max().item(),
        "B_D2H_predictions_equal_A": True,
        "host_memory_policy": "pageable CPU input and output",
    }


def instrumented_e2e_ms(model, cpu_inputs, gpu_buffers, cpu_output,
                         forward_start: torch.cuda.Event,
                         forward_end: torch.cuda.Event) -> tuple[float, float]:
    """Primary B boundary plus diagnostic forward Events inside that boundary."""
    torch.cuda.synchronize()
    host_start_ns = time.perf_counter_ns()
    for key in ("input_ids", "attention_mask"):
        gpu_buffers[key].copy_(cpu_inputs[key], non_blocking=False)
    forward_start.record()
    gpu_logits = model(**gpu_buffers).logits
    forward_end.record()
    cpu_output.copy_(gpu_logits, non_blocking=False)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter_ns() - host_start_ns) / 1e6
    forward_ms = forward_start.elapsed_time(forward_end)
    if not all(math.isfinite(value) and value > 0 for value in (total_ms, forward_ms)):
        raise RuntimeError("Invalid M3.5 E2E latency")
    return total_ms, forward_ms


def nearest_rank_p95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="one preserved pair per order per batch in separate smoke files")
    args = parser.parse_args()
    measured_pairs = 1 if args.smoke else MEASURED_PAIRS_PER_ORDER
    warmup_pairs = 2 if args.smoke else WARMUP_PAIRS_PER_ORDER
    label = "m35_smoke" if args.smoke else "m35_diagnostic"

    configure()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    environment_path = RESULTS / "environment.json"
    environment = read_json(environment_path)
    if environment.get("primary_sequence_length") != PRIMARY_SEQUENCE_LENGTH:
        raise RuntimeError("Primary sequence length metadata mismatch")
    if environment.get("dataset_revision") != DATASET_REVISION:
        raise RuntimeError("Dataset revision metadata mismatch")
    m3_raw_path = RESULTS / "raw_runs.csv"
    m3_sha_before = sha256(m3_raw_path)
    pool = load_pool()
    if pool._fingerprint != environment.get("pool_fingerprint"):
        raise RuntimeError("Dataset pool fingerprint changed")
    tokenizer = load_tokenizer()
    cpu_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).float().eval().cpu()
    model_revision = getattr(cpu_model.config, "_commit_hash", None)
    if model_revision != environment.get("model_revision"):
        raise RuntimeError("Model revision changed since M3")
    gpu_model = copy.deepcopy(cpu_model).to("cuda").eval()
    session_id = uuid.uuid4().hex
    environment["torch_get_num_threads"] = torch.get_num_threads()
    environment["torch_get_num_interop_threads"] = torch.get_num_interop_threads()
    write_json(environment_path, environment)
    metadata = {
        "session_id": session_id,
        "kind": label,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_sizes": list(BATCH_SIZES_PROTOTYPE),
        "sequence_length": PRIMARY_SEQUENCE_LENGTH,
        "pairs_per_order_per_batch": measured_pairs,
        "warmup_pairs_per_order_per_batch": warmup_pairs,
        "pair_schedule": "alternate A->B and B->A; batch 8 starts B->A",
        "A": "resident GPU input, CUDA Events around forward",
        "B": "pageable CPU input, blocking H2D, forward with internal CUDA Events, blocking logits D2H; synchronized host total",
        "torch_get_num_threads": torch.get_num_threads(),
        "torch_get_num_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "M3_raw_sha256_before": m3_sha_before,
        "workload_equivalence": {},
    }
    columns = ("session_id", "timestamp_utc", "batch_size", "sequence_length",
               "pair_id", "pair_index", "execution_order", "A_forward_event_ms",
               "B_e2e_total_host_ms", "B_forward_event_ms")
    raw_path = RESULTS / f"{label}_raw.csv"
    collected = defaultdict(lambda: defaultdict(list))
    with raw_path.open("a", newline="") as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=columns)
        if handle.tell() == 0:
            writer.writeheader()
        for size in BATCH_SIZES_PROTOTYPE:
            cpu_inputs = make_batch(pool, tokenizer, size, PRIMARY_SEQUENCE_LENGTH)
            gpu_inputs = {key: value.to("cuda") for key, value in cpu_inputs.items()}
            gpu_buffers = {key: torch.empty_like(value, device="cuda")
                           for key, value in cpu_inputs.items()}
            cpu_output = torch.empty((size, 2), dtype=torch.float32)
            a_start = torch.cuda.Event(enable_timing=True)
            a_end = torch.cuda.Event(enable_timing=True)
            b_start = torch.cuda.Event(enable_timing=True)
            b_end = torch.cuda.Event(enable_timing=True)
            metadata["workload_equivalence"][str(size)] = check_workload(
                cpu_model, gpu_model, cpu_inputs, gpu_inputs, gpu_buffers,
                cpu_output, model_revision)

            def A() -> float:
                return gpu_forward_event_ms(gpu_model, gpu_inputs, a_start, a_end)

            def B() -> tuple[float, float]:
                return instrumented_e2e_ms(gpu_model, cpu_inputs, gpu_buffers,
                                           cpu_output, b_start, b_end)

            orders = (("A->B", "B->A") if size == 1 else ("B->A", "A->B"))
            for _ in range(warmup_pairs):
                for order in orders:
                    if order == "A->B":
                        A(); B()
                    else:
                        B(); A()
            for pair_id in range(1, measured_pairs + 1):
                for position, order in enumerate(orders, start=1):
                    if order == "A->B":
                        a_ms = A()
                        b_total_ms, b_forward_ms = B()
                    else:
                        b_total_ms, b_forward_ms = B()
                        a_ms = A()
                    pair_index = (pair_id - 1) * 2 + position
                    writer.writerow({
                        "session_id": session_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "batch_size": size,
                        "sequence_length": PRIMARY_SEQUENCE_LENGTH,
                        "pair_id": pair_id,
                        "pair_index": pair_index,
                        "execution_order": order,
                        "A_forward_event_ms": f"{a_ms:.9f}",
                        "B_e2e_total_host_ms": f"{b_total_ms:.9f}",
                        "B_forward_event_ms": f"{b_forward_ms:.9f}",
                    })
                    handle.flush()
                    group = collected[(size, order)]
                    group["A_forward_event_ms"].append(a_ms)
                    group["B_e2e_total_host_ms"].append(b_total_ms)
                    group["B_forward_event_ms"].append(b_forward_ms)
                    group["B_total_minus_A_ms"].append(b_total_ms - a_ms)
                    group["B_forward_minus_A_ms"].append(b_forward_ms - a_ms)
            for key in ("input_ids", "attention_mask"):
                if not torch.equal(gpu_inputs[key], gpu_buffers[key]):
                    raise RuntimeError(f"Input changed during M3.5: batch {size}, {key}")
    m3_sha_after = sha256(m3_raw_path)
    if m3_sha_before != m3_sha_after:
        raise RuntimeError("M3 raw results changed")
    metadata["M3_raw_sha256_after"] = m3_sha_after
    summary = {}
    for (size, order), metrics in sorted(collected.items()):
        summary.setdefault(str(size), {})[order] = {
            name: {"count": len(values), "median_ms": statistics.median(values),
                   "p95_ms": nearest_rank_p95(values)}
            for name, values in metrics.items()
        }
    metadata["summary"] = summary
    metadata["raw_path"] = str(raw_path.relative_to(ROOT))
    metadata["raw_rows_in_session"] = len(BATCH_SIZES_PROTOTYPE) * 2 * measured_pairs
    summary_path = RESULTS / f"{label}_{session_id}_summary.json"
    write_json(summary_path, metadata)
    print(json.dumps({"session_id": session_id, "kind": label,
                      "raw_rows_in_session": metadata["raw_rows_in_session"],
                      "raw_path": metadata["raw_path"],
                      "summary_path": str(summary_path.relative_to(ROOT)),
                      "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
