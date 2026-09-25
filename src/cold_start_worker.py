"""One cold CPU or GPU job. Spawn this file in a new Python process per run.

The clock starts after imports, configuration, and pretokenized input-file load,
immediately before cached model loading. No CUDA context exists before the GPU
job clock starts. No warm-up or torch.cuda.empty_cache() is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


def ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 872:
        raise ValueError("N must be in 1..872")
    if hashlib.sha256(args.inputs.read_bytes()).hexdigest() != args.inputs_sha256:
        raise RuntimeError("Pretokenized input artifact changed")
    saved = torch.load(args.inputs, map_location="cpu", weights_only=True)
    tokens = {key: saved[key][:args.count] for key in ("input_ids", "attention_mask")}
    if any(tuple(value.shape) != (args.count, 64) or value.dtype != torch.int64
           or value.is_pinned() for value in tokens.values()):
        raise RuntimeError("Invalid pretokenized input policy")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = torch.empty((args.count, 2), dtype=torch.float32, device="cpu")
    if output.is_pinned():
        raise RuntimeError("Cold-job output must use pageable CPU memory")
    gpu_inputs = None
    with torch.inference_mode():
        # First torch.cuda.* call is below, after the total-job clock starts.
        total_start = time.perf_counter_ns()
        initially_uninitialized = not torch.cuda.is_initialized()
        if not initially_uninitialized:
            raise RuntimeError("Fresh worker already has a CUDA context")
        load_start = time.perf_counter_ns()
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=args.model_revision, local_files_only=True,
        ).float().eval().cpu()
        load_end = time.perf_counter_ns()
        if (getattr(model.config, "_commit_hash", None) != args.model_revision
                or not all(p.dtype == torch.float32 for p in model.parameters())):
            raise RuntimeError("Pinned model identity or FP32 policy mismatch")
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA context initialized during model loading")
        model_load_ms = ms(load_start, load_end)
        device_prepare_ms = 0.0
        if args.device == "gpu":
            prep_start = time.perf_counter_ns()
            torch.cuda.init()
            model = model.to("cuda")
            gpu_inputs = {key: torch.empty((1, 64), dtype=torch.int64, device="cuda")
                          for key in ("input_ids", "attention_mask")}
            torch.cuda.synchronize()
            device_prepare_ms = ms(prep_start, time.perf_counter_ns())

        def infer_one(index: int) -> None:
            destination = output[index:index + 1]
            if args.device == "cpu":
                inputs = {key: value[index:index + 1] for key, value in tokens.items()}
                destination.copy_(model(**inputs).logits)
            else:
                for key in ("input_ids", "attention_mask"):
                    gpu_inputs[key].copy_(tokens[key][index:index + 1], non_blocking=False)
                destination.copy_(model(**gpu_inputs).logits, non_blocking=False)

        first_start = time.perf_counter_ns()
        infer_one(0)
        if args.device == "gpu":
            torch.cuda.synchronize()
        first_forward_ms = ms(first_start, time.perf_counter_ns())
        remaining_start = time.perf_counter_ns()
        for index in range(1, args.count):
            infer_one(index)
        if args.device == "gpu" and args.count > 1:
            torch.cuda.synchronize()
        end = time.perf_counter_ns()
        remaining_inference_ms = ms(remaining_start, end) if args.count > 1 else 0.0
        total_job_ms = ms(total_start, end)
    if (not torch.isfinite(output).all().item()
            or not all(math.isfinite(value) and value >= 0 for value in
                       (model_load_ms, device_prepare_ms, first_forward_ms,
                        remaining_inference_ms, total_job_ms))
            or min(model_load_ms, first_forward_ms, total_job_ms) <= 0):
        raise RuntimeError("Invalid cold-job output or timing")
    labels = output.argmax(-1).tolist()
    if any(label not in (0, 1) for label in labels):
        raise RuntimeError("Invalid predicted class")
    report = {
        "worker_pid": os.getpid(),
        "worker_parent_pid": os.getppid(),
        "python_executable": sys.executable,
        "device": args.device,
        "sample_count": args.count,
        "batch_size": 1,
        "forward_invocations": args.count,
        "model_revision": args.model_revision,
        "inputs_sha256": args.inputs_sha256,
        "gpu_uninitialized_before_job": initially_uninitialized,
        "cpu_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "model_load_ms": model_load_ms,
        "device_prepare_ms": device_prepare_ms,
        "first_forward_ms": first_forward_ms,
        "remaining_inference_ms": remaining_inference_ms,
        "total_job_ms": total_job_ms,
        "throughput_samples_per_sec": args.count * 1000 / total_job_ms,
        "prediction_sha256": hashlib.sha256(bytes(labels)).hexdigest(),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
