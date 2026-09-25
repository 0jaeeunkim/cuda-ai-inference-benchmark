"""M0-M3 CLI. Run with the project's .venv Python from the repository root."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402

from data import (BATCH_SIZES_PRIMARY, BATCH_SIZES_PROTOTYPE, DATASET_CONFIG,
                  DATASET_ID, DATASET_REVISION, DATASET_SPLIT, MODEL_ID, load_pool,
                  load_tokenizer, make_batch, token_lengths,
                  PRIMARY_SEQUENCE_LENGTH)  # noqa: E402
from timing import (cpu_forward_ms, gpu_end_to_end_ms,
                    gpu_forward_event_ms)  # noqa: E402

RESULTS = ROOT / "results"
CPU_THREADS = 8
WARMUP = 5
RUNS = 30


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def configure() -> None:
    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment() -> dict:
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the project virtual environment")
    gpu = torch.cuda.get_device_properties(0)
    probe = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    result = probe @ probe
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise RuntimeError("CUDA FP32 matrix multiply yielded nonfinite values")
    packages = {}
    for name in ("torch", "transformers", "datasets", "tokenizers", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "os": platform.platform(),
        "os_release": Path("/etc/os-release").read_text().splitlines()[0],
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), None),
        "logical_cpus": os.cpu_count(),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "gpu_model": gpu.name,
        "gpu_memory_bytes": gpu.total_memory,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda_arch_list": torch.cuda.get_arch_list(),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"]),
        "nvidia_smi_cuda_version": command_output(["nvidia-smi"]),
        "cuda_toolkit_nvcc": command_output(["/usr/local/cuda/bin/nvcc", "--version"]),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_fp32_matmul_smoke": "passed",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "tf32_enabled": False,
        "dtype": "float32",
        "packages": packages,
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "pool_selection": "all validation rows in published order; each batch uses prefix rows 0..batch_size-1",
        "batch_sizes_primary_planned": list(BATCH_SIZES_PRIMARY),
        "batch_sizes_prototype": list(BATCH_SIZES_PROTOTYPE),
        "warmup_iterations": WARMUP,
        "measured_iterations": RUNS,
    }


def m0() -> None:
    meta = environment()
    pool = load_pool()
    tokenizer = load_tokenizer()
    stats = token_lengths(pool, tokenizer, RESULTS / "token_lengths.csv")
    meta["pool_count"] = len(pool)
    meta["pool_fingerprint"] = pool._fingerprint
    meta["tokenizer_class"] = type(tokenizer).__name__
    meta["tokenizer_revision"] = tokenizer.init_kwargs.get("_commit_hash")
    stats["primary_length_selected"] = True
    meta["token_lengths"] = stats
    meta["primary_sequence_length"] = PRIMARY_SEQUENCE_LENGTH
    meta["primary_length_reason"] = "64 preserves all 872 SST-2 validation samples; 32 truncates 208, and 128 adds unnecessary padding"
    write_json(RESULTS / "environment.json", meta)
    print(json.dumps({"m0": "passed", "token_lengths": stats,
                      "pool_fingerprint": pool._fingerprint}, indent=2))


def load_workload(sequence_length: int):
    configure()
    meta = read_json(RESULTS / "environment.json")
    if not meta.get("token_lengths"):
        raise RuntimeError("M0 has not passed; run --stage m0 first")
    if meta.get("primary_sequence_length") != PRIMARY_SEQUENCE_LENGTH or sequence_length != PRIMARY_SEQUENCE_LENGTH:
        raise RuntimeError(f"Primary fixed sequence length must be {PRIMARY_SEQUENCE_LENGTH}")
    pool = load_pool()
    if pool._fingerprint != meta["pool_fingerprint"]:
        raise RuntimeError("Dataset fingerprint changed after M0")
    tokenizer = load_tokenizer()
    inputs = {size: make_batch(pool, tokenizer, size, sequence_length)
              for size in BATCH_SIZES_PROTOTYPE}
    cpu_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    cpu_model = cpu_model.float().eval().cpu()
    if any(parameter.dtype != torch.float32 for parameter in cpu_model.parameters()):
        raise RuntimeError("Model contains non-FP32 parameters")
    meta["prototype_sequence_length"] = sequence_length
    meta["prototype_length_status"] = "primary fixed length selected by user"
    meta["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    meta["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    meta["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    meta["e2e_host_memory"] = "pageable"
    meta["e2e_allocation_policy"] = "preallocated host output and GPU input buffers outside timed interval"
    meta["model_revision"] = cpu_model.config._commit_hash
    write_json(RESULTS / "environment.json", meta)
    return inputs, cpu_model


def cpu_check(inputs, cpu_model) -> dict:
    checks = {}
    with torch.inference_mode():
        for size, batch in inputs.items():
            logits = cpu_model(**batch).logits
            valid = (list(logits.shape) == [size, 2]
                     and bool(torch.isfinite(logits).all().item())
                     and list(batch["input_ids"].shape) == [size, PRIMARY_SEQUENCE_LENGTH]
                     and list(batch["attention_mask"].shape) == [size, PRIMARY_SEQUENCE_LENGTH]
                     and batch["input_ids"].dtype == torch.long
                     and batch["attention_mask"].dtype == torch.long
                     and bool(((batch["attention_mask"] == 0) |
                               (batch["attention_mask"] == 1)).all().item()))
            if not valid:
                raise RuntimeError(f"CPU smoke failed for batch {size}")
            checks[str(size)] = {
                "passed": True,
                "logits_shape": list(logits.shape),
                "predicted_labels": logits.argmax(dim=-1).tolist(),
                "input_shape": list(batch["input_ids"].shape),
                "input_ids_sum": int(batch["input_ids"].sum()),
                "attention_mask_sum": int(batch["attention_mask"].sum()),
            }
    return checks


def gpu_check(inputs, cpu_model, gpu_model) -> dict:
    checks = {}
    with torch.inference_mode():
        for size, cpu_batch in inputs.items():
            gpu_batch = {key: value.to("cuda") for key, value in cpu_batch.items()}
            input_match = all(torch.equal(cpu_batch[key], gpu_batch[key].cpu())
                              for key in cpu_batch)
            cpu_logits = cpu_model(**cpu_batch).logits
            gpu_logits = gpu_model(**gpu_batch).logits.cpu()
            prediction_match = torch.equal(cpu_logits.argmax(-1), gpu_logits.argmax(-1))
            shape_match = list(gpu_logits.shape) == [size, 2]
            finite = bool(torch.isfinite(gpu_logits).all().item())
            if not (input_match and prediction_match and shape_match and finite):
                raise RuntimeError(f"GPU correctness failed for batch {size}")
            checks[str(size)] = {
                "passed": True,
                "selected_row_indices": list(range(size)),
                "tokenized_input_match": input_match,
                "gpu_logits_shape": list(gpu_logits.shape),
                "gpu_logits_finite": finite,
                "prediction_match": prediction_match,
                "cpu_predicted_labels": cpu_logits.argmax(-1).tolist(),
                "gpu_predicted_labels": gpu_logits.argmax(-1).tolist(),
                "max_abs_logits_difference": (cpu_logits - gpu_logits).abs().max().item(),
            }
    return checks


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def m1_m2_m3(stage: str, sequence_length: int) -> None:
    inputs, cpu_model = load_workload(sequence_length)
    validation = read_json(RESULTS / "validation.json")
    validation["sequence_length"] = sequence_length
    validation["length_status"] = "primary fixed length selected by user"
    validation["cpu"] = cpu_check(inputs, cpu_model)
    write_json(RESULTS / "validation.json", validation)
    print("M1 CPU smoke and both batches passed", flush=True)
    if stage == "m1":
        return

    gpu_model = copy.deepcopy(cpu_model).to("cuda").eval()
    validation["gpu"] = gpu_check(inputs, cpu_model, gpu_model)
    write_json(RESULTS / "validation.json", validation)
    print("M2 GPU smoke and CPU/GPU correctness passed", flush=True)
    if stage == "m2":
        return

    session_id = uuid.uuid4().hex
    raw_path = RESULTS / "raw_runs.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("session_id", "timestamp_utc", "batch_size", "sequence_length",
               "run", "measurement_type", "latency_ms", "latency_per_sample_ms",
               "throughput_samples_per_sec", "prediction_match")
    summary = {}
    with raw_path.open("a", newline="") as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=columns)
        if handle.tell() == 0:
            writer.writeheader()
        for size, cpu_batch in inputs.items():
            gpu_batch = {key: value.to("cuda") for key, value in cpu_batch.items()}
            gpu_buffers = {key: torch.empty_like(value, device="cuda")
                           for key, value in cpu_batch.items()}
            cpu_logits_buffer = torch.empty((size, 2), dtype=torch.float32)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Small timing smoke before the required warm-up and measured runs.
            for measure in (
                lambda: cpu_forward_ms(cpu_model, cpu_batch),
                lambda: gpu_forward_event_ms(gpu_model, gpu_batch, start_event, end_event),
                lambda: gpu_end_to_end_ms(gpu_model, cpu_batch, gpu_buffers, cpu_logits_buffer),
            ):
                measure()
            reference_logits = cpu_model(**cpu_batch).logits
            if not torch.equal(reference_logits.argmax(-1), cpu_logits_buffer.argmax(-1)):
                raise RuntimeError(f"End-to-end output mismatch for batch {size}")
            for _ in range(WARMUP):
                cpu_forward_ms(cpu_model, cpu_batch)
                gpu_forward_event_ms(gpu_model, gpu_batch, start_event, end_event)
                gpu_end_to_end_ms(gpu_model, cpu_batch, gpu_buffers, cpu_logits_buffer)

            samples = {name: [] for name in ("cpu", "gpu_inference_only", "gpu_end_to_end")}
            for run in range(1, RUNS + 1):
                measured = {
                    "cpu": cpu_forward_ms(cpu_model, cpu_batch),
                    "gpu_inference_only": gpu_forward_event_ms(
                        gpu_model, gpu_batch, start_event, end_event),
                    "gpu_end_to_end": gpu_end_to_end_ms(
                        gpu_model, cpu_batch, gpu_buffers, cpu_logits_buffer),
                }
                for name, latency in measured.items():
                    samples[name].append(latency)
                    writer.writerow({
                        "session_id": session_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "batch_size": size,
                        "sequence_length": sequence_length,
                        "run": run,
                        "measurement_type": name,
                        "latency_ms": f"{latency:.9f}",
                        "latency_per_sample_ms": f"{latency / size:.9f}",
                        "throughput_samples_per_sec": f"{size * 1000 / latency:.9f}",
                        "prediction_match": True,
                    })
                handle.flush()
            summary[str(size)] = {
                name: {"count": len(values), "median_ms": statistics.median(values),
                       "p95_ms": percentile_nearest_rank(values, 0.95)}
                for name, values in samples.items()
            }
    validation["m3"] = {"passed": True, "session_id": session_id,
                        "warmup_iterations_per_path": WARMUP,
                        "measured_iterations_per_condition": RUNS,
                        "raw_rows_in_session": len(BATCH_SIZES_PROTOTYPE) * 3 * RUNS,
                        "summary_for_report_only": summary,
                        "timing_boundary": {
                            "cpu": "perf_counter_ns around CPU forward only",
                            "gpu_inference_only": "CUDA Events around forward with resident GPU inputs; end event synchronized before elapsed_time",
                            "gpu_end_to_end": "synchronized perf_counter_ns: pretokenized CPU input copy into preallocated GPU buffers, forward, logits D2H copy into preallocated CPU buffer, final synchronize",
                        }}
    write_json(RESULTS / "validation.json", validation)
    print(json.dumps({"m3": validation["m3"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("m0", "m1", "m2", "m3"), required=True)
    parser.add_argument("--sequence-length", type=int, choices=(PRIMARY_SEQUENCE_LENGTH,))
    args = parser.parse_args()
    if args.stage == "m0":
        m0()
    else:
        if args.sequence_length is None:
            parser.error("--sequence-length is required for M1-M3")
        m1_m2_m3(args.stage, args.sequence_length)


if __name__ == "__main__":
    main()
