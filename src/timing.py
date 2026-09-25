"""Timing boundaries for the M3 prototype.

Inputs are tokenized before entering any measured function. The model is in
eval mode and callers execute these functions inside torch.inference_mode().
"""

from __future__ import annotations

import math
import time

import torch


def cpu_forward_ms(model, cpu_inputs: dict[str, torch.Tensor]) -> float:
    start_ns = time.perf_counter_ns()
    _ = model(**cpu_inputs).logits
    return _checked((time.perf_counter_ns() - start_ns) / 1e6)


def gpu_forward_event_ms(model, gpu_inputs: dict[str, torch.Tensor],
                         start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    """CUDA Events enclose only forward; inputs already live on GPU.

    end.synchronize() blocks until this recorded interval has completed before
    elapsed_time reads the event timestamps. No host-timer approximation is used.
    """
    start.record()
    _ = model(**gpu_inputs).logits
    end.record()
    end.synchronize()
    return _checked(start.elapsed_time(end))


def gpu_end_to_end_ms(model, cpu_inputs: dict[str, torch.Tensor],
                      gpu_buffers: dict[str, torch.Tensor],
                      cpu_logits_buffer: torch.Tensor) -> float:
    """Synchronized host interval: CPU tensors -> H2D -> forward -> D2H.

    Device input and host output allocations are done by the caller once.
    Non-blocking copies are disabled. The final synchronize guarantees that
    all GPU work and the required output transfer have finished.
    """
    torch.cuda.synchronize()
    start_ns = time.perf_counter_ns()
    for key in ("input_ids", "attention_mask"):
        gpu_buffers[key].copy_(cpu_inputs[key], non_blocking=False)
    gpu_logits = model(**gpu_buffers).logits
    cpu_logits_buffer.copy_(gpu_logits, non_blocking=False)
    torch.cuda.synchronize()
    return _checked((time.perf_counter_ns() - start_ns) / 1e6)


def _checked(latency_ms: float) -> float:
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError(f"Invalid latency: {latency_ms}")
    return latency_ms
