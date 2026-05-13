"""Roofline helpers for Mixture-of-Experts operator modeling."""

from __future__ import annotations

import math

import torch

from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType


def dtype_nbytes(dtype: torch.dtype | str) -> int:
    """Return storage bytes per element for the dtype used by the MoE model."""
    if isinstance(dtype, str):
        normalized = dtype.lower().replace("torch.", "")
        if normalized in {"nvfp4", "fp4"}:
            return 1
        if normalized in {"float8_e4m3fn", "float8_e5m2", "fp8"}:
            return 1
        if normalized in {"float16", "bfloat16", "fp16", "bf16", "half"}:
            return 2
        if normalized in {"float32", "fp32"}:
            return 4
        raise ValueError(f"Unsupported dtype string for MoE cost model: {dtype}")

    return torch.empty((), dtype=dtype).element_size()


def _non_negative(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _time_from_flops_ms(num_flops: int, peak_tflops: float) -> float:
    if peak_tflops <= 0:
        raise ValueError(f"peak_tflops must be positive, got {peak_tflops}")
    if num_flops <= 0:
        return 0.0
    return num_flops / peak_tflops / 1e9


def estimate_gemm_ms(
    m: int,
    n: int,
    k: int,
    hw_info: HardwareInfo,
    dtype: torch.dtype | str,
) -> float:
    """Estimate an MxK by KxN GEMM runtime in milliseconds."""
    m = _non_negative(m, "m")
    n = _non_negative(n, "n")
    k = _non_negative(k, "k")
    if m == 0 or n == 0 or k == 0:
        return 0.0

    is_large_op = min(m, n, k) >= 512
    peak_tflops = hw_info.get_peak_tflops(OperatorType.GEMM, dtype, is_large_op)
    return _time_from_flops_ms(2 * m * n * k, peak_tflops)


def estimate_math_ms(num_ops: int, hw_info: HardwareInfo) -> float:
    """Estimate vector/scalar math runtime in milliseconds."""
    num_ops = _non_negative(num_ops, "num_ops")
    peak_tflops = hw_info.get_peak_tflops(OperatorType.MATH, torch.float32)
    return _time_from_flops_ms(num_ops, peak_tflops)


def estimate_memory_ms(num_bytes: int, hw_info: HardwareInfo) -> float:
    """Estimate memory movement runtime in milliseconds."""
    num_bytes = _non_negative(num_bytes, "num_bytes")
    if num_bytes == 0:
        return 0.0
    peak_gbps = hw_info.get_peak_memory_bandwidth_gbps()
    if peak_gbps <= 0:
        raise ValueError(f"peak memory bandwidth must be positive, got {peak_gbps}")
    return num_bytes / peak_gbps / 1e6


def estimate_router_ms(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    hw_info: HardwareInfo,
    dtype: torch.dtype | str,
) -> float:
    """Estimate router gate projection plus top-k selection cost."""
    num_tokens = _non_negative(num_tokens, "num_tokens")
    hidden_size = _non_negative(hidden_size, "hidden_size")
    num_experts = _non_negative(num_experts, "num_experts")
    top_k = _non_negative(top_k, "top_k")
    if num_tokens == 0 or hidden_size == 0 or num_experts == 0 or top_k == 0:
        return 0.0

    gate_ms = estimate_gemm_ms(num_tokens, num_experts, hidden_size, hw_info, dtype)
    topk_ops = num_tokens * num_experts + num_tokens * top_k * math.ceil(math.log2(num_experts + 1))
    return gate_ms + estimate_math_ms(topk_ops, hw_info)


def estimate_dispatch_ms(
    num_assignments: int,
    hidden_size: int,
    hw_info: HardwareInfo,
    dtype: torch.dtype | str,
) -> float:
    """Estimate token dispatch gather/scatter memory traffic."""
    num_assignments = _non_negative(num_assignments, "num_assignments")
    hidden_size = _non_negative(hidden_size, "hidden_size")
    return estimate_memory_ms(2 * num_assignments * hidden_size * dtype_nbytes(dtype), hw_info)


def estimate_expert_ffn_ms(
    num_active_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    hw_info: HardwareInfo,
    dtype: torch.dtype | str,
) -> float:
    """Estimate routed SwiGLU expert FFN runtime."""
    num_active_tokens = _non_negative(num_active_tokens, "num_active_tokens")
    hidden_size = _non_negative(hidden_size, "hidden_size")
    intermediate_size = _non_negative(intermediate_size, "intermediate_size")
    if num_active_tokens == 0 or hidden_size == 0 or intermediate_size == 0:
        return 0.0

    up_gate_ms = estimate_gemm_ms(
        num_active_tokens,
        2 * intermediate_size,
        hidden_size,
        hw_info,
        dtype,
    )
    down_ms = estimate_gemm_ms(
        num_active_tokens,
        hidden_size,
        intermediate_size,
        hw_info,
        dtype,
    )
    activation_ms = estimate_math_ms(2 * num_active_tokens * intermediate_size, hw_info)
    return up_gate_ms + activation_ms + down_ms


def estimate_combine_ms(
    num_assignments: int,
    hidden_size: int,
    hw_info: HardwareInfo,
    dtype: torch.dtype | str,
) -> float:
    """Estimate weighted expert-output combine memory and math cost."""
    num_assignments = _non_negative(num_assignments, "num_assignments")
    hidden_size = _non_negative(hidden_size, "hidden_size")
    memory_ms = estimate_memory_ms(2 * num_assignments * hidden_size * dtype_nbytes(dtype), hw_info)
    math_ms = estimate_math_ms(num_assignments * hidden_size, hw_info)
    return memory_ms + math_ms
