"""Real-kernel measurement (GPU). The only component that runs real kernels."""
from __future__ import annotations

import statistics
from typing import Any

import torch

_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}


def median_of_reps(samples: list[float], warmup: int = 2) -> float:
    """Median over samples after discarding the first `warmup` (pure; testable)."""
    kept = samples[warmup:] if len(samples) > warmup else samples
    return float(statistics.median(kept))


def _gemm_kernel(a, b, dtype: str):
    """Run the right GEMM kernel for the dtype. bf16/fp16 -> `@`; fp8 -> _scaled_mm."""
    if dtype.startswith("fp8"):
        scale = torch.ones(1, device=a.device)
        return torch._scaled_mm(a, b.t().contiguous().t(), scale_a=scale, scale_b=scale,
                                out_dtype=torch.bfloat16)
    return a @ b


def measure_gemm(M: int, K: int, N: int, dtype: str, reps: int = 5,
                 warmup: int = 2, hw_info: Any = None) -> dict[str, Any]:
    """Time one MxKxN GEMM on the current CUDA device (median over reps) AND record
    the analytical anchor components (tensor_ns, mem_ns) via the SAME analytical_bound
    code inference uses — guaranteeing the calibrate-time anchor == the inference-time
    anchor (train/inference parity, spec section 2)."""
    dt = _DTYPE[dtype]
    a = torch.randn(M, K, device="cuda").to(dt)
    b = torch.randn(K, N, device="cuda").to(dt)
    samples = []
    for _ in range(reps + warmup):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = _gemm_kernel(a, b, dtype)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e6)   # ms -> ns

    from ..compute.predictor.analytical import analytical_bound
    from ..operator_graph import OperatorType
    if hw_info is None:
        from ..config import get_hardware_info
        hw_info, _ = get_hardware_info()
    aten = torch.ops.aten
    out = torch.empty(M, N, dtype=dt, device="cuda")
    bound = analytical_bound(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)
    return {"M": M, "K": K, "N": N, "dtype": dtype,
            "latency_ns": median_of_reps(samples, warmup),
            "tensor_ns": bound.tensor_ns, "mem_ns": bound.mem_ns}
