"""Feature schema — single source of truth for inference (estimator) and training
(profiler). All features are blackbox observables: shapes, dtypes, op identity,
op math, device constants. SCHEMA_VERSION is pinned in the bundle manifest and
asserted at load."""
from __future__ import annotations

import math
from typing import Any

import torch

from .router import Family
from .analytical import AnalyticalBound

SCHEMA_VERSION = "1.0.0-gemm"

_DTYPE_STR = {
    torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8_e4m3", torch.float8_e5m2: "fp8_e5m2",
}


def _log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0


def _dtype_str(t: torch.Tensor) -> str:
    return _DTYPE_STR.get(t.dtype, str(t.dtype))


def _gemm_dims(func_packet, args) -> tuple[int, int, int, int]:
    """Return (M, K, N, batch). Handles mm/addmm/bmm/matmul; batch=1 for plain GEMM."""
    import torch as _t
    aten = _t.ops.aten
    if func_packet == aten.addmm and len(args) >= 3:
        a, b = args[1], args[2]
    elif func_packet in (aten.mm, aten.matmul) and len(args) >= 2:
        a, b = args[0], args[1]
    elif func_packet == aten.bmm and len(args) >= 2:
        a, b = args[0], args[1]
        return a.shape[1], a.shape[2], b.shape[2], a.shape[0]
    else:
        a, b = args[0], args[1]
    return a.shape[-2], a.shape[-1], b.shape[-1], 1


def featurize(func_packet, args, kwargs, out, family: Family,
              bound: AnalyticalBound) -> dict[str, Any]:
    """Build the feature row for one operator. GEMM + universal columns (MVP)."""
    flat = [t for t in (out if isinstance(out, (list, tuple)) else [out])
            if isinstance(t, torch.Tensor)]
    out_t = flat[0] if flat else (args[0] if args and isinstance(args[0], torch.Tensor) else None)
    bytes_total = 0
    for t in list(args) + flat:
        if isinstance(t, torch.Tensor):
            bytes_total += t.numel() * t.element_size()
    ai = (bound.tensor_ns / bound.mem_ns) if bound.mem_ns > 0 else 0.0

    row: dict[str, Any] = {
        "log_tensor_ns": _log(bound.tensor_ns),
        "log_fma_ns": _log(bound.fma_ns),
        "log_sfu_ns": _log(bound.sfu_ns),
        "log_mem_ns": _log(bound.mem_ns),
        "log_launch_floor_ns": _log(bound.launch_ns),
        "log_anchor_ns": _log(bound.t_an_ns),
        "arithmetic_intensity": ai,
        "log_bytes_total": _log(bytes_total),
        "working_set_bytes": float(bytes_total),
        "dtype": _dtype_str(out_t) if out_t is not None else "fp32",
        "bits_per_element": (out_t.element_size() * 8) if out_t is not None else 32,
    }
    if family is Family.GEMM:
        m, k, n, batch = _gemm_dims(func_packet, args)
        row.update({"M": int(m), "K": int(k), "N": int(n), "batched_dim": int(batch),
                    "log_M": _log(m), "log_N": _log(n), "log_K": _log(k)})
        for name, d in (("M", m), ("N", n), ("K", k)):
            for mod in (8, 16, 64, 128):
                row[f"{name}_mod_{mod}"] = int(d) % mod
    return row


# Column order per family (categoricals listed in CATEGORICAL). Pinned in the manifest.
CATEGORICAL = ["dtype"]


def feature_columns(family: Family) -> list[str]:
    """Deterministic column order for the LightGBM feature matrix."""
    universal = ["log_tensor_ns", "log_fma_ns", "log_sfu_ns", "log_mem_ns",
                 "log_launch_floor_ns", "log_anchor_ns", "arithmetic_intensity",
                 "log_bytes_total", "working_set_bytes", "dtype", "bits_per_element"]
    if family is Family.GEMM:
        gemm = ["M", "N", "K", "batched_dim", "log_M", "log_N", "log_K"]
        gemm += [f"{n}_mod_{m}" for n in ("M", "N", "K") for m in (8, 16, 64, 128)]
        return universal + gemm
    return universal
