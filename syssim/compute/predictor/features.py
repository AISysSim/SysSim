"""Feature schema — single source of truth for inference (estimator) and training
(profiler). All features are blackbox observables: shapes, dtypes, op identity,
op math, device constants. SCHEMA_VERSION is pinned in the bundle manifest and
asserted at load."""
from __future__ import annotations

import math
from typing import Any

import torch

from .router import Family
from .roofline import Roofline

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


def _op_subtype(func_packet) -> str:
    return str(func_packet).rsplit(".", 1)[-1] if func_packet is not None else "unknown"


def _mod_aligns(prefix: str, value: int, row: dict) -> None:
    for mod in (8, 16, 64, 128):
        row[f"{prefix}_mod_{mod}"] = int(value) % mod


def _attention_feats(func_packet, args, kwargs, out) -> dict:
    """B, H_q, H_kv, gqa, S_q, S_kv, D_head, is_causal, variant (q/k are (B,H,S,D))."""
    q = args[0] if args and isinstance(args[0], torch.Tensor) else None
    k = args[1] if len(args) > 1 and isinstance(args[1], torch.Tensor) else q
    r: dict[str, Any] = {"variant": _op_subtype(func_packet)}
    if q is not None and q.dim() == 4:
        B, Hq, Sq, D = (int(s) for s in q.shape)
        Hkv = int(k.shape[1]) if (k is not None and k.dim() == 4) else Hq
        Skv = int(k.shape[2]) if (k is not None and k.dim() == 4) else Sq
        r.update({"B": B, "H_q": Hq, "H_kv": Hkv,
                  "gqa_ratio": (float(Hq) / float(Hkv)) if Hkv else 1.0,
                  "S_q": Sq, "S_kv": Skv, "D_head": D,
                  "log_S_q": _log(Sq), "log_S_kv": _log(Skv), "log_D_head": _log(D)})
        for name, d in (("S_q", Sq), ("S_kv", Skv), ("D_head", D)):
            _mod_aligns(name, d, r)
    causal = kwargs.get("is_causal", args[4] if len(args) > 4 else False)
    r["is_causal"] = int(bool(causal))
    return r


def _norm_feats(func_packet, args, kwargs, out) -> dict:
    """outer_dims_product, norm_dim, op_subtype, has_weight/bias, groups."""
    x = args[0] if args and isinstance(args[0], torch.Tensor) else None
    norm_shape = args[1] if len(args) > 1 else None
    norm_dim = 1
    if isinstance(norm_shape, (list, tuple)) and norm_shape:
        for s in norm_shape:
            norm_dim *= int(s)
    elif isinstance(x, torch.Tensor) and x.dim() > 0:
        norm_dim = int(x.shape[-1])
    total = x.numel() if isinstance(x, torch.Tensor) else 0
    outer = (total // norm_dim) if norm_dim else 0
    weight = args[2] if len(args) > 2 else None
    bias = args[3] if len(args) > 3 else None
    return {
        "outer_dims_product": int(outer), "norm_dim": int(norm_dim),
        "log_outer_dims_product": _log(outer), "log_norm_dim": _log(norm_dim),
        "op_subtype": _op_subtype(func_packet),
        "has_weight": int(weight is not None), "has_bias": int(bias is not None),
        "groups": int(kwargs.get("groups", 1) or 1),
    }


def _elementwise_feats(func_packet, args, out) -> dict:
    """total_elements, num_operands, op_subtype, is_inplace."""
    ref = out if isinstance(out, torch.Tensor) else (
        args[0] if args and isinstance(args[0], torch.Tensor) else None)
    total = ref.numel() if isinstance(ref, torch.Tensor) else 0
    num_operands = sum(1 for a in args if isinstance(a, torch.Tensor))
    name = _op_subtype(func_packet)
    return {
        "total_elements": int(total), "log_total_elements": _log(total),
        "num_operands": int(num_operands), "op_subtype": name,
        "is_inplace": int(name.endswith("_")),
    }


def _reduction_feats(func_packet, args, out) -> dict:
    """input_volume, reduced_axis_size, num_non_reduced_elements, op_subtype."""
    x = args[0] if args and isinstance(args[0], torch.Tensor) else None
    dim = args[1] if len(args) > 1 and isinstance(args[1], int) else -1
    vol = x.numel() if isinstance(x, torch.Tensor) else 0
    red = int(x.shape[dim]) if (isinstance(x, torch.Tensor) and x.dim() > 0) else 1
    non_red = (vol // red) if red else 0
    return {
        "input_volume": int(vol), "reduced_axis_size": int(red),
        "num_non_reduced_elements": int(non_red), "log_input_volume": _log(vol),
        "op_subtype": _op_subtype(func_packet), "is_causal_softmax": 0,
    }


def featurize(func_packet, args, kwargs, out, family: Family,
              bound: Roofline) -> dict[str, Any]:
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
        "log_anchor_ns": _log(bound.roofline_ns),
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
    elif family is Family.ATTENTION:
        row.update(_attention_feats(func_packet, args, kwargs, out))
    elif family is Family.NORMALIZATION:
        row.update(_norm_feats(func_packet, args, kwargs, out))
    elif family is Family.ELEMENTWISE:
        row.update(_elementwise_feats(func_packet, args, out))
    elif family is Family.REDUCTION:
        row.update(_reduction_feats(func_packet, args, out))
    return row


# Column order per family (categoricals listed in CATEGORICAL). Pinned in the manifest.
CATEGORICAL = ["dtype", "op_subtype", "variant"]


def feature_columns(family: Family) -> list[str]:
    """Deterministic column order for the LightGBM feature matrix."""
    universal = ["log_tensor_ns", "log_fma_ns", "log_sfu_ns", "log_mem_ns",
                 "log_launch_floor_ns", "log_anchor_ns", "arithmetic_intensity",
                 "log_bytes_total", "working_set_bytes", "dtype", "bits_per_element"]
    if family is Family.GEMM:
        gemm = ["M", "N", "K", "batched_dim", "log_M", "log_N", "log_K"]
        gemm += [f"{n}_mod_{m}" for n in ("M", "N", "K") for m in (8, 16, 64, 128)]
        return universal + gemm
    if family is Family.ATTENTION:
        attn = ["B", "H_q", "H_kv", "gqa_ratio", "S_q", "S_kv", "D_head",
                "log_S_q", "log_S_kv", "log_D_head", "is_causal", "variant"]
        attn += [f"{n}_mod_{m}" for n in ("S_q", "S_kv", "D_head") for m in (8, 16, 64, 128)]
        return universal + attn
    if family is Family.NORMALIZATION:
        return universal + ["outer_dims_product", "norm_dim", "log_outer_dims_product",
                            "log_norm_dim", "op_subtype", "has_weight", "has_bias", "groups"]
    if family is Family.ELEMENTWISE:
        return universal + ["total_elements", "log_total_elements", "num_operands",
                            "op_subtype", "is_inplace"]
    if family is Family.REDUCTION:
        return universal + ["input_volume", "reduced_axis_size", "num_non_reduced_elements",
                            "log_input_volume", "op_subtype", "is_causal_softmax"]
    return universal
