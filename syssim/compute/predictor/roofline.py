"""The roofline runtime bound for operators.

One roofline formulation: the binding demand among the Tensor (MMA), FMA (FP32
vector), SFU (transcendental), and memory pipelines, plus an optional kernel
launch floor. ``roofline(...)`` returns a ``Roofline`` whose ``roofline_ns`` is
that single bound; the core ``RooflineEstimator`` returns it directly and the
hybrid predictor uses it as the residual anchor and OOD rail.

Pure analytics — blackbox (shapes + op math + device constants); no kernel
internals. Lives under ``compute/predictor/`` and imports no lightgbm, so the
default roofline path stays free of that dependency.

Units: TFLOP/s and GB/s for peaks; ns internally; ms is applied by callers.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._ordered_set import OrderedSet

from ...operator_graph import OperatorType
from ...config import HardwareInfo
from ..flop_counter import flop_registry, instruction_mix


# ── Unit conversion constants ────────────────────────────────────────────────
TERA_TO_UNIT = 1e12          # 1 TFLOP = 10^12 FLOP
SECONDS_TO_NS = 1e9          # 1 second = 10^9 nanoseconds
TFLOPS_TO_FLOPS = TERA_TO_UNIT

# Operations with all dims ≥ this use the tensor-unit peak; smaller ones the
# conservative peak (launch-overhead dominated).
LARGE_GEMM_THRESHOLD = 512

_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)

aten = torch.ops.aten

_FLOAT_TYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                torch.float8_e4m3fn, torch.float8_e5m2)

# No fall-back kernel needed/exists for view ops.
_VIEW_OPS = OrderedSet(
    [
        aten.lift_fresh,
        aten.t,
        aten.transpose,
        aten.view,
        aten.detach,
        aten._unsafe_view,
        aten.split,
        aten.adjoint,
        aten.as_strided,
        aten.diagonal,
        aten.expand,
        aten.expand_as,
        aten.movedim,
        aten.permute,
        aten.select,
        aten.squeeze,
        aten.mT,
        aten.mH,
        aten.real,
        aten.imag,
        aten.view_as,
        aten.unflatten,
        aten.unfold,
        aten.unbind,
        aten.unsqueeze,
        aten.vsplit,
        aten.hsplit,
        aten.split_with_sizes,
        aten.swapaxes,
        aten.swapdims,
        aten.chunk,
    ]
)

# Tensor-create ops are zero-time for benchmarking purposes.
_CREATE_OPS = OrderedSet(
    [
        aten.randint,
        aten.randn,
        aten.rand,
        aten.randn_like,
        aten.rand_like,
        aten.randint_like,
        aten.arange,
        aten.ones_like,
        aten.zeros_like,
    ]
)

_IGNORE_OPS = _VIEW_OPS | _CREATE_OPS

_GEMM_OPS = OrderedSet(
    [
        aten.mm,
        aten.addmm,
        aten.bmm,
        aten.matmul,
        aten.linear,
    ]
)

_ATTN_OPS = frozenset({
    aten._scaled_dot_product_efficient_attention,
    aten._scaled_dot_product_flash_attention,
    aten._scaled_dot_product_flash_attention_for_cpu,
    aten._scaled_dot_product_cudnn_attention,
    aten._flash_attention_forward,
    aten._efficient_attention_forward,
})


def get_num_bytes(t: torch.Tensor) -> int:
    num_bytes = t.untyped_storage().nbytes()
    return math.ceil(num_bytes / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE


def _is_large_tensor_core_op(func_packet, args, op_type: OperatorType) -> bool:
    """Whether an op is large enough (all dims ≥ 512) for full tensor-unit peak.

    Small ops are launch-overhead dominated and use the conservative peak.
    """
    if op_type == OperatorType.GEMM:
        if func_packet == aten.mm and len(args) >= 2:
            a, b = args[0], args[1]
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dim() == 2 and b.dim() == 2:
                m, k = a.shape
                k2, n = b.shape
                return m >= LARGE_GEMM_THRESHOLD and n >= LARGE_GEMM_THRESHOLD and k >= LARGE_GEMM_THRESHOLD

        elif func_packet == aten.addmm and len(args) >= 3:
            a, b = args[1], args[2]
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dim() == 2 and b.dim() == 2:
                m, k = a.shape
                k2, n = b.shape
                return m >= LARGE_GEMM_THRESHOLD and n >= LARGE_GEMM_THRESHOLD and k >= LARGE_GEMM_THRESHOLD

        elif func_packet == aten.bmm and len(args) >= 2:
            a, b = args[0], args[1]
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dim() == 3 and b.dim() == 3:
                batch, m, k = a.shape
                _, k2, n = b.shape
                return m >= LARGE_GEMM_THRESHOLD and n >= LARGE_GEMM_THRESHOLD and k >= LARGE_GEMM_THRESHOLD

        elif func_packet == aten.matmul and len(args) >= 2:
            a, b = args[0], args[1]
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if a.dim() >= 2 and b.dim() >= 2:
                    m = a.shape[-2] if a.dim() >= 2 else 1
                    k = a.shape[-1]
                    n = b.shape[-1] if b.dim() >= 2 else 1
                    return m >= LARGE_GEMM_THRESHOLD and n >= LARGE_GEMM_THRESHOLD and k >= LARGE_GEMM_THRESHOLD

    elif op_type == OperatorType.ATTN:
        if len(args) >= 1 and isinstance(args[0], torch.Tensor) and args[0].dim() == 4:
            b, h, s, d = args[0].shape
            return b * h * s >= 4096 and s >= 512

    return False


def get_roofline_compute_time(
    func_packet, args, kwargs, out, out_dtypes, hw_info: HardwareInfo, op_type: OperatorType
) -> float:
    """Compute-bound ceiling for an op, in ns: (FLOPs / peak_FLOP_s) × 1e9.

    Two-tier peak: large ops (all dims ≥ 512) use the tensor-unit peak, small
    ops the conservative peak. Returns 0.0 if the op is not in the FLOP registry.
    """
    if func_packet in flop_registry:
        if len(out_dtypes) != 1:
            raise AssertionError(
                f"Only support single out dtype got {out_dtypes} for {func_packet}"
            )
        dtype = out_dtypes.pop()
        is_large_op = _is_large_tensor_core_op(func_packet, args, op_type)
        peak_tflops = hw_info.get_peak_tflops(op_type, dtype, is_large_op)
        peak_gpu_flops = peak_tflops * TFLOPS_TO_FLOPS
        flop_count_func = flop_registry[func_packet]
        flop_count = flop_count_func(*args, **kwargs, out_val=out)
        return (flop_count / peak_gpu_flops) * SECONDS_TO_NS
    return 0.0


def get_roofline_transfer_time(
    flat_args_kwargs, flat_outs, hw_info: HardwareInfo
) -> float:
    """Memory-bound ceiling for an op, in ns: total_bytes / peak_bw_gbps.

    Dimensionally: bytes / (GB/s numeric) = ns, since 1 GB = 1e9 bytes.
    """
    read_bytes = sum(
        get_num_bytes(t) for t in flat_args_kwargs if isinstance(t, torch.Tensor)
    )
    write_bytes = sum(
        get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
    )
    return (read_bytes + write_bytes) / hw_info.get_peak_memory_bandwidth_gbps()


@dataclass(frozen=True)
class Roofline:
    """Per-pipeline analytical demands (ns) and the single binding bound."""
    tensor_ns: float
    fma_ns: float
    sfu_ns: float
    mem_ns: float
    launch_ns: float

    @property
    def roofline_ns(self) -> float:
        """The roofline: the binding pipeline OR the launch floor, whichever is larger."""
        return max(self.tensor_ns, self.fma_ns, self.sfu_ns, self.mem_ns, self.launch_ns)


def roofline(
    func_packet: Any,
    args: tuple,
    kwargs: dict,
    out: Any,
    hw_info: HardwareInfo,
    op_type: OperatorType,
    *,
    t_launch_ns: float = 0.0,
) -> Roofline:
    """Compute the per-pipeline demands for one operator.

    For GEMM/ATTN, ``get_roofline_compute_time`` uses the tensor (MMA) peak -> that
    is ``tensor_ns``. ``fma_ns``/``sfu_ns`` come from the op's instruction mix
    (non-MMA FP32-vector / transcendental counts) — both 0 for GEMM (pure MMA),
    nonzero for softmax/gelu/norm. ``mem_ns`` is the memory-bound time. ``launch_ns``
    is the device launch floor (0 for the bare roofline; the measured ``t_launch``
    for the hybrid anchor).
    """
    if func_packet is None or func_packet in _IGNORE_OPS:
        return Roofline(0.0, 0.0, 0.0, 0.0, 0.0)

    flat_args, _ = tree_flatten((args, kwargs))
    flat_outs, _ = tree_flatten(out)
    out_dtypes = {t.dtype for t in flat_outs
                  if isinstance(t, torch.Tensor) and t.dtype in _FLOAT_TYPES}

    tensor_ns = get_roofline_compute_time(
        func_packet, args, kwargs, out, set(out_dtypes), hw_info, op_type
    )
    mem_ns = get_roofline_transfer_time(flat_args, flat_outs, hw_info)
    # FMA (non-MMA FP32 vector) and SFU (transcendental) demands from the op's
    # instruction mix; both 0 for GEMM (pure MMA — tensor_ns already covers it).
    _mma, fp32_flops, transc_ops = instruction_mix(func_packet, args, kwargs, out)
    fma_peak = hw_info.peak_tflops_math * 1e12            # TFLOP/s -> FLOP/s
    sfu_peak = (hw_info.sfu_peak or 0.0) * 1e12
    fma_ns = (fp32_flops / fma_peak * 1e9) if fma_peak > 0 else 0.0
    sfu_ns = (transc_ops / sfu_peak * 1e9) if sfu_peak > 0 else 0.0
    return Roofline(
        tensor_ns=float(tensor_ns),
        fma_ns=float(fma_ns),
        sfu_ns=float(sfu_ns),
        mem_ns=float(mem_ns),
        launch_ns=float(t_launch_ns),
    )
