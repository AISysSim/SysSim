"""Roofline-based runtime prediction for operators.

Provides op classification sets, roofline compute/transfer time estimation,
and an estimate_runtime entry point decoupled from OperatorNode.

Unit System:
- Storage: TFLOP/s (10^12 FLOP/s), GB/s (10^9 bytes/s)
- Time (internal): nanoseconds (ns)
- Time (external): milliseconds (ms)
- Conversions: See constants below
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._ordered_set import OrderedSet

from ..operator_graph import OperatorType
from ..config import ExecutionMode, HardwareInfo
from .flop_counter import flop_registry, sdpa_flop_count


# ============================================================================
# Unit Conversion Constants
# ============================================================================
# These constants ensure correct dimensional analysis throughout the codebase.
# Always use these instead of magic numbers to make units explicit.

TERA_TO_UNIT = 1e12          # 1 TFLOP = 10^12 FLOP (tera = trillion)
GIGA_TO_UNIT = 1e9           # 1 GB = 10^9 bytes (giga = billion)
PETA_TO_TERA = 1000.0        # 1 PFLOP = 1000 TFLOP (peta = quadrillion)
SECONDS_TO_NS = 1e9          # 1 second = 10^9 nanoseconds
NS_TO_MS = 1e6               # 1 millisecond = 10^6 nanoseconds

# Common combined conversions
TFLOPS_TO_FLOPS = TERA_TO_UNIT    # Alias for clarity
GBPS_TO_BPS = GIGA_TO_UNIT        # Alias for clarity

# Threshold for large tensor unit operations
# Operations with all dimensions ≥ this threshold use tensor unit peak
# Smaller operations use conservative peak due to launch overhead
LARGE_GEMM_THRESHOLD = 512

_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)

aten = torch.ops.aten

_FLOAT_TYPES = OrderedSet(
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]
)

# No fall-back kernel needed/exists for view ops
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

# We can ignore benchmarking tensor create ops
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


@dataclass
class ConstraintTime:
    """Single roofline constraint."""
    work_type: str  # "math", "memory"
    unit_level: str  # "device"
    time_ms: float
    work_amount: float  # W_k(x)
    capacity: float  # C_{k,l}(h)


@dataclass
class RooflineResult:
    """Multi-dimensional roofline result."""
    t_roofline_ms: float  # max of all constraints
    constraints: list[ConstraintTime]
    dominant_constraint: tuple[str, str]  # (work_type, unit_level)

    def get_constraint_ratios(self) -> dict[tuple[str, str], float]:
        """Return r_{k,l} = T_{k,l} / T_roofline for all constraints."""
        if self.t_roofline_ms == 0:
            return {(c.work_type, c.unit_level): 0.0 for c in self.constraints}
        return {
            (c.work_type, c.unit_level): c.time_ms / self.t_roofline_ms
            for c in self.constraints
        }


def get_num_bytes(t: torch.Tensor) -> int:
    num_bytes = t.untyped_storage().nbytes()
    mem_consumed = (
        math.ceil(num_bytes / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
    )
    return mem_consumed


def _is_large_tensor_core_op(func_packet, args, op_type: OperatorType) -> bool:
    """Check if operation is large enough for full tensor unit utilization.

    Large operations (all dims ≥ 512) can achieve near-peak tensor unit performance.
    Small operations are dominated by kernel launch overhead (~7 μs) and use
    conservative peak estimates.

    Args:
        func_packet: PyTorch operator function packet
        args: Operator arguments
        op_type: OperatorType

    Returns:
        True if all dimensions ≥ LARGE_GEMM_THRESHOLD (512), False otherwise
    """
    if op_type == OperatorType.GEMM:
        # Extract M, N, K dimensions based on operator type
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
                # Handle various matmul cases (2D, 3D, batched)
                if a.dim() >= 2 and b.dim() >= 2:
                    m = a.shape[-2] if a.dim() >= 2 else 1
                    k = a.shape[-1]
                    n = b.shape[-1] if b.dim() >= 2 else 1
                    return m >= LARGE_GEMM_THRESHOLD and n >= LARGE_GEMM_THRESHOLD and k >= LARGE_GEMM_THRESHOLD

    elif op_type == OperatorType.ATTN:
        # Attention ops are usually large enough for tensor units if batch*heads*seq is large
        if len(args) >= 1 and isinstance(args[0], torch.Tensor) and args[0].dim() == 4:
            b, h, s, d = args[0].shape
            # Heuristic: total query tokens ≥ 4K and seq_len ≥ 512
            return b * h * s >= 4096 and s >= 512

    return False


def get_roofline_compute_time(
    func_packet, args, kwargs, out, out_dtypes, hw_info: HardwareInfo, op_type: OperatorType
) -> float:
    """Estimate compute-bound roofline time for an operator with size-aware peak selection.

    Calculates the minimum time required to execute the FLOPs for this operation
    at the hardware's peak FLOP/s. This represents the compute-bound ceiling.

    For GEMM/ATTN operators, uses two-tier peak selection:
    - Large ops (all dims ≥ 512): Use tensor unit peak (e.g., 1979 TFLOP/s)
    - Small ops (any dim < 512): Use conservative peak (e.g., 535 TFLOP/s)

    Args:
        func_packet: PyTorch operator function packet (e.g., aten.mm)
        args: Operator positional arguments
        kwargs: Operator keyword arguments
        out: Operator output tensor(s)
        out_dtypes: Set of output data types (used to select peak FLOP/s)
        hw_info: Hardware specs with peak_tflops_mm, peak_tflops_math in TFLOP/s
        op_type: OperatorType (GEMM/ATTN use peak_tflops_mm, others use peak_tflops_math)

    Returns:
        Compute time in nanoseconds (ns).

        Calculation: (FLOPs / peak_FLOP_s) × 1e9
        - Uses FLOP counting registry (flop_counter.py)
        - Converts TFLOP/s → FLOP/s via × 1e12 (TERA_TO_UNIT)
        - Converts seconds → ns via × 1e9 (SECONDS_TO_NS)
        Returns 0.0 if operator not in FLOP registry.

    Example:
        64×64×64 FP16 GEMM on GH200:
        - FLOPs: 2 × 64³ = 524,288
        - is_large_op: False (64 < 512)
        - Peak: 535 TFLOP/s (conservative) = 535e12 FLOP/s
        - Time: 524,288 / 535e12 × 1e9 = 0.98 ns

        2048×2048×8192 FP16 GEMM on GH200:
        - FLOPs: 2 × 2048 × 2048 × 8192 = 68.7 GFLOP
        - is_large_op: True (all dims ≥ 512)
        - Peak: 1979 TFLOP/s (tensor unit) = 1979e12 FLOP/s
        - Time: 68.7e9 / 1979e12 × 1e9 = 34.7 μs
    """
    if func_packet in flop_registry:
        if len(out_dtypes) != 1:
            raise AssertionError(
                f"Only support single out dtype got {out_dtypes} for {func_packet}"
            )
        dtype = out_dtypes.pop()

        # Determine if this is a large tensor unit operation
        is_large_op = _is_large_tensor_core_op(func_packet, args, op_type)

        # Select appropriate peak (1979 for large, 535 for small on GH200)
        peak_tflops = hw_info.get_peak_tflops(op_type, dtype, is_large_op)
        peak_gpu_flops = peak_tflops * TFLOPS_TO_FLOPS  # × 1e12

        flop_count_func = flop_registry[func_packet]
        flop_count = flop_count_func(*args, **kwargs, out_val=out)
        compute_time_ns = (flop_count / peak_gpu_flops) * SECONDS_TO_NS
        return compute_time_ns
    return 0.0


def get_roofline_transfer_time(
    flat_args_kwargs, flat_outs, hw_info: HardwareInfo
) -> float:
    """Estimate memory-bound roofline time for an operator.

    Calculates the minimum time required to transfer input and output data
    at the hardware's peak memory bandwidth. This represents the memory-bound ceiling.

    Args:
        flat_args_kwargs: Flattened list of operator arguments (may include tensors)
        flat_outs: Flattened list of operator outputs (may include tensors)
        hw_info: Hardware specs with peak_memory_bandwidth_gbps in GB/s

    Returns:
        Transfer time in nanoseconds (ns).

        Calculation: total_bytes / peak_bw_gbps
        - This formula is dimensionally correct due to the numeric representation:
          * peak_bw_gbps is stored as a float (e.g., 3350.0) representing GB/s
          * bytes / (GB/s) = bytes / GB/s = (bytes × s) / GB
          * Since 1 GB = 1e9 bytes, this simplifies to: time_ns = bytes / bw_numeric
        - Mathematically equivalent to: (bytes / (bw_gbps × 1e9)) × 1e9
        - See unit tests in test_unit_consistency.py for verification

    Example:
        64×64×64 FP16 GEMM on H100:
        - Bytes: 3 matrices × 64² × 2 bytes = 24,576 bytes
        - Peak BW: 3350 GB/s = 3.35e12 bytes/s
        - Time: 24,576 / 3350 = 7.34 ns
    """
    read_bytes = sum(
        get_num_bytes(t) for t in flat_args_kwargs if isinstance(t, torch.Tensor)
    )
    write_bytes = sum(
        get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
    )
    counted_bytes = read_bytes + write_bytes
    transfer_time_ns = counted_bytes / hw_info.get_peak_memory_bandwidth_gbps()
    return transfer_time_ns


def _decode_attention_compute_ns(
    args: tuple, hw_info: HardwareInfo, cache_seq_len: int,
) -> float:
    """Estimate compute time for decode attention with KV cache override.

    Q keeps its traced shape (seq_len=1), but K/V use cache_seq_len.
    Returns estimated time in nanoseconds.
    """
    q = args[0]
    if not isinstance(q, torch.Tensor) or q.dim() != 4:
        return 0.0
    b, h, s_q, d = q.shape
    # Override K/V shapes to reflect full KV cache
    q_shape = (b, h, s_q, d)
    k_shape = (b, h, cache_seq_len, d)
    v_shape = (b, h, cache_seq_len, d)
    flop_count = sdpa_flop_count(q_shape, k_shape, v_shape)
    peak_flops = hw_info.get_peak_tflops(OperatorType.ATTN, q.dtype) * 1e12
    if peak_flops == 0:
        return 0.0
    return (flop_count / peak_flops) * 1e9


def _decode_attention_transfer_ns(
    args: tuple, hw_info: HardwareInfo, cache_seq_len: int,
) -> float:
    """Estimate memory transfer time for decode attention with KV cache.

    Accounts for reading the full KV cache rather than just the traced tensors.
    Returns estimated time in nanoseconds.
    """
    q = args[0]
    if not isinstance(q, torch.Tensor) or q.dim() != 4:
        return 0.0
    b, h, s_q, d = q.shape
    element_size = q.element_size()
    # Q read
    q_bytes = b * h * s_q * d * element_size
    # KV cache read (both K and V)
    kv_bytes = 2 * b * h * cache_seq_len * d * element_size
    # Output write
    out_bytes = b * h * s_q * d * element_size
    total_bytes = q_bytes + kv_bytes + out_bytes
    bw = hw_info.get_peak_memory_bandwidth_gbps()
    if bw == 0:
        return 0.0
    return total_bytes / bw
    

def roofline_estimate(
    func_packet: Any,
    args: tuple,
    kwargs: dict,
    out: Any,
    hw_info: HardwareInfo,
    op_type: OperatorType,
    execution_mode: ExecutionMode | None = None,
    cache_seq_len: int = 0,
) -> RooflineResult:
    """Estimate runtime for an operator using multi-dimensional roofline model.

    Implements the roofline performance model: T_roofline = max(T_compute, T_memory).
    Computes both compute-bound time (FLOPs / peak_FLOP_s) and memory-bound time
    (bytes / peak_bandwidth), returning the maximum (bottleneck).

    Args:
        func_packet: The torch operation (e.g., aten.mm)
        args: Positional arguments to the operation
        kwargs: Keyword arguments to the operation
        out: Output tensor(s) from the operation
        hw_info: Hardware configuration:
            - peak_tflops_mm: Peak TFLOP/s for matrix multiply (GEMM, ATTN)
            - peak_tflops_math: Peak TFLOP/s for vector unit
            - peak_memory_bandwidth_gbps: Peak GB/s for memory transfers
        op_type: The OperatorType (GEMM, ATTN, MATH, etc.)
        execution_mode: Optional execution mode (TRAINING, PREFILL, DECODE).
                       DECODE mode uses cache-aware estimates for ATTN ops.
        cache_seq_len: KV cache sequence length for decode mode.

    Returns:
        RooflineResult with:
        - t_roofline_ms: Estimated time in milliseconds (max of all constraints)
        - constraints: List of ConstraintTime objects (compute, memory)
        - dominant_constraint: Tuple (work_type, unit_level) of bottleneck

    Example:
        Large GEMM (4096×4096×4096 FP16) on H100:
        - Compute: 137.4 GFLOPs / 989 TFLOP/s = 0.139 ms
        - Memory: 100.7 MB / 3350 GB/s = 0.030 ms
        - Roofline: max(0.139, 0.030) = 0.139 ms (compute-bound)
    """
    if func_packet is None or func_packet in _IGNORE_OPS:
        return RooflineResult(
            t_roofline_ms=0.0,
            constraints=[],
            dominant_constraint=("none", "none"),
        )

    constraints = []

    # Decode attention: override FLOPs and memory with KV cache-aware estimates
    if (
        execution_mode == ExecutionMode.DECODE
        and op_type == OperatorType.ATTN
        and cache_seq_len > 0
    ):
        compute_ns = _decode_attention_compute_ns(args, hw_info, cache_seq_len)
        transfer_ns = _decode_attention_transfer_ns(args, hw_info, cache_seq_len)

        # Extract work amounts for decode attention
        q = args[0]
        if isinstance(q, torch.Tensor) and q.dim() == 4:
            b, h, s_q, d = q.shape
            q_shape = (b, h, s_q, d)
            k_shape = (b, h, cache_seq_len, d)
            v_shape = (b, h, cache_seq_len, d)
            flop_count = sdpa_flop_count(q_shape, k_shape, v_shape)

            element_size = q.element_size()
            q_bytes = b * h * s_q * d * element_size
            kv_bytes = 2 * b * h * cache_seq_len * d * element_size
            out_bytes = b * h * s_q * d * element_size
            total_bytes = q_bytes + kv_bytes + out_bytes
        else:
            flop_count = 0
            total_bytes = 0

        # Build constraint list
        peak_flops = hw_info.get_peak_tflops(op_type, args[0].dtype if isinstance(args[0], torch.Tensor) else torch.float32) * 1e12
        constraints.append(ConstraintTime(
            work_type="math",
            unit_level="device",
            time_ms=compute_ns / 1e6,
            work_amount=flop_count,
            capacity=peak_flops,
        ))
        constraints.append(ConstraintTime(
            work_type="memory",
            unit_level="device",
            time_ms=transfer_ns / 1e6,
            work_amount=total_bytes,
            capacity=hw_info.get_peak_memory_bandwidth_gbps(),
        ))
    else:
        # Standard roofline path
        flat_args, _ = tree_flatten((args, kwargs))
        flat_outs, _ = tree_flatten(out)

        out_dtypes = {
            t.dtype
            for t in flat_outs
            if isinstance(t, torch.Tensor) and t.dtype in _FLOAT_TYPES
        }

        # Compute constraint
        compute_ns = get_roofline_compute_time(
            func_packet, args, kwargs, out, out_dtypes.copy(), hw_info, op_type
        )
        if compute_ns > 0 and out_dtypes:
            dtype = next(iter(out_dtypes))
            flop_count_func = flop_registry.get(func_packet)
            flop_count = flop_count_func(*args, **kwargs, out_val=out) if flop_count_func else 0
            peak_flops = hw_info.get_peak_tflops(op_type, dtype) * 1e12
            constraints.append(ConstraintTime(
                work_type="math",
                unit_level="device",
                time_ms=compute_ns / 1e6,
                work_amount=flop_count,
                capacity=peak_flops,
            ))

        # Memory constraint
        transfer_ns = get_roofline_transfer_time(flat_args, flat_outs, hw_info)
        read_bytes = sum(get_num_bytes(t) for t in flat_args if isinstance(t, torch.Tensor))
        write_bytes = sum(get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor))
        constraints.append(ConstraintTime(
            work_type="memory",
            unit_level="device",
            time_ms=transfer_ns / 1e6,
            work_amount=read_bytes + write_bytes,
            capacity=hw_info.get_peak_memory_bandwidth_gbps(),
        ))

    # Find max constraint
    if not constraints:
        return RooflineResult(0.0, [], ("none", "none"))

    max_constraint = max(constraints, key=lambda c: c.time_ms)
    t_roofline_ms = max_constraint.time_ms
    dominant = (max_constraint.work_type, max_constraint.unit_level)

    return RooflineResult(
        t_roofline_ms=t_roofline_ms,
        constraints=constraints,
        dominant_constraint=dominant,
    )


def estimate_runtime(
    func_packet: Any,
    args: tuple,
    kwargs: dict,
    out: Any,
    hw_info: HardwareInfo,
    op_type: OperatorType,
    execution_mode: ExecutionMode | None = None,
    cache_seq_len: int = 0,
) -> float:
    """Estimate runtime using hybrid roofline + efficiency model.

    Args:
        func_packet: The torch operation
        args: Positional arguments
        kwargs: Keyword arguments
        out: Output tensor(s)
        hw_info: Hardware configuration
        op_type: The OperatorType
        execution_mode: Optional execution mode
        cache_seq_len: KV cache sequence length

    Returns:
        Estimated runtime in milliseconds.
    """
    # Transparent boundary: delegate to the estimator resolved from hw_info
    # (RooflineEstimator by default; custom backends via hw_info.estimator).
    return hw_info.build_estimator().estimate_op(
        func_packet, args, kwargs, out, op_type, execution_mode, cache_seq_len
    )
