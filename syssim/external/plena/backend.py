"""PLENA cycle-level performance estimation backend.

Provides PLENAEstimator as a drop-in replacement for GPU roofline estimation,
mapping PyTorch operators to PLENA instruction cycles.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import torch

from ...config import ExecutionMode
from ...operator_graph import OperatorType

if TYPE_CHECKING:
    from .config import PLENAConfig

aten = torch.ops.aten

# Op classification sets (from compute_cost_predictor.py)
_GEMM_OPS = frozenset({
    aten.mm,
    aten.addmm,
    aten.bmm,
    aten.matmul,
    aten.linear,
})

_ATTN_OPS = frozenset({
    aten._scaled_dot_product_efficient_attention,
    aten._scaled_dot_product_flash_attention,
    aten._scaled_dot_product_flash_attention_for_cpu,
    aten._scaled_dot_product_cudnn_attention,
    aten._flash_attention_forward,
    aten._efficient_attention_forward,
})


@dataclass
class PLENAHardwareInfo:
    """PLENA hardware information for cycle-based estimation.

    Wraps PLENA PerfModel and frequency for cycle → time conversion.
    Can be used as a type marker to distinguish PLENA from GPU hardware.

    Attributes:
        frequency_hz: Clock frequency in Hz for cycle-to-time conversion
        perf_model: PLENA PerfModel instance (lazy-loaded)
    """

    frequency_hz: float = 1e9
    _perf_model: Any = None
    _config: Optional["PLENAConfig"] = None

    @classmethod
    def from_config(cls, config: "PLENAConfig") -> "PLENAHardwareInfo":
        """Create PLENAHardwareInfo from PLENAConfig.

        Args:
            config: PLENAConfig with paths to PLENA configuration files

        Returns:
            PLENAHardwareInfo with lazy-loaded PerfModel
        """
        info = cls(frequency_hz=config.frequency_hz)
        info._config = config
        return info

    @property
    def perf_model(self):
        """Lazy-load PerfModel on first access."""
        if self._perf_model is None:
            if self._config is None:
                raise RuntimeError("PLENAHardwareInfo requires config to load PerfModel")
            self._perf_model = _load_perf_model(self._config)
        return self._perf_model


def _load_perf_model(config: "PLENAConfig"):
    """Load PLENA PerfModel from configuration."""
    # Add PLENA_Simulator to path if needed
    plena_base = Path(config.settings_path).parent
    plena_perf_dir = plena_base / "analytic_models" / "performance"
    if str(plena_perf_dir) not in sys.path:
        sys.path.insert(0, str(plena_perf_dir))

    from perf_model import PerfModel, load_hardware_config_from_toml

    hardware_config = load_hardware_config_from_toml(config.settings_path)
    return PerfModel(hardware_config, config.isa_lib_path)


def _extract_gemm_dims(func_packet, args: tuple) -> tuple[int, int, int, int]:
    """Extract M, N, K dimensions and batch from GEMM op args.

    Returns:
        (batch, M, N, K) - batch=1 for non-batched ops
    """
    if func_packet == aten.mm and len(args) >= 2:
        a, b = args[0], args[1]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dim() == 2 and b.dim() == 2:
                m, k = a.shape
                k2, n = b.shape
                return (1, m, n, k)

    elif func_packet == aten.addmm and len(args) >= 3:
        a, b = args[1], args[2]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dim() == 2 and b.dim() == 2:
                m, k = a.shape
                k2, n = b.shape
                return (1, m, n, k)

    elif func_packet == aten.bmm and len(args) >= 2:
        a, b = args[0], args[1]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dim() == 3 and b.dim() == 3:
                batch, m, k = a.shape
                _, k2, n = b.shape
                return (batch, m, n, k)

    elif func_packet == aten.matmul and len(args) >= 2:
        a, b = args[0], args[1]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dim() == 2 and b.dim() == 2:
                m, k = a.shape
                k2, n = b.shape
                return (1, m, n, k)
            elif a.dim() == 3 and b.dim() == 3:
                batch, m, k = a.shape
                _, k2, n = b.shape
                return (batch, m, n, k)
            elif a.dim() >= 2 and b.dim() >= 2:
                # Handle broadcast matmul
                m = a.shape[-2]
                k = a.shape[-1]
                n = b.shape[-1]
                batch = 1
                for dim in a.shape[:-2]:
                    batch *= dim
                return (batch, m, n, k)

    elif func_packet == aten.linear and len(args) >= 2:
        x, w = args[0], args[1]
        if isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor):
            # Linear: x @ w.T, w is (out_features, in_features)
            if x.dim() == 2:
                m, k = x.shape
                n, k2 = w.shape
                return (1, m, n, k)
            elif x.dim() == 3:
                batch, seq, k = x.shape
                n, k2 = w.shape
                return (batch * seq, 1, n, k)

    return (1, 1, 1, 1)


def _extract_attention_dims(args: tuple) -> tuple[int, int, int, int, int]:
    """Extract attention dimensions from SDPA args.

    Returns:
        (batch, num_heads, seq_len, kv_len, head_dim)
    """
    if len(args) >= 3:
        q, k, v = args[0], args[1], args[2]
        if isinstance(q, torch.Tensor) and q.dim() == 4:
            b, h, s, d = q.shape
            # K/V may have different seq_len (cross-attention)
            kv_len = k.shape[2] if isinstance(k, torch.Tensor) and k.dim() == 4 else s
            return (b, h, s, kv_len, d)
    return (1, 1, 1, 1, 1)


def estimate_plena_cycles_for_gemm(
    perf_model,
    batch: int,
    m: int,
    n: int,
    k: int,
) -> int:
    """Estimate PLENA cycles for GEMM operation.

    Maps M×K @ K×N matrix multiply to PLENA M_MM instruction cycles.

    Args:
        perf_model: PLENA PerfModel instance
        batch: Batch size (for batched matmul)
        m: M dimension (rows of left matrix)
        n: N dimension (cols of right matrix)
        k: K dimension (inner dimension)

    Returns:
        Estimated cycle count
    """
    blen = perf_model.blen
    mlen = perf_model.mlen
    instr = perf_model.instr

    # Number of MLEN×MLEN tiles
    m_tiles = math.ceil(m / blen)
    n_tiles = math.ceil(n / blen)
    k_tiles = math.ceil(k / mlen)

    # Each tile requires M_MM instruction
    cycles = batch * m_tiles * n_tiles * k_tiles * instr["M_MM"]

    return cycles


def estimate_plena_cycles_for_attention(
    perf_model,
    batch: int,
    num_heads: int,
    seq_len: int,
    kv_len: int,
    head_dim: int,
    mode: str = "prefill",
) -> int:
    """Estimate PLENA cycles for attention operation.

    Uses PLENA's flash_attention method for accurate cycle counting.

    Args:
        perf_model: PLENA PerfModel instance
        batch: Batch size
        num_heads: Number of attention heads
        seq_len: Query sequence length
        kv_len: Key/Value sequence length
        head_dim: Head dimension
        mode: "prefill" or "decode"

    Returns:
        Estimated cycle count
    """
    # Use PLENA's native flash_attention implementation
    # Assume GQA with num_kv_heads = num_heads for simplicity
    # (real implementation would need to detect GQA config)
    return perf_model.flash_attention(
        num_attention_heads=num_heads,
        num_kv_heads=num_heads,  # Assume no GQA unless detected
        head_dim=head_dim,
        seq_len=seq_len,
        kv_size=kv_len,
        batch_size=batch,
        mode=mode,
    )


def estimate_plena_cycles_for_math(
    perf_model,
    num_elements: int,
    op_multiplier: float = 1.0,
) -> int:
    """Estimate PLENA cycles for vector/math operations.

    Maps element-wise ops to V_BASIC instruction cycles.

    Args:
        perf_model: PLENA PerfModel instance
        num_elements: Number of elements to process
        op_multiplier: Multiplier for complex ops (e.g., 6 for SiLU)

    Returns:
        Estimated cycle count
    """
    vlen = perf_model.vlen
    instr = perf_model.instr

    num_vectors = math.ceil(num_elements / vlen)
    cycles = int(num_vectors * instr["V_BASIC"] * op_multiplier)

    return cycles


class PLENAEstimator:
    """PLENA cycle-based estimator for PyTorch operators.

    Drop-in replacement for GPU roofline estimation. Maps PyTorch ops
    to PLENA instructions and returns estimated runtime in milliseconds.

    Example:
        >>> from syssim.external.plena import PLENAConfig
        >>> config = PLENAConfig.from_plena_submodule()
        >>> estimator = PLENAEstimator(config)
        >>> time_ms = estimator.estimate_runtime(func_packet, args, kwargs, out, op_type)
    """

    def __init__(self, config: "PLENAConfig"):
        """Initialize PLENAEstimator.

        Args:
            config: PLENAConfig with paths to PLENA configuration files
        """
        self.config = config
        self.hw_info = PLENAHardwareInfo.from_config(config)
        self._perf_model = None

    @property
    def perf_model(self):
        """Lazy-load PerfModel on first access."""
        if self._perf_model is None:
            self._perf_model = _load_perf_model(self.config)
        return self._perf_model

    def cycles_to_ms(self, cycles: int) -> float:
        """Convert cycle count to milliseconds."""
        seconds = cycles / self.config.frequency_hz
        return seconds * 1000.0

    def estimate_op(
        self,
        func_packet: Any,
        args: tuple,
        kwargs: dict,
        out: Any,
        op_type: OperatorType,
        execution_mode: Optional[ExecutionMode] = None,
        cache_seq_len: int = 0,
    ) -> float:
        """Estimate runtime for an operator using PLENA cycle model.

        Args:
            func_packet: PyTorch operator function packet
            args: Operator arguments
            kwargs: Operator keyword arguments
            out: Operator output
            op_type: OperatorType classification
            execution_mode: ExecutionMode (PREFILL, DECODE, etc.)
            cache_seq_len: KV cache sequence length for decode mode

        Returns:
            Estimated runtime in milliseconds
        """
        cycles = 0

        if op_type == OperatorType.GEMM and func_packet in _GEMM_OPS:
            batch, m, n, k = _extract_gemm_dims(func_packet, args)
            cycles = estimate_plena_cycles_for_gemm(self.perf_model, batch, m, n, k)

        elif op_type == OperatorType.ATTN and func_packet in _ATTN_OPS:
            batch, num_heads, seq_len, kv_len, head_dim = _extract_attention_dims(args)

            # Handle decode mode with KV cache
            if execution_mode == ExecutionMode.DECODE and cache_seq_len > 0:
                kv_len = cache_seq_len

            mode = "decode" if execution_mode == ExecutionMode.DECODE else "prefill"
            cycles = estimate_plena_cycles_for_attention(
                self.perf_model, batch, num_heads, seq_len, kv_len, head_dim, mode
            )

        elif op_type == OperatorType.MATH:
            # Estimate based on output tensor size
            num_elements = 0
            if isinstance(out, torch.Tensor):
                num_elements = out.numel()
            elif isinstance(out, (tuple, list)):
                for t in out:
                    if isinstance(t, torch.Tensor):
                        num_elements += t.numel()

            # Determine op complexity multiplier
            op_multiplier = _get_math_op_multiplier(func_packet)
            cycles = estimate_plena_cycles_for_math(self.perf_model, num_elements, op_multiplier)

        # Convert cycles to milliseconds
        return self.cycles_to_ms(cycles)


def _get_math_op_multiplier(func_packet) -> float:
    """Get complexity multiplier for math operations.

    Different ops have different instruction requirements:
    - Simple: add, mul, relu → 1x
    - Medium: gelu, softmax → 4x
    - Complex: silu, layernorm → 6x
    """
    if func_packet is None:
        return 1.0

    packet_name = str(func_packet).lower()

    # Complex ops (multiple vector instructions)
    if any(x in packet_name for x in ["silu", "swiglu", "layer_norm", "rms_norm"]):
        return 6.0

    # Medium complexity
    if any(x in packet_name for x in ["gelu", "softmax", "sigmoid"]):
        return 4.0

    # Simple ops
    return 1.0
