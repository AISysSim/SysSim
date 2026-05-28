"""Multi-pipeline analytical runtime bound.

Generalizes single-ceiling roofline to the binding pipeline among Tensor (MMA),
FMA (FP32 vector), SFU (transcendentals), and memory, plus an optional launch
floor. Used by the core ``RooflineEstimator`` (returned directly) and by the
hybrid predictor (as the residual anchor). Pure analytics — blackbox (shapes +
op math + device constants); no kernel internals.

Lives under ``compute/predictor/`` (not core ``compute/``) by project decision;
the core ``RooflineEstimator`` imports it from here, and the package ``__init__``
is lazy so importing this module never pulls in lightgbm.

Units: ns internally (matches the roofline helpers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from ...config import HardwareInfo
from ...operator_graph import OperatorType
from ..compute_cost_predictor import (
    get_roofline_compute_time,
    get_roofline_transfer_time,
    _IGNORE_OPS,
)

_FLOAT_TYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                torch.float8_e4m3fn, torch.float8_e5m2)


@dataclass(frozen=True)
class AnalyticalBound:
    """Per-pipeline analytical demands (ns) and the binding bound."""
    tensor_ns: float
    fma_ns: float
    sfu_ns: float
    mem_ns: float
    launch_ns: float

    @property
    def roofline_hw_ns(self) -> float:
        """True physical lower bound: binding hardware pipeline (excludes launch)."""
        return max(self.tensor_ns, self.fma_ns, self.sfu_ns, self.mem_ns)

    @property
    def t_an_ns(self) -> float:
        """Anchor: binding pipeline OR the launch floor, whichever is larger."""
        return max(self.roofline_hw_ns, self.launch_ns)


def analytical_bound(
    func_packet: Any,
    args: tuple,
    kwargs: dict,
    out: Any,
    hw_info: HardwareInfo,
    op_type: OperatorType,
    *,
    t_launch_ns: float = 0.0,
) -> AnalyticalBound:
    """Compute the per-pipeline analytical demands for one operator.

    For GEMM/ATTN, ``get_roofline_compute_time`` uses the tensor (MMA) peak -> that
    is ``tensor_ns``. ``fma_ns``/``sfu_ns`` are 0 in the MVP (GEMM emits no
    FP32-vector or transcendental ops); they are populated for the memory-bound
    families in a follow-up plan once ``flop_counter`` emits an instruction mix.
    ``mem_ns`` is the memory-bound time. ``launch_ns`` is the device launch floor
    (0 for the analytical default; the measured ``t_launch`` for the hybrid anchor).
    """
    if func_packet is None or func_packet in _IGNORE_OPS:
        return AnalyticalBound(0.0, 0.0, 0.0, 0.0, 0.0)

    flat_args, _ = tree_flatten((args, kwargs))
    flat_outs, _ = tree_flatten(out)
    out_dtypes = {t.dtype for t in flat_outs
                  if isinstance(t, torch.Tensor) and t.dtype in _FLOAT_TYPES}

    tensor_ns = get_roofline_compute_time(
        func_packet, args, kwargs, out, set(out_dtypes), hw_info, op_type
    )
    mem_ns = get_roofline_transfer_time(flat_args, flat_outs, hw_info)
    # MVP: no instruction-mix split yet -> FMA/SFU demands are zero for GEMM.
    return AnalyticalBound(
        tensor_ns=float(tensor_ns),
        fma_ns=0.0,
        sfu_ns=0.0,
        mem_ns=float(mem_ns),
        launch_ns=float(t_launch_ns),
    )
