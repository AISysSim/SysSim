"""Rule-based operator-family router (no training). Blackbox: keys off the
dispatched aten op + OperatorType only."""
from __future__ import annotations

from enum import Enum

import torch

from ...operator_graph import OperatorType

aten = torch.ops.aten


class Family(Enum):
    GEMM = "gemm"
    ATTENTION = "attention"
    NORMALIZATION = "normalization"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"


_NORM_OPS = frozenset({
    aten.native_layer_norm, aten.native_group_norm,
} | ({aten._fused_rms_norm} if hasattr(aten, "_fused_rms_norm") else set()))

_REDUCTION_OPS = frozenset({
    aten.sum, aten.mean, aten.max, aten.argmax, aten.var,
    aten._softmax, aten._log_softmax,
})


def route(func_packet, op_type: OperatorType) -> Family:
    if op_type == OperatorType.GEMM:
        return Family.GEMM
    if op_type == OperatorType.ATTN:
        return Family.ATTENTION
    if func_packet in _NORM_OPS:
        return Family.NORMALIZATION
    if func_packet in _REDUCTION_OPS:
        return Family.REDUCTION
    return Family.ELEMENTWISE
