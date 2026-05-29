"""Pluggable per-operator runtime estimators.

The tracer is transparent to this module: it calls ``estimate_runtime(...)``,
which delegates to the estimator resolved from ``hw_info`` (see
HardwareInfo.build_estimator). The default is ``RooflineEstimator``
(syssim/compute/roofline_estimator.py); custom backends (e.g. the PLENA backend
under syssim/external/plena) implement the same protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..config import ExecutionMode, HardwareInfo
from ..operator_graph import OperatorType


@runtime_checkable
class Estimator(Protocol):
    """Estimates one operator's runtime in milliseconds."""

    def estimate_op(
        self, func_packet: Any, args: tuple, kwargs: dict, out: Any,
        op_type: Any, execution_mode: Any = None, cache_seq_len: int = 0,
    ) -> float:
        ...


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
    """Estimate one operator's runtime in milliseconds.

    Transparent boundary the tracer calls: delegate to the estimator resolved
    from ``hw_info`` (RooflineEstimator by default; custom backends via
    ``hw_info.estimator``).
    """
    return hw_info.build_estimator().estimate_op(
        func_packet, args, kwargs, out, op_type, execution_mode, cache_seq_len
    )
