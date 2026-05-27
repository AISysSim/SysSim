"""Pluggable per-operator runtime estimators.

The tracer is transparent to this module: it calls
`compute_cost_predictor.estimate_runtime(...)`, which delegates to the
estimator resolved from `hw_info` (see HardwareInfo.build_estimator). The
default is RooflineEstimator; custom backends (e.g. PLENA under
syssim/external/plena) implement the same protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .compute_cost_predictor import roofline_estimate, efficiency_estimate


@runtime_checkable
class Estimator(Protocol):
    """Estimates one operator's runtime in milliseconds."""

    def estimate_op(
        self, func_packet: Any, args: tuple, kwargs: dict, out: Any,
        op_type: Any, execution_mode: Any = None, cache_seq_len: int = 0,
    ) -> float:
        ...


class RooflineEstimator:
    """Default estimator: GPU roofline ceiling / ML-predicted efficiency."""

    def __init__(self, hw_info: Any) -> None:
        self._hw_info = hw_info

    def estimate_op(
        self, func_packet: Any, args: tuple, kwargs: dict, out: Any,
        op_type: Any, execution_mode: Any = None, cache_seq_len: int = 0,
    ) -> float:
        roofline_result = roofline_estimate(
            func_packet, args, kwargs, out, self._hw_info, op_type,
            execution_mode, cache_seq_len,
        )
        efficiency = efficiency_estimate(
            func_packet, args, kwargs, out, self._hw_info, op_type,
            roofline_result, execution_mode, cache_seq_len,
        )
        if efficiency == 0:
            return 0.0
        return roofline_result.t_roofline_ms / efficiency
