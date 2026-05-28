"""Pluggable per-operator runtime estimators.

The tracer is transparent to this module: it calls
`compute_cost_predictor.estimate_runtime(...)`, which delegates to the
estimator resolved from `hw_info` (see HardwareInfo.build_estimator). The
default is RooflineEstimator; custom backends (e.g. PLENA under
syssim/external/plena) implement the same protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Estimator(Protocol):
    """Estimates one operator's runtime in milliseconds."""

    def estimate_op(
        self, func_packet: Any, args: tuple, kwargs: dict, out: Any,
        op_type: Any, execution_mode: Any = None, cache_seq_len: int = 0,
    ) -> float:
        ...


class RooflineEstimator:
    """Default estimator: the multi-pipeline analytical bound (no efficiency model)."""

    def __init__(self, hw_info: Any) -> None:
        self._hw_info = hw_info

    def estimate_op(
        self, func_packet: Any, args: tuple, kwargs: dict, out: Any,
        op_type: Any, execution_mode: Any = None, cache_seq_len: int = 0,
    ) -> float:
        from .predictor.analytical import analytical_bound
        bound = analytical_bound(
            func_packet, args, kwargs, out, self._hw_info, op_type)
        return bound.t_an_ns / 1e6   # ns -> ms
