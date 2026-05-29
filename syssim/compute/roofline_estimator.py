"""RooflineEstimator: the default per-op estimator — the bare roofline bound."""

from __future__ import annotations

from typing import Any


class RooflineEstimator:
    """Default estimator: the roofline bound in ms (no learned correction)."""

    def __init__(self, hw_info: Any) -> None:
        self._hw_info = hw_info

    def estimate_op(
        self, func_packet: Any, args: tuple, kwargs: dict, out: Any,
        op_type: Any, execution_mode: Any = None, cache_seq_len: int = 0,
    ) -> float:
        from .predictor.roofline import roofline
        return roofline(
            func_packet, args, kwargs, out, self._hw_info, op_type).roofline_ns / 1e6
