"""HybridEstimator: T_an * exp(residual) with an OOD roofline rail. Never raises."""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .analytical import analytical_bound
from . import features as F
from .router import route
from .bundle import Bundle, load_bundle

log = logging.getLogger(__name__)


class HybridEstimator:
    """Per-op estimator: analytical anchor + learned residual (LightGBM)."""

    def __init__(self, bundle: Bundle, hw_info: Any) -> None:
        self._bundle = bundle
        self._hw_info = hw_info

    @classmethod
    def load(cls, path: str, hw_info: Any) -> "HybridEstimator":
        return cls(load_bundle(path), hw_info)

    def estimate_op(self, func_packet, args, kwargs, out, op_type,
                    execution_mode=None, cache_seq_len=0) -> float:
        try:
            family = route(func_packet, op_type)
            t_launch = self._bundle.t_launch_ns.get(family.value, 0.0)
            bound = analytical_bound(func_packet, args, kwargs, out, self._hw_info,
                                     op_type, t_launch_ns=t_launch)
            if bound.t_an_ns <= 0:
                return 0.0
            booster = self._bundle.models.get(family.value)
            if booster is None:
                return bound.t_an_ns / 1e6          # analytical fallback
            row = F.featurize(func_packet, args, kwargs, out, family, bound)
            cols = self._bundle.feature_columns.get(family.value) or F.feature_columns(family)
            x = self._encode(row, cols)
            resid = float(booster.predict(x)[0])
            pred_ns = bound.t_an_ns * math.exp(resid)
            return max(pred_ns, bound.roofline_hw_ns) / 1e6   # OOD rail; ns->ms
        except Exception as e:                      # never crash a simulation
            log.debug("HybridEstimator fell back to analytical: %s", e)
            try:
                b = analytical_bound(func_packet, args, kwargs, out, self._hw_info, op_type)
                return b.t_an_ns / 1e6
            except Exception:
                return 0.0

    def _encode(self, row: dict, cols: list[str]) -> np.ndarray:
        codes = self._bundle.categorical_codes
        vals = []
        for c in cols:
            v = row.get(c, 0.0)
            if c in codes:                          # categorical -> integer code
                v = codes[c].get(str(v), -1)
            vals.append(float(v))
        return np.array([vals], dtype=np.float64)
