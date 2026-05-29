"""TreeEstimator: roofline x exp(learned residual) with an OOD roofline rail. Never raises."""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .predictor.roofline import roofline
from .predictor import features as F
from .predictor.router import route
from .predictor.tree_model import TreeModel, load_tree_model

log = logging.getLogger(__name__)


class TreeEstimator:
    """Per-op estimator: roofline anchor + learned residual correction."""

    def __init__(self, model: TreeModel, hw_info: Any) -> None:
        self._model = model
        self._hw_info = hw_info

    @classmethod
    def load(cls, path: str, hw_info: Any) -> "TreeEstimator":
        return cls(load_tree_model(path), hw_info)

    def estimate_op(self, func_packet, args, kwargs, out, op_type,
                    execution_mode=None, cache_seq_len=0) -> float:
        try:
            family = route(func_packet, op_type)
            t_launch = self._model.t_launch_ns.get(family.value, 0.0)
            rl = roofline(func_packet, args, kwargs, out, self._hw_info,
                          op_type, t_launch_ns=t_launch)
            if rl.roofline_ns <= 0:
                return 0.0
            # No calibrated tree for this family -> residual 0 -> the bare roofline.
            resid = 0.0
            booster = self._model.models.get(family.value)
            if booster is not None:
                row = F.featurize(func_packet, args, kwargs, out, family, rl)
                cols = self._model.feature_columns.get(family.value) or F.feature_columns(family)
                resid = float(booster.predict(self._encode(row, cols))[0])
            # OOD rail: never below the roofline (which already includes the launch floor).
            return max(rl.roofline_ns * math.exp(resid), rl.roofline_ns) / 1e6
        except Exception as e:                      # never crash a simulation
            log.debug("tree -> roofline: %s", e)
            try:
                return roofline(func_packet, args, kwargs, out,
                                self._hw_info, op_type).roofline_ns / 1e6
            except Exception:
                return 0.0

    def _encode(self, row: dict, cols: list[str]) -> np.ndarray:
        codes = self._model.categorical_codes
        vals = []
        for c in cols:
            v = row.get(c, 0.0)
            if c in codes:                          # categorical -> integer code
                v = codes[c].get(str(v), -1)
            vals.append(float(v))
        return np.array([vals], dtype=np.float64)
