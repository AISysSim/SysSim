"""Hybrid kernel-runtime predictor package.

Lazy exports (PEP 562 ``__getattr__``): importing a submodule such as
``.analytical`` must NOT pull in ``HybridEstimator -> bundle -> lightgbm``. This
lets the core default ``RooflineEstimator`` import the analytical bound from here
without making lightgbm a hard dependency of the default (analytical) path.
``router`` and ``hybrid_estimator`` land in later tasks; their branches below are
dormant until then.
"""
from __future__ import annotations

__all__ = ["analytical_bound", "AnalyticalBound", "Family", "route", "HybridEstimator"]


def __getattr__(name):
    if name in ("analytical_bound", "AnalyticalBound"):
        from . import analytical
        return getattr(analytical, name)
    if name in ("Family", "route"):
        from . import router
        return getattr(router, name)
    if name == "HybridEstimator":
        from .hybrid_estimator import HybridEstimator
        return HybridEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
