"""Kernel-runtime predictor implementation: the roofline bound, per-op feature
blocks, the family router, and the trained LightGBM model loader.

Importing this package stays lightgbm-free (roofline + router only); the model
loader (tree_model) is imported directly where the calibrated path is used.
"""
from __future__ import annotations

from .roofline import roofline, Roofline
from .router import Family, route

__all__ = ["roofline", "Roofline", "Family", "route"]
