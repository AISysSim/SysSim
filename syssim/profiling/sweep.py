"""Margin sampling: a few log-spaced points just outside the observed token range
so a slightly-larger model interpolates rather than hitting LightGBM's flat
extrapolation. (Trace-anchored points are the catalog; this only adds margin.)"""
from __future__ import annotations


def margin_token_points(observed_min: int, observed_max: int, n: int = 3) -> list[int]:
    """Return n log-spaced points below observed_min and n above observed_max."""
    below = [max(1, int(observed_min / (2 ** (i + 1)))) for i in range(n)]
    above = [int(observed_max * (2 ** (i + 1))) for i in range(n)]
    return sorted(set(below + above))
