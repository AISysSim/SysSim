"""TreeModel: manifest + per-family LightGBM boosters loaded from data/<device>/.

LightGBM is the residual model today; a different model kind would live in a
sibling module (e.g. mlp_model.py) implementing the same load shape.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import lightgbm as lgb

from . import features as F


@dataclass
class TreeModel:
    device: str
    sfu_peak: float
    t_launch_ns: dict[str, float]
    categorical_codes: dict[str, dict[str, int]]
    feature_columns: dict[str, list[str]]
    models: dict[str, lgb.Booster] = field(default_factory=dict)


def load_tree_model(path: str) -> TreeModel:
    with open(os.path.join(path, "manifest.json")) as f:
        m = json.load(f)
    if m.get("schema_version") != F.SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version {m.get('schema_version')!r} != "
            f"code SCHEMA_VERSION {F.SCHEMA_VERSION!r} (stale model)"
        )
    models: dict[str, lgb.Booster] = {}
    for fam, kind in (m.get("families") or {}).items():
        if kind != "tree":
            continue
        model_path = os.path.join(path, f"{fam}_model.lgb")
        if os.path.exists(model_path):
            models[fam] = lgb.Booster(model_file=model_path)
        # else: missing file -> family falls back to the roofline at predict time
    return TreeModel(
        device=m["device"],
        sfu_peak=float(m.get("sfu_peak", 0.0)),
        t_launch_ns={k: float(v) for k, v in (m.get("t_launch_ns") or {}).items()},
        categorical_codes=m.get("categorical_codes", {}),
        feature_columns=m.get("feature_columns", {}),
        models=models,
    )
