"""Calibrate (CPU): fit the GEMM residual model + analytical constants from the
committed profiling data, write the bundle. No GPU."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from ..compute.predictor import features as F


def _ape(y_true, y_pred):
    return np.median(np.abs(y_pred - y_true) / np.maximum(y_true, 1e-12))


def calibrate_gemm(*, data_dir: str, device: str, sfu_peak: float,
                   target: str = "residual") -> dict[str, Any]:
    """Fit the GEMM model from data_dir/prof/gemm.parquet -> bundle in data_dir."""
    df = pd.read_parquet(os.path.join(data_dir, "prof", "gemm.parquet"))
    latency_ns = df["latency_ns"].to_numpy()
    # Launch floor = asymptotic small-shape latency. Compute it FIRST so the anchor
    # here is identical to inference's bound.t_an_ns = max(tensor, fma, sfu, mem, launch)
    # (fma=sfu=0 for GEMM). This is the train/inference parity guarantee (spec section 2).
    t_launch_ns = float(latency_ns.min())
    t_an_ns = np.maximum.reduce([df["tensor_ns"].to_numpy(),
                                 df["mem_ns"].to_numpy(),
                                 np.full(len(df), t_launch_ns)])
    y = np.log(latency_ns) - np.log(np.maximum(t_an_ns, 1e-12))   # residual

    # GEMM feature columns from the measured shapes. `dtype` is encoded to the SAME
    # integer code HybridEstimator._encode applies at inference (numeric, not an
    # lgb-categorical), so training and inference featurize identically.
    _DTYPE_CODES = {"bf16": 0, "fp16": 1, "fp8_e4m3": 2, "fp8_e5m2": 3, "fp32": 4}
    feat = pd.DataFrame({
        "log_anchor_ns": np.log(np.maximum(t_an_ns, 1e-12)),
        "log_M": np.log(df["M"]), "log_N": np.log(df["N"]), "log_K": np.log(df["K"]),
        "M": df["M"], "N": df["N"], "K": df["K"],
        "dtype": df["dtype"].map(_DTYPE_CODES).fillna(-1).astype(float),
    })
    cols = list(feat.columns)
    # Held-out split by unique (M,N,K,dtype) shape (random, seeded): identical
    # measured shapes never leak across train/val, and the held-out set interpolates
    # within the swept range -- not the degenerate "largest-configs-only" tail split.
    rng = np.random.default_rng(0)
    keys = list(zip(df["M"].tolist(), df["N"].tolist(), df["K"].tolist(), df["dtype"].tolist()))
    uniq = sorted(set(keys))
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * 0.15)))
    val_keys = {uniq[int(i)] for i in perm[:n_val]}
    is_val = np.array([k in val_keys for k in keys])
    tr = np.where(~is_val)[0]; va = np.where(is_val)[0]
    ds = lgb.Dataset(feat.iloc[tr], label=y[tr])
    booster = lgb.train(
        {"objective": "regression", "metric": "mae", "num_leaves": 63,
         "learning_rate": 0.05, "min_data_in_leaf": 10, "verbose": -1},
        ds, num_boost_round=300,
        valid_sets=[lgb.Dataset(feat.iloc[va], label=y[va])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    booster.save_model(os.path.join(data_dir, "gemm_model.lgb"))

    pred_resid = booster.predict(feat.iloc[va])
    # Inference applies the OOD rail max(pred, roofline_hw); roofline_hw == t_an_ns
    # here (fma=sfu=0, launch already folded in), so mirror it for honest metrics.
    pred_ns = np.maximum(t_an_ns[va] * np.exp(pred_resid), t_an_ns[va])
    metrics = {
        "median_ape": float(_ape(latency_ns[va], pred_ns)),
        "mean_signed_log_error": float(np.mean(np.log(pred_ns) - np.log(latency_ns[va]))),
    }

    manifest = {
        "device": device, "schema_version": F.SCHEMA_VERSION,
        "lightgbm_version": lgb.__version__,
        "sfu_peak": sfu_peak, "t_launch_ns": {"gemm": t_launch_ns},
        "categorical_codes": {"dtype": {"bf16": 0, "fp16": 1, "fp8_e4m3": 2,
                                        "fp8_e5m2": 3, "fp32": 4}},
        "families": {"gemm": "tree"},
        "feature_columns": {"gemm": cols},
        "metrics": {"gemm": metrics},
    }
    json.dump(manifest, open(os.path.join(data_dir, "manifest.json"), "w"), indent=2)
    return metrics
