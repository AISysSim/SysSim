"""Calibrate (CPU): fit ONE regularized LightGBM residual tree per family from the
committed profiling parquets, then merge that family's entry into the bundle manifest.
No GPU.

Features come from the SAME features.featurize used at inference: each parquet row's op
is reconstructed with meta tensors and a Roofline rebuilt from the recorded anchor
components, so train-time and inference-time feature rows are identical (parity). One
uniform strategy for all families — a residual tree, regularized to the row count, which
subsumes a near-constant fit for the memory-bound families without overfitting."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch

from ..compute.predictor import features as F
from ..compute.predictor.router import Family
from ..compute.predictor.roofline import Roofline

_DTYPE_CODES = {"bf16": 0, "fp16": 1, "fp8_e4m3": 2, "fp8_e5m2": 3, "fp32": 4}
_META = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8_e4m3": torch.float8_e4m3fn,
         "fp8_e5m2": torch.float8_e5m2, "fp32": torch.float32}
_METRIC_COLS = {"latency_ns", "tensor_ns", "mem_ns", "fma_ns", "sfu_ns"}


def _categorical_codes() -> dict:
    return {
        "dtype": dict(_DTYPE_CODES),
        "op_subtype": {"native_layer_norm": 0, "gelu": 1, "silu": 2, "add": 3, "mul": 4,
                       "_softmax": 5, "_log_softmax": 6, "_fused_rms_norm": 7,
                       "copy_": 8, "masked_fill": 9, "masked_fill_": 10},
        "variant": {"_scaled_dot_product_flash_attention": 0,
                    "_scaled_dot_product_efficient_attention": 1,
                    "_scaled_dot_product_cudnn_attention": 2},
    }


def _reconstruct_op(family: str, row: dict):
    """Rebuild (func_packet, args, out) with meta tensors from a parquet row's shapes,
    so featurize() yields the EXACT inference feature row."""
    aten = torch.ops.aten
    dt = _META.get(row.get("dtype"), torch.float32)
    if family == "gemm":
        M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
        b = row.get("batch", 1)
        batch = 1 if b is None or (isinstance(b, float) and b != b) else int(b)
        if batch > 1:    # batched attention score matmul -> aten.bmm
            return (aten.bmm,
                    (torch.empty(batch, M, K, dtype=dt, device="meta"),
                     torch.empty(batch, K, N, dtype=dt, device="meta")),
                    torch.empty(batch, M, N, dtype=dt, device="meta"))
        return (aten.mm,
                (torch.empty(M, K, dtype=dt, device="meta"),
                 torch.empty(K, N, dtype=dt, device="meta")),
                torch.empty(M, N, dtype=dt, device="meta"))
    if family == "attention":
        B, Hq, Hkv = int(row["B"]), int(row["H_q"]), int(row["H_kv"])
        S, D = int(row["S"]), int(row["D"])
        q = torch.empty(B, Hq, S, D, dtype=dt, device="meta")
        k = torch.empty(B, Hkv, S, D, dtype=dt, device="meta")
        v = torch.empty(B, Hkv, S, D, dtype=dt, device="meta")
        return (aten._scaled_dot_product_flash_attention,
                (q, k, v, 0.0, bool(row["causal"])), q)
    if family == "normalization":
        tokens, hidden = int(row["tokens"]), int(row["hidden"])
        x = torch.empty(tokens, hidden, dtype=dt, device="meta")
        w = torch.empty(hidden, dtype=dt, device="meta")
        # Distinguish rmsnorm from layernorm so featurize's op_subtype differs — they
        # have different latencies and must not be conflated under one func.
        if row.get("op_subtype") == "rmsnorm" and hasattr(aten, "_fused_rms_norm"):
            return aten._fused_rms_norm, (x, [hidden], w, 1e-6), x
        bias = None if row.get("op_subtype") == "rmsnorm" else torch.empty(hidden, dtype=dt, device="meta")
        return aten.native_layer_norm, (x, [hidden], w, bias, 1e-5), x
    if family == "elementwise":
        total = int(row["total_elements"])
        x = torch.empty(total, dtype=dt, device="meta")
        sub = row.get("op_subtype", "gelu")
        if sub == "copy_":
            return aten.copy_, (torch.empty(total, dtype=dt, device="meta"), x), x
        if sub in ("masked_fill", "masked_fill_"):
            mask = torch.empty(total, dtype=torch.bool, device="meta")
            func = aten.masked_fill_ if sub == "masked_fill_" else aten.masked_fill
            return func, (x, mask, 0.0), x
        func = {"gelu": aten.gelu, "silu": aten.silu, "add": aten.add, "mul": aten.mul}.get(sub, aten.gelu)
        args = (x, torch.empty(total, dtype=dt, device="meta")) if sub in ("add", "mul") else (x,)
        return func, args, x
    if family == "reduction":
        B, H, S = int(row["B"]), int(row["H"]), int(row["S"])
        x = torch.empty(B, H, S, S, dtype=dt, device="meta")
        return aten._softmax, (x, -1, False), x
    raise ValueError(f"unknown family: {family}")


def _build_features(family: str, df: pd.DataFrame, t_launch: float):
    fam = Family(family)
    feats, anchors = [], []
    for _, r in df.iterrows():
        rd = r.to_dict()
        bound = Roofline(
            tensor_ns=float(rd.get("tensor_ns", 0.0) or 0.0),
            fma_ns=float(rd.get("fma_ns", 0.0) or 0.0),
            sfu_ns=float(rd.get("sfu_ns", 0.0) or 0.0),
            mem_ns=float(rd.get("mem_ns", 0.0) or 0.0),
            launch_ns=t_launch)
        func, args, out = _reconstruct_op(family, rd)
        feats.append(F.featurize(func, args, {}, out, fam, bound))
        anchors.append(bound.roofline_ns)
    return feats, np.asarray(anchors, dtype=float)


def _encode(feats: list, cols: list, codes: dict) -> pd.DataFrame:
    data: dict = {c: [] for c in cols}
    for row in feats:
        for c in cols:
            v = row.get(c, 0.0)
            if c in codes:
                v = codes[c].get(str(v), -1)
            data[c].append(float(v))
    return pd.DataFrame(data)


def _shape_keys(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c not in _METRIC_COLS]
    return [tuple(t) for t in df[cols].itertuples(index=False, name=None)]


def calibrate_family(family: str, *, data_dir: str, device: str, sfu_peak: float,
                     target: str = "residual") -> dict[str, Any]:
    """Fit `family`'s residual tree from data_dir/prof/<family>.parquet; merge its entry
    into data_dir/manifest.json (preserving other families). Returns metrics incl. p95."""
    df = pd.read_parquet(os.path.join(data_dir, "prof", f"{family}.parquet"))
    latency = df["latency_ns"].to_numpy()
    t_launch = float(latency.min())
    feats, anchor = _build_features(family, df, t_launch)
    y = np.log(latency) - np.log(np.maximum(anchor, 1e-12))
    cols = F.feature_columns(Family(family))
    codes = _categorical_codes()
    X = _encode(feats, cols, codes)

    # Held-out split by unique shape signature (random, seeded; no shape leaks).
    rng = np.random.default_rng(0)
    keys = _shape_keys(df)
    uniq = sorted(set(keys), key=repr)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * 0.15)))
    val_keys = {uniq[int(i)] for i in perm[:n_val]}
    is_val = np.array([k in val_keys for k in keys])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    if len(tr) == 0 or len(va) == 0:        # too few shapes to split -> evaluate in-sample
        tr = va = np.arange(len(df))

    n = len(tr)
    # op_subtype/dtype/variant are encoded as integer codes but are genuinely categorical;
    # tell LightGBM so it splits per-op (not on code thresholds) — otherwise one residual is
    # averaged across ops of different efficiency (e.g. mem-bound copy_/masked_fill lumped
    # with compute-bound gelu/silu in elementwise).
    cat_cols = [c for c in cols if c in F.CATEGORICAL]
    params = {"objective": "regression", "metric": "mae", "verbose": -1,
              "learning_rate": 0.05,
              "num_leaves": max(3, min(63, n // 4)),         # regularize to row count
              "min_data_in_leaf": max(1, min(20, n // 10))}
    booster = lgb.train(params, lgb.Dataset(X.iloc[tr], label=y[tr], categorical_feature=cat_cols),
                        num_boost_round=300,
                        valid_sets=[lgb.Dataset(X.iloc[va], label=y[va], categorical_feature=cat_cols)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    booster.save_model(os.path.join(data_dir, f"{family}_model.lgb"))

    pred = np.maximum(anchor[va] * np.exp(booster.predict(X.iloc[va])), anchor[va])  # OOD rail
    ape = np.abs(pred - latency[va]) / np.maximum(latency[va], 1e-12)
    metrics = {
        "median_ape": float(np.median(ape)),
        "p95_ape": float(np.percentile(ape, 95)),
        "mean_signed_log_error": float(np.mean(np.log(pred) - np.log(latency[va]))),
    }

    man_path = os.path.join(data_dir, "manifest.json")
    m = json.load(open(man_path)) if os.path.exists(man_path) else {
        "device": device, "lightgbm_version": lgb.__version__, "sfu_peak": sfu_peak,
        "t_launch_ns": {}, "families": {}, "feature_columns": {}, "metrics": {}}
    m["schema_version"] = F.SCHEMA_VERSION
    m["categorical_codes"] = codes
    m.setdefault("t_launch_ns", {})[family] = t_launch
    m.setdefault("families", {})[family] = "tree"
    m.setdefault("feature_columns", {})[family] = cols
    m.setdefault("metrics", {})[family] = metrics
    json.dump(m, open(man_path, "w"), indent=2)
    return metrics


def calibrate_gemm(*, data_dir: str, device: str, sfu_peak: float,
                   target: str = "residual") -> dict[str, Any]:
    """Back-compat wrapper — GEMM via the uniform per-family path."""
    return calibrate_family("gemm", data_dir=data_dir, device=device,
                            sfu_peak=sfu_peak, target=target)
