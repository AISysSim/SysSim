"""Calibrate (CPU) from in-context layer profiles: fit one residual tree per family from the
real per-op signatures+times captured by measure_layer, then merge into the bundle manifest.

This is the in-context counterpart to calibrate.py. Instead of reconstructing ops from synthetic
per-family shape templates, it reconstructs the EXACT op the profiler saw — operator, per-arg
shapes+dtypes (incl. fp32 upcasts and bool masks), kwargs, and output — as meta tensors, then runs
the SAME route/roofline/featurize the simulator uses at inference. Train-time and inference-time
feature rows are therefore identical by construction, and the learned residual log(t_real/anchor)
reflects the true in-context kernel efficiency (which isolated microbenchmarks mispredict for the
explicit-attention memory-bound ops). No GPU.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import lightgbm as lgb
import torch

from ..compute.predictor import features as F
from ..compute.predictor.roofline import roofline, _GEMM_OPS, _ATTN_OPS, _IGNORE_OPS
from ..compute.predictor.router import route, Family
from ..operator_graph import OperatorType
from .calibrate import _categorical_codes

aten = torch.ops.aten

_DT = {
    "torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16,
    "torch.float32": torch.float32, "torch.float64": torch.float64,
    "torch.bool": torch.bool, "torch.uint8": torch.uint8, "torch.int8": torch.int8,
    "torch.int32": torch.int32, "torch.int64": torch.int64, "torch.long": torch.int64,
    "torch.float8_e4m3fn": torch.float8_e4m3fn, "torch.float8_e5m2": torch.float8_e5m2,
}


def _mk(elem):
    """Reconstruct one serialized arg/kwarg/output element as a meta tensor / scalar / list."""
    if not isinstance(elem, dict):
        return elem
    if "t" in elem:
        return torch.empty([int(d) for d in elem["t"]],
                           dtype=_DT.get(elem.get("dt"), torch.bfloat16), device="meta")
    if "seq" in elem:
        return [_mk(e) for e in elem["seq"]]
    return elem.get("v")


def _reconstruct(sig: dict):
    func = getattr(aten, sig["op"])
    args = tuple(_mk(a) for a in sig.get("args", []))
    kwargs = {k: _mk(v) for k, v in sig.get("kwargs", {}).items()}
    out = _mk(sig["out"])
    return func, args, kwargs, out


def _op_type(func_packet) -> OperatorType:
    if func_packet in _GEMM_OPS:
        return OperatorType.GEMM
    if func_packet in _ATTN_OPS:
        return OperatorType.ATTN
    return OperatorType.MATH


def _build_hw_info(hw_yaml: str):
    from ..training.spec import load_hardware_yaml
    from ..config import HardwareInfo
    hw = load_hardware_yaml(hw_yaml)
    return HardwareInfo(
        peak_tflops_mm=hw.peak_tflops_mm, peak_tflops_math=hw.peak_tflops_math,
        peak_memory_bandwidth_gbps=hw.peak_memory_bandwidth_GBps,
        peak_tflops_mm_fp8=hw.peak_tflops_mm_fp8, peak_tflops_mm_fp4=hw.peak_tflops_mm_fp4,
        sfu_peak=hw.sfu_peak)


def _load_rows(json_paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in json_paths:
        rows.extend(json.load(open(p)))
    return rows


def _routed(rows: list[dict]):
    """Yield (family, func, args, kwargs, out, latency_ns) for every reconstructable row."""
    for r in rows:
        try:
            func, args, kwargs, out = _reconstruct(r)
        except Exception:
            continue
        if func in _IGNORE_OPS:
            continue
        ot = _op_type(func)
        yield route(func, ot).value, func, args, kwargs, out, ot, float(r["per_instance_ns"])


def _encode(feats: list[dict], cols: list[str], codes: dict) -> np.ndarray:
    out = np.zeros((len(feats), len(cols)), dtype=np.float64)
    for i, row in enumerate(feats):
        for j, c in enumerate(cols):
            v = row.get(c, 0.0)
            if c in codes:
                v = codes[c].get(str(v), -1)
            out[i, j] = float(v)
    return out


def _build_codes(rows: list[dict]) -> dict:
    """Categorical codes for the manifest: the base codes (preserved verbatim so already-fit
    families keep their encoding) extended with every op_subtype/variant seen in-context, each
    appended at a fresh integer. The synthetic worklist only covered ~11 op_subtypes; the real
    layer dispatches many more (native_dropout, *_backward, embedding_dense_backward, baddbmm, ...)
    which would otherwise all collide to code -1 and become indistinguishable to the tree."""
    codes = _categorical_codes()
    subs = codes["op_subtype"]
    nxt = max(subs.values(), default=-1) + 1
    for _fam, func, _a, _k, _o, _ot, _lat in _routed(rows):
        name = str(func).rsplit(".", 1)[-1]
        if name not in subs:
            subs[name] = nxt
            nxt += 1
    return codes


def calibrate_family_from_rows(family: str, rows: list[dict], *, data_dir: str,
                               hw_info, codes: dict, device: str = "gh200") -> dict[str, Any]:
    """Fit `family`'s residual tree from in-context rows; merge its entry into the manifest."""
    items = [(func, args, kwargs, out, ot, lat)
             for fam, func, args, kwargs, out, ot, lat in _routed(rows) if fam == family]
    if not items:
        raise ValueError(f"no rows routed to family {family}")
    latency = np.array([it[5] for it in items], dtype=float)
    t_launch = float(latency.min())

    fam = Family(family)
    feats, anchors = [], []
    for func, args, kwargs, out, ot, _lat in items:
        rl = roofline(func, args, kwargs, out, hw_info, ot, t_launch_ns=t_launch)
        feats.append(F.featurize(func, args, kwargs, out, fam, rl))
        anchors.append(rl.roofline_ns)
    anchor = np.asarray(anchors, dtype=float)
    keep = anchor > 0
    if not keep.all():
        items = [it for it, k in zip(items, keep) if k]
        feats = [f for f, k in zip(feats, keep) if k]
        anchor = anchor[keep]
        latency = latency[keep]

    y = np.log(latency) - np.log(np.maximum(anchor, 1e-12))
    cols = F.feature_columns(fam)
    X = _encode(feats, cols, codes)

    # Held-out split by unique op signature (op_subtype + shape), seeded; no shape leaks.
    sig_keys = [(feats[i].get("op_subtype", feats[i].get("variant", "")),
                 feats[i].get("log_anchor_ns", 0.0)) for i in range(len(feats))]
    rng = np.random.default_rng(0)
    uniq = sorted(set(sig_keys), key=repr)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * 0.15)))
    val_keys = {uniq[int(i)] for i in perm[:n_val]}
    is_val = np.array([k in val_keys for k in sig_keys])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    if len(tr) == 0 or len(va) == 0:
        tr = va = np.arange(len(items))

    n = len(tr)
    cat_cols = [c for c in cols if c in F.CATEGORICAL]
    Xdf_cols = list(cols)
    import pandas as pd
    Xtr = pd.DataFrame(X[tr], columns=Xdf_cols)
    Xva = pd.DataFrame(X[va], columns=Xdf_cols)
    # The in-context residual is ~constant within an op_subtype (e.g. masked_fill std 0.05), so the
    # ideal model is one leaf per op plus a few size bins. Keep min_data_in_leaf small so 10-16
    # sample ops can isolate (a per-op constant cannot overfit a flat target), and give enough
    # leaves to cover all op_subtypes.
    params = {"objective": "regression", "metric": "mae", "verbose": -1,
              "learning_rate": 0.05, "num_leaves": max(15, min(127, n // 2)),
              "min_data_in_leaf": 2}
    booster = lgb.train(params, lgb.Dataset(Xtr, label=y[tr], categorical_feature=cat_cols),
                        num_boost_round=300,
                        valid_sets=[lgb.Dataset(Xva, label=y[va], categorical_feature=cat_cols)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    booster.save_model(os.path.join(data_dir, f"{family}_model.lgb"))

    pred = np.maximum(anchor[va] * np.exp(booster.predict(Xva)), anchor[va])
    ape = np.abs(pred - latency[va]) / np.maximum(latency[va], 1e-12)
    metrics = {"median_ape": float(np.median(ape)), "p95_ape": float(np.percentile(ape, 95)),
               "n_train": int(n), "n_val": int(len(va)),
               "mean_signed_log_error": float(np.mean(np.log(pred) - np.log(latency[va])))}

    man_path = os.path.join(data_dir, "manifest.json")
    m = json.load(open(man_path)) if os.path.exists(man_path) else {
        "device": device, "lightgbm_version": lgb.__version__,
        "t_launch_ns": {}, "families": {}, "feature_columns": {}, "metrics": {}}
    m["schema_version"] = F.SCHEMA_VERSION
    m["categorical_codes"] = codes
    m.setdefault("t_launch_ns", {})[family] = t_launch
    m.setdefault("families", {})[family] = "tree"
    m.setdefault("feature_columns", {})[family] = cols
    m.setdefault("metrics", {})[family] = metrics
    json.dump(m, open(man_path, "w"), indent=2)
    return metrics


def _rows_from_parquet(parquet_path: str) -> list[dict]:
    """Read profile.parquet (model, op, count, per_instance_ns, signature=JSON{args,kwargs,out})
    into the row dicts _routed/_reconstruct consume (op + args + kwargs + out + latency)."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    rows = []
    for op, sig_json, cnt, ns in zip(df["op"], df["signature"], df["count"], df["per_instance_ns"]):
        sig = json.loads(sig_json)
        rows.append({"op": op, "args": sig.get("args", []), "kwargs": sig.get("kwargs", {}),
                     "out": sig.get("out"), "count": int(cnt), "per_instance_ns": float(ns)})
    return rows


def calibrate_from_parquet(parquet_path: str, *, data_dir: str, hardware: str,
                           families: list[str], device: str = "gh200") -> dict[str, dict]:
    """Fit each family's residual tree from <data>/profile.parquet and merge into the manifest."""
    rows = _rows_from_parquet(parquet_path)
    print(f"calibrate: loaded {len(rows)} (op, shape) rows from {parquet_path}", flush=True)
    hw = _build_hw_info(hardware)
    codes = _build_codes(rows)
    print(f"calibrate: built feature codes ({len(codes.get('op_subtype', {}))} op subtypes); "
          f"families = {', '.join(families)}", flush=True)
    os.makedirs(data_dir, exist_ok=True)
    metrics = {}
    for family in families:
        print(f"calibrate: fitting {family} ...", flush=True)
        result = calibrate_family_from_rows(
            family, rows, data_dir=data_dir, hw_info=hw, codes=codes, device=device)
        metrics[family] = result
        print(f"calibrate: {family} done -> median_ape={result['median_ape'] * 100:.1f}%  "
              f"p95={result['p95_ape'] * 100:.1f}%  (train={result['n_train']} val={result['n_val']})",
              flush=True)
    return metrics


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("layer_json", nargs="+", help="measure_layer JSON outputs (one per model)")
    ap.add_argument("--data-dir", default="data/gh200_focused")
    ap.add_argument("--hardware", default="examples/configs/hardware/isambard_gh200_4gpu.yaml")
    ap.add_argument("--families", default="elementwise,reduction")
    args = ap.parse_args()
    rows = _load_rows(args.layer_json)
    hw = _build_hw_info(args.hardware)
    codes = _build_codes(rows)
    for family in args.families.split(","):
        family = family.strip()
        mt = calibrate_family_from_rows(family, rows, data_dir=args.data_dir, hw_info=hw, codes=codes)
        print("  %-14s median_ape=%.1f%% p95=%.1f%% (train=%d val=%d)"
              % (family, mt["median_ape"] * 100, mt["p95_ape"] * 100, mt["n_train"], mt["n_val"]))


if __name__ == "__main__":
    main()
