import json
import os

import numpy as np
import pandas as pd

from syssim.profiling.calibrate import calibrate_gemm


def _toy_rows(n=400):
    rng = np.random.default_rng(0)
    M = rng.integers(128, 8192, n); K = rng.integers(128, 8192, n); N = rng.integers(128, 8192, n)
    flops = 2.0 * M * K * N
    t_an_ns = flops / (1979e12) * 1e9            # tensor-bound anchor (ns)
    # Learnable efficiency: a smooth function of GEMM size, so the residual model
    # has real signal to fit and the gate is meaningful (not split-fragile noise).
    size = np.log(M.astype(float) * N * K)
    eff = 0.55 + 0.35 / (1.0 + np.exp(-(size - 20.0)))   # residual = -log(eff)
    latency_ns = t_an_ns / eff
    return pd.DataFrame({"M": M, "K": K, "N": N, "dtype": "bf16",
                         "direction": "fwd", "tp": 1,
                         "tensor_ns": t_an_ns, "mem_ns": t_an_ns * 0.1,
                         "latency_ns": latency_ns})


def test_calibrate_writes_bundle_and_low_error(tmp_path):
    prof = tmp_path / "prof"; prof.mkdir()
    _toy_rows().to_parquet(prof / "gemm.parquet")
    metrics = calibrate_gemm(data_dir=str(tmp_path), device="gh200",
                             sfu_peak=247.25)
    assert os.path.exists(tmp_path / "gemm_model.lgb")
    m = json.load(open(tmp_path / "manifest.json"))
    assert m["schema_version"]
    assert m["families"]["gemm"] == "tree"
    assert metrics["median_ape"] < 0.10          # in-distribution gate
    assert abs(metrics["mean_signed_log_error"]) < 0.03   # bias gate
