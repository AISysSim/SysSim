import json
import os

import numpy as np
import pandas as pd

from syssim.cli import main

_HW = os.path.join(os.path.dirname(__file__), "..", "examples", "configs", "hardware",
                   "isambard_gh200_4gpu.yaml")


def _gemm_row(m, k, n, latency_ns):
    """One profile.parquet row for an aten::mm [m,k]x[k,n]->[m,n] in bf16 (new schema: op + signature
    JSON of args/kwargs/out, plus count + per-instance GPU ns)."""
    bf16 = "torch.bfloat16"
    return {
        "op": "mm",
        "count": 1,
        "per_instance_ns": latency_ns,
        "signature": json.dumps({
            "args": [{"t": [m, k], "dt": bf16}, {"t": [k, n], "dt": bf16}],
            "kwargs": {},
            "out": {"t": [m, n], "dt": bf16},
        }),
    }


def test_calibrate_subcommand_builds_bundle(tmp_path):
    """calibrate reads <data>/profile.parquet (op-signature rows) and writes a per-family residual
    model + manifest."""
    rng = np.random.default_rng(0)
    n = 200
    m = rng.integers(128, 8192, n); k = rng.integers(128, 8192, n); nn = rng.integers(128, 8192, n)
    anchor_ns = 2.0 * m * k * nn / 1979e12 * 1e9               # roofline-ish anchor
    latency_ns = anchor_ns / rng.uniform(0.6, 0.9, n)         # measured = anchor / efficiency
    rows = [_gemm_row(int(m[i]), int(k[i]), int(nn[i]), float(latency_ns[i])) for i in range(n)]
    pd.DataFrame(rows).to_parquet(tmp_path / "profile.parquet")

    rc = main(["calibrate", "--data", str(tmp_path), "--hardware", _HW, "--families", "gemm"])
    assert rc == 0
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "gemm_model.lgb").exists()


def test_profile_dry_run_builds_jobs(capsys):
    """profile --dry-run builds the (layer config x tensor-parallel) job list from the spec without
    touching the GPU."""
    rc = main(["profile", "--out", str("/tmp/ignore"), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "job" in out.lower()
    assert "configs" in out
