import numpy as np
import pandas as pd

from syssim.cli import main


def test_calibrate_subcommand_builds_bundle(tmp_path, capsys):
    prof = tmp_path / "prof"; prof.mkdir()
    rng = np.random.default_rng(0); n = 200
    M = rng.integers(128, 8192, n); K = rng.integers(128, 8192, n); N = rng.integers(128, 8192, n)
    t_an = 2.0 * M * K * N / 1979e12 * 1e9
    pd.DataFrame({"M": M, "K": K, "N": N, "dtype": "bf16", "direction": "fwd", "tp": 1,
                  "tensor_ns": t_an, "mem_ns": t_an * 0.1,
                  "latency_ns": t_an / rng.uniform(0.6, 0.9, n)}).to_parquet(prof / "gemm.parquet")
    rc = main(["calibrate", "--device", "gh200", "--data", str(tmp_path), "--families", "gemm"])
    assert rc == 0
    assert (tmp_path / "manifest.json").exists()


def test_profile_dry_run_builds_worklist(capsys):
    rc = main(["profile", "--device", "gh200", "--out", "/tmp/ignore",
               "--families", "gemm,attention,normalization,elementwise,reduction",
               "--num-workers", "4", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "work-list" in out
    for fam in ("gemm", "attention", "normalization", "elementwise", "reduction"):
        assert fam in out          # per-family counts shown
