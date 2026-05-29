import json
import os
import shutil

import pytest

from syssim.profiling.calibrate import calibrate_family

# Calibrate against the committed reference parquets (copied to a tmp dir so the
# real bundle is not clobbered). Asserts calibrate_family runs + emits p95 per family.
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gh200")
FAMILIES = ["attention", "normalization", "elementwise", "reduction"]


@pytest.mark.parametrize("fam", FAMILIES)
def test_calibrate_family_writes_model_and_p95(fam, tmp_path):
    src = os.path.join(DATA, "prof", f"{fam}.parquet")
    if not os.path.exists(src):
        pytest.skip(f"no reference parquet for {fam}")
    prof = tmp_path / "prof"; prof.mkdir()
    shutil.copy(src, prof / f"{fam}.parquet")
    metrics = calibrate_family(fam, data_dir=str(tmp_path), device="gh200", sfu_peak=247.25)
    assert os.path.exists(tmp_path / f"{fam}_model.lgb")
    assert "p95_ape" in metrics and "median_ape" in metrics and "mean_signed_log_error" in metrics
    m = json.load(open(tmp_path / "manifest.json"))
    assert m["families"][fam] == "tree"
    assert m["feature_columns"][fam]                  # non-empty per-family columns
