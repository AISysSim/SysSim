import json
import os
import shutil

import pytest

from syssim.compute.predictor.bundle import load_bundle

FIX = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "predictors", "gh200_tiny")


def test_load_bundle_reads_manifest_and_models():
    b = load_bundle(FIX)
    assert b.device == "gh200_tiny"
    assert b.sfu_peak == 247.25
    assert b.t_launch_ns["gemm"] == 5000.0
    assert "gemm" in b.models                 # booster loaded
    assert b.categorical_codes["dtype"]["bf16"] == 0


def test_schema_mismatch_is_hard_error(tmp_path):
    shutil.copytree(FIX, tmp_path / "b")
    m = json.load(open(tmp_path / "b" / "manifest.json"))
    m["schema_version"] = "999"
    json.dump(m, open(tmp_path / "b" / "manifest.json", "w"))
    with pytest.raises(ValueError, match="schema"):
        load_bundle(str(tmp_path / "b"))


def test_partial_bundle_missing_family_is_ok(tmp_path):
    shutil.copytree(FIX, tmp_path / "b")
    os.remove(tmp_path / "b" / "gemm_model.lgb")
    m = json.load(open(tmp_path / "b" / "manifest.json"))
    m["families"] = {}                         # no tree families present
    json.dump(m, open(tmp_path / "b" / "manifest.json", "w"))
    b = load_bundle(str(tmp_path / "b"))
    assert "gemm" not in b.models              # falls back to analytical at predict time
