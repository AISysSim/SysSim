import json
import os
import shutil

import pytest

from syssim.compute.predictor.tree_model import load_tree_model

FIX = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "predictors", "gh200_tiny")


def test_load_tree_model_reads_manifest_and_models():
    m = load_tree_model(FIX)
    assert m.device == "gh200_tiny"
    assert m.sfu_peak == 247.25
    assert m.t_launch_ns["gemm"] == 5000.0
    assert "gemm" in m.models                 # booster loaded
    assert m.categorical_codes["dtype"]["bf16"] == 0


def test_schema_mismatch_is_hard_error(tmp_path):
    shutil.copytree(FIX, tmp_path / "b")
    m = json.load(open(tmp_path / "b" / "manifest.json"))
    m["schema_version"] = "999"
    json.dump(m, open(tmp_path / "b" / "manifest.json", "w"))
    with pytest.raises(ValueError, match="schema"):
        load_tree_model(str(tmp_path / "b"))


def test_partial_model_missing_family_is_ok(tmp_path):
    shutil.copytree(FIX, tmp_path / "b")
    os.remove(tmp_path / "b" / "gemm_model.lgb")
    m = json.load(open(tmp_path / "b" / "manifest.json"))
    m["families"] = {}                         # no tree families present
    json.dump(m, open(tmp_path / "b" / "manifest.json", "w"))
    model = load_tree_model(str(tmp_path / "b"))
    assert "gemm" not in model.models          # falls back to the roofline at predict time
