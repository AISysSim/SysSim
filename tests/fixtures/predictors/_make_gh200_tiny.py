# tests/fixtures/predictors/_make_gh200_tiny.py — run once; commit the outputs.
# Builds a tiny committed fixture bundle (1-feature GEMM residual booster + a
# minimal manifest) for the bundle / hybrid-estimator unit tests.
import json
import os

import numpy as np
import lightgbm as lgb

here = os.path.join(os.path.dirname(__file__), "gh200_tiny")
os.makedirs(os.path.join(here, "prof"), exist_ok=True)
# Tiny GEMM residual model: 1 feature, trivial data.
X = np.array([[10.0], [12.0], [14.0]])
y = np.array([0.2, 0.25, 0.3])
ds = lgb.Dataset(X, label=y)
booster = lgb.train({"objective": "regression", "num_leaves": 3,
                     "min_data_in_leaf": 1, "verbose": -1}, ds, num_boost_round=2)
booster.save_model(os.path.join(here, "gemm_model.lgb"))
manifest = {
    "device": "gh200_tiny", "schema_version": "1.0.0-gemm",
    "lightgbm_version": lgb.__version__,
    "env": {"driver": "test", "cuda": "test", "torch": "test"},
    "spec_hash": "test", "sfu_peak": 247.25,
    "t_launch_ns": {"gemm": 5000.0},
    "categorical_codes": {"dtype": {"bf16": 0, "fp16": 1, "fp8_e4m3": 2, "fp8_e5m2": 3, "fp32": 4}},
    "families": {"gemm": "tree"},
    "feature_columns": {"gemm": ["log_anchor_ns"]},
}
json.dump(manifest, open(os.path.join(here, "manifest.json"), "w"), indent=2)
print("wrote", here)
