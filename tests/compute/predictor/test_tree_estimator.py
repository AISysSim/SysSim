import math
import os

import torch

from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType
from syssim.compute.tree_estimator import TreeEstimator
from syssim.compute.predictor.roofline import roofline
from syssim.compute.predictor.tree_model import TreeModel

aten = torch.ops.aten
FIX = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "predictors", "gh200_tiny")
HW = HardwareInfo(peak_tflops_mm=1979.0, peak_tflops_math=989.0,
                  peak_memory_bandwidth_gbps=3350.0)


def _mm(m, k, n):
    return (aten.mm,
            (torch.empty(m, k, dtype=torch.bfloat16, device="cpu"),
             torch.empty(k, n, dtype=torch.bfloat16, device="cpu")),
            {}, torch.empty(m, n, dtype=torch.bfloat16, device="cpu"))


def test_estimate_op_returns_finite_ms_and_respects_rail():
    est = TreeEstimator.load(FIX, HW)
    fp, args, kw, out = _mm(4096, 4096, 4096)
    ms = est.estimate_op(fp, args, kw, out, OperatorType.GEMM)
    assert ms > 0 and math.isfinite(ms)
    # OOD rail: never below the roofline (launch-inclusive) bound.
    rb = roofline(fp, args, kw, out, HW, OperatorType.GEMM).roofline_ns / 1e6
    assert ms >= rb - 1e-12


def test_missing_family_falls_back_to_roofline():
    est = TreeEstimator.load(FIX, HW)
    est._model.models.clear()            # simulate an uncalibrated model
    fp, args, kw, out = _mm(2048, 2048, 2048)
    ms = est.estimate_op(fp, args, kw, out, OperatorType.GEMM)
    assert ms > 0 and math.isfinite(ms)


def test_never_raises_on_bad_input():
    est = TreeEstimator.load(FIX, HW)
    # malformed args should not raise; returns a finite (possibly 0) ms
    ms = est.estimate_op(aten.mm, (), {}, None, OperatorType.GEMM)
    assert math.isfinite(ms)


def test_rail_clamps_to_launch_inclusive_roofline():
    """The OOD rail floors at roofline_ns, which includes the launch floor. A
    negative-residual model must NOT pull a tiny launch-bound op below that floor
    (regression: the rail previously clamped to the launch-less hw roofline)."""

    class _NegBooster:                   # always predicts a large negative residual
        def predict(self, x):
            return [-20.0]

    launch = 1.0e5                       # 100 us launch floor for elementwise
    x = torch.empty(1024, dtype=torch.bfloat16, device="cpu")
    codes = {"dtype": {"bf16": 0, "fp16": 1, "fp8_e4m3": 2, "fp8_e5m2": 3, "fp32": 4},
             "op_subtype": {"gelu": 1, "silu": 2, "add": 3, "mul": 4}}
    model = TreeModel(device="t", sfu_peak=0.0,
                      t_launch_ns={"elementwise": launch},
                      categorical_codes=codes, feature_columns={},
                      models={"elementwise": _NegBooster()})
    est = TreeEstimator(model, HW)
    ms = est.estimate_op(aten.gelu, (x,), {}, x, OperatorType.MATH)

    rl = roofline(aten.gelu, (x,), {}, x, HW, OperatorType.MATH, t_launch_ns=launch)
    assert rl.roofline_ns == launch                  # launch dominates this tiny op
    assert ms == launch / 1e6                         # pinned at the floor, not below
