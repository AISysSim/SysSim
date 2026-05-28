import math
import os

import torch

from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType
from syssim.compute.predictor import HybridEstimator

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
    est = HybridEstimator.load(FIX, HW)
    fp, args, kw, out = _mm(4096, 4096, 4096)
    ms = est.estimate_op(fp, args, kw, out, OperatorType.GEMM)
    assert ms > 0 and math.isfinite(ms)
    # OOD rail: never below the hw roofline bound
    from syssim.compute.predictor.analytical import analytical_bound
    rb = analytical_bound(fp, args, kw, out, HW, OperatorType.GEMM).roofline_hw_ns / 1e6
    assert ms >= rb - 1e-12


def test_missing_family_falls_back_to_analytical():
    est = HybridEstimator.load(FIX, HW)
    est._bundle.models.clear()           # simulate partial bundle
    fp, args, kw, out = _mm(2048, 2048, 2048)
    ms = est.estimate_op(fp, args, kw, out, OperatorType.GEMM)
    assert ms > 0 and math.isfinite(ms)


def test_never_raises_on_bad_input():
    est = HybridEstimator.load(FIX, HW)
    # malformed args should not raise; returns a finite (possibly 0) ms
    ms = est.estimate_op(aten.mm, (), {}, None, OperatorType.GEMM)
    assert math.isfinite(ms)
