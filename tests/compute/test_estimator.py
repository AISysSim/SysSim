"""Unit tests for the pluggable estimator interface."""

import torch

from syssim.compute.estimator import Estimator, RooflineEstimator
from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType
from syssim.compute.predictor.analytical import analytical_bound


def _hw():
    return HardwareInfo(
        peak_tflops_mm=1979.0, peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )


def test_roofline_estimator_conforms_to_protocol():
    est = RooflineEstimator(_hw())
    assert isinstance(est, Estimator)          # runtime-checkable protocol
    assert hasattr(est, "estimate_op")


class _StubEstimator:
    def __init__(self, ms: float):
        self.ms = ms

    def estimate_op(self, func_packet, args, kwargs, out, op_type,
                    execution_mode=None, cache_seq_len=0):
        return self.ms


def test_build_estimator_defaults_to_roofline():
    hw = _hw()
    est = hw.build_estimator()
    assert isinstance(est, RooflineEstimator)
    assert hw.build_estimator() is est            # cached: same instance


def test_build_estimator_returns_custom_when_set():
    stub = _StubEstimator(3.0)
    hw = HardwareInfo(
        peak_tflops_mm=1979.0, peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0, estimator=stub,
    )
    assert hw.build_estimator() is stub


def test_estimate_runtime_delegates_to_hw_estimator():
    # The transparent boundary the tracer calls dispatches to hw_info's estimator.
    from syssim.compute.compute_cost_predictor import estimate_runtime
    from syssim.operator_graph import OperatorType
    hw = HardwareInfo(
        peak_tflops_mm=1979.0, peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0, estimator=_StubEstimator(5.0),
    )
    assert estimate_runtime(None, (), {}, None, hw, OperatorType.MATH) == 5.0


def test_roofline_estimator_returns_pure_analytical_bound():
    aten = torch.ops.aten
    hw = HardwareInfo(peak_tflops_mm=1979.0, peak_tflops_math=989.0,
                      peak_memory_bandwidth_gbps=3350.0)
    a = torch.empty(4096, 4096, dtype=torch.bfloat16, device="cpu")
    b = torch.empty(4096, 4096, dtype=torch.bfloat16, device="cpu")
    out = torch.empty(4096, 4096, dtype=torch.bfloat16, device="cpu")
    est = RooflineEstimator(hw)
    ms = est.estimate_op(aten.mm, (a, b), {}, out, OperatorType.GEMM)
    expected_ms = analytical_bound(
        aten.mm, (a, b), {}, out, hw, OperatorType.GEMM).t_an_ns / 1e6
    assert ms == expected_ms          # no efficiency divide
    assert ms > 0
