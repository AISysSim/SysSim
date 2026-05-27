"""Unit tests for the pluggable estimator interface."""

from syssim.compute.estimator import Estimator, RooflineEstimator
from syssim.config import HardwareInfo


def _hw():
    return HardwareInfo(
        peak_tflops_mm=1979.0, peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )


def test_roofline_estimator_conforms_to_protocol():
    est = RooflineEstimator(_hw())
    assert isinstance(est, Estimator)          # runtime-checkable protocol
    assert hasattr(est, "estimate_op")
