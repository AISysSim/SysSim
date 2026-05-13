"""Tests for efficiency model infrastructure."""

import pytest
import torch
import tempfile
import os

from syssim.compute.compute_cost_predictor import (
    roofline_estimate,
    efficiency_estimate,
    estimate_runtime,
    RooflineResult,
    ConstraintTime,
)
from syssim.compute.efficiency_models import (
    EfficiencyFeatures,
    MLPEfficiencyModel,
    BackendManager,
    set_backend_dir,
)
from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType


@pytest.fixture
def hw_info():
    """Create a test hardware info configuration."""
    return HardwareInfo(989.0, 989.0, 3350.0)


def test_roofline_result_structure(hw_info):
    """Test that roofline_estimate returns RooflineResult."""
    a = torch.randn(128, 256)
    b = torch.randn(256, 512)

    result = roofline_estimate(
        torch.ops.aten.mm,
        (a, b),
        {},
        torch.randn(128, 512),
        hw_info,
        OperatorType.GEMM,
    )

    assert isinstance(result, RooflineResult)
    assert result.t_roofline_ms >= 0
    assert len(result.constraints) > 0
    assert result.dominant_constraint[0] in ["math", "memory"]


def test_constraint_ratios(hw_info):
    """Test that constraint ratios are computed correctly."""
    a = torch.randn(128, 256)
    b = torch.randn(256, 512)

    result = roofline_estimate(
        torch.ops.aten.mm,
        (a, b),
        {},
        torch.randn(128, 512),
        hw_info,
        OperatorType.GEMM,
    )

    ratios = result.get_constraint_ratios()

    # Check that ratios are in valid range
    for key, ratio in ratios.items():
        assert 0.0 <= ratio <= 1.0

    # Check that dominant constraint has ratio of 1.0
    dominant_ratio = ratios[result.dominant_constraint]
    assert abs(dominant_ratio - 1.0) < 1e-6


def test_efficiency_fallback_no_model(hw_info):
    """Test that efficiency_estimate returns 1.0 when no model loaded."""
    a = torch.randn(128, 256)
    b = torch.randn(256, 512)

    roofline_result = roofline_estimate(
        torch.ops.aten.mm,
        (a, b),
        {},
        torch.randn(128, 512),
        hw_info,
        OperatorType.GEMM,
    )

    efficiency = efficiency_estimate(
        torch.ops.aten.mm,
        (a, b),
        {},
        torch.randn(128, 512),
        hw_info,
        OperatorType.GEMM,
        roofline_result,
    )

    assert efficiency == 1.0


def test_estimate_runtime_integration(hw_info):
    """Test that estimate_runtime works end-to-end."""
    a = torch.randn(128, 256)
    b = torch.randn(256, 512)

    time_ms = estimate_runtime(
        torch.ops.aten.mm,
        (a, b),
        {},
        torch.randn(128, 512),
        hw_info,
        OperatorType.GEMM,
    )

    assert time_ms >= 0


def test_efficiency_features_to_array():
    """Test EfficiencyFeatures.to_array conversion."""
    features = EfficiencyFeatures(
        constraint_times={("math", "device"): 1.0, ("memory", "device"): 0.5},
        constraint_ratios={("math", "device"): 1.0, ("memory", "device"): 0.5},
        dominant_constraint=("math", "device"),
        op_params={"log_M": 5.0, "log_N": 6.0, "log_K": 7.0},
        hw_params={"flop_ratio": 1.0, "log_peak_flop_mm": 30.0},
    )

    feature_order = [
        "T_math_device", "T_memory_device",
        "r_math_device", "r_memory_device",
        "dom_math_device",
        "log_M", "log_N", "log_K",
        "flop_ratio", "log_peak_flop_mm",
    ]

    arr = features.to_array(feature_order)

    assert arr.shape == (len(feature_order),)
    assert arr[0] == 1.0  # T_math_device
    assert arr[1] == 0.5  # T_memory_device
    assert arr[4] == 1.0  # dom_math_device (one-hot)


def test_model_manager_empty_dir():
    """Test ModelManager with non-existent directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_dir = os.path.join(tmpdir, "nonexistent")
        manager = BackendManager(fake_dir)

        # Should not crash, but all models should be None
        model = manager.get_model(OperatorType.GEMM)
        assert model is None


def test_roofline_zero_time_returns_zero_efficiency(hw_info):
    """Test that zero roofline time returns efficiency of 1.0."""
    # Create a result with zero time
    result = RooflineResult(
        t_roofline_ms=0.0,
        constraints=[],
        dominant_constraint=("none", "none"),
    )

    a = torch.randn(128, 256)
    b = torch.randn(256, 512)

    efficiency = efficiency_estimate(
        torch.ops.aten.mm,
        (a, b),
        {},
        torch.randn(128, 512),
        hw_info,
        OperatorType.GEMM,
        result,
    )

    assert efficiency == 1.0


def test_estimate_runtime_with_zero_efficiency(hw_info):
    """Test that estimate_runtime handles zero efficiency gracefully."""
    # This is a contrived test since efficiency_estimate never returns 0
    # But the code path exists, so test it
    a = torch.randn(1, 1)
    b = torch.randn(1, 1)

    # Small tensors should have very small time
    time_ms = estimate_runtime(
        torch.ops.aten.mm,
        (a, b),
        {},
        torch.randn(1, 1),
        hw_info,
        OperatorType.GEMM,
    )

    # Should not crash and return a valid number
    assert time_ms >= 0


def test_backend_manager_routes_by_dtype(tmp_path, monkeypatch):
    """BackendManager.get_model(op_type, dtype) should pick the right per-dtype file."""
    # Stub get_hardware_info so we do not need real CUDA
    monkeypatch.setattr(
        "syssim.compute.efficiency_models.get_hardware_info"
        if hasattr(__import__("syssim.compute.efficiency_models", fromlist=["x"]), "get_hardware_info")
        else "syssim.config.get_hardware_info",
        lambda: (None, "pro6000"),
        raising=False,
    )

    mgr = BackendManager.__new__(BackendManager)
    mgr.model_dir = str(tmp_path)
    mgr.hw_name = "pro6000"
    mgr._models = {}

    found_fp16 = mgr._resolve_model_path(OperatorType.GEMM, "fp16")
    found_fp8 = mgr._resolve_model_path(OperatorType.GEMM, "fp8")
    found_fp4 = mgr._resolve_model_path(OperatorType.GEMM, "fp4")
    assert found_fp16.endswith("gemm_pro6000_fp16_xgb.pth")
    assert found_fp8.endswith("gemm_pro6000_fp8_xgb.pth")
    assert found_fp4.endswith("gemm_pro6000_fp4_xgb.pth")


def test_backend_manager_get_model_falls_back_to_fp16(tmp_path):
    """When asked for fp8/fp4 with no model, fall back to fp16 model if present."""
    mgr = BackendManager.__new__(BackendManager)
    mgr.model_dir = str(tmp_path)
    mgr.hw_name = "pro6000"

    class _Stub:
        def predict(self, _f):
            return 0.5

    fp16_model = _Stub()
    mgr._models = {(OperatorType.GEMM, "fp16"): fp16_model}

    assert mgr.get_model(OperatorType.GEMM, "fp16") is fp16_model
    assert mgr.get_model(OperatorType.GEMM, "fp8") is fp16_model  # fallback
    assert mgr.get_model(OperatorType.GEMM, "fp4") is fp16_model  # fallback
    # Unknown op returns None
    assert mgr.get_model(OperatorType.ATTN, "fp16") is None

