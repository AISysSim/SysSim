"""Tests for roofline efficiency bug fix (>100% efficiency).

Verifies:
1. Size-based peak selection (large vs small ops)
2. No efficiency violations in profiled data
3. MLP predictions stay bounded in [0, 1]
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from syssim.compute.compute_cost_predictor import (
    _is_large_tensor_core_op,
    aten,
)
from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType


class TestPeakSelection:
    """Test two-tier peak FLOP/s selection."""

    def test_large_gemm_uses_tensor_core_peak(self):
        """Large GEMMs (all dims ≥ 512) should use tensor unit peak."""
        hw = HardwareInfo(1979.0, 33.5, 4900.0, peak_tflops_mm_conservative=535.0)

        # Create large tensors (2048 x 8192) x (8192 x 2048)
        a = torch.randn(2048, 8192, dtype=torch.float16)
        b = torch.randn(8192, 2048, dtype=torch.float16)

        is_large = _is_large_tensor_core_op(aten.mm, (a, b), OperatorType.GEMM)
        assert is_large, "2048x8192 GEMM should be classified as large"

        peak = hw.get_peak_tflops(OperatorType.GEMM, torch.float16, is_large)
        assert peak == 1979.0, f"Expected peak 1979, got {peak}"

    def test_small_gemm_uses_conservative_peak(self):
        """Small GEMMs (any dim < 512) should use conservative peak."""
        hw = HardwareInfo(1979.0, 33.5, 4900.0, peak_tflops_mm_conservative=535.0)

        # Create small tensors (64 x 64) x (64 x 64)
        a = torch.randn(64, 64, dtype=torch.float16)
        b = torch.randn(64, 64, dtype=torch.float16)

        is_large = _is_large_tensor_core_op(aten.mm, (a, b), OperatorType.GEMM)
        assert not is_large, "64x64 GEMM should be classified as small"

        peak = hw.get_peak_tflops(OperatorType.GEMM, torch.float16, is_large)
        assert peak == 535.0, f"Expected conservative peak 535, got {peak}"

    def test_threshold_boundary(self):
        """Test behavior at threshold boundary (512)."""
        HardwareInfo(1979.0, 33.5, 4900.0, peak_tflops_mm_conservative=535.0)

        # Exactly at threshold: 512 x 512 x 512 (should be large)
        a = torch.randn(512, 512, dtype=torch.float16)
        b = torch.randn(512, 512, dtype=torch.float16)
        is_large = _is_large_tensor_core_op(aten.mm, (a, b), OperatorType.GEMM)
        assert is_large, "512x512 GEMM should be classified as large"

        # Just below threshold: 511 x 512 x 512 (should be small)
        a = torch.randn(511, 512, dtype=torch.float16)
        b = torch.randn(512, 512, dtype=torch.float16)
        is_large = _is_large_tensor_core_op(aten.mm, (a, b), OperatorType.GEMM)
        assert not is_large, "511x512 GEMM should be classified as small"

    def test_addmm_classification(self):
        """Test addmm operator classification."""
        # Large addmm
        bias = torch.randn(1024, 2048, dtype=torch.float16)
        a = torch.randn(1024, 4096, dtype=torch.float16)
        b = torch.randn(4096, 2048, dtype=torch.float16)
        is_large = _is_large_tensor_core_op(aten.addmm, (bias, a, b), OperatorType.GEMM)
        assert is_large

        # Small addmm
        bias = torch.randn(64, 128, dtype=torch.float16)
        a = torch.randn(64, 256, dtype=torch.float16)
        b = torch.randn(256, 128, dtype=torch.float16)
        is_large = _is_large_tensor_core_op(aten.addmm, (bias, a, b), OperatorType.GEMM)
        assert not is_large

    def test_bmm_classification(self):
        """Test batched matmul classification."""
        # Large bmm (batch doesn't affect threshold)
        a = torch.randn(16, 1024, 2048, dtype=torch.float16)
        b = torch.randn(16, 2048, 1024, dtype=torch.float16)
        is_large = _is_large_tensor_core_op(aten.bmm, (a, b), OperatorType.GEMM)
        assert is_large

        # Small bmm
        a = torch.randn(16, 64, 128, dtype=torch.float16)
        b = torch.randn(16, 128, 64, dtype=torch.float16)
        is_large = _is_large_tensor_core_op(aten.bmm, (a, b), OperatorType.GEMM)
        assert not is_large

    def test_backward_compatibility(self):
        """HardwareInfo without conservative peak should use same peak for both."""
        hw = HardwareInfo(989.0, 33.5, 3350.0)  # No conservative peak specified

        # Should use 989.0 for both large and small
        assert hw.get_peak_tflops(OperatorType.GEMM, torch.float16, is_large_op=True) == 989.0
        assert hw.get_peak_tflops(OperatorType.GEMM, torch.float16, is_large_op=False) == 989.0


class TestProfilingDataBounds:
    """Test that profiled data has no efficiency violations."""

    def test_no_efficiency_violations_in_csv(self):
        """Profiled data should have no efficiencies > 100%."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        data_dir = Path(__file__).parent.parent / "data" / "trained_models"
        if not data_dir.exists():
            pytest.skip(f"Data directory not found: {data_dir}")

        csv_files = list(data_dir.glob("*_data.csv"))
        if not csv_files:
            pytest.skip("No CSV data files found")

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            if "efficiency" not in df.columns:
                continue

            bad = df[df["efficiency"] > 1.0]
            assert len(bad) == 0, (
                f"{csv_path.name} has {len(bad)} configs with efficiency > 1.0:\n"
                f"{bad[['M', 'N', 'K', 'efficiency']].head() if 'M' in df.columns else bad.head()}"
            )

            # Also check that all efficiencies are positive
            assert (df["efficiency"] > 0).all(), f"{csv_path.name} has non-positive efficiencies"

    def test_efficiency_statistics(self):
        """Verify efficiency statistics are reasonable."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        data_dir = Path(__file__).parent.parent / "data" / "trained_models"
        gemm_csv = data_dir / "gemm_mlp_gh200_data.csv"

        if not gemm_csv.exists():
            pytest.skip(f"GEMM data file not found: {gemm_csv}")

        df = pd.read_csv(gemm_csv)
        if "efficiency" not in df.columns:
            pytest.skip("No efficiency column in data")

        # Efficiency should be in reasonable range
        assert df["efficiency"].min() > 0, "Min efficiency should be positive"
        assert df["efficiency"].max() <= 1.0, "Max efficiency should be ≤ 1.0"

        # Mean efficiency should be reasonable (typically 10-40%)
        mean_eff = df["efficiency"].mean()
        assert 0.01 < mean_eff < 0.9, f"Mean efficiency {mean_eff:.3f} seems unreasonable"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMLPPredictions:
    """Test that MLP predictions stay bounded."""

    def test_mlp_predictions_bounded(self):
        """MLP with Sigmoid should never predict efficiency > 1."""
        try:
            from syssim.compute.efficiency_models import MLPEfficiencyModel
        except ImportError:
            pytest.skip("efficiency_models not available")

        model_path = Path(__file__).parent.parent / "data" / "trained_models" / "gemm_mlp_gh200.pth"
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            feature_order = checkpoint["feature_order"]
            model_obj = MLPEfficiencyModel(str(model_path), feature_order)
        except Exception as e:
            pytest.skip(f"Failed to load model: {e}")

        # Test on random features (7 features for GEMM)
        num_features = len(feature_order)
        X_test = torch.randn(100, num_features)

        # Get predictions
        model_obj.model.eval()
        with torch.no_grad():
            predictions = model_obj.model(X_test).numpy().flatten()

        # Verify bounds
        assert np.all(predictions > 0), "All predictions should be positive"
        assert np.all(predictions <= 1.0), f"All predictions should be ≤ 1.0, got max={predictions.max():.4f}"

        # Verify Sigmoid is working (predictions should span a range)
        assert predictions.std() > 0.01, "Predictions should have some variance"


class TestOperatorTypeClassification:
    """Test operator type classification for non-GEMM ops."""

    def test_math_ops_use_math_peak(self):
        """MATH operators should use peak_tflops_math."""
        hw = HardwareInfo(1979.0, 33.5, 4900.0, peak_tflops_mm_conservative=535.0)

        peak = hw.get_peak_tflops(OperatorType.MATH, torch.float32, is_large_op=False)
        assert peak == 33.5, "MATH ops should use peak_tflops_math"

        # is_large_op should be ignored for MATH
        peak_large = hw.get_peak_tflops(OperatorType.MATH, torch.float32, is_large_op=True)
        assert peak_large == 33.5, "MATH ops should ignore is_large_op"

    def test_collective_ops_use_math_peak(self):
        """COLLECTIVE operators should use peak_tflops_math."""
        hw = HardwareInfo(1979.0, 33.5, 4900.0, peak_tflops_mm_conservative=535.0)

        peak = hw.get_peak_tflops(OperatorType.COLLECTIVE, torch.float32, is_large_op=False)
        assert peak == 33.5

    def test_attention_classification(self):
        """Test attention operator classification."""
        # Large attention (batch*heads*seq ≥ 4096, seq ≥ 512)
        q = torch.randn(2, 8, 1024, 64, dtype=torch.float16)  # b=2, h=8, s=1024, d=64
        # batch * heads * seq = 2 * 8 * 1024 = 16384 ≥ 4096, seq=1024 ≥ 512
        is_large = _is_large_tensor_core_op(aten._scaled_dot_product_flash_attention, (q,), OperatorType.ATTN)
        assert is_large

        # Small attention (seq < 512)
        q = torch.randn(16, 8, 256, 64, dtype=torch.float16)  # b=16, h=8, s=256, d=64
        # batch * heads * seq = 16 * 8 * 256 = 32768 ≥ 4096, but seq=256 < 512
        is_large = _is_large_tensor_core_op(aten._scaled_dot_product_flash_attention, (q,), OperatorType.ATTN)
        assert not is_large


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
