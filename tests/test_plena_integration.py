"""Integration tests for PLENA op-level estimation (Approach A)."""

import pytest
from pathlib import Path

import torch
import torch.nn as nn

# Skip all tests if PLENA_Simulator submodule or CUDA is not available
PLENA_DIR = Path(__file__).parent.parent / "PLENA_Simulator"
PLENA_AVAILABLE = (
    PLENA_DIR.exists()
    and (PLENA_DIR / "plena_settings.toml").exists()
    and (PLENA_DIR / "analytic_models" / "performance" / "customISA_lib.json").exists()
)
CUDA_AVAILABLE = torch.cuda.is_available()

requires_plena = pytest.mark.skipif(
    not PLENA_AVAILABLE,
    reason="PLENA_Simulator submodule not initialized",
)

requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available for tracing",
)

requires_plena_and_cuda = pytest.mark.skipif(
    not (PLENA_AVAILABLE and CUDA_AVAILABLE),
    reason="Requires both PLENA submodule and CUDA",
)


# ── trace_model_for_plena Tests ────────────────────────────────────────────


@requires_plena_and_cuda
class TestTraceModelForPLENA:
    """Tests for trace_model_for_plena op-level tracing."""

    @pytest.fixture
    def plena_config(self):
        """Create PLENAConfig for testing."""
        from syssim.config_plena import PLENAConfig

        return PLENAConfig.from_plena_submodule()

    def test_trace_linear(self, plena_config):
        """Tracing Linear should produce operators with PLENA estimates."""
        from syssim import trace_model_for_plena, OperatorType

        model = nn.Linear(256, 128)
        x = torch.randn(32, 256).cuda()

        graph = trace_model_for_plena(model, x, plena_config, mode="prefill")

        assert len(graph) > 0

        # Should have GEMM operators
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        assert len(gemm_ops) >= 1

        # GEMM should have positive time estimate
        for op in gemm_ops:
            assert op.estimated_time_ms > 0.0

    def test_trace_mlp(self, plena_config):
        """Tracing MLP should produce operators."""
        from syssim import trace_model_for_plena, OperatorType

        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        x = torch.randn(32, 256).cuda()

        graph = trace_model_for_plena(model, x, plena_config, mode="prefill")

        # Should have multiple operators
        assert len(graph) >= 3

        # Should have 2 GEMM ops
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        assert len(gemm_ops) == 2

    def test_critical_path_positive(self, plena_config):
        """Critical path should be positive."""
        from syssim import trace_model_for_plena

        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        x = torch.randn(16, 128).cuda()

        graph = trace_model_for_plena(model, x, plena_config)
        cp = graph.compute_critical_path()

        assert cp > 0.0

    def test_decode_mode(self, plena_config):
        """Decode mode should work."""
        from syssim import trace_model_for_plena

        model = nn.Linear(256, 128)
        x = torch.randn(1, 256).cuda()  # seq_len=1 for decode

        graph = trace_model_for_plena(model, x, plena_config, mode="decode")
        assert len(graph) > 0

    def test_invalid_mode_raises(self, plena_config):
        """Invalid mode should raise ValueError."""
        from syssim import trace_model_for_plena

        model = nn.Linear(64, 32)
        x = torch.randn(8, 64).cuda()

        with pytest.raises(ValueError, match="Invalid inference mode"):
            trace_model_for_plena(model, x, plena_config, mode="invalid")


# ── Export Tests ───────────────────────────────────────────────────────────


class TestExports:
    """Tests for module exports."""

    def test_syssim_exports_plena_config(self):
        """syssim should export PLENAConfig."""
        from syssim import PLENAConfig

        assert PLENAConfig is not None

    def test_syssim_exports_is_plena_hardware(self):
        """syssim should export is_plena_hardware."""
        from syssim import is_plena_hardware

        assert callable(is_plena_hardware)

    def test_syssim_exports_trace_model_for_plena(self):
        """syssim should export trace_model_for_plena."""
        from syssim import trace_model_for_plena

        assert callable(trace_model_for_plena)
