"""Integration tests for PLENA model-level estimation."""

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


# ── PLENAModelConfig Tests ─────────────────────────────────────────────────


class TestPLENAModelConfig:
    """Tests for PLENAModelConfig model configuration."""

    def test_basic_config(self):
        """Basic config creation should work."""
        from syssim.integrations.plena import PLENAModelConfig

        config = PLENAModelConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            num_hidden_layers=32,
            intermediate_size=14336,
            vocab_size=128256,
        )

        assert config.hidden_size == 4096
        assert config.num_attention_heads == 32
        assert config.num_kv_heads == 8
        assert config.head_dim == 128  # 4096 // 32

    def test_custom_head_dim(self):
        """Custom head_dim should override calculation."""
        from syssim.integrations.plena import PLENAModelConfig

        config = PLENAModelConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            num_hidden_layers=32,
            intermediate_size=14336,
            vocab_size=128256,
            head_dim=64,  # Override
        )

        assert config.head_dim == 64

    def test_to_model_param_dict(self):
        """to_model_param_dict should return proper format."""
        from syssim.integrations.plena import PLENAModelConfig

        config = PLENAModelConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            num_hidden_layers=32,
            intermediate_size=14336,
            vocab_size=128256,
        )

        params = config.to_model_param_dict()
        assert params["hidden_size"] == 4096
        assert params["num_key_value_heads"] == 8
        assert "head_dim" not in params  # Not in PLENA format


# ── PLENAInferenceResult Tests ─────────────────────────────────────────────


class TestPLENAInferenceResult:
    """Tests for PLENAInferenceResult data class."""

    def test_basic_result(self):
        """Basic result creation should work."""
        from syssim.integrations.plena import PLENAInferenceResult

        result = PLENAInferenceResult(
            prefill_ms=10.0,
            decode_ms=100.0,
            ttft_ms=15.0,
            tps=500.0,
        )

        assert result.prefill_ms == 10.0
        assert result.decode_ms == 100.0
        assert result.ttft_ms == 15.0
        assert result.tps == 500.0

    def test_conversion_properties(self):
        """Conversion properties should work correctly."""
        from syssim.integrations.plena import PLENAInferenceResult

        result = PLENAInferenceResult(
            prefill_ms=1000.0,  # 1 second
            decode_ms=2000.0,  # 2 seconds
            ttft_ms=1100.0,
            tps=500.0,
        )

        assert result.prefill_s == pytest.approx(1.0)
        assert result.decode_s == pytest.approx(2.0)


# ── estimate_plena_inference Tests ─────────────────────────────────────────


@requires_plena
class TestEstimatePLENAInference:
    """Tests for estimate_plena_inference layer-level estimation."""

    @pytest.fixture
    def llama_8b_config(self):
        """Llama-3.1-8B-like config for testing."""
        from syssim.integrations.plena import PLENAModelConfig

        return PLENAModelConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            num_hidden_layers=32,
            intermediate_size=14336,
            vocab_size=128256,
        )

    @pytest.fixture
    def plena_paths(self):
        """Return PLENA config file paths."""
        settings = str(PLENA_DIR / "plena_settings.toml")
        isa_lib = str(PLENA_DIR / "analytic_models" / "performance" / "customISA_lib.json")
        return settings, isa_lib

    def test_basic_estimation(self, llama_8b_config, plena_paths):
        """Basic estimation should return valid result."""
        from syssim.integrations.plena import estimate_plena_inference

        settings, isa_lib = plena_paths
        result = estimate_plena_inference(
            model_config=llama_8b_config,
            settings_path=settings,
            isa_lib_path=isa_lib,
            batch_size=1,
            input_seq_len=512,
            output_seq_len=64,
        )

        assert result.prefill_ms > 0
        assert result.decode_ms > 0
        assert result.ttft_ms > 0
        assert result.tps > 0

    def test_batch_size_scaling(self, llama_8b_config, plena_paths):
        """Larger batch should have higher TPS."""
        from syssim.integrations.plena import estimate_plena_inference

        settings, isa_lib = plena_paths

        result_b1 = estimate_plena_inference(
            model_config=llama_8b_config,
            settings_path=settings,
            isa_lib_path=isa_lib,
            batch_size=1,
            input_seq_len=512,
            output_seq_len=64,
        )

        result_b4 = estimate_plena_inference(
            model_config=llama_8b_config,
            settings_path=settings,
            isa_lib_path=isa_lib,
            batch_size=4,
            input_seq_len=512,
            output_seq_len=64,
        )

        # Batch 4 should have higher TPS than batch 1
        assert result_b4.tps > result_b1.tps

    def test_longer_input_slower_prefill(self, llama_8b_config, plena_paths):
        """Longer input should have slower prefill."""
        from syssim.integrations.plena import estimate_plena_inference

        settings, isa_lib = plena_paths

        result_short = estimate_plena_inference(
            model_config=llama_8b_config,
            settings_path=settings,
            isa_lib_path=isa_lib,
            batch_size=1,
            input_seq_len=256,
            output_seq_len=64,
        )

        result_long = estimate_plena_inference(
            model_config=llama_8b_config,
            settings_path=settings,
            isa_lib_path=isa_lib,
            batch_size=1,
            input_seq_len=2048,
            output_seq_len=64,
        )

        assert result_long.prefill_ms > result_short.prefill_ms


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


# ── Comparison with PLENA Standalone ───────────────────────────────────────


@requires_plena
class TestCompareWithStandalone:
    """Compare SysSim PLENA integration with standalone llama_model.py."""

    @pytest.fixture
    def llama_8b_config(self):
        """Llama-3.1-8B-like config."""
        from syssim.integrations.plena import PLENAModelConfig

        return PLENAModelConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            num_hidden_layers=32,
            intermediate_size=14336,
            vocab_size=128256,
        )

    def test_same_result_as_standalone(self, llama_8b_config):
        """SysSim should produce same results as standalone llama_model.py."""
        import sys
        import json
        import tempfile

        from syssim.integrations.plena import estimate_plena_inference

        # Add PLENA to path
        plena_perf_dir = PLENA_DIR / "analytic_models" / "performance"
        if str(plena_perf_dir) not in sys.path:
            sys.path.insert(0, str(plena_perf_dir))

        from llama_model import LLaMAModel
        from perf_model import load_hardware_config_from_toml

        settings_path = str(PLENA_DIR / "plena_settings.toml")
        isa_lib_path = str(PLENA_DIR / "analytic_models" / "performance" / "customISA_lib.json")

        # Test parameters
        batch_size = 1
        input_seq_len = 512
        output_seq_len = 64

        # Run SysSim version
        syssim_result = estimate_plena_inference(
            model_config=llama_8b_config,
            settings_path=settings_path,
            isa_lib_path=isa_lib_path,
            batch_size=batch_size,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            verbose=False,
        )

        # Run standalone version
        model_params = llama_8b_config.to_model_param_dict()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(model_params, f)
            model_config_path = f.name

        hardware_config = load_hardware_config_from_toml(settings_path)
        standalone_model = LLaMAModel(
            model_config_path=model_config_path,
            hardware_config=hardware_config,
            custom_isa_path=isa_lib_path,
            batch_size=batch_size,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
        )
        standalone_ttft, standalone_tps = standalone_model.compute_performance(verbose=False)

        # Compare results (should be very close)
        assert syssim_result.ttft_ms == pytest.approx(standalone_ttft * 1000, rel=0.01)
        assert syssim_result.tps == pytest.approx(standalone_tps, rel=0.01)


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

    def test_syssim_exports_plena_model_config(self):
        """syssim should export PLENAModelConfig."""
        from syssim import PLENAModelConfig

        assert PLENAModelConfig is not None

    def test_syssim_exports_plena_inference_result(self):
        """syssim should export PLENAInferenceResult."""
        from syssim import PLENAInferenceResult

        assert PLENAInferenceResult is not None

    def test_syssim_exports_estimate_plena_inference(self):
        """syssim should export estimate_plena_inference."""
        from syssim import estimate_plena_inference

        assert callable(estimate_plena_inference)

    def test_syssim_exports_trace_hf_model_for_plena(self):
        """syssim should export trace_hf_model_for_plena."""
        from syssim import trace_hf_model_for_plena

        assert callable(trace_hf_model_for_plena)
