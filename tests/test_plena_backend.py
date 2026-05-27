"""Unit tests for PLENA backend."""

import pytest
from pathlib import Path

# Skip all tests if PLENA_Simulator submodule is not available
PLENA_DIR = Path(__file__).parent.parent / "PLENA_Simulator"
PLENA_AVAILABLE = (
    PLENA_DIR.exists()
    and (PLENA_DIR / "plena_settings.toml").exists()
    and (PLENA_DIR / "analytic_models" / "performance" / "customISA_lib.json").exists()
)

requires_plena = pytest.mark.skipif(
    not PLENA_AVAILABLE,
    reason="PLENA_Simulator submodule not initialized",
)


# ── PLENAConfig Tests ──────────────────────────────────────────────────────


class TestPLENAConfig:
    """Tests for PLENAConfig configuration loading."""

    @requires_plena
    def test_from_plena_submodule(self):
        """PLENAConfig.from_plena_submodule() should auto-locate config files."""
        from syssim.config_plena import PLENAConfig

        config = PLENAConfig.from_plena_submodule()

        assert Path(config.settings_path).exists()
        assert Path(config.isa_lib_path).exists()
        assert config.frequency_hz == 1e9

    @requires_plena
    def test_custom_frequency(self):
        """PLENAConfig should accept custom frequency."""
        from syssim.config_plena import PLENAConfig

        config = PLENAConfig.from_plena_submodule(frequency_hz=2e9)
        assert config.frequency_hz == 2e9

    def test_invalid_settings_path(self):
        """PLENAConfig should raise for non-existent settings file."""
        from syssim.config_plena import PLENAConfig

        with pytest.raises(FileNotFoundError, match="settings"):
            PLENAConfig(
                settings_path="/nonexistent/plena_settings.toml",
                isa_lib_path="/nonexistent/customISA_lib.json",
            )

    @requires_plena
    def test_invalid_isa_path(self):
        """PLENAConfig should raise for non-existent ISA lib file."""
        from syssim.config_plena import PLENAConfig

        settings_path = str(PLENA_DIR / "plena_settings.toml")
        with pytest.raises(FileNotFoundError, match="ISA"):
            PLENAConfig(
                settings_path=settings_path,
                isa_lib_path="/nonexistent/customISA_lib.json",
            )


# ── PLENAHardwareInfo Tests ────────────────────────────────────────────────


class TestPLENAHardwareInfo:
    """Tests for PLENAHardwareInfo hardware wrapper."""

    @requires_plena
    def test_from_config(self):
        """PLENAHardwareInfo should wrap config."""
        from syssim.config_plena import PLENAConfig
        from syssim.compute.plena_backend import PLENAHardwareInfo

        config = PLENAConfig.from_plena_submodule()
        hw_info = PLENAHardwareInfo.from_config(config)

        assert hw_info.frequency_hz == 1e9
        assert hw_info._config is not None

    @requires_plena
    def test_lazy_load_perf_model(self):
        """PerfModel should be lazy-loaded on first access."""
        from syssim.config_plena import PLENAConfig
        from syssim.compute.plena_backend import PLENAHardwareInfo

        config = PLENAConfig.from_plena_submodule()
        hw_info = PLENAHardwareInfo.from_config(config)

        # Not loaded yet
        assert hw_info._perf_model is None

        # Access triggers load
        perf = hw_info.perf_model
        assert perf is not None
        assert hw_info._perf_model is perf


# ── is_plena_hardware Tests ────────────────────────────────────────────────


class TestIsPLENAHardware:
    """Tests for is_plena_hardware type check."""

    @requires_plena
    def test_plena_hardware_returns_true(self):
        """is_plena_hardware should return True for PLENAHardwareInfo."""
        from syssim.config_plena import PLENAConfig, is_plena_hardware
        from syssim.compute.plena_backend import PLENAHardwareInfo

        config = PLENAConfig.from_plena_submodule()
        hw_info = PLENAHardwareInfo.from_config(config)

        assert is_plena_hardware(hw_info) is True

    def test_gpu_hardware_returns_false(self):
        """is_plena_hardware should return False for GPU HardwareInfo."""
        from syssim import HardwareInfo
        from syssim.config_plena import is_plena_hardware

        hw_info = HardwareInfo(
            peak_tflops_mm=989.0,
            peak_tflops_math=989.0,
            peak_memory_bandwidth_gbps=3350.0,
        )

        assert is_plena_hardware(hw_info) is False


# ── Cycle Estimation Tests ─────────────────────────────────────────────────


@requires_plena
class TestCycleEstimation:
    """Tests for PLENA cycle estimation functions."""

    @pytest.fixture
    def perf_model(self):
        """Load PLENA PerfModel for testing."""
        from syssim.config_plena import PLENAConfig, get_plena_perf_model

        config = PLENAConfig.from_plena_submodule()
        return get_plena_perf_model(config)

    def test_gemm_cycles_positive(self, perf_model):
        """GEMM estimation should return positive cycles."""
        from syssim.compute.plena_backend import estimate_plena_cycles_for_gemm

        cycles = estimate_plena_cycles_for_gemm(
            perf_model, batch=1, m=128, n=256, k=512
        )
        assert cycles > 0

    def test_gemm_cycles_scale_with_size(self, perf_model):
        """Larger GEMM should have more cycles."""
        from syssim.compute.plena_backend import estimate_plena_cycles_for_gemm

        cycles_small = estimate_plena_cycles_for_gemm(
            perf_model, batch=1, m=64, n=64, k=64
        )
        cycles_large = estimate_plena_cycles_for_gemm(
            perf_model, batch=1, m=512, n=512, k=512
        )

        assert cycles_large > cycles_small

    def test_attention_cycles_positive(self, perf_model):
        """Attention estimation should return positive cycles."""
        from syssim.compute.plena_backend import estimate_plena_cycles_for_attention

        cycles = estimate_plena_cycles_for_attention(
            perf_model,
            batch=1,
            num_heads=8,
            seq_len=256,
            kv_len=256,
            head_dim=64,
            mode="prefill",
        )
        assert cycles > 0

    def test_attention_decode_vs_prefill(self, perf_model):
        """Decode mode should have different (typically fewer) cycles for same seq_len."""
        from syssim.compute.plena_backend import estimate_plena_cycles_for_attention

        cycles_prefill = estimate_plena_cycles_for_attention(
            perf_model,
            batch=1,
            num_heads=8,
            seq_len=1,  # Decode: seq_len=1
            kv_len=1024,  # But large KV cache
            head_dim=64,
            mode="prefill",
        )
        cycles_decode = estimate_plena_cycles_for_attention(
            perf_model,
            batch=1,
            num_heads=8,
            seq_len=1,
            kv_len=1024,
            head_dim=64,
            mode="decode",
        )

        # Both should be positive
        assert cycles_prefill > 0
        assert cycles_decode > 0

    def test_math_cycles_positive(self, perf_model):
        """Math estimation should return positive cycles."""
        from syssim.compute.plena_backend import estimate_plena_cycles_for_math

        cycles = estimate_plena_cycles_for_math(perf_model, num_elements=1024)
        assert cycles > 0

    def test_math_cycles_scale_with_elements(self, perf_model):
        """More elements should have more cycles."""
        from syssim.compute.plena_backend import estimate_plena_cycles_for_math

        cycles_small = estimate_plena_cycles_for_math(perf_model, num_elements=256)
        cycles_large = estimate_plena_cycles_for_math(perf_model, num_elements=4096)

        assert cycles_large > cycles_small


# ── PLENAEstimator Tests ───────────────────────────────────────────────────


@requires_plena
class TestPLENAEstimator:
    """Tests for PLENAEstimator runtime estimation."""

    @pytest.fixture
    def estimator(self):
        """Create PLENAEstimator for testing."""
        from syssim.config_plena import PLENAConfig
        from syssim.compute.plena_backend import PLENAEstimator

        config = PLENAConfig.from_plena_submodule()
        return PLENAEstimator(config)

    def test_cycles_to_ms(self, estimator):
        """Cycle to millisecond conversion should be correct."""
        # 1e9 cycles at 1 GHz = 1 second = 1000 ms
        ms = estimator.cycles_to_ms(int(1e9))
        assert ms == pytest.approx(1000.0)

        # 1e6 cycles at 1 GHz = 1 ms
        ms = estimator.cycles_to_ms(int(1e6))
        assert ms == pytest.approx(1.0)

    def test_estimate_runtime_returns_float(self, estimator):
        """estimate_runtime should return a float."""
        import torch
        from syssim import OperatorType

        # Simulate a math operation
        out = torch.randn(128, 256)
        time_ms = estimator.estimate_runtime(
            func_packet=None,
            args=(),
            kwargs={},
            out=out,
            op_type=OperatorType.MATH,
        )

        assert isinstance(time_ms, float)
        assert time_ms >= 0.0

    def test_estimate_runtime_math_positive(self, estimator):
        """Math ops should have positive runtime."""
        import torch
        from syssim import OperatorType

        out = torch.randn(1024, 1024)
        time_ms = estimator.estimate_runtime(
            func_packet=None,
            args=(),
            kwargs={},
            out=out,
            op_type=OperatorType.MATH,
        )

        assert time_ms > 0.0


# ── Dimension Extraction Tests ─────────────────────────────────────────────


class TestDimensionExtraction:
    """Tests for operator dimension extraction."""

    def test_extract_gemm_dims_mm(self):
        """_extract_gemm_dims should handle aten.mm."""
        import torch
        from syssim.compute.plena_backend import _extract_gemm_dims

        aten = torch.ops.aten
        a = torch.randn(32, 64)
        b = torch.randn(64, 128)

        batch, m, n, k = _extract_gemm_dims(aten.mm, (a, b))
        assert batch == 1
        assert m == 32
        assert n == 128
        assert k == 64

    def test_extract_gemm_dims_addmm(self):
        """_extract_gemm_dims should handle aten.addmm."""
        import torch
        from syssim.compute.plena_backend import _extract_gemm_dims

        aten = torch.ops.aten
        bias = torch.randn(128)
        a = torch.randn(32, 64)
        b = torch.randn(64, 128)

        batch, m, n, k = _extract_gemm_dims(aten.addmm, (bias, a, b))
        assert batch == 1
        assert m == 32
        assert n == 128
        assert k == 64

    def test_extract_gemm_dims_bmm(self):
        """_extract_gemm_dims should handle aten.bmm."""
        import torch
        from syssim.compute.plena_backend import _extract_gemm_dims

        aten = torch.ops.aten
        a = torch.randn(4, 32, 64)
        b = torch.randn(4, 64, 128)

        batch, m, n, k = _extract_gemm_dims(aten.bmm, (a, b))
        assert batch == 4
        assert m == 32
        assert n == 128
        assert k == 64

    def test_extract_attention_dims(self):
        """_extract_attention_dims should extract SDPA dims."""
        import torch
        from syssim.compute.plena_backend import _extract_attention_dims

        q = torch.randn(2, 8, 256, 64)
        k = torch.randn(2, 8, 512, 64)
        v = torch.randn(2, 8, 512, 64)

        batch, num_heads, seq_len, kv_len, head_dim = _extract_attention_dims((q, k, v))
        assert batch == 2
        assert num_heads == 8
        assert seq_len == 256
        assert kv_len == 512
        assert head_dim == 64


# ── Math Op Multiplier Tests ───────────────────────────────────────────────


class TestMathOpMultiplier:
    """Tests for math operation complexity multiplier."""

    def test_simple_ops(self):
        """Simple ops should have multiplier 1.0."""
        from syssim.compute.plena_backend import _get_math_op_multiplier

        import torch
        aten = torch.ops.aten

        assert _get_math_op_multiplier(aten.relu) == 1.0
        assert _get_math_op_multiplier(aten.add) == 1.0
        assert _get_math_op_multiplier(None) == 1.0

    def test_medium_ops(self):
        """Medium complexity ops should have multiplier 4.0."""
        from syssim.compute.plena_backend import _get_math_op_multiplier

        import torch
        aten = torch.ops.aten

        assert _get_math_op_multiplier(aten.gelu) == 4.0
        assert _get_math_op_multiplier(aten.softmax) == 4.0
        assert _get_math_op_multiplier(aten.sigmoid) == 4.0

    def test_complex_ops(self):
        """Complex ops should have multiplier 6.0."""
        from syssim.compute.plena_backend import _get_math_op_multiplier

        import torch
        aten = torch.ops.aten

        assert _get_math_op_multiplier(aten.silu) == 6.0
        assert _get_math_op_multiplier(aten.layer_norm) == 6.0
