"""Structural tests for Hugging Face integration (no model loading required)."""

import pytest

from syssim import HardwareInfo, SimulatorConfig


def test_integration_module_imports():
    """Test that integration module can be imported."""
    from syssim.integrations import huggingface

    assert huggingface is not None


def test_function_exports():
    """Test that HF functions are exported from main package."""
    from syssim import trace_hf_model_for_training, trace_hf_training_step

    assert callable(trace_hf_model_for_training)
    assert callable(trace_hf_training_step)


def test_hf_available_flag():
    """Test HF_AVAILABLE flag reflects transformers installation."""
    from syssim.integrations.huggingface import HF_AVAILABLE

    # Just check it's a boolean
    assert isinstance(HF_AVAILABLE, bool)


def test_config_creation():
    """Test that config can be created for HF integration."""
    hw = HardwareInfo(989.0, 989.0, 3350.0)
    config = SimulatorConfig(hw_info=hw)
    assert config is not None
    assert config.hw_info.peak_tflops_mm == 989.0


def test_helper_functions_exist():
    """Test that helper functions are defined."""
    from syssim.integrations.huggingface import _create_lm_loss_fn

    assert callable(_create_lm_loss_fn)


def test_imports_dont_fail():
    """Test that imports don't crash (even if transformers unavailable)."""
    try:
        from syssim.integrations.huggingface import (
            trace_hf_model_for_training,
            trace_hf_training_step,
        )

        # If transformers unavailable, functions should still be defined
        assert callable(trace_hf_model_for_training)
        assert callable(trace_hf_training_step)
    except Exception as e:
        pytest.fail(f"Imports should not fail: {e}")
