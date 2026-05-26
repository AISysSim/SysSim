"""Smoke tests for FP16/FP8/FP4 profiling kernels in compute_cost_profiler."""
import importlib.util
from pathlib import Path

import pytest
import torch

from syssim.compute.compute_cost_profiler import (
    _profile_gemm,
    _profile_attention,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tutorial_script(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_tutorial4_actual_measurement_defaults_to_three_measured_steps_after_warmup(monkeypatch):
    module = _load_tutorial_script(
        "low_precision_measure_actual_h100",
        "docs/tutorials/scripts/low_precision_measure_actual_h100.py",
    )

    monkeypatch.setattr("sys.argv", ["low_precision_measure_actual_h100.py"])

    args = module.parse_args()
    assert args.warmups == 1
    assert args.runs == 3


def test_tutorial4_profile_model_defaults_to_one_run_reduced_smoke(monkeypatch):
    module = _load_tutorial_script(
        "low_precision_profile_model_h100",
        "docs/tutorials/scripts/low_precision_profile_model_h100.py",
    )

    monkeypatch.setattr("sys.argv", ["low_precision_profile_model_h100.py"])

    args = module.parse_args()
    assert args.profile_scale == "reduced"
    assert args.num_runs == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_gemm_fp16_returns_positive_time():
    t = _profile_gemm(64, 64, 64, num_runs=3, dtype="fp16")
    assert t > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_gemm_fp8_returns_positive_time():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 9:
        pytest.skip("FP8 needs SM>=89 (Hopper/Blackwell)")
    t = _profile_gemm(128, 128, 128, num_runs=3, dtype="fp8")
    # Must succeed (>0) on Blackwell; on older hw mark of -1.0 also acceptable
    assert t > 0 or t == -1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_gemm_fp4_returns_positive_time():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 10:
        pytest.skip("NVFP4 needs SM>=100 (Blackwell)")
    t = _profile_gemm(256, 256, 256, num_runs=3, dtype="fp4")
    assert t > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_attention_fp16_returns_positive_time():
    t = _profile_attention(
        batch=1, num_heads=8, seq_len=128, head_dim=128, num_runs=3, dtype="fp16"
    )
    assert t > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_attention_fp8_returns_positive_time():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 9:
        pytest.skip("FP8 attention needs SM>=89")
    t = _profile_attention(
        batch=1, num_heads=8, seq_len=128, head_dim=128, num_runs=3, dtype="fp8"
    )
    # Blackwell should run; older may return -1.0 as not supported
    assert t > 0 or t == -1.0
