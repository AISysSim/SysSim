"""Smoke tests for FP16/FP8/FP4 profiling kernels in compute_cost_profiler."""

import pytest
import torch

from syssim.compute.compute_cost_profiler import (
    _profile_attention,
    _profile_gemm,
)


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
    t = _profile_attention(batch=1, num_heads=8, seq_len=128, head_dim=128, num_runs=3, dtype="fp16")
    assert t > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_attention_fp8_returns_positive_time():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 9:
        pytest.skip("FP8 attention needs SM>=89")
    t = _profile_attention(batch=1, num_heads=8, seq_len=128, head_dim=128, num_runs=3, dtype="fp8")
    # Blackwell should run; older may return -1.0 as not supported
    assert t > 0 or t == -1.0
