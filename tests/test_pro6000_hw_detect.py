"""Tests for RTX PRO 6000 (Blackwell) hardware detection and per-dtype peaks."""
import pytest
import torch

from syssim.config import HardwareInfo, get_hardware_info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pro6000_detected_when_present():
    name = torch.cuda.get_device_name(0).lower()
    if "rtx pro 6000" not in name and "blackwell" not in name:
        pytest.skip("Not running on Pro 6000")

    hw, hw_name = get_hardware_info()
    assert hw_name == "pro6000"
    assert hw.peak_tflops_mm > 3000  # FP16 dense
    assert hw.peak_memory_bandwidth_gbps > 1500
    # FP8 / FP4 peaks must be set
    assert hw.peak_tflops_mm_fp8 is not None and hw.peak_tflops_mm_fp8 > hw.peak_tflops_mm
    assert hw.peak_tflops_mm_fp4 is not None and hw.peak_tflops_mm_fp4 > hw.peak_tflops_mm_fp8


def test_hardware_info_has_per_dtype_peaks():
    hw = HardwareInfo(
        peak_tflops_mm=3752.0,
        peak_tflops_math=117.0,
        peak_memory_bandwidth_gbps=1792.0,
        peak_tflops_mm_fp8=7504.0,
        peak_tflops_mm_fp4=15008.0,
    )
    assert hw.get_peak_tflops_mm_for_dtype(torch.float16) == 3752.0
    assert hw.get_peak_tflops_mm_for_dtype(torch.bfloat16) == 3752.0
    assert hw.get_peak_tflops_mm_for_dtype(torch.float8_e4m3fn) == 7504.0
    assert hw.get_peak_tflops_mm_for_dtype(torch.float8_e5m2) == 7504.0
    assert hw.get_peak_tflops_mm_for_dtype("nvfp4") == 15008.0


def test_hardware_info_falls_back_to_fp16_peak_when_unset():
    hw = HardwareInfo(
        peak_tflops_mm=989.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )
    # FP8/FP4 unset -> fall through to FP16 peak
    assert hw.get_peak_tflops_mm_for_dtype(torch.float8_e4m3fn) == 989.0
    assert hw.get_peak_tflops_mm_for_dtype("nvfp4") == 989.0
    assert hw.get_peak_tflops_mm_for_dtype(torch.float16) == 989.0
