import torch

from syssim.config import HardwareInfo, get_hardware_info
from syssim.training.spec import HardwareConfig


def test_hardware_info_has_sfu_peak_default():
    hw = HardwareInfo(peak_tflops_mm=1979.0, peak_tflops_math=989.0,
                      peak_memory_bandwidth_gbps=3350.0)
    # sfu_peak defaults to a documented fraction of the math (FMA) peak
    assert hw.sfu_peak == 989.0 / 4.0


def test_hardware_info_sfu_peak_override():
    hw = HardwareInfo(peak_tflops_mm=1979.0, peak_tflops_math=989.0,
                      peak_memory_bandwidth_gbps=3350.0, sfu_peak=250.0)
    assert hw.sfu_peak == 250.0


def test_hardwareconfig_accepts_sfu_peak():
    hw = HardwareConfig(peak_tflops_mm=1979.0, peak_tflops_math=989.0,
                        peak_memory_bandwidth_GBps=3350.0, gpus_per_node=4,
                        sfu_peak=250.0)
    assert hw.sfu_peak == 250.0
