"""Hybrid-vs-analytical end-to-end integration (CUDA + Megatron guarded).

Skips cleanly when CUDA, Megatron, or the gh200 bundle are unavailable — matching
the existing tests/training suite. Validates that HybridEstimator plugs into
simulate() via HardwareConfig.estimator and produces a finite step time that
differs from the pure-analytical baseline.
"""
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("hybrid integration needs CUDA (tracer)", allow_module_level=True)
pytest.importorskip("megatron")

import syssim
from syssim.compute.predictor import HybridEstimator
from syssim.config import HardwareInfo
from syssim.training.spec import load_hardware_yaml

BUNDLE = "data/gh200"
MODEL = "examples/configs/models/qwen3-1_7b.yaml"
HW_YAML = "examples/configs/hardware/single_h100.yaml"


@pytest.mark.skipif(not os.path.exists(os.path.join(BUNDLE, "manifest.json")),
                    reason="gh200 bundle not built")
def test_hybrid_vs_analytical_step_time_differs_and_finite():
    par = syssim.ParallelismConfig(tp=1, dp=1)
    tr = syssim.TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")

    base = syssim.simulate(model=MODEL, hardware=HW_YAML, parallelism=par, training=tr)

    hw_cfg = load_hardware_yaml(HW_YAML)
    hw_info = HardwareInfo(
        peak_tflops_mm=hw_cfg.peak_tflops_mm,
        peak_tflops_math=hw_cfg.peak_tflops_math,
        peak_memory_bandwidth_gbps=hw_cfg.peak_memory_bandwidth_GBps,
        peak_tflops_mm_fp8=hw_cfg.peak_tflops_mm_fp8,
        sfu_peak=hw_cfg.sfu_peak,
    )
    hw_cfg.estimator = HybridEstimator.load(BUNDLE, hw_info)
    hyb = syssim.simulate(model=MODEL, hardware=hw_cfg, parallelism=par, training=tr)

    assert hyb.step_time_ms > 0
    assert hyb.step_time_ms != base.step_time_ms   # the learned residual moved it
