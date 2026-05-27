"""Smoke tests for the ARIA tutorial Colab.

Pure-Python tests run anywhere. CUDA-gated integration tests are marked
with @requires_cuda; they run in Colab.

Run locally:   pytest demo/smoke_test.py -v
Run in Colab:  !pytest demo/smoke_test.py -v
"""

from __future__ import annotations

import pytest

import demo.helpers as helpers


def test_helpers_importable():
    assert helpers.helpers_loaded() == "ok"


from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_YAML = REPO_ROOT / "demo" / "configs" / "models" / "llama3-8b.yaml"


def test_llama3_8b_yaml_loads():
    from syssim.training.spec import load_model_yaml
    cfg = load_model_yaml(str(LLAMA_YAML))
    assert cfg.num_layers == 32
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.num_query_groups == 8       # GQA: 8 KV heads
    assert cfg.ffn_hidden_size == 14336
    assert cfg.vocab_size == 128256
    assert cfg.swiglu is True
    assert cfg.rope is True
    assert cfg.tie_word_embeddings is False


MI300X_YAML = REPO_ROOT / "demo" / "configs" / "hardware" / "mi300x.yaml"


def test_mi300x_yaml_loads():
    from syssim.training.spec import load_hardware_yaml
    hw = load_hardware_yaml(str(MI300X_YAML))
    assert hw.peak_tflops_mm == 1307          # FP16 matrix peak (MI300X spec)
    assert hw.peak_tflops_mm_fp8 == 2615      # FP8 matrix peak (~2x FP16)
    assert hw.peak_memory_bandwidth_GBps == 5300   # HBM3 bandwidth
    assert hw.gpu_memory_GB == 192
    assert hw.gpus_per_node == 8


import csv
import os
import tempfile


def _read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def test_synthesize_gemm_csv_schema():
    with tempfile.TemporaryDirectory() as tmp:
        out = helpers.synthesize_gemm_csv(
            out_path=Path(tmp) / "gemm.csv",
            peak_tflops=1307, peak_bw_GBps=5300, dtype_bytes=2, seed=42,
        )
        fields, rows = _read_csv(out)
        assert fields == ["M", "N", "K", "t_measured_ms"]
        assert len(rows) >= 100
        for r in rows[:5]:
            assert float(r["t_measured_ms"]) > 0
            assert int(r["M"]) > 0 and int(r["N"]) > 0 and int(r["K"]) > 0


def test_synthesize_attn_csv_schema():
    with tempfile.TemporaryDirectory() as tmp:
        out = helpers.synthesize_attn_csv(
            out_path=Path(tmp) / "attn.csv",
            peak_tflops=1307, peak_bw_GBps=5300, dtype_bytes=2, seed=42,
        )
        fields, rows = _read_csv(out)
        assert fields == ["bs", "seq", "nh", "nkv", "hd", "t_measured_ms"]
        assert len(rows) >= 50


def test_synthesize_rmsnorm_csv_schema():
    with tempfile.TemporaryDirectory() as tmp:
        out = helpers.synthesize_rmsnorm_csv(
            out_path=Path(tmp) / "rmsnorm.csv",
            peak_bw_GBps=5300, dtype_bytes=2, seed=42,
        )
        fields, rows = _read_csv(out)
        assert fields == ["seq", "dim", "t_measured_ms"]
        assert len(rows) >= 30


def test_synthesize_gemm_csv_seed_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        a = helpers.synthesize_gemm_csv(Path(tmp) / "a.csv", 1307, 5300, 2, seed=42)
        b = helpers.synthesize_gemm_csv(Path(tmp) / "b.csv", 1307, 5300, 2, seed=42)
        assert a.read_text() == b.read_text()


def test_constant_estimator_protocol():
    from syssim.compute.estimator import Estimator
    est = helpers.ConstantEstimator(constant_ms=1.0)
    assert isinstance(est, Estimator)
    assert est.estimate_op(None, (), {}, None, None) == 1.0
    assert est.estimate_op("anything", (1, 2), {"a": 3}, None, None) == 1.0


def test_constant_estimator_custom_value():
    est = helpers.ConstantEstimator(constant_ms=2.5)
    assert est.estimate_op(None, (), {}, None, None) == 2.5


MI300X_PEAKS = {
    "peak_tflops_mm": 1307.0,
    "peak_tflops_math": 163.4,
    "peak_memory_bandwidth_gbps": 5300.0,
    "peak_tflops_mm_fp8": 2615.0,
    "peak_tflops_mm_fp4": None,
}


def test_simulated_hardware_overrides_get_hardware_info():
    """Inside the context, syssim.config.get_hardware_info returns our HW."""
    import syssim.config as sc
    import syssim.compute.compute_cost_profiler as ccp

    orig_sc = sc.get_hardware_info
    orig_ccp = ccp.get_hardware_info

    with helpers.simulated_hardware("mi300x_test", MI300X_PEAKS) as (hw, name):
        assert name == "mi300x_test"
        assert hw.peak_tflops_mm == 1307.0
        # Patched in BOTH places — local binding in ccp matters
        hw_sc, name_sc = sc.get_hardware_info()
        hw_ccp, name_ccp = ccp.get_hardware_info()
        assert name_sc == "mi300x_test" and name_ccp == "mi300x_test"
        assert hw_sc.peak_tflops_mm == 1307.0
        assert hw_ccp.peak_tflops_mm == 1307.0

    # Restored after exit
    assert sc.get_hardware_info is orig_sc
    assert ccp.get_hardware_info is orig_ccp


def test_simulated_hardware_restores_on_exception():
    import syssim.config as sc
    orig = sc.get_hardware_info
    with pytest.raises(ValueError, match="intentional"):
        with helpers.simulated_hardware("x", MI300X_PEAKS):
            raise ValueError("intentional")
    assert sc.get_hardware_info is orig


# ---------------------------------------------------------------------------
# CUDA-gated integration tests (skipped on Mac; run in Colab)
# ---------------------------------------------------------------------------

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA (run in Colab)")

QWEN3_8B_YAML = REPO_ROOT / "examples" / "configs" / "models" / "qwen3-8b.yaml"
DGX_H100_YAML = REPO_ROOT / "examples" / "configs" / "hardware" / "dgx_h100.yaml"

H100_FP8_PEAKS = {
    "peak_tflops_mm": 1979.0,
    "peak_tflops_math": 989.0,
    "peak_memory_bandwidth_gbps": 3350.0,
    "peak_tflops_mm_fp8": 3958.0,
    "peak_tflops_mm_fp4": None,
}


@requires_cuda
def test_simulate_qwen3_8b_on_h100():
    """§1 smoke: simulate Qwen3-8B on H100."""
    import syssim
    r = syssim.simulate(
        model=str(QWEN3_8B_YAML),
        hardware=str(DGX_H100_YAML),
        parallelism=syssim.ParallelismConfig(tp=2, dp=4),
        training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    )
    assert r.step_time_ms > 0
    assert 0 < r.mfu < 1
    assert r.peak_memory_gb > 0


@requires_cuda
def test_simulate_llama3_8b_on_h100():
    """§1 smoke: simulate Llama-3-8B on H100 (validates demo/configs/models/llama3-8b.yaml end-to-end)."""
    import syssim
    r = syssim.simulate(
        model=str(LLAMA_YAML),
        hardware=str(DGX_H100_YAML),
        parallelism=syssim.ParallelismConfig(tp=2, dp=4),
        training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    )
    assert r.step_time_ms > 0


@requires_cuda
def test_simulate_mi300x_roofline():
    """§3a smoke: simulate on MI300X with default (no efficiency model) — pure roofline."""
    import syssim
    r = syssim.simulate(
        model=str(QWEN3_8B_YAML),
        hardware=str(MI300X_YAML),
        parallelism=syssim.ParallelismConfig(tp=8),
        training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    )
    assert r.step_time_ms > 0


@requires_cuda
def test_train_mi300x_predictor_and_swap():
    """§3c smoke: synth MI300X data → train predictors → set dir → simulate."""
    import syssim
    from syssim.compute.compute_cost_profiler import train_efficiency_model
    from syssim.api import set_efficiency_model_dir

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prof_dir = tmp / "profiling"
        prof_dir.mkdir()
        model_dir = tmp / "models"
        model_dir.mkdir()

        gemm_csv = helpers.synthesize_gemm_csv(prof_dir / "gemm.csv", 1307, 5300, 2)
        attn_csv = helpers.synthesize_attn_csv(prof_dir / "attn.csv", 1307, 5300, 2)
        rms_csv = helpers.synthesize_rmsnorm_csv(prof_dir / "rms.csv", 5300, 2)

        with helpers.simulated_hardware("mi300x", MI300X_PEAKS) as (_, hw_name):
            # Train ONE predictor per operator
            train_efficiency_model(
                "gemm", gemm_csv,
                str(model_dir / f"gemm_{hw_name}_fp16_xgb.pth"),
                backend="xgboost", dtype="fp16",
            )
            train_efficiency_model(
                "attn", attn_csv,
                str(model_dir / f"attn_{hw_name}_fp16_xgb.pth"),
                backend="xgboost", dtype="fp16",
            )
            train_efficiency_model(
                "rmsnorm", rms_csv,
                str(model_dir / f"rmsnorm_{hw_name}_fp16_xgb.pth"),
                backend="xgboost", dtype="fp16",
            )
            set_efficiency_model_dir(str(model_dir))

            r = syssim.simulate(
                model=str(QWEN3_8B_YAML),
                hardware=str(MI300X_YAML),
                parallelism=syssim.ParallelismConfig(tp=8),
                training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
            )
            assert r.step_time_ms > 0

        # Reset predictor outside context
        set_efficiency_model_dir("")


@requires_cuda
def test_train_h100_fp8_predictor_and_swap():
    """§4c smoke: same as above but for H100 FP8."""
    import syssim
    from syssim.compute.compute_cost_profiler import train_efficiency_model
    from syssim.api import set_efficiency_model_dir

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prof_dir = tmp / "profiling"
        prof_dir.mkdir()
        model_dir = tmp / "models"
        model_dir.mkdir()

        gemm_csv = helpers.synthesize_gemm_csv(prof_dir / "gemm.csv", 3958, 3350, 1)
        attn_csv = helpers.synthesize_attn_csv(prof_dir / "attn.csv", 3958, 3350, 1)
        rms_csv = helpers.synthesize_rmsnorm_csv(prof_dir / "rms.csv", 3350, 1)

        with helpers.simulated_hardware("h100", H100_FP8_PEAKS) as (_, hw_name):
            train_efficiency_model(
                "gemm", gemm_csv,
                str(model_dir / f"gemm_{hw_name}_fp8_xgb.pth"),
                backend="xgboost", dtype="fp8",
            )
            train_efficiency_model(
                "attn", attn_csv,
                str(model_dir / f"attn_{hw_name}_fp8_xgb.pth"),
                backend="xgboost", dtype="fp8",
            )
            train_efficiency_model(
                "rmsnorm", rms_csv,
                str(model_dir / f"rmsnorm_{hw_name}_fp8_xgb.pth"),
                backend="xgboost", dtype="fp8",
            )
            set_efficiency_model_dir(str(model_dir))
            os.environ["SYSSIM_FORCE_DTYPE"] = "fp8"
            try:
                r = syssim.simulate(
                    model=str(QWEN3_8B_YAML),
                    hardware=str(DGX_H100_YAML),
                    parallelism=syssim.ParallelismConfig(tp=2, dp=4),
                    training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="fp8"),
                )
                assert r.step_time_ms > 0
            finally:
                del os.environ["SYSSIM_FORCE_DTYPE"]
        set_efficiency_model_dir("")


@requires_cuda
def test_constant_estimator_in_simulate():
    """§5 smoke: plug ConstantEstimator into HardwareConfig, simulate."""
    import syssim
    from syssim.training.spec import load_hardware_yaml

    hw = load_hardware_yaml(str(DGX_H100_YAML))
    hw.estimator = helpers.ConstantEstimator(constant_ms=1.0)
    r = syssim.simulate(
        model=str(QWEN3_8B_YAML),
        hardware=hw,
        parallelism=syssim.ParallelismConfig(tp=2, dp=4),
        training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    )
    assert r.step_time_ms > 0
