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
