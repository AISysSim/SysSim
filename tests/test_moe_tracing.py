"""Tests for MoE model tracing."""

from types import SimpleNamespace

import pytest
import torch

from syssim import (
    HardwareInfo,
    MoERuntimeConfig,
    OperatorType,
    SimulatorConfig,
    build_moe_operator_graph,
    extract_hf_moe_spec,
    trace_hf_moe_model_for_training,
)

try:
    from transformers import AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from transformers import Qwen3MoeConfig
    QWEN3_MOE_AVAILABLE = True
except ImportError:
    QWEN3_MOE_AVAILABLE = False

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for tracing",
)

requires_hf = pytest.mark.skipif(
    not HF_AVAILABLE,
    reason="transformers not installed",
)

requires_qwen3_moe = pytest.mark.skipif(
    not QWEN3_MOE_AVAILABLE,
    reason="transformers version lacks Qwen3MoeConfig",
)


# Small MoE config for fast testing
SMALL_MOE_CONFIG = dict(
    num_hidden_layers=2,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    vocab_size=1000,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    hidden_act="silu",
    num_experts=4,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
    norm_topk_prob=True,
    router_aux_loss_coef=0.001,
    attention_bias=False,
    tie_word_embeddings=False,
)


@pytest.fixture
def hw():
    return HardwareInfo(
        peak_tflops_mm=989.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )


@pytest.fixture
def config(hw):
    return SimulatorConfig(hw_info=hw)


def _fake_config(**overrides):
    values = dict(
        num_hidden_layers=2,
        hidden_size=256,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=1000,
        decoder_sparse_step=1,
        model_type="qwen3_moe",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _build_moe_model(config_dict):
    """Build an MoE model on meta device."""
    model_cfg = Qwen3MoeConfig(**config_dict)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_cfg, torch_dtype=torch.bfloat16)
    return model


class TestMoEOperatorTracing:
    def test_fake_config_builds_stage_nodes(self, config):
        spec = extract_hf_moe_spec(_fake_config())
        runtime = MoERuntimeConfig(batch_size=1, seq_len=32)
        graph = build_moe_operator_graph(spec, runtime, config)

        op_types = {op.op_type for op in graph.operators.values()}
        assert OperatorType.MOE_ROUTER in op_types
        assert OperatorType.MOE_DISPATCH in op_types
        assert OperatorType.MOE_EXPERT in op_types
        assert OperatorType.MOE_COMBINE in op_types
        assert graph.compute_critical_path() > 0.0

    def test_fake_config_ep_builds_collectives(self, config):
        spec = extract_hf_moe_spec(_fake_config())
        runtime = MoERuntimeConfig(batch_size=1, seq_len=32, expert_parallel_size=2)
        graph = build_moe_operator_graph(spec, runtime, config)

        assert sum(op.op_type == OperatorType.COLLECTIVE for op in graph.operators.values()) == 4
        assert "collective: 4" in graph.summary()


@requires_cuda
@requires_hf
@requires_qwen3_moe
class TestQwen3MoEIntegration:
    """Test MoE operator graph construction with tiny Qwen3 MoE configs."""

    def test_qwen3_moe_graph_has_stage_nodes(self, config):
        model = _build_moe_model(SMALL_MOE_CONFIG)
        model.train()

        input_ids = torch.randint(0, SMALL_MOE_CONFIG["vocab_size"], (1, 32))
        inputs = {"input_ids": input_ids, "labels": input_ids.clone()}

        graph = trace_hf_moe_model_for_training(model, inputs, config)
        op_types = {op.op_type for op in graph.operators.values()}
        assert OperatorType.MOE_ROUTER in op_types
        assert OperatorType.MOE_DISPATCH in op_types
        assert OperatorType.MOE_EXPERT in op_types
        assert OperatorType.MOE_COMBINE in op_types
        assert graph.compute_critical_path() > 0.0

    def test_qwen3_moe_spec_extraction(self):
        model = _build_moe_model(SMALL_MOE_CONFIG)
        spec = extract_hf_moe_spec(model)
        assert spec.num_layers == SMALL_MOE_CONFIG["num_hidden_layers"]
        assert spec.hidden_size == SMALL_MOE_CONFIG["hidden_size"]
        assert spec.intermediate_size == SMALL_MOE_CONFIG["moe_intermediate_size"]
        assert spec.num_experts == SMALL_MOE_CONFIG["num_experts"]
        assert spec.top_k == SMALL_MOE_CONFIG["num_experts_per_tok"]
