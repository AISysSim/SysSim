"""Tests for MoE (Mixture of Experts) model tracing."""

import pytest
import torch

from syssim import (
    HardwareInfo,
    SimulatorConfig,
    OperatorType,
)
from syssim.integrations.huggingface import trace_hf_model_for_training

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

# Equivalent dense config (same dims, no MoE)
SMALL_DENSE_CONFIG = dict(
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
    num_experts=1,
    num_experts_per_tok=1,
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


def _build_moe_model(config_dict):
    """Build an MoE model on meta device."""
    model_cfg = Qwen3MoeConfig(**config_dict)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_cfg, torch_dtype=torch.bfloat16)
    return model


@requires_cuda
@requires_hf
@requires_qwen3_moe
class TestMoETracing:
    """Test MoE model tracing with small configs."""

    def test_moe_traces_without_error(self, config):
        """Trace a tiny MoE model and verify no crash."""
        model = _build_moe_model(SMALL_MOE_CONFIG)
        model.train()

        input_ids = torch.randint(0, SMALL_MOE_CONFIG["vocab_size"], (1, 32))
        inputs = {"input_ids": input_ids, "labels": input_ids.clone()}

        graph = trace_hf_model_for_training(model, inputs, config)
        assert len(graph.operators) > 0, "Graph should contain operators"

    def test_moe_critical_path_positive(self, config):
        """MoE model should have positive critical path time."""
        model = _build_moe_model(SMALL_MOE_CONFIG)
        model.train()

        input_ids = torch.randint(0, SMALL_MOE_CONFIG["vocab_size"], (1, 32))
        inputs = {"input_ids": input_ids, "labels": input_ids.clone()}

        graph = trace_hf_model_for_training(model, inputs, config)
        cp = graph.compute_critical_path()
        assert cp > 0.0, "Critical path should be positive"

    def test_moe_has_gemm_ops(self, config):
        """MoE model should have GEMM ops (from attention + expert FFNs)."""
        model = _build_moe_model(SMALL_MOE_CONFIG)
        model.train()

        input_ids = torch.randint(0, SMALL_MOE_CONFIG["vocab_size"], (1, 32))
        inputs = {"input_ids": input_ids, "labels": input_ids.clone()}

        graph = trace_hf_model_for_training(model, inputs, config)
        op_types = {op.op_type for op in graph.operators.values()}
        assert OperatorType.GEMM in op_types, "Should have GEMM operations"

    def test_moe_has_attention_ops(self, config):
        """MoE model should have attention ops."""
        model = _build_moe_model(SMALL_MOE_CONFIG)
        model.train()

        input_ids = torch.randint(0, SMALL_MOE_CONFIG["vocab_size"], (1, 32))
        inputs = {"input_ids": input_ids, "labels": input_ids.clone()}

        graph = trace_hf_model_for_training(model, inputs, config)
        op_types = {op.op_type for op in graph.operators.values()}
        assert OperatorType.ATTN in op_types, "Should have attention operations"

    def test_moe_has_more_gemms_than_dense(self, config):
        """MoE model should have more GEMM ops than equivalent dense model
        because of multiple expert FFN layers."""
        # MoE model
        moe_model = _build_moe_model(SMALL_MOE_CONFIG)
        moe_model.train()
        input_ids = torch.randint(0, SMALL_MOE_CONFIG["vocab_size"], (1, 32))
        inputs = {"input_ids": input_ids, "labels": input_ids.clone()}
        moe_graph = trace_hf_model_for_training(moe_model, inputs, config)

        # Dense model (single expert)
        dense_model = _build_moe_model(SMALL_DENSE_CONFIG)
        dense_model.train()
        input_ids_d = torch.randint(0, SMALL_DENSE_CONFIG["vocab_size"], (1, 32))
        inputs_d = {"input_ids": input_ids_d, "labels": input_ids_d.clone()}
        dense_graph = trace_hf_model_for_training(dense_model, inputs_d, config)

        moe_gemms = sum(
            1 for op in moe_graph.operators.values()
            if op.op_type == OperatorType.GEMM
        )
        dense_gemms = sum(
            1 for op in dense_graph.operators.values()
            if op.op_type == OperatorType.GEMM
        )

        assert moe_gemms > dense_gemms, (
            f"MoE should have more GEMMs ({moe_gemms}) than dense ({dense_gemms})"
        )

    def test_moe_summary_contains_expected_types(self, config):
        """Summary should list gemm and math op types."""
        model = _build_moe_model(SMALL_MOE_CONFIG)
        model.train()

        input_ids = torch.randint(0, SMALL_MOE_CONFIG["vocab_size"], (1, 32))
        inputs = {"input_ids": input_ids, "labels": input_ids.clone()}

        graph = trace_hf_model_for_training(model, inputs, config)
        summary = graph.summary()
        assert "gemm" in summary, "Summary should contain gemm ops"
        assert "math" in summary, "Summary should contain math ops"

    def test_moe_param_count(self):
        """MoE model should have more parameters than dense equivalent."""
        moe_model = _build_moe_model(SMALL_MOE_CONFIG)
        dense_model = _build_moe_model(SMALL_DENSE_CONFIG)

        moe_params = sum(p.numel() for p in moe_model.parameters())
        dense_params = sum(p.numel() for p in dense_model.parameters())

        assert moe_params > dense_params, (
            f"MoE total params ({moe_params}) should exceed dense ({dense_params})"
        )
