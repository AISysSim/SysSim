"""Phase 1 MoE config + provider plumbing tests.

Covers the additive MoE architecture fields on ModelConfig, the EP/ETP kwargs on
ParallelismConfig (world_size unchanged), model-YAML acceptance of MoE keys, and
(GPU/megatron-gated) the provider forwarding + trace-safe flags.
"""
import textwrap

import pytest

from syssim.training.spec import (
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
    load_model_yaml,
)


def test_model_config_accepts_and_round_trips_moe_fields():
    m = ModelConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, num_query_groups=4,
        ffn_hidden_size=128, seq_length=32, max_position_embeddings=32, vocab_size=128,
        num_experts=8, moe_router_topk=2, moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=64, moe_layer_freq=1,
        moe_token_dispatcher_type="alltoall",
    )
    assert m.num_experts == 8
    assert m.moe_router_topk == 2
    assert m.moe_ffn_hidden_size == 256
    assert m.moe_shared_expert_intermediate_size == 64
    assert m.moe_layer_freq == 1
    assert m.moe_token_dispatcher_type == "alltoall"


def test_model_config_dense_defaults_moe_fields_none():
    m = ModelConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, num_query_groups=4,
        ffn_hidden_size=128, seq_length=32, max_position_embeddings=32, vocab_size=128,
    )
    assert m.num_experts is None
    assert m.moe_router_topk is None
    assert m.moe_ffn_hidden_size is None
    assert m.moe_shared_expert_intermediate_size is None
    assert m.moe_layer_freq is None
    assert m.moe_token_dispatcher_type is None


def test_load_model_yaml_accepts_moe_keys(tmp_path):
    p = tmp_path / "moe_model.yaml"
    p.write_text(textwrap.dedent("""
        num_layers: 2
        hidden_size: 64
        num_attention_heads: 4
        num_query_groups: 4
        ffn_hidden_size: 128
        seq_length: 32
        max_position_embeddings: 32
        vocab_size: 128
        num_experts: 8
        moe_router_topk: 2
        moe_ffn_hidden_size: 256
        moe_layer_freq: 1
        moe_token_dispatcher_type: alltoall
    """).strip())
    m = load_model_yaml(str(p))
    assert m.num_experts == 8
    assert m.moe_router_topk == 2
    assert m.moe_ffn_hidden_size == 256
    assert m.moe_token_dispatcher_type == "alltoall"


def test_load_model_yaml_still_rejects_disallowed_key(tmp_path):
    p = tmp_path / "bad_model.yaml"
    p.write_text(textwrap.dedent("""
        num_layers: 2
        hidden_size: 64
        num_attention_heads: 4
        num_query_groups: 4
        ffn_hidden_size: 128
        seq_length: 32
        max_position_embeddings: 32
        vocab_size: 128
        expert_model_parallel_size: 8
    """).strip())
    with pytest.raises(ValueError, match="disallowed key"):
        load_model_yaml(str(p))


def test_parallelism_ep_etp_short_kwargs():
    par = ParallelismConfig(tp=2, dp=4, ep=8, etp=2)
    assert par.expert_model_parallel_size == 8
    assert par.expert_tensor_parallel_size == 2
    assert par.expert_group_size == 16
    # world_size is unchanged by EP: tp*dp*cp*pp
    assert par.world_size == 8


def test_parallelism_ep_long_name_wins():
    par = ParallelismConfig(ep=4, expert_model_parallel_size=8,
                            etp=1, expert_tensor_parallel_size=2)
    assert par.expert_model_parallel_size == 8
    assert par.expert_tensor_parallel_size == 2


def test_parallelism_ep_defaults_to_one():
    par = ParallelismConfig(tp=2, dp=4)
    assert par.expert_model_parallel_size == 1
    assert par.expert_tensor_parallel_size == 1
    assert par.expert_group_size == 1
    assert par.world_size == 8


def test_parallelism_ep_must_be_positive():
    with pytest.raises(ValueError, match="expert_model_parallel_size must be >= 1"):
        ParallelismConfig(ep=0)
    with pytest.raises(ValueError, match="expert_tensor_parallel_size must be >= 1"):
        ParallelismConfig(etp=0)


# ---- GPU/megatron-gated provider plumbing ----

def _init_fake_parallel_state():
    import torch
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from syssim.training.dist_setup import init_fake_process_group

    init_fake_process_group(world_size=1, rank=0)
    torch.cuda.set_device(0)
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            create_gloo_process_groups=False,
        )
    model_parallel_cuda_manual_seed(42)


def test_resolve_provider_moe_forwards_fields_and_trace_safe_flags():
    pytest.importorskip("megatron.core")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from syssim.training.sources import resolve_megatron_provider
    from syssim.training.dist_setup import destroy_process_group

    _init_fake_parallel_state()
    try:
        model = ModelConfig(
            num_layers=2, hidden_size=256, num_attention_heads=8, num_query_groups=8,
            ffn_hidden_size=512, seq_length=128, max_position_embeddings=128, vocab_size=1024,
            num_experts=4, moe_router_topk=2, moe_ffn_hidden_size=512,
            moe_token_dispatcher_type="alltoall",
        )
        par = ParallelismConfig(tp=1, dp=1, ep=1)
        tr = TrainingConfig(micro_batch=2, global_batch=2, dtype="bf16")
        provider = resolve_megatron_provider(model, par, tr)

        assert provider.num_moe_experts == 4
        assert provider.moe_ffn_hidden_size == 512
        assert provider.moe_router_topk == 2
        assert provider.moe_token_dispatcher_type == "alltoall"
        # trace-safe internal flags
        assert provider.moe_expert_capacity_factor == 1.0
        assert provider.moe_pad_expert_input_to_capacity is True
        assert provider.moe_router_force_load_balancing is True
        assert provider.moe_grouped_gemm is False
        # moe layer freq defaulted to 1
        assert provider.moe_layer_freq == 1
    finally:
        from megatron.core import parallel_state
        parallel_state.destroy_model_parallel()
        destroy_process_group()


def test_resolve_provider_dense_has_no_experts():
    pytest.importorskip("megatron.core")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from syssim.training.sources import resolve_megatron_provider
    from syssim.training.dist_setup import destroy_process_group

    _init_fake_parallel_state()
    try:
        model = ModelConfig(
            num_layers=2, hidden_size=256, num_attention_heads=8, num_query_groups=8,
            ffn_hidden_size=512, seq_length=128, max_position_embeddings=128, vocab_size=1024,
        )
        par = ParallelismConfig(tp=1, dp=1)
        tr = TrainingConfig(micro_batch=2, global_batch=2, dtype="bf16")
        provider = resolve_megatron_provider(model, par, tr)
        assert getattr(provider, "num_moe_experts", None) is None
    finally:
        from megatron.core import parallel_state
        parallel_state.destroy_model_parallel()
        destroy_process_group()
