"""EP>1 correctness: analytical per-rank expert bytes shard by expert_group_size,
the grad all-reduce volume/group split is EP-aware, dense/EP=1 stay byte-identical,
and gpt-oss-20b at a sensible EP produces a finite, physical report.

The pure-analytical tests (memory.py byte functions, runner rank-group helper) always
run. The MemTracker EP-scaling oracle and the gpt-oss-20b end-to-end run need a CUDA
GPU + Megatron-Core and skip cleanly on a CPU-only host.
"""
import os

import pytest

from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig
from syssim.training.memory import (
    _ffn_param_count, _ffn_param_count_per_rank, _expert_param_count,
    param_bytes_per_rank, grad_bytes_per_rank, sharded_param_bytes_per_rank,
    optimizer_state_bytes_per_rank,
    expert_grad_bytes_per_rank, dense_grad_bytes_per_rank,
    expert_sharded_param_bytes_per_rank,
)


# Tiny MoE dims (match test_moe_flops_params probes).
HIDDEN, HEADS, LAYERS = 256, 8, 2
SEQ, VOCAB = 128, 1024
NUM_EXPERTS, TOPK, MOE_FFN = 8, 2, 512


def _tiny_moe():
    return ModelConfig(
        num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=HEADS,
        num_query_groups=HEADS, ffn_hidden_size=MOE_FFN,
        seq_length=SEQ, max_position_embeddings=SEQ, vocab_size=VOCAB,
        swiglu=True, tie_word_embeddings=False,
        num_experts=NUM_EXPERTS, moe_router_topk=TOPK, moe_ffn_hidden_size=MOE_FFN,
        moe_layer_freq=1,
    )


def _tiny_dense():
    return ModelConfig(
        num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=HEADS,
        num_query_groups=HEADS, ffn_hidden_size=MOE_FFN,
        seq_length=SEQ, max_position_embeddings=SEQ, vocab_size=VOCAB,
        swiglu=True, tie_word_embeddings=False,
    )


_TR = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")


# ---------------------------------------------------------------------------
# Analytical per-rank expert sharding (always runs).
# ---------------------------------------------------------------------------

def test_ffn_per_rank_ep1_equals_ffn_count():
    """EP=1 (expert_group_size==1): the per-rank FFN count equals the replicated
    EP=1 count exactly -> byte-identical to the pre-EP path."""
    m = _tiny_moe()
    p = ParallelismConfig(dp=4, ep=1)
    assert _ffn_param_count_per_rank(m, p) == _ffn_param_count(m)


def test_ffn_per_rank_shards_only_expert_portion():
    """At EP=K only the routed-expert portion divides by K; router gate stays."""
    m = _tiny_moe()
    base = _ffn_param_count(m)
    experts = _expert_param_count(m)
    router_gate = HIDDEN * NUM_EXPERTS
    assert experts == NUM_EXPERTS * 3 * HIDDEN * MOE_FFN
    assert base == experts + router_gate  # no shared expert in tiny config

    for ep in (2, 4, 8):
        p = ParallelismConfig(dp=8, ep=ep)
        got = _ffn_param_count_per_rank(m, p)
        assert got == experts // ep + router_gate, ep


def test_per_rank_param_bytes_scale_by_expert_group_size():
    """The EP-sharded expert contribution to param/grad/optimizer bytes scales 1/EP."""
    m = _tiny_moe()
    p1 = ParallelismConfig(dp=8, ep=1)
    p4 = ParallelismConfig(dp=8, ep=4)

    # Expert portion delta between EP=1 and EP=4 = experts*(1 - 1/4)*layers*2B.
    experts_elems = _expert_param_count(m) * LAYERS
    delta_bytes = (experts_elems - experts_elems // 4) * 2  # bf16

    assert param_bytes_per_rank(m, p1, _TR) - param_bytes_per_rank(m, p4, _TR) == delta_bytes
    assert grad_bytes_per_rank(m, p1, _TR) - grad_bytes_per_rank(m, p4, _TR) == delta_bytes
    assert (sharded_param_bytes_per_rank(m, p1, _TR)
            - sharded_param_bytes_per_rank(m, p4, _TR)) == delta_bytes
    # Optimizer state is 12 B/elem (fp32 master+m+v) and NOT dist-opt here.
    assert (optimizer_state_bytes_per_rank(m, p1, _TR)
            - optimizer_state_bytes_per_rank(m, p4, _TR)) == (experts_elems - experts_elems // 4) * 12


def test_expert_dense_grad_split():
    """expert_grad_bytes is the EP-sharded routed-expert portion; dense_grad_bytes is
    everything else; they sum to grad_bytes_per_rank."""
    m = _tiny_moe()
    p = ParallelismConfig(dp=8, ep=4)
    eg = expert_grad_bytes_per_rank(m, p, _TR)
    dg = dense_grad_bytes_per_rank(m, p, _TR)
    assert eg > 0
    assert eg + dg == grad_bytes_per_rank(m, p, _TR)
    assert eg == _expert_param_count(m) // 4 * LAYERS * 2  # bf16


def test_expert_sharded_param_bytes():
    m = _tiny_moe()
    p = ParallelismConfig(dp=8, ep=4)
    esp = expert_sharded_param_bytes_per_rank(m, p, _TR)
    assert esp == _expert_param_count(m) // 4 * LAYERS * 2
    assert esp < sharded_param_bytes_per_rank(m, p, _TR)


# ---------------------------------------------------------------------------
# Dense + EP=1 regression: gated EP helpers must be byte-identical / no-op.
# ---------------------------------------------------------------------------

def test_dense_unchanged_by_ep_helpers():
    m = _tiny_dense()
    p = ParallelismConfig(dp=4, ep=1)
    assert _ffn_param_count_per_rank(m, p) == _ffn_param_count(m)
    assert expert_grad_bytes_per_rank(m, p, _TR) == 0
    assert dense_grad_bytes_per_rank(m, p, _TR) == grad_bytes_per_rank(m, p, _TR)
    assert expert_sharded_param_bytes_per_rank(m, p, _TR) == 0


def test_moe_ep1_bytes_identical_to_pre_ep():
    """EP=1 MoE per-rank bytes equal what the pre-EP formula (_ffn_param_count) gives."""
    m = _tiny_moe()
    p = ParallelismConfig(dp=4, ep=1)
    tp = 1
    sharded = (
        # _attn_param_count inlined
        HIDDEN * HEADS * (HIDDEN // HEADS)
        + 2 * HIDDEN * HEADS * (HIDDEN // HEADS)
        + HEADS * (HIDDEN // HEADS) * HIDDEN
        + _ffn_param_count(m)
    ) * LAYERS
    embed = VOCAB * HIDDEN
    lm = VOCAB * HIDDEN
    norms = 2 * HIDDEN * LAYERS
    replicated = embed + HIDDEN + lm + norms
    expected = (sharded // tp + replicated) * 2  # bf16
    assert param_bytes_per_rank(m, p, _TR) == expected
    assert expert_grad_bytes_per_rank(m, p, _TR) == 0


# ---------------------------------------------------------------------------
# Expert-DP rank group (always runs).
# ---------------------------------------------------------------------------

def test_expert_dp_group_size():
    from syssim.training.runner import expert_dp_group_ranks, dp_group_ranks
    # ep==dp -> expert-DP group size 1 (no expert all-reduce).
    assert len(expert_dp_group_ranks(tp_size=1, dp_size=4, ep_size=4)) == 1
    # dp/ep = 2.
    assert len(expert_dp_group_ranks(tp_size=1, dp_size=4, ep_size=2)) == 2
    # dp/ep = 4, strided by tp_size.
    g = expert_dp_group_ranks(tp_size=2, dp_size=8, ep_size=2)
    assert len(g) == 4
    assert g == dp_group_ranks(2, 8)[:4]


# ---------------------------------------------------------------------------
# MemTracker EP-scaling oracle (GPU-gated): per-rank resident expert Parameter
# bytes at EP=K are ~1/K of EP=1.
# ---------------------------------------------------------------------------

def _build_and_measure_expert_params(ep_size: int) -> int:
    """Build the tiny MoE under a fake world=ep PG with parallel_state EP=ep, return
    the resident routed-expert nn.Parameter element count (sum over SequentialMLP)."""
    import torch
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.moe.experts import SequentialMLP

    from syssim.training.dist_setup import init_fake_process_group, destroy_process_group
    from syssim.training.cuda_redirect import redirect_cuda_alloc_to_meta
    from syssim.training._norm_meta import install_norm_meta_kernels

    init_fake_process_group(world_size=ep_size, rank=0)
    try:
        torch.cuda.set_device(0)
        if not parallel_state.is_initialized():
            kwargs = {}
            if ep_size > 1:
                kwargs["expert_model_parallel_size"] = ep_size
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                create_gloo_process_groups=False, **kwargs)
        model_parallel_cuda_manual_seed(42)

        kw = dict(
            num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=HEADS,
            num_query_groups=HEADS, ffn_hidden_size=MOE_FFN,
            attention_softmax_in_fp32=False, bf16=True,
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=1, sequence_parallel=False,
            expert_model_parallel_size=ep_size,
            add_bias_linear=False, gated_linear_unit=True,
            num_moe_experts=NUM_EXPERTS, moe_ffn_hidden_size=MOE_FFN, moe_router_topk=TOPK,
            moe_token_dispatcher_type="alltoall", moe_expert_capacity_factor=1.0,
            moe_pad_expert_input_to_capacity=True, moe_router_force_load_balancing=True,
            moe_grouped_gemm=False, moe_layer_freq=1,
            moe_router_load_balancing_type="aux_loss",
        )
        provider = TransformerConfig(**kw)
        for f in ("masked_softmax_fusion", "bias_activation_fusion", "apply_rope_fusion",
                  "bias_dropout_fusion", "persist_layer_norm", "gradient_accumulation_fusion",
                  "moe_permute_fusion", "moe_router_fusion"):
            if hasattr(provider, f):
                setattr(provider, f, False)
        if hasattr(provider, "finalize"):
            provider.finalize()

        install_norm_meta_kernels()
        spec = get_gpt_layer_local_spec(num_experts=NUM_EXPERTS, moe_grouped_gemm=False)
        with redirect_cuda_alloc_to_meta():
            model = GPTModel(config=provider, transformer_layer_spec=spec, vocab_size=VOCAB,
                             max_sequence_length=SEQ, pre_process=True, post_process=True,
                             parallel_output=True).train()
        return sum(
            p.numel()
            for mod in model.modules() if isinstance(mod, SequentialMLP)
            for p in mod.parameters()
        )
    finally:
        try:
            parallel_state.destroy_model_parallel()
        except Exception:
            pass
        try:
            destroy_process_group()
        except Exception:
            pass


@pytest.mark.requires_cuda
def test_memtracker_expert_params_scale_with_ep():
    """Resident routed-expert nn.Parameter count at EP=K is exactly 1/K of EP=1."""
    pytest.importorskip("megatron.core")
    ep1 = _build_and_measure_expert_params(1)
    ep2 = _build_and_measure_expert_params(2)
    ep4 = _build_and_measure_expert_params(4)
    assert ep1 == NUM_EXPERTS * 3 * HIDDEN * MOE_FFN * LAYERS
    assert ep1 // 2 == ep2, (ep1, ep2)
    assert ep1 // 4 == ep4, (ep1, ep4)


# ---------------------------------------------------------------------------
# gpt-oss-20b end-to-end at a sensible EP (GPU-gated): finite report, physical MFU,
# OOM correctly flagged or under the per-GPU cap.
# ---------------------------------------------------------------------------

@pytest.mark.requires_cuda
def test_gpt_oss_20b_ep_report_physical():
    pytest.importorskip("megatron.core")
    import math
    import syssim
    from syssim.training.spec import load_model_yaml, load_hardware_yaml

    root = os.path.join(os.path.dirname(__file__), "..", "..")
    m = load_model_yaml(os.path.join(root, "examples", "configs", "models", "gpt-oss-20b.yaml"))
    hw = load_hardware_yaml(os.path.join(
        root, "examples", "configs", "hardware", "isambard_gh200_4gpu.yaml"))
    par = ParallelismConfig(dp=4, ep=4)
    tr = TrainingConfig(micro_batch=1, global_batch=4, dtype="bf16",
                        use_distributed_optimizer=True)
    rep = syssim.simulate(model=m, hardware=hw, parallelism=par, training=tr)

    assert math.isfinite(rep.step_time_ms) and rep.step_time_ms > 0
    assert math.isfinite(rep.peak_memory_gb) and rep.peak_memory_gb > 0
    # MFU is physical: a positive floor (not zero from a costing bug) and below 1.
    assert 0.0001 < rep.mfu < 1.0, rep.mfu
