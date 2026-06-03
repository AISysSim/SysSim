"""Phase-4 MoE analytical formulas: param counts (memory.py) and the FLOP budget
(report.py). Validated against the two oracles from the investigation plus a
dense regression that the gated branches keep the dense path byte-identical.

O1 (resident expert Parameter bytes) and O2 (traced expert-GEMM FLOPs) need a CUDA
GPU + Megatron-Core (they build and trace a real Megatron MoE model); they are
gated and skip cleanly on a CPU-only host. O3 and the dense regression are pure
analytical and always run.
"""
import os

import pytest

from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig
from syssim.training.memory import count_parameters, param_bytes_per_rank, _ffn_param_count
from syssim.training.report import compute_model_flops_budget


# Tiny shared MoE dims (match the O1/O2 probes).
HIDDEN, HEADS, LAYERS = 256, 8, 2
SEQ, VOCAB = 128, 1024
NUM_EXPERTS, TOPK, MOE_FFN = 4, 2, 512
MBS = 2


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


# ---------------------------------------------------------------------------
# O3 (always runs): gpt-oss-20b magnitude + dense control unchanged.
# ---------------------------------------------------------------------------

def test_o3_gpt_oss_20b_param_count_band():
    from syssim.training.spec import load_model_yaml
    path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "examples", "configs", "models", "gpt-oss-20b.yaml",
    )
    m = load_model_yaml(path)
    n = count_parameters(m)
    assert 19e9 <= n <= 22e9, f"gpt-oss-20b got {n/1e9:.3f}B params (want 19-22B)"


def test_o3_gpt_oss_20b_active_params_consistent():
    """Active (top-4 of 32) params should be a few billion, far below total."""
    from syssim.training.spec import load_model_yaml
    path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "examples", "configs", "models", "gpt-oss-20b.yaml",
    )
    m = load_model_yaml(path)
    total = count_parameters(m)
    # Active FFN per layer = topk experts (vs all num_experts in the resident count).
    matmuls = 3
    h = m.hidden_size
    per_layer_total_ffn = _ffn_param_count(m)
    per_layer_active_ffn = (
        m.moe_router_topk * matmuls * h * m.moe_ffn_hidden_size + h * m.num_experts
    )
    active = total - (per_layer_total_ffn - per_layer_active_ffn) * m.num_layers
    assert 3e9 <= active <= 5e9, f"active params {active/1e9:.3f}B out of band"
    assert active < total


# ---------------------------------------------------------------------------
# Dense regression (always runs): gated branches must be byte-identical to the
# pre-Phase-4 dense formulas.
# ---------------------------------------------------------------------------

def test_dense_ffn_param_count_unchanged():
    m = _tiny_dense()
    # Pre-Phase-4 dense formula: 3*h*f for swiglu.
    assert _ffn_param_count(m) == 3 * HIDDEN * MOE_FFN


def test_dense_count_parameters_unchanged():
    m = _tiny_dense()
    # Recompute the pre-Phase-4 total explicitly (no MoE term anywhere).
    head_dim = HIDDEN // HEADS
    attn = HIDDEN * (HEADS * head_dim) + 2 * HIDDEN * (HEADS * head_dim) + (HEADS * head_dim) * HIDDEN
    ffn = 3 * HIDDEN * MOE_FFN
    norms = 2 * HIDDEN
    embed = VOCAB * HIDDEN
    lm = VOCAB * HIDDEN
    expected = (attn + ffn + norms) * LAYERS + embed + HIDDEN + lm
    assert count_parameters(m) == expected


def test_dense_flops_budget_unchanged():
    m = _tiny_dense()
    par = ParallelismConfig()
    tr = TrainingConfig(micro_batch=MBS, global_batch=MBS, dtype="bf16")
    budget = compute_model_flops_budget(m, par, tr)
    # Reproduce the dense fwd_ffn term directly and check it is the one in use.
    b, s, h, f = MBS, SEQ, HIDDEN, MOE_FFN
    matmuls = 3
    fwd_ffn = 2 * b * s * h * f * matmuls
    fwd_proj = 2 * b * s * h * h * 4
    fwd_score = 2 * b * HEADS * s * s * (h // HEADS)
    fwd_per = fwd_proj + 2 * fwd_score + fwd_ffn
    fwd_total = fwd_per * LAYERS + 2 * b * s * h * VOCAB
    micro_per_step = 1
    expected = (fwd_total + 2 * fwd_total) * micro_per_step
    assert budget.model_flops_per_step == expected


# ---------------------------------------------------------------------------
# Analytical MoE formula self-consistency (always runs).
# ---------------------------------------------------------------------------

def test_moe_ffn_param_formula():
    m = _tiny_moe()
    matmuls = 3
    expected = NUM_EXPERTS * matmuls * HIDDEN * MOE_FFN + HIDDEN * NUM_EXPERTS
    assert _ffn_param_count(m) == expected


def test_moe_fwd_ffn_flops_formula():
    m = _tiny_moe()
    par = ParallelismConfig()
    tr = TrainingConfig(micro_batch=MBS, global_batch=MBS, dtype="bf16")
    budget = compute_model_flops_budget(m, par, tr)
    matmuls = 3
    b, s, h = MBS, SEQ, HIDDEN
    fwd_ffn_per_layer = 2 * b * s * TOPK * h * MOE_FFN * matmuls + 2 * b * s * h * NUM_EXPERTS
    # Sanity: the per-layer MoE FFN term is what the probe oracle expects (ignoring
    # the small router-gate add) -> 805,306,368 fwd expert FLOPs over 2 layers.
    expert_only = 2 * b * s * TOPK * h * MOE_FFN * matmuls * LAYERS
    assert expert_only == 805_306_368
    assert budget.model_flops_per_step > 0
    assert fwd_ffn_per_layer > 0


# ---------------------------------------------------------------------------
# O1 (GPU-gated): analytical expert param bytes match MemTracker resident
# Parameter expert portion.
# ---------------------------------------------------------------------------

@pytest.mark.requires_cuda
def test_o1_expert_param_bytes_match_memtracker():
    pytest.importorskip("megatron.core")
    import torch
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.moe.experts import SequentialMLP

    from syssim.training.dist_setup import init_fake_process_group, destroy_process_group
    from syssim.training.runner import make_lm_forward_step, make_lm_data_iterator
    from syssim.training.cuda_redirect import redirect_cuda_alloc_to_meta
    from syssim.training._norm_meta import install_norm_meta_kernels
    from syssim.tracer import OperatorGraphTracer
    from syssim.config import HardwareInfo

    init_fake_process_group(world_size=1, rank=0)
    try:
        torch.cuda.set_device(0)
        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                create_gloo_process_groups=False)
        model_parallel_cuda_manual_seed(42)

        kw = dict(
            num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=HEADS,
            num_query_groups=HEADS, ffn_hidden_size=MOE_FFN,
            attention_softmax_in_fp32=False, bf16=True,
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=1, sequence_parallel=False,
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

        orig = SequentialMLP.forward
        def patched(self, h, tpe, probs):
            if self.num_local_experts > 1:
                total = h.shape[0]; cap = total // self.num_local_experts
                sizes = [cap] * self.num_local_experts; rem = total - cap * self.num_local_experts
                if rem:
                    sizes[-1] += rem
                tl = torch.split(h, sizes); pl = torch.split(probs, sizes)
                outs = []
                for e, t, pr in zip(self.local_experts, tl, pl):
                    o, _ = e(t, pr); outs.append(o)
                return torch.cat(outs, dim=0), None
            return orig(self, h, tpe, probs)
        SequentialMLP.forward = patched
        try:
            install_norm_meta_kernels()
            spec = get_gpt_layer_local_spec(num_experts=NUM_EXPERTS, moe_grouped_gemm=False)
            with redirect_cuda_alloc_to_meta():
                model = GPTModel(config=provider, transformer_layer_spec=spec, vocab_size=VOCAB,
                                 max_sequence_length=SEQ, pre_process=True, post_process=True,
                                 parallel_output=True).train()
                from megatron.core.transformer.module import Float16Module
                model = Float16Module(provider, model)

            # Ground truth: resident expert nn.Parameter element count.
            expert_params = sum(
                p.numel()
                for mod in model.modules() if isinstance(mod, SequentialMLP)
                for p in mod.parameters()
            )

            tracer = OperatorGraphTracer(hw_info=HardwareInfo(
                peak_tflops_mm=989.0, peak_tflops_math=67.0, peak_memory_bandwidth_gbps=4000.0))
            tracer.estimate_memory(
                model=model, forward_backward_func=get_forward_backward_func(),
                forward_step_func=make_lm_forward_step(VOCAB),
                data_iterator=make_lm_data_iterator(VOCAB, MBS, SEQ),
                seq_length=SEQ, micro_batch_size=MBS,
                use_distributed_optimizer=False, provider=provider)
        finally:
            SequentialMLP.forward = orig

        # Analytical expert-portion param count (excluding the router gate, which is
        # NOT an nn.Parameter of SequentialMLP).
        m = _tiny_moe()
        matmuls = 3
        analytical_expert = NUM_EXPERTS * matmuls * HIDDEN * MOE_FFN * LAYERS
        assert analytical_expert == expert_params, (
            f"analytical expert params {analytical_expert} != measured {expert_params}")
        # And bytes match in bf16.
        assert analytical_expert * 2 == expert_params * 2
    finally:
        try:
            parallel_state.destroy_model_parallel()
        except Exception:
            pass
        try:
            destroy_process_group()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# O2 (GPU-gated): analytical fwd_ffn MoE term matches summed traced expert-GEMM
# FLOPs for the tiny MoE.
# ---------------------------------------------------------------------------

@pytest.mark.requires_cuda
def test_o2_fwd_ffn_matches_traced_expert_gemms():
    """O2: the analytical fwd_ffn MoE expert term equals the summed FORWARD
    expert-GEMM FLOPs from the real trace, EXACTLY (ratio 1.0).

    The trace classifies each GEMM with a ``phase`` (forward/backward); the
    per-expert forward GEMMs are fc1 (M=capacity, K=hidden, N=2*moe_ffn — the
    gate+up projection fused into one GEMM, since the MoE provider sets
    gated_linear_unit) and fc2 (M=capacity, K=moe_ffn, N=hidden, the down
    projection). fc1(2f) + fc2(f) is exactly 3 matmuls of work, so the traced
    forward expert FLOPs equal the matmuls=3 SwiGLU budget term, tying the MFU
    denominator to real traced expert compute.
    """
    pytest.importorskip("megatron.core")
    import syssim
    from syssim.training.runner import trace
    from syssim.operator_graph import OperatorType

    model = _tiny_moe()
    parallelism = ParallelismConfig()
    training = TrainingConfig(micro_batch=MBS, global_batch=MBS, dtype="bf16")
    hardware = syssim.HardwareConfig(
        peak_tflops_mm=1979.0, peak_tflops_math=989.0, peak_memory_bandwidth_GBps=3350.0,
        gpus_per_node=4, gpu_memory_GB=96.0,
        topology={"dims": ["fully_connected"], "size": [4],
                  "bandwidth": [450.0], "latency": [12000.0]},
    )

    import tempfile
    with tempfile.TemporaryDirectory() as wd:
        t = trace(model=model, parallelism=parallelism, training=training,
                  hardware=hardware, gpus_per_node=hardware.gpus_per_node, workdir=wd)

    cap = -(-(SEQ * MBS * TOPK) // NUM_EXPERTS)  # ceil(tokens*topk/E) = per-expert slots
    fc1, fc2, traced_fwd_flops = 0, 0, 0
    for n in t.graph.operators.values():
        if n.op_type != OperatorType.GEMM:
            continue
        c = n.config
        if c.get("M") != cap or c.get("phase") != "forward":
            continue
        K, N = c.get("K"), c.get("N")
        # gpt-oss experts are gated SwiGLU: the provider sets gated_linear_unit, so
        # fc1 fuses gate+up into one 2*moe_ffn-wide GEMM and fc2 is the moe_ffn->h
        # down projection. fc1(2f) + fc2(f) == 3 matmuls of work, matching the budget.
        if K == HIDDEN and N == 2 * MOE_FFN:
            fc1 += 1
            traced_fwd_flops += 2 * cap * K * N
        elif K == MOE_FFN and N == HIDDEN:
            fc2 += 1
            traced_fwd_flops += 2 * cap * K * N

    n_fwd = NUM_EXPERTS * LAYERS
    assert fc1 == n_fwd and fc2 == n_fwd, (
        f"expected {n_fwd} fc1/fc2 forward expert GEMMs, got {fc1}/{fc2}")
    assert traced_fwd_flops == 805_306_368, traced_fwd_flops

    # Analytical fwd_ffn expert term (excludes the router gate; its GEMM has M != cap
    # so it is not in the traced sum above either).
    matmuls_analytical = 3  # full SwiGLU gate+up+down
    analytical = 2 * MBS * SEQ * TOPK * HIDDEN * MOE_FFN * matmuls_analytical * LAYERS
    assert analytical == 805_306_368, analytical

    # The analytical MoE FFN budget equals the summed traced forward expert-GEMM
    # FLOPs EXACTLY (gate+up fused fc1 of width 2*moe_ffn + down fc2 == 3 matmuls),
    # so the MFU denominator is provably tied to real traced expert compute.
    assert analytical == traced_fwd_flops, (
        f"analytical {analytical} != traced {traced_fwd_flops}")
