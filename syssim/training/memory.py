"""Analytical per-rank memory accounting.

Megatron-style sharding assumed:
  - TP shards attention + FFN weight matrices.
  - Norms, embeddings, lm_head are NOT TP-sharded (replicated across TP group).
  - DP / CP replicate; ZeRO is out of scope for v1.
  - Sequence-parallel + context-parallel reduce activation memory by the
    corresponding factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .spec import ModelConfig, ParallelismConfig, TrainingConfig


def _attn_param_count(m: ModelConfig) -> int:
    """Count attention layer parameters (QKV + output projection)."""
    h = m.hidden_size
    head_dim = h // m.num_attention_heads
    q_params = h * (m.num_attention_heads * head_dim)
    kv_params = 2 * h * (m.num_query_groups * head_dim)
    o_params = (m.num_attention_heads * head_dim) * h
    return q_params + kv_params + o_params


def _expert_param_count(m: ModelConfig) -> int:
    """Per-layer parameter count of the routed experts ONLY (no router gate, no
    shared expert). This is the portion that shards across the expert-parallel
    group; the router gate and shared expert stay replicated. Returns 0 for a
    dense model (``num_experts`` None)."""
    if m.num_experts is None:
        return 0
    matmuls = 3 if m.swiglu else 2
    return m.num_experts * matmuls * m.hidden_size * m.moe_ffn_hidden_size


def _ffn_param_count(m: ModelConfig) -> int:
    """Per-layer FFN parameter count (EP=1 / replicated view).

    Dense: SwiGLU has 3 matrices, standard has 2.

    MoE (``num_experts`` set): every expert holds a full FFN, so the per-layer
    FFN params are ``num_experts * matmuls * h * moe_ffn_hidden_size`` plus the
    router gate (``h * num_experts``) and an optional shared expert
    (``matmuls * h * moe_shared_expert_intermediate_size``). This is the EP=1
    (fully-replicated) per-layer count; the per-rank byte functions divide the
    routed-expert portion by ``expert_group_size`` at EP>1 via
    ``_ffn_param_count_per_rank`` (this helper itself stays EP-invariant, so the
    FLOP budget and the EP=1 byte path remain byte-identical). ``moe_layer_freq``
    is honored only as a presence flag here: with the Phase-1 freq semantics
    (freq>=1 => every layer is MoE for gpt-oss) all layers are MoE, matching the
    per-layer convention of the dense path.
    """
    h = m.hidden_size
    if m.num_experts is not None:
        matmuls = 3 if m.swiglu else 2
        experts = _expert_param_count(m)
        router_gate = h * m.num_experts
        shared = 0
        if m.moe_shared_expert_intermediate_size:
            shared = matmuls * h * m.moe_shared_expert_intermediate_size
        return experts + router_gate + shared
    f = m.ffn_hidden_size
    return 3 * h * f if m.swiglu else 2 * h * f


def _ffn_param_count_per_rank(m: ModelConfig, p: ParallelismConfig) -> int:
    """Per-rank, per-layer FFN parameter count with expert-parallel sharding.

    At EP=1 (``expert_group_size == 1``) this equals ``_ffn_param_count(m)``
    exactly (dense and EP=1 stay byte-identical). At EP>1 only the routed-expert
    portion shards by ``expert_group_size`` (= ep*etp); the router gate and the
    shared expert stay replicated across the expert-parallel group. Used by the
    per-rank param/grad/optimizer byte functions so MemTracker-independent
    analytical memory and the DP all-reduce/optimizer timing honor EP.
    """
    egs = p.expert_group_size
    if m.num_experts is None or egs <= 1:
        return _ffn_param_count(m)
    return _ffn_param_count(m) - _expert_param_count(m) + _expert_param_count(m) // egs


def _norm_param_count(m: ModelConfig) -> int:
    """Count RMSNorm parameters (pre-attention + pre-FFN per layer)."""
    return 2 * m.hidden_size


def count_parameters(m: ModelConfig) -> int:
    """Count total trainable parameters for a decoder-only LM.

    Includes:
      - Per-layer: attention (QKV + output), FFN, norms
      - Embeddings: embedding matrix (tied with lm_head if enabled)
      - Final norm: single RMSNorm after all layers
      - LM head: separate only if not tied to embeddings

    Args:
        m: ModelConfig with all Megatron fields populated.

    Returns:
        Total parameter count.
    """
    per_layer = _attn_param_count(m) + _ffn_param_count(m) + _norm_param_count(m)
    embed = m.vocab_size * m.hidden_size
    final_norm = m.hidden_size
    lm_head = 0 if m.tie_word_embeddings else m.vocab_size * m.hidden_size
    return per_layer * m.num_layers + embed + final_norm + lm_head


def _weight_dtype_bytes(tr: TrainingConfig) -> int:
    if tr.fp8:
        return 1
    if tr.bf16 or tr.fp16:
        return 2
    return 4


def param_bytes_per_rank(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> int:
    """Per-rank weight memory in bytes (accounting for TP sharding).

    Sharded parameters (attention + FFN per layer) are divided by TP group size.
    Replicated parameters (norms, embeddings, lm_head) are kept on all ranks.

    Args:
        m: ModelConfig.
        p: ParallelismConfig.
        tr: TrainingConfig (for dtype).

    Returns:
        Bytes per rank for weights.
    """
    tp = p.tensor_model_parallel_size
    sharded = (_attn_param_count(m) + _ffn_param_count_per_rank(m, p)) * m.num_layers
    embed = m.vocab_size * m.hidden_size
    final_norm = m.hidden_size
    lm_head = 0 if m.tie_word_embeddings else m.vocab_size * m.hidden_size
    norms = _norm_param_count(m) * m.num_layers
    replicated = embed + final_norm + lm_head + norms
    elements = sharded // tp + replicated
    return elements * _weight_dtype_bytes(tr)


def sharded_param_bytes_per_rank(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> int:
    """Bytes of the TP-sharded (per-layer attention + FFN) parameters on one rank.

    This is the set the distributed optimizer (ZeRO-1) shards across the data-parallel group and
    all-gathers each step; the replicated parameters (embeddings, lm_head, norms) are NOT
    DP-gathered. Verified against the real NCCL all-gather size on GH200 (datatype bf16).
    """
    tp = p.tensor_model_parallel_size
    sharded = (_attn_param_count(m) + _ffn_param_count_per_rank(m, p)) * m.num_layers
    return (sharded // tp) * _weight_dtype_bytes(tr)


def expert_sharded_param_bytes_per_rank(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> int:
    """Per-rank bytes of the routed-expert TP-sharded params (the EP-sharded portion
    the distributed optimizer all-gathers over the expert-DP group, size dp/ep).
    Returns 0 for dense / EP=1. TP folds via expert_group_size (= ep*etp)."""
    egs = p.expert_group_size
    if m.num_experts is None or egs <= 1:
        return 0
    tp = p.tensor_model_parallel_size
    return (_expert_param_count(m) // egs * m.num_layers // tp) * _weight_dtype_bytes(tr)


def grad_bytes_per_rank(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> int:
    """Per-rank gradient memory in bytes.

    Gradients have the same shape as their corresponding parameters.

    Args:
        m: ModelConfig.
        p: ParallelismConfig.
        tr: TrainingConfig (for dtype).

    Returns:
        Bytes per rank for gradients.
    """
    return param_bytes_per_rank(m, p, tr)


def expert_grad_bytes_per_rank(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> int:
    """Per-rank gradient bytes of the routed experts ONLY (the EP-sharded portion).

    At EP>1 these reduce over the expert-DATA-parallel group (size dp/ep), NOT the
    full DP group. Returns 0 for a dense model or EP=1 (``expert_group_size <= 1``),
    so the dense / EP=1 grad-reduction path is unchanged. The expert weights are TP-
    sharded too (etp folds into expert_group_size); attention TP is excluded here.
    """
    egs = p.expert_group_size
    if m.num_experts is None or egs <= 1:
        return 0
    per_rank_expert_elems = _expert_param_count(m) // egs * m.num_layers
    return per_rank_expert_elems * _weight_dtype_bytes(tr)


def dense_grad_bytes_per_rank(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> int:
    """Per-rank gradient bytes of everything that reduces over the FULL DP group:
    attention, router gate, shared expert, norms, embeddings, lm_head — i.e. all
    grad bytes EXCEPT the routed-expert portion. Equals ``grad_bytes_per_rank`` at
    EP=1 / dense (``expert_grad_bytes_per_rank`` is 0 there)."""
    return grad_bytes_per_rank(m, p, tr) - expert_grad_bytes_per_rank(m, p, tr)


def optimizer_state_bytes_per_rank(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> int:
    """Per-rank Adam optimizer state in bytes (mixed precision).

    Adam maintains fp32 master weights + momentum (m) + variance (v): 12 bytes per parameter
    (4 master + 4 m + 4 v). Parameters shard across TP. With the distributed optimizer (ZeRO-1,
    `tr.use_distributed_optimizer`), the optimizer state is additionally sharded across the
    data-parallel group.

    Args:
        m: ModelConfig.
        p: ParallelismConfig.
        tr: TrainingConfig (dtype + distributed-optimizer flag).

    Returns:
        Bytes per rank for optimizer state.
    """
    tp = p.tensor_model_parallel_size
    sharded_elems = (_attn_param_count(m) + _ffn_param_count_per_rank(m, p)) * m.num_layers // tp
    embed = m.vocab_size * m.hidden_size
    final_norm = m.hidden_size
    lm_head = 0 if m.tie_word_embeddings else m.vocab_size * m.hidden_size
    norms = _norm_param_count(m) * m.num_layers
    replicated_elems = embed + final_norm + lm_head + norms
    total = (sharded_elems + replicated_elems) * 12
    if tr.use_distributed_optimizer:
        total //= p.data_parallel_size
    return total


@dataclass
class MemoryBreakdown:
    param_bytes: int
    grad_bytes: int
    optimizer_state_bytes: int
    activation_bytes: int
    pp_stage_memory_gb: Optional[list[float]] = None

    @property
    def peak_bytes(self) -> int:
        return self.param_bytes + self.grad_bytes + self.optimizer_state_bytes + self.activation_bytes

    @property
    def peak_gb(self) -> float:
        if self.pp_stage_memory_gb:
            return max(self.pp_stage_memory_gb)
        return self.peak_bytes / 1e9


def activation_bytes_per_rank(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> int:
    """Sum of dominant per-layer activations (norms + QKV + attn scores + FFN gate/up)."""
    b = tr.micro_batch_size
    s = m.seq_length
    h = m.hidden_size
    elem = _weight_dtype_bytes(tr)
    a_h = m.num_attention_heads

    per_layer_full = (
        6 * b * s * h                      # norms + Q/K/V + attn-out
        + b * a_h * s * s                  # softmax(QK^T)
        + 2 * b * s * m.ffn_hidden_size   # SwiGLU gate + up
    ) * elem
    per_layer_selective = (
        6 * b * s * h
        + 2 * b * s * m.ffn_hidden_size
    ) * elem

    if tr.recompute_granularity == "full":
        total = per_layer_full           # only one layer alive at a time
    elif tr.recompute_granularity == "selective":
        total = per_layer_selective * m.num_layers
    else:
        total = per_layer_full * m.num_layers

    sp_div = p.tensor_model_parallel_size if p.sequence_parallel else 1
    cp_div = p.context_parallel_size
    return total // (sp_div * cp_div)


def peak_memory_gb_per_rank(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> MemoryBreakdown:
    return MemoryBreakdown(
        param_bytes           = param_bytes_per_rank(m, p, tr),
        grad_bytes            = grad_bytes_per_rank(m, p, tr),
        optimizer_state_bytes = optimizer_state_bytes_per_rank(m, p, tr),
        activation_bytes      = activation_bytes_per_rank(m, p, tr),
    )


def _layers_per_stage(num_layers: int, pp_size: int) -> list[int]:
    """Megatron's default even split: first `num_layers % pp_size` stages get one extra layer."""
    base, rem = divmod(num_layers, pp_size)
    return [base + (1 if i < rem else 0) for i in range(pp_size)]


def per_pp_stage_peak_memory_gb(
    m: ModelConfig, p: ParallelismConfig, tr: TrainingConfig,
) -> list[float]:
    """Per-PP-stage peak memory (GB), one entry per pipeline stage.

    First stage owns the embedding; last stage owns the final norm + lm_head
    (when not tied); middle stages own only their transformer layers. Activations,
    grads, and optimizer state scale with the per-stage layer count.
    """
    pp = p.pipeline_model_parallel_size
    if pp == 1:
        return [peak_memory_gb_per_rank(m, p, tr).peak_gb]

    tp = p.tensor_model_parallel_size
    elem = _weight_dtype_bytes(tr)
    layers_per_stage = _layers_per_stage(m.num_layers, pp)

    embed_params = m.vocab_size * m.hidden_size
    final_norm_params = m.hidden_size
    lm_head_params = 0 if m.tie_word_embeddings else m.vocab_size * m.hidden_size
    norm_per_layer = _norm_param_count(m)
    sharded_per_layer = (_attn_param_count(m) + _ffn_param_count(m)) // tp

    out: list[float] = []
    for stage_idx, layers in enumerate(layers_per_stage):
        per_stage = ModelConfig(
            num_layers=layers,
            hidden_size=m.hidden_size,
            num_attention_heads=m.num_attention_heads,
            num_query_groups=m.num_query_groups,
            ffn_hidden_size=m.ffn_hidden_size,
            seq_length=m.seq_length,
            max_position_embeddings=m.max_position_embeddings,
            vocab_size=m.vocab_size,
            swiglu=m.swiglu, rope=m.rope, rope_theta=m.rope_theta,
            tie_word_embeddings=m.tie_word_embeddings,
            rms_norm_eps=m.rms_norm_eps,
        )

        # Stage-local transformer layers
        sharded_elems = sharded_per_layer * layers
        replicated_elems = norm_per_layer * layers
        # First stage: embedding (replicated, not TP-sharded)
        if stage_idx == 0:
            replicated_elems += embed_params
        # Last stage: final norm + lm_head
        if stage_idx == pp - 1:
            replicated_elems += final_norm_params + lm_head_params

        param_b = (sharded_elems + replicated_elems) * elem
        grad_b = param_b
        opt_b = (sharded_elems + replicated_elems) * 12
        act_b = activation_bytes_per_rank(per_stage, p, tr)

        out.append((param_b + grad_b + opt_b + act_b) / 1e9)
    return out
