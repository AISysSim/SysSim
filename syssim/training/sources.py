"""Model source constructors: HFModel (HuggingFace via Megatron-Bridge) + CustomModel placeholder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HFModel:
    """HuggingFace-source model spec.

    Architecture is resolved lazily via `megatron.bridge.AutoBridge.from_hf_config`
    at trace time — no weights are downloaded. `overrides` is applied to the
    resolved Megatron provider before `finalize()`.
    """
    huggingface: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomModel:
    """Reserved API symbol for the deferred custom-nn.Module source.

    v1 raises NotImplementedError at construction time. The eventual implementation
    will accept (module, forward_backward_func, forward_step_func, data_iterator)
    and pass them to OperatorGraphTracer.trace(...) — the same call shape as
    Megatron sources.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CustomModel support is planned but not implemented in v1. "
            "Use a model YAML or HFModel for now."
        )


def resolve_megatron_provider(
    model: "ModelConfig | HFModel",
    parallelism: "ParallelismConfig",
    training: "TrainingConfig",
):
    """Return a finalized Megatron `TransformerConfig` for the given source.

    For ModelConfig: builds a TransformerConfig directly from architecture fields.
    For HFModel: resolves via `megatron.bridge.AutoBridge.from_hf_config`.

    Applies parallelism + dtype and calls finalize() before return.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig
    from .spec import ModelConfig

    if isinstance(model, HFModel):
        try:
            from transformers import AutoConfig
            from megatron.bridge import AutoBridge
        except ImportError as e:
            raise ImportError(
                "HFModel requires transformers + megatron-bridge "
                "(installed by `pip install -e .`)."
            ) from e
        hf_config = AutoConfig.from_pretrained(model.huggingface)
        bridge = AutoBridge.from_hf_config(hf_config)
        provider = bridge.to_megatron_provider()
        for k, v in model.overrides.items():
            setattr(provider, k, v)
    elif isinstance(model, ModelConfig):
        provider = TransformerConfig(
            num_layers=model.num_layers,
            hidden_size=model.hidden_size,
            num_attention_heads=model.num_attention_heads,
            num_query_groups=model.num_query_groups,
            kv_channels=model.kv_channels,   # None -> Megatron derives hidden_size // heads
            ffn_hidden_size=model.ffn_hidden_size,
            attention_softmax_in_fp32=False,
        )
        # MoE architecture — only forwarded when this is an MoE model. Dense models
        # (num_experts is None) leave the provider untouched (byte-identical path).
        if model.num_experts is not None:
            provider.num_moe_experts = model.num_experts
            provider.moe_ffn_hidden_size = model.moe_ffn_hidden_size
            provider.moe_router_topk = model.moe_router_topk if model.moe_router_topk is not None else 1
            provider.moe_shared_expert_intermediate_size = model.moe_shared_expert_intermediate_size
            provider.moe_layer_freq = model.moe_layer_freq if model.moe_layer_freq is not None else 1
            provider.moe_token_dispatcher_type = model.moe_token_dispatcher_type or "alltoall"
            # gpt-oss experts are gated SwiGLU. The provider does not derive
            # gated_linear_unit from swiglu, so set it explicitly here (MoE path
            # only): without it the expert MLP traces as a non-gated 2-matmul FFN
            # (fc1 width = moe_ffn), which undercounts the real gate+up projection
            # and is inconsistent with the matmuls=3 analytical FLOP/param formulas.
            # With it, fc1 is 2*moe_ffn wide and the traced expert FLOPs equal the
            # SwiGLU budget exactly.
            if model.swiglu:
                import torch.nn.functional as _F
                provider.gated_linear_unit = True
                provider.activation_func = _F.silu
            # Trace-safe INTERNAL flags (not user config): force the static drop_and_pad
            # alltoall branch so the MoE forward traces under fake tensors with uniform
            # per-expert load, and keep experts sequential (fake-safe, FLOP-counted natively).
            provider.moe_expert_capacity_factor = 1.0
            provider.moe_pad_expert_input_to_capacity = True
            provider.moe_router_force_load_balancing = True
            provider.moe_grouped_gemm = False
            # Drop linear bias on the MoE path. With add_bias_linear, the expert MLP
            # applies the router probs to the bias via an in-place `output += ...` on a
            # tensor-parallel autograd view, which torch forbids during the fake-tensor
            # backward trace (inplace-on-view of a custom Function output). Bias is a
            # negligible-FLOP/param term; disabling it keeps the MoE forward traceable.
            provider.add_bias_linear = False
    else:
        raise TypeError(f"unsupported model source: {type(model).__name__}")

    provider.tensor_model_parallel_size = parallelism.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = 1
    provider.sequence_parallel = parallelism.sequence_parallel
    provider.context_parallel_size = parallelism.context_parallel_size
    # Expert parallelism — only set when requested, so a dense run leaves these at
    # the provider defaults (keeps the dense provider byte-identical to today).
    if parallelism.expert_model_parallel_size > 1:
        provider.expert_model_parallel_size = parallelism.expert_model_parallel_size
    if parallelism.expert_tensor_parallel_size > 1:
        provider.expert_tensor_parallel_size = parallelism.expert_tensor_parallel_size
    provider.bf16 = training.bf16
    provider.fp16 = training.fp16
    # pipeline_dtype is required by Megatron p2p when PP > 1 (recv_prev path).
    # Set it to match the compute dtype even when pipeline_model_parallel_size is
    # patched to 1 per-stage, because the schedule still issues recv_forward on
    # non-first stages.
    if parallelism.pipeline_model_parallel_size > 1:
        import torch
        if training.bf16:
            provider.pipeline_dtype = torch.bfloat16
        elif training.fp16:
            provider.pipeline_dtype = torch.float16
        else:
            provider.pipeline_dtype = torch.float32
    # Activation recompute (checkpointing): flow it onto the traced model so the MemTracker
    # physically measures the reduced activation footprint, matching what real megatron stores.
    rg = training.recompute_granularity
    if rg == "full":
        provider.recompute_granularity = "full"
        provider.recompute_method = "uniform"
        provider.recompute_num_layers = 1
    elif rg == "selective":
        provider.recompute_granularity = "selective"

    # Disable fused custom kernels for tracing. SysSim traces under FakeTensorMode; fused ops run
    # real CUDA kernels that read fake-tensor data -> CUDA illegal-memory-access (surfacing later at
    # the next sync, e.g. attention's RNG fork). Forcing the torch-native (aten) paths keeps the
    # graph FakeTensor-safe and lets SysSim cost-model these ops directly. (apex/TE fused norms come
    # from the layer spec, not these flags, and are handled by _norm_meta meta kernels.)
    for _flag in ("masked_softmax_fusion", "bias_activation_fusion", "apply_rope_fusion",
                  "bias_dropout_fusion", "persist_layer_norm", "gradient_accumulation_fusion",
                  "moe_permute_fusion", "moe_router_fusion"):
        if hasattr(provider, _flag):
            setattr(provider, _flag, False)
    if hasattr(provider, "finalize"):
        provider.finalize()
    return provider
