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
                "HFModel requires `pip install -e '.[huggingface]'` "
                "(installs transformers + megatron-bridge)."
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
            ffn_hidden_size=model.ffn_hidden_size,
            attention_softmax_in_fp32=False,
        )
    else:
        raise TypeError(f"unsupported model source: {type(model).__name__}")

    provider.tensor_model_parallel_size = parallelism.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = 1
    provider.sequence_parallel = parallelism.sequence_parallel
    provider.context_parallel_size = parallelism.context_parallel_size
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
    if hasattr(provider, "finalize"):
        provider.finalize()
    return provider
