"""Convenience wrappers for Hugging Face Transformers training."""

from dataclasses import replace
from typing import Any, Callable
import torch

try:
    from transformers import PreTrainedModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedModel = Any  # Type stub for when transformers not installed

from ..api import trace_model_for_training
from ..config import ExecutionMode, SimulatorConfig
from ..moe import (
    MoERuntimeConfig,
    build_moe_operator_graph,
    extract_hf_moe_spec,
)
from ..operator_graph import OperatorGraph


def trace_hf_model_for_training(
    model: PreTrainedModel,
    inputs: dict[str, torch.Tensor] | Any,
    config: SimulatorConfig,
    loss_fn: Callable | None = None,
    labels: torch.Tensor | None = None,
) -> OperatorGraph:
    """Trace a Hugging Face model for training (forward + backward).

    Args:
        model: PreTrainedModel (e.g., GPT2LMHeadModel, LlamaForCausalLM)
        inputs: Model inputs as dict
            - {"input_ids": tensor, "attention_mask": tensor}
            - Can include "labels" for built-in loss
        config: SimulatorConfig with hardware info
        loss_fn: Optional custom loss function. If None, uses model's built-in loss
        labels: Optional labels tensor (alternative to including in inputs)

    Returns:
        OperatorGraph with forward + backward operations traced

    Example:
        >>> from transformers import GPT2LMHeadModel, AutoTokenizer
        >>> from syssim import HardwareInfo, SimulatorConfig
        >>>
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>>
        >>> inputs = tokenizer("Hello world", return_tensors="pt")
        >>> inputs["labels"] = inputs["input_ids"].clone()
        >>>
        >>> hw = HardwareInfo(989.0, 989.0, 3350.0)
        >>> config = SimulatorConfig(hw_info=hw)
        >>>
        >>> graph = trace_hf_model_for_training(model, inputs, config)
        >>> print(f"Training step time: {graph.compute_critical_path():.2f} ms")
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "transformers package is required for Hugging Face integration. "
            "Install with: pip install transformers"
        )

    # Convert BatchEncoding to dict if needed
    if hasattr(inputs, "data"):
        inputs = dict(inputs.data)

    # Add labels if provided separately
    if labels is not None and "labels" not in inputs:
        inputs = dict(inputs)
        inputs["labels"] = labels

    # Ensure training mode — tracer converts params/inputs to fake CUDA internally
    model.train()

    # Define loss function
    if loss_fn is None:
        # Use model's built-in loss (requires labels in inputs)
        if "labels" in inputs:
            loss_fn = lambda out: out.loss if hasattr(out, "loss") else out[0]
        else:
            # Fallback: language modeling loss (shift logits/labels)
            loss_fn = _create_lm_loss_fn(inputs["input_ids"])

    # Call core tracer — it converts model params and inputs to fake CUDA
    return trace_model_for_training(model, inputs, config, loss_fn=loss_fn)


def trace_hf_training_step(
    model: PreTrainedModel,
    batch: dict[str, torch.Tensor],
    config: SimulatorConfig,
    use_mixed_precision: bool = False,
) -> OperatorGraph:
    """Trace a single training step (forward + loss + backward).

    Args:
        model: PreTrainedModel in training mode
        batch: Training batch with input_ids, attention_mask, labels
        config: SimulatorConfig
        use_mixed_precision: Whether to use FP16/BF16 (affects tensor dtypes)

    Returns:
        OperatorGraph for one training iteration

    Example:
        >>> from transformers import GPT2LMHeadModel, AutoTokenizer
        >>> from syssim import HardwareInfo, SimulatorConfig
        >>>
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>>
        >>> batch = tokenizer(
        ...     ["Hello world", "Machine learning"],
        ...     return_tensors="pt",
        ...     padding=True
        ... )
        >>> batch["labels"] = batch["input_ids"].clone()
        >>>
        >>> hw = HardwareInfo(989.0, 989.0, 3350.0)
        >>> config = SimulatorConfig(hw_info=hw)
        >>>
        >>> graph = trace_hf_training_step(model, batch, config)
        >>> print(f"Batch size: {batch['input_ids'].shape[0]}")
        >>> print(f"Training step time: {graph.compute_critical_path():.2f} ms")
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "transformers package is required for Hugging Face integration. "
            "Install with: pip install transformers"
        )

    if use_mixed_precision:
        # Convert inputs to half precision
        batch = {
            k: v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
            for k, v in batch.items()
        }

    return trace_hf_model_for_training(model, batch, config)


def trace_hf_moe_model_for_training(
    model: object,
    inputs: dict[str, torch.Tensor] | object,
    config: SimulatorConfig,
    runtime: MoERuntimeConfig | None = None,
) -> OperatorGraph:
    """Build a MoE operator graph for a Hugging Face causal LM training step."""
    spec = extract_hf_moe_spec(model)
    if runtime is None:
        runtime = _runtime_from_inputs(inputs, ExecutionMode.TRAINING, _model_dtype(model))
    else:
        runtime = replace(runtime, mode=ExecutionMode.TRAINING)
    return build_moe_operator_graph(spec, runtime, config)


def trace_hf_moe_model_for_inference(
    model: object,
    inputs: dict[str, torch.Tensor] | object,
    config: SimulatorConfig,
    mode: str = "prefill",
    runtime: MoERuntimeConfig | None = None,
) -> OperatorGraph:
    """Build a MoE operator graph for Hugging Face prefill or decode."""
    if mode not in {"prefill", "decode"}:
        raise ValueError("mode must be 'prefill' or 'decode'")

    exec_mode = ExecutionMode.PREFILL if mode == "prefill" else ExecutionMode.DECODE
    spec = extract_hf_moe_spec(model)
    if runtime is None:
        runtime = _runtime_from_inputs(inputs, exec_mode, _model_dtype(model))
    else:
        runtime = replace(runtime, mode=exec_mode)
    return build_moe_operator_graph(spec, runtime, config)


def _create_lm_loss_fn(input_ids: torch.Tensor) -> Callable:
    """Create language modeling loss function (shift logits/labels)."""
    def lm_loss(outputs):
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        # Shift so tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # Cross entropy
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss
    return lm_loss


def _runtime_from_inputs(
    inputs: dict[str, torch.Tensor] | object,
    mode: ExecutionMode,
    dtype: torch.dtype | str,
) -> MoERuntimeConfig:
    if hasattr(inputs, "data"):
        inputs = dict(inputs.data)
    if not isinstance(inputs, dict) or "input_ids" not in inputs:
        raise ValueError("MoE runtime inference requires inputs['input_ids'] or an explicit runtime")

    input_ids = inputs["input_ids"]
    if not hasattr(input_ids, "shape") or len(input_ids.shape) < 2:
        raise ValueError("MoE runtime inference requires inputs['input_ids'] with shape [batch, seq]")
    return MoERuntimeConfig(
        batch_size=int(input_ids.shape[0]),
        seq_len=int(input_ids.shape[1]),
        mode=mode,
        dtype=dtype,
    )


def _model_dtype(model: object) -> torch.dtype | str:
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    config = getattr(model, "config", None)
    dtype = getattr(config, "torch_dtype", None)
    return dtype if dtype is not None else torch.bfloat16
