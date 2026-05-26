"""Public API for rlsysim tracing and simulation."""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from .config import ExecutionMode, SimulatorConfig
from .tracer import OperatorGraphTracer
from .operator_graph import OperatorGraph

if TYPE_CHECKING:
    from .config_plena import PLENAConfig


@dataclass
class TrainingMemoryEstimate:
    """Coarse training memory estimate in decimal GB."""

    parameter_gb: float
    gradient_gb: float
    optimizer_state_gb: float
    activation_peak_gb: float | None = None
    workspace_gb: float = 0.0
    fp32_master_gb: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_model_state_gb(self) -> float:
        return (
            self.parameter_gb
            + self.gradient_gb
            + self.optimizer_state_gb
            + self.fp32_master_gb
        )

    @property
    def total_peak_gb(self) -> float | None:
        if self.activation_peak_gb is None:
            return None
        return self.total_model_state_gb + self.activation_peak_gb + self.workspace_gb

    def summary(self) -> str:
        activation = (
            "unavailable"
            if self.activation_peak_gb is None
            else f"{self.activation_peak_gb:.2f} GB"
        )
        total = (
            "unavailable"
            if self.total_peak_gb is None
            else f"{self.total_peak_gb:.2f} GB"
        )
        return "\n".join(
            [
                f"Parameters       : {self.parameter_gb:.2f} GB",
                f"Gradients        : {self.gradient_gb:.2f} GB",
                f"Optimizer states : {self.optimizer_state_gb:.2f} GB",
                f"FP32 master      : {self.fp32_master_gb:.2f} GB",
                f"Activation peak  : {activation}",
                f"Workspace        : {self.workspace_gb:.2f} GB",
                f"Model states     : {self.total_model_state_gb:.2f} GB",
                f"Total peak       : {total}",
            ]
        )


@dataclass
class TrainingPerformanceResult:
    """Result returned by estimate_training_performance."""

    wall_time_ms: float
    graph: OperatorGraph
    memory: TrainingMemoryEstimate
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    forward_ms: float | None = None
    backward_ms: float | None = None
    optimizer_ms: float | None = None


def trace_model_for_training(
    model: nn.Module,
    example_inputs: Any,
    config: SimulatorConfig,
    loss_fn: Any = None,
) -> OperatorGraph:
    """Trace a PyTorch model for training (forward + backward).

    Args:
        model: The PyTorch model to trace.
        example_inputs: Example inputs for shape inference (tensor, tuple, list, or dict).
        config: SimulatorConfig with HardwareInfo for roofline estimation.
        loss_fn: Callable that reduces the model output to a scalar for backward.
                 Defaults to ``lambda out: out.sum()``.

    Returns:
        An OperatorGraph containing the traced operations and their dependencies.
    """
    tracer = OperatorGraphTracer(
        hw_info=config.hw_info,
        execution_mode=ExecutionMode.TRAINING,
        cache_seq_len=0,
    )
    return tracer.trace(model, example_inputs, forward_backward=True, loss_fn=loss_fn)


def estimate_training_performance(
    model: nn.Module | str,
    example_inputs: Any | None = None,
    accelerator: str = "plena",
    accelerator_config: Any | None = None,
    batch_size: int | None = None,
    seq_len: int | None = None,
    dtype: str | torch.dtype = "bf16",
    optimizer: str = "adamw",
    include_fp32_master_weights: bool = False,
    loss_fn: Any = None,
    num_parameters_override: int | None = None,
) -> TrainingPerformanceResult:
    """Estimate one training step and model-state memory.

    PLENA uses the PLENA cycle estimator instead of the GPU roofline +
    efficiency-model path.
    """
    if isinstance(model, str):
        model = _load_hf_model_on_meta(model, dtype)

    if example_inputs is None:
        if batch_size is None or seq_len is None:
            raise ValueError("batch_size and seq_len are required when example_inputs is not provided")
        example_inputs = _make_lm_training_batch(model, batch_size, seq_len)

    accelerator_key = accelerator.lower()
    if accelerator_key != "plena":
        raise ValueError("estimate_training_performance currently supports accelerator='plena'")
    if accelerator_config is None:
        raise ValueError("accelerator_config is required for PLENA estimates")

    from .compute.plena_backend import PLENAEstimator

    tracer = OperatorGraphTracer(
        hw_info=None,
        execution_mode=ExecutionMode.TRAINING,
        cache_seq_len=0,
        plena_estimator=PLENAEstimator(accelerator_config),
    )
    graph = tracer.trace(model, example_inputs, forward_backward=True, loss_fn=loss_fn)
    wall_time_ms = graph.compute_critical_path()

    memory = _estimate_training_memory(
        model,
        dtype=dtype,
        optimizer=optimizer,
        include_fp32_master_weights=include_fp32_master_weights,
        num_parameters_override=num_parameters_override,
    )

    assumptions = [
        "PLENA op latency uses simulator cycle estimates.",
        "Single-accelerator estimate; no distributed communication is included.",
        "Optimizer update time is not modeled.",
    ]
    warnings = list(memory.warnings)

    return TrainingPerformanceResult(
        wall_time_ms=wall_time_ms,
        graph=graph,
        memory=memory,
        assumptions=assumptions,
        warnings=warnings,
        optimizer_ms=None,
    )


def trace_model_for_inference(
    model: nn.Module,
    example_inputs: Any,
    config: SimulatorConfig,
    mode: str = "prefill",
) -> OperatorGraph:
    """Trace a PyTorch model for inference (forward only).

    Args:
        model: The PyTorch model to trace.
        example_inputs: Example inputs for shape inference (tensor, tuple, list, or dict).
        config: SimulatorConfig with HardwareInfo for roofline estimation.
        mode: Inference mode, either "prefill" or "decode".

    Returns:
        An OperatorGraph containing the traced operations and their dependencies.
    """
    mode_map = {
        "prefill": ExecutionMode.PREFILL,
        "decode": ExecutionMode.DECODE,
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid inference mode '{mode}', expected 'prefill' or 'decode'")
    execution_mode = mode_map[mode]
    cache_seq_len = config.cache_seq_len if execution_mode == ExecutionMode.DECODE else 0

    tracer = OperatorGraphTracer(
        hw_info=config.hw_info,
        execution_mode=execution_mode,
        cache_seq_len=cache_seq_len,
    )
    return tracer.trace(model, example_inputs, forward_backward=False, loss_fn=None)


def set_efficiency_model_dir(model_dir: str) -> None:
    """Configure directory containing trained efficiency models.

    Args:
        model_dir: Path to directory with model files (*.pth).

    Example:
        >>> from syssim import set_efficiency_model_dir, trace_model_for_inference
        >>> set_efficiency_model_dir("./trained_models")
        >>> graph = trace_model_for_inference(model, inputs, config)
    """
    from .compute.efficiency_models import set_backend_dir
    set_backend_dir(model_dir)


def trace_model_for_plena(
    model: nn.Module,
    example_inputs: Any,
    plena_config: "PLENAConfig",
    mode: str = "prefill",
) -> OperatorGraph:
    """Trace a PyTorch model with PLENA cycle-based estimation.

    Uses PLENA's analytic performance model instead of GPU roofline
    to estimate operator execution times on the PLENA accelerator.

    Args:
        model: The PyTorch model to trace.
        example_inputs: Example inputs for shape inference (tensor, tuple, list, or dict).
        plena_config: PLENAConfig with paths to PLENA configuration files.
        mode: Inference mode, either "prefill" or "decode".

    Returns:
        An OperatorGraph containing the traced operations with PLENA cycle estimates.

    Example:
        >>> from syssim import trace_model_for_plena
        >>> from syssim.config_plena import PLENAConfig
        >>>
        >>> config = PLENAConfig.from_plena_submodule()
        >>> graph = trace_model_for_plena(model, inputs, config, mode="prefill")
        >>> print(f"Critical path: {graph.compute_critical_path():.2f} ms")
    """
    from .compute.plena_backend import PLENAEstimator

    mode_map = {
        "prefill": ExecutionMode.PREFILL,
        "decode": ExecutionMode.DECODE,
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid inference mode '{mode}', expected 'prefill' or 'decode'")
    execution_mode = mode_map[mode]

    plena_estimator = PLENAEstimator(plena_config)

    tracer = OperatorGraphTracer(
        hw_info=None,
        execution_mode=execution_mode,
        cache_seq_len=0,
        plena_estimator=plena_estimator,
    )
    return tracer.trace(model, example_inputs, forward_backward=False, loss_fn=None)


def _load_hf_model_on_meta(model_name: str, dtype: str | torch.dtype) -> nn.Module:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("transformers is required when model is provided as a string") from exc

    torch_dtype = _torch_dtype(dtype)
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            cfg,
            dtype=torch_dtype,
            trust_remote_code=True,
        )
    model.train()
    return model


def _make_lm_training_batch(model: nn.Module, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    cfg = getattr(model, "config", None)
    vocab_size = int(getattr(cfg, "vocab_size", 32000))
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_key = dtype.lower()
    if dtype_key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_key in ("fp16", "float16", "half"):
        return torch.float16
    if dtype_key in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _dtype_nbytes(dtype: str | torch.dtype) -> int:
    return torch.empty((), dtype=_torch_dtype(dtype)).element_size()


def _estimate_training_memory(
    model: nn.Module,
    dtype: str | torch.dtype,
    optimizer: str,
    include_fp32_master_weights: bool,
    num_parameters_override: int | None = None,
) -> TrainingMemoryEstimate:
    num_parameters = (
        num_parameters_override
        if num_parameters_override is not None
        else sum(p.numel() for p in model.parameters())
    )
    parameter_gb = num_parameters * _dtype_nbytes(dtype) / 1e9
    gradient_gb = parameter_gb

    optimizer_key = optimizer.lower()
    if optimizer_key in ("adam", "adamw"):
        optimizer_state_gb = num_parameters * 8 / 1e9
    elif optimizer_key in ("sgd", "none"):
        optimizer_state_gb = 0.0
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    fp32_master_gb = num_parameters * 4 / 1e9 if include_fp32_master_weights else 0.0
    warnings = [
        "Activation peak memory is not yet modeled by graph liveness analysis; reporting model-state memory only."
    ]
    return TrainingMemoryEstimate(
        parameter_gb=parameter_gb,
        gradient_gb=gradient_gb,
        optimizer_state_gb=optimizer_state_gb,
        fp32_master_gb=fp32_master_gb,
        activation_peak_gb=None,
        warnings=warnings,
    )
