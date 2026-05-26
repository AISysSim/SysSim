"""Public API for rlsysim tracing and simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as nn

from .config import ExecutionMode, SimulatorConfig
from .operator_graph import OperatorGraph
from .tracer import OperatorGraphTracer

if TYPE_CHECKING:
    from .config_plena import PLENAConfig


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

    # Create PLENA estimator
    plena_estimator = PLENAEstimator(plena_config)

    tracer = OperatorGraphTracer(
        hw_info=None,  # Not using GPU roofline
        execution_mode=execution_mode,
        cache_seq_len=0,
        plena_estimator=plena_estimator,
    )
    return tracer.trace(model, example_inputs, forward_backward=False, loss_fn=None)
