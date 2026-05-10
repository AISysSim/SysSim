"""Diffusers integration helpers."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ..api import trace_model_for_inference
from ..config import SimulatorConfig
from ..operator_graph import OperatorGraph


def trace_diffusers_model_for_inference(
    model_or_pipeline: nn.Module | Any,
    example_inputs: Any,
    config: SimulatorConfig,
    *,
    component: str | None = None,
) -> OperatorGraph:
    """Trace a Diffusers model component for inference.

    This is a thin adapter over ``trace_model_for_inference(..., mode="prefill")``.
    Pass either an ``nn.Module`` component directly, or a pipeline plus the
    component attribute name to trace.

    Args:
        model_or_pipeline: Diffusers ``nn.Module`` component, or a pipeline
            object containing the component.
        example_inputs: Example tensor inputs for the component forward pass.
        config: SimulatorConfig with HardwareInfo.
        component: Optional attribute to select from ``model_or_pipeline``, e.g.
            ``"transformer"`` or ``"vae"``.

    Returns:
        OperatorGraph containing the traced forward-pass operations.
    """
    model = getattr(model_or_pipeline, component) if component is not None else model_or_pipeline
    if model is None:
        raise ValueError(f"Diffusers component '{component}' is None")
    if not isinstance(model, nn.Module):
        raise TypeError(
            "trace_diffusers_model_for_inference expects an nn.Module component. "
            "Pass a pipeline component name such as component='transformer', "
            "or pass the component module directly."
        )
    return trace_model_for_inference(model, example_inputs, config, mode="prefill")
