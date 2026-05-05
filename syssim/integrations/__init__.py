"""Integration modules for popular frameworks."""

from .huggingface import (
    trace_hf_model_for_training,
    trace_hf_training_step,
)
from .diffusers import trace_diffusers_model_for_inference

__all__ = [
    "trace_diffusers_model_for_inference",
    "trace_hf_model_for_training",
    "trace_hf_training_step",
]
