"""Mixture-of-Experts operator graph construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from syssim.compute.moe_cost import (
    dtype_nbytes,
    estimate_combine_ms,
    estimate_dispatch_ms,
    estimate_expert_ffn_ms,
    estimate_memory_ms,
    estimate_router_ms,
)
from syssim.config import ExecutionMode, HardwareInfo, SimulatorConfig
from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType


@dataclass(frozen=True)
class MoEModelSpec:
    """Model-level MoE structure extracted from a config."""

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    vocab_size: int | None = None
    decoder_sparse_step: int = 1
    first_sparse_layer: int = 0
    name: str = "moe_model"

    def sparse_layer_indices(self) -> tuple[int, ...]:
        self.validate()
        return tuple(range(self.first_sparse_layer, self.num_layers, self.decoder_sparse_step))

    def validate(self) -> None:
        errors: list[str] = []
        if self.num_layers <= 0:
            errors.append("num_layers must be positive")
        if self.hidden_size <= 0:
            errors.append("hidden_size must be positive")
        if self.intermediate_size <= 0:
            errors.append("intermediate_size must be positive")
        if self.num_experts <= 0:
            errors.append("num_experts must be positive")
        if self.top_k <= 0:
            errors.append("top_k must be positive")
        if self.num_experts > 0 and self.top_k > self.num_experts:
            errors.append("top_k must be <= num_experts")
        if self.decoder_sparse_step <= 0:
            errors.append("decoder_sparse_step must be positive")
        if self.first_sparse_layer < 0:
            errors.append("first_sparse_layer must be non-negative")
        if self.num_layers > 0 and self.first_sparse_layer >= self.num_layers:
            errors.append("first_sparse_layer must be < num_layers")
        if self.vocab_size is not None and self.vocab_size <= 0:
            errors.append("vocab_size must be positive when provided")
        if errors:
            raise ValueError("; ".join(errors))


@dataclass(frozen=True)
class MoERuntimeConfig:
    """Runtime shape and distributed settings for MoE operator modeling."""

    batch_size: int
    seq_len: int
    mode: ExecutionMode = ExecutionMode.TRAINING
    dtype: torch.dtype | str = torch.bfloat16
    expert_parallel_size: int = 1
    capacity_factor: float = 1.0
    tokens_per_expert: tuple[int, ...] | None = None

    def num_tokens(self) -> int:
        return self.batch_size * self.seq_len

    def num_assignments(self, top_k: int) -> int:
        return self.num_tokens() * top_k

    def validate(self, spec: MoEModelSpec) -> None:
        spec.validate()
        errors: list[str] = []
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.seq_len <= 0:
            errors.append("seq_len must be positive")
        if not isinstance(self.mode, ExecutionMode):
            errors.append("mode must be an ExecutionMode")
        if self.expert_parallel_size <= 0:
            errors.append("expert_parallel_size must be positive")
        if self.expert_parallel_size > spec.num_experts:
            errors.append("expert_parallel_size must be <= num_experts")
        if self.capacity_factor <= 0:
            errors.append("capacity_factor must be positive")
        if self.tokens_per_expert is not None:
            if len(self.tokens_per_expert) != spec.num_experts:
                errors.append("tokens_per_expert length must equal num_experts")
            if any(tokens < 0 for tokens in self.tokens_per_expert):
                errors.append("tokens_per_expert values must be non-negative")
            expected = self.num_assignments(spec.top_k)
            actual = sum(self.tokens_per_expert)
            if actual != expected:
                errors.append(
                    f"tokens_per_expert sum must equal batch_size * seq_len * top_k "
                    f"({expected}), got {actual}"
                )
        if errors:
            raise ValueError("; ".join(errors))


def extract_hf_moe_spec(model_or_config: object, name: str | None = None) -> MoEModelSpec:
    """Extract a MoE spec from a Hugging Face model or config object."""
    config = getattr(model_or_config, "config", model_or_config)
    required_fields = (
        "num_hidden_layers",
        "hidden_size",
        "moe_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
    )
    missing = [field for field in required_fields if not hasattr(config, field)]
    if missing:
        raise ValueError(
            "Missing required Hugging Face MoE config fields: "
            + ", ".join(sorted(missing))
        )

    model_name = name
    if model_name is None:
        model_name = getattr(config, "_name_or_path", None) or getattr(config, "model_type", None)
    if not model_name:
        model_name = "hf_moe_model"

    spec = MoEModelSpec(
        num_layers=int(getattr(config, "num_hidden_layers")),
        hidden_size=int(getattr(config, "hidden_size")),
        intermediate_size=int(getattr(config, "moe_intermediate_size")),
        num_experts=int(getattr(config, "num_experts")),
        top_k=int(getattr(config, "num_experts_per_tok")),
        vocab_size=_optional_int(getattr(config, "vocab_size", None)),
        decoder_sparse_step=int(getattr(config, "decoder_sparse_step", 1)),
        first_sparse_layer=int(getattr(config, "first_sparse_layer", 0)),
        name=str(model_name),
    )
    spec.validate()
    return spec


def build_moe_operator_graph(
    spec: MoEModelSpec,
    runtime: MoERuntimeConfig,
    config: SimulatorConfig,
    graph_name: str | None = None,
    topology: object | None = None,
    loggp: object | None = None,
) -> OperatorGraph:
    """Build a MoE `OperatorGraph` from config-level structure."""
    if (topology is None) != (loggp is None):
        raise ValueError("topology and loggp must be supplied together")

    spec.validate()
    runtime.validate(spec)

    graph = OperatorGraph(graph_name or spec.name)
    previous_combine: str | None = None
    num_tokens = runtime.num_tokens()
    num_assignments = runtime.num_assignments(spec.top_k)
    tokens_per_expert = _tokens_per_expert(spec, runtime)
    effective_expert_tokens = max(tokens_per_expert) * spec.num_experts
    dtype = _dtype_name(runtime.dtype)
    time_multiplier = _mode_time_multiplier(runtime.mode)

    for layer_idx in spec.sparse_layer_indices():
        common = {
            "layer_idx": layer_idx,
            "num_tokens": num_tokens,
            "num_assignments": num_assignments,
            "hidden_size": spec.hidden_size,
            "num_experts": spec.num_experts,
            "top_k": spec.top_k,
            "dtype": dtype,
            "mode": runtime.mode.value,
            "capacity_factor": runtime.capacity_factor,
            "expert_parallel_size": runtime.expert_parallel_size,
        }

        router_name = f"layer_{layer_idx:03d}_moe_router"
        dispatch_name = f"layer_{layer_idx:03d}_moe_dispatch"
        expert_name = f"layer_{layer_idx:03d}_moe_expert"
        combine_name = f"layer_{layer_idx:03d}_moe_combine"

        router_deps = [previous_combine] if previous_combine is not None else []
        graph.add_operator(OperatorNode(
            name=router_name,
            op_type=OperatorType.MOE_ROUTER,
            data_deps=router_deps,
            estimated_time_ms=time_multiplier * estimate_router_ms(
                num_tokens,
                spec.hidden_size,
                spec.num_experts,
                spec.top_k,
                config.hw_info,
                runtime.dtype,
            ),
            config={
                **common,
                "stage": "router",
                "router_shape": (spec.hidden_size, spec.num_experts),
            },
        ))

        graph.add_operator(OperatorNode(
            name=dispatch_name,
            op_type=OperatorType.MOE_DISPATCH,
            data_deps=[router_name],
            estimated_time_ms=time_multiplier * estimate_dispatch_ms(
                num_assignments,
                spec.hidden_size,
                config.hw_info,
                runtime.dtype,
            ),
            config={
                **common,
                "stage": "dispatch",
                "tokens_per_expert": tokens_per_expert,
            },
        ))

        expert_deps = [dispatch_name]
        if runtime.expert_parallel_size > 1:
            a2a_dispatch_name = f"layer_{layer_idx:03d}_moe_alltoall_dispatch"
            alltoall_ms = estimate_moe_alltoall_ms(
                runtime,
                spec.hidden_size,
                config.hw_info,
                topology=topology,
                loggp=loggp,
                top_k=spec.top_k,
            )
            graph.add_operator(OperatorNode(
                name=a2a_dispatch_name,
                op_type=OperatorType.COLLECTIVE,
                data_deps=[dispatch_name],
                estimated_time_ms=time_multiplier * alltoall_ms,
                config={
                    **common,
                    "stage": "alltoall_dispatch",
                    "payload_bytes": _moe_alltoall_payload_bytes(
                        runtime,
                        spec.hidden_size,
                        spec.top_k,
                    ),
                    "network_model": _network_model_name(topology, loggp),
                },
            ))
            expert_deps = [a2a_dispatch_name]

        graph.add_operator(OperatorNode(
            name=expert_name,
            op_type=OperatorType.MOE_EXPERT,
            data_deps=expert_deps,
            estimated_time_ms=time_multiplier * estimate_expert_ffn_ms(
                effective_expert_tokens,
                spec.hidden_size,
                spec.intermediate_size,
                config.hw_info,
                runtime.dtype,
            ),
            config={
                **common,
                "stage": "expert",
                "intermediate_size": spec.intermediate_size,
                "tokens_per_expert": tokens_per_expert,
                "max_tokens_per_expert": max(tokens_per_expert),
                "effective_expert_tokens": effective_expert_tokens,
            },
        ))

        combine_deps = [expert_name]
        if runtime.expert_parallel_size > 1:
            a2a_combine_name = f"layer_{layer_idx:03d}_moe_alltoall_combine"
            alltoall_ms = estimate_moe_alltoall_ms(
                runtime,
                spec.hidden_size,
                config.hw_info,
                topology=topology,
                loggp=loggp,
                top_k=spec.top_k,
            )
            graph.add_operator(OperatorNode(
                name=a2a_combine_name,
                op_type=OperatorType.COLLECTIVE,
                data_deps=[expert_name],
                estimated_time_ms=time_multiplier * alltoall_ms,
                config={
                    **common,
                    "stage": "alltoall_combine",
                    "payload_bytes": _moe_alltoall_payload_bytes(
                        runtime,
                        spec.hidden_size,
                        spec.top_k,
                    ),
                    "network_model": _network_model_name(topology, loggp),
                },
            ))
            combine_deps = [a2a_combine_name]

        graph.add_operator(OperatorNode(
            name=combine_name,
            op_type=OperatorType.MOE_COMBINE,
            data_deps=combine_deps,
            estimated_time_ms=time_multiplier * estimate_combine_ms(
                num_assignments,
                spec.hidden_size,
                config.hw_info,
                runtime.dtype,
            ),
            config={
                **common,
                "stage": "combine",
                "tokens_per_expert": tokens_per_expert,
            },
        ))

        previous_combine = combine_name

    graph.validate()
    return graph


def estimate_moe_alltoall_ms(
    runtime: MoERuntimeConfig,
    hidden_size: int,
    hw_info: HardwareInfo,
    topology: object | None = None,
    loggp: object | None = None,
    top_k: int = 1,
) -> float:
    """Estimate one expert-parallel MoE all-to-all in milliseconds."""
    if (topology is None) != (loggp is None):
        raise ValueError("topology and loggp must be supplied together")
    if runtime.expert_parallel_size <= 1:
        return 0.0

    payload_bytes = _moe_alltoall_payload_bytes(runtime, hidden_size, top_k)
    if topology is None and loggp is None:
        return estimate_memory_ms(payload_bytes, hw_info)

    from syssim.network import alltoall, simulate

    ranks = list(range(runtime.expert_parallel_size))
    ops = alltoall(ranks, float(payload_bytes), tag_prefix="moe_alltoall")
    result = simulate(ops, topology, loggp)
    return result.makespan * 1e3


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _dtype_name(dtype: torch.dtype | str) -> str:
    if isinstance(dtype, str):
        return dtype
    return str(dtype).replace("torch.", "")


def _mode_time_multiplier(mode: ExecutionMode) -> float:
    return 3.0 if mode == ExecutionMode.TRAINING else 1.0


def _tokens_per_expert(spec: MoEModelSpec, runtime: MoERuntimeConfig) -> tuple[int, ...]:
    if runtime.tokens_per_expert is not None:
        return runtime.tokens_per_expert

    total = runtime.num_assignments(spec.top_k)
    base = total // spec.num_experts
    remainder = total % spec.num_experts
    return tuple(base + (1 if idx < remainder else 0) for idx in range(spec.num_experts))


def _moe_alltoall_payload_bytes(
    runtime: MoERuntimeConfig,
    hidden_size: int,
    top_k: int,
) -> int:
    return math.ceil(runtime.num_assignments(top_k) * hidden_size * dtype_nbytes(runtime.dtype))


def _network_model_name(topology: object | None, loggp: object | None) -> str:
    if topology is None and loggp is None:
        return "memory_roofline"
    return "loggp_alltoall"
