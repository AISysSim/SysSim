"""SimulationReport dataclass + aggregation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimulationReport:
    step_time_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    collective_total_ms: float
    collective_exposed_ms: float
    by_op_type_ms: dict[str, float]

    model_flops_per_step: int
    achieved_tflops: float
    mfu: float
    hfu: float

    param_bytes: int
    grad_bytes: int
    optimizer_state_bytes: int
    activation_bytes: int
    peak_memory_gb: float

    # NEW: per-stage breakdowns (length 1 when PP=1)
    pp_stage_memory_gb: list = field(default_factory=list)
    per_pp_rank_step_time_ms: list = field(default_factory=list)

    model: Any = None
    parallelism: Any = None
    training: Any = None
    hardware: Any = None
    bottlenecks: Any = None   # Bottlenecks; None if not assembled

    def __repr__(self) -> str:
        lines = [
            "SimulationReport(",
            f"  step_time_ms        = {self.step_time_ms:.4f}",
            f"    forward_ms        = {self.forward_ms:.4f}",
            f"    backward_ms       = {self.backward_ms:.4f}",
            f"    optimizer_ms      = {self.optimizer_ms:.4f}",
            f"  collective_total_ms = {self.collective_total_ms:.4f}",
            f"  collective_exposed  = {self.collective_exposed_ms:.4f}",
            f"  achieved_tflops     = {self.achieved_tflops:.2f}",
            f"  mfu                 = {self.mfu*100:.2f}%",
            f"  hfu                 = {self.hfu*100:.2f}%",
            f"  peak_memory_gb      = {self.peak_memory_gb:.3f}",
        ]
        if self.bottlenecks is not None:
            for bl in str(self.bottlenecks).split("\n"):
                lines.append(f"  {bl}")
        lines.append(")")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "step_time_ms": self.step_time_ms,
            "forward_ms": self.forward_ms,
            "backward_ms": self.backward_ms,
            "optimizer_ms": self.optimizer_ms,
            "collective_total_ms": self.collective_total_ms,
            "collective_exposed_ms": self.collective_exposed_ms,
            "by_op_type_ms": dict(self.by_op_type_ms),
            "model_flops_per_step": self.model_flops_per_step,
            "achieved_tflops": self.achieved_tflops,
            "mfu": self.mfu,
            "hfu": self.hfu,
            "param_bytes": self.param_bytes,
            "grad_bytes": self.grad_bytes,
            "optimizer_state_bytes": self.optimizer_state_bytes,
            "activation_bytes": self.activation_bytes,
            "peak_memory_gb": self.peak_memory_gb,
            "pp_stage_memory_gb": self.pp_stage_memory_gb,
            "per_pp_rank_step_time_ms": self.per_pp_rank_step_time_ms,
            "bottlenecks": (self.bottlenecks.__dict__ if self.bottlenecks else None),
        }

    def to_json(self, path: str | None = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            with open(path, "w") as f:
                f.write(s)
        return s

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "to_dataframe requires pandas; install with `pip install pandas`"
            ) from e
        return pd.DataFrame([self.to_dict()])


@dataclass
class Bottlenecks:
    # Compute / runtime
    top_ops_by_time: list = field(default_factory=list)   # list[(op_name, ms)] top-5
    dominant_op_type: str = ""                             # op type with most total ms
    longest_collective: tuple | None = None               # (op_name, ms) or None

    # Memory
    peak_module: str = ""
    memory_by_type: dict = field(default_factory=dict)     # category -> bytes
    binding_stage: int = 0                                 # PP stage with the peak memory

    # OOM (only meaningful when gpu_memory_GB is known)
    oom: bool = False
    oom_capacity_gb: float = 0.0
    oom_required_gb: float = 0.0
    oom_excess_gb: float = 0.0

    def __repr__(self) -> str:
        lines = ["Bottlenecks(",
                 f"  dominant_op_type = {self.dominant_op_type}",
                 f"  top_ops_by_time  = {self.top_ops_by_time[:3]}",
                 f"  peak_module      = {self.peak_module}"]
        if self.oom:
            lines.append(f"  OOM! required {self.oom_required_gb:.2f} GB > "
                         f"capacity {self.oom_capacity_gb:.2f} GB "
                         f"(excess {self.oom_excess_gb:.2f} GB)")
            lines.append(f"  memory_by_type   = {self.memory_by_type}")
        lines.append(")")
        return "\n".join(lines)


def build_bottlenecks(graph, *, peak_memory_gb, gpu_memory_GB,
                      peak_module, memory_by_type, binding_stage):
    """Assemble runtime + memory bottlenecks. OOM fields populated only when
    gpu_memory_GB is provided and peak exceeds it."""
    from ..operator_graph import OperatorType
    ops = list(graph.operators.values())
    ranked = sorted(ops, key=lambda o: o.estimated_time_ms, reverse=True)
    top_ops = [(o.name, o.estimated_time_ms) for o in ranked[:5] if o.estimated_time_ms > 0]
    by_type = aggregate_by_op_type(graph)
    dominant = max(by_type, key=by_type.get) if by_type else ""
    colls = [(o.name, o.estimated_time_ms) for o in ops
             if o.op_type == OperatorType.COLLECTIVE]
    longest_coll = max(colls, key=lambda t: t[1]) if colls else None

    b = Bottlenecks(
        top_ops_by_time=top_ops,
        dominant_op_type=dominant,
        longest_collective=longest_coll,
        peak_module=peak_module,
        memory_by_type=dict(memory_by_type),
        binding_stage=binding_stage,
    )
    if gpu_memory_GB is not None and gpu_memory_GB > 0 and peak_memory_gb > gpu_memory_GB:
        b.oom = True
        b.oom_capacity_gb = gpu_memory_GB
        b.oom_required_gb = peak_memory_gb
        b.oom_excess_gb = peak_memory_gb - gpu_memory_GB
    return b


def aggregate_by_op_type(graph) -> dict[str, float]:
    out: dict[str, float] = {}
    for op in graph.operators.values():
        out[op.op_type.value] = out.get(op.op_type.value, 0.0) + op.estimated_time_ms
    return out


@dataclass
class ModelFlopsBudget:
    model_flops_per_step: float
    hardware_flops_per_step: float


def compute_efficiency(
    budget: ModelFlopsBudget,
    step_time_ms: float,
    world_size: int,
    peak_tflops_mm: float,
) -> dict[str, float]:
    if step_time_ms <= 0 or world_size <= 0 or peak_tflops_mm <= 0:
        return {"achieved_tflops": 0.0, "mfu": 0.0, "hfu": 0.0}
    step_s = step_time_ms / 1000.0
    achieved = budget.model_flops_per_step / step_s / 1e12 / world_size
    hw = budget.hardware_flops_per_step / step_s / 1e12 / world_size
    return {
        "achieved_tflops": achieved,
        "mfu": achieved / peak_tflops_mm,
        "hfu": hw / peak_tflops_mm,
    }


def compute_model_flops_budget(model, parallelism, training) -> ModelFlopsBudget:
    """Standard transformer FLOP formula: ~24 b s h^2 + 4 b s^2 h per layer fwd."""
    b = training.micro_batch_size
    s = model.seq_length
    h = model.hidden_size
    f = model.ffn_hidden_size

    ffn_matmuls = 3 if model.swiglu else 2
    fwd_ffn = 2 * b * s * h * f * ffn_matmuls
    fwd_proj = 2 * b * s * h * h * 4
    fwd_score = 2 * b * model.num_attention_heads * s * s * (h // model.num_attention_heads)
    fwd_av = fwd_score
    fwd_attn = fwd_proj + fwd_score + fwd_av
    fwd_per = fwd_attn + fwd_ffn
    fwd_total = fwd_per * model.num_layers + 2 * b * s * h * model.vocab_size
    bwd_total = 2 * fwd_total

    micro_per_step = max(
        1,
        training.global_batch_size
        // (training.micro_batch_size * parallelism.data_parallel_size),
    )
    model_flops = (fwd_total + bwd_total) * micro_per_step
    if training.recompute_granularity == "full":
        hw_flops = model_flops + fwd_total * micro_per_step
    elif training.recompute_granularity == "selective":
        hw_flops = model_flops + 0.3 * fwd_total * micro_per_step
    else:
        hw_flops = model_flops
    return ModelFlopsBudget(model_flops, hw_flops)
