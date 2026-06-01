"""Cross-product sweep over simulate() kwargs."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from .runner import simulate
from .spec import ParallelismConfig, TrainingConfig


@dataclass
class SweepRow:
    inputs: dict[str, Any]
    report: Any
    metrics: dict[str, float]


class Sweep:
    def __init__(self, rows: list[SweepRow]):
        self.rows = rows

    def best(self, metric: str) -> SweepRow | None:
        if not self.rows:
            return None
        return max(self.rows, key=lambda r: r.metrics.get(metric, float("-inf")))

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("to_dataframe requires pandas") from e
        return pd.DataFrame([{**r.inputs, **r.metrics} for r in self.rows])


def _apply_path(model, par: ParallelismConfig, tr: TrainingConfig, path: str, value):
    section, _, field = path.partition(".")
    if section == "model":
        # Model-level axis (e.g. "model.seq_length"): resolve the model YAML to a ModelConfig
        # and override the one field. Lets the sweep vary architecture knobs that don't live on
        # ParallelismConfig/TrainingConfig.
        import dataclasses
        from .spec import load_model_yaml, ModelConfig
        mc = load_model_yaml(model) if isinstance(model, str) else model
        if not isinstance(mc, ModelConfig):
            raise ValueError(
                f"sweeping {path!r} needs a model YAML path or ModelConfig, got {type(model).__name__}"
            )
        if field not in {f.name for f in dataclasses.fields(mc)}:
            raise ValueError(f"unknown ModelConfig field in sweep path {path!r}: {field!r}")
        return dataclasses.replace(mc, **{field: value}), par, tr
    if section == "parallelism":
        kwargs = {
            "tp": par.tensor_model_parallel_size, "dp": par.data_parallel_size,
            "sp": par.sequence_parallel, "cp": par.context_parallel_size,
            "pp": par.pipeline_model_parallel_size,
        }
        kwargs[field] = value
        return model, ParallelismConfig(**kwargs), tr
    if section == "training":
        # Reconstruct using dtype to avoid the "exactly one active flag" problem
        # when overwriting a single flag without clearing the others.
        current_dtype = "bf16" if tr.bf16 else ("fp16" if tr.fp16 else "fp8")
        kwargs = {
            "micro_batch": tr.micro_batch_size,
            "global_batch": tr.global_batch_size,
            "dtype": current_dtype,
            "recompute_granularity": tr.recompute_granularity,
        }
        if field in ("fp16", "bf16", "fp8"):
            # Switching dtype: derive new dtype string
            flag_to_dtype = {"fp16": "fp16", "bf16": "bf16", "fp8": "fp8"}
            if value:
                kwargs["dtype"] = flag_to_dtype[field]
            # If setting a flag to False we keep current_dtype (no-op for the active flag)
        elif field == "dtype":
            kwargs["dtype"] = value
        else:
            kwargs[field] = value
        return model, par, TrainingConfig(**kwargs)
    raise ValueError(f"unknown sweep path section: {section!r}")


def sweep(*, model, hardware, parallelism=None, training=None, over: dict, workdir=None) -> Sweep:
    parallelism = parallelism or ParallelismConfig()
    training    = training or TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    paths = list(over.keys())
    grid = list(itertools.product(*[over[p] for p in paths]))

    rows: list[SweepRow] = []
    for combo in grid:
        mdl, par, tr = model, parallelism, training
        chosen = {}
        for path, value in zip(paths, combo):
            mdl, par, tr = _apply_path(mdl, par, tr, path, value)
            chosen[path] = value
        report = simulate(model=mdl, hardware=hardware,
                          parallelism=par, training=tr, workdir=workdir)
        rows.append(SweepRow(
            inputs=chosen,
            report=report,
            metrics={"step_time_ms": report.step_time_ms, "mfu": report.mfu,
                     "peak_memory_gb": report.peak_memory_gb},
        ))
    return Sweep(rows)
