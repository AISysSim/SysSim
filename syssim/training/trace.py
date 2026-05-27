"""Trace artifact: OperatorGraph + provenance + simulate_on entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..operator_graph import OperatorGraph


@dataclass
class Trace:
    """The cached graph from one trace run, plus provenance.

    `simulate_on(hardware)` is the load-bearing entry point: it injects the
    DP all-reduce + optimizer step (both depend on hardware bandwidth), runs
    the discrete-event simulator, and returns a SimulationReport. The cached
    `graph` is treated as immutable — `simulate_on` works on a copy.
    """
    graph: OperatorGraph
    model: Any
    parallelism: Any
    training: Any
    gpus_per_node: int
    per_stage_profiles: list = None  # list[MemoryProfile], one per PP stage (None if not captured)

    def simulate_on(self, hardware) -> "SimulationReport":
        from .runner import _simulate_on_hardware
        return _simulate_on_hardware(self, hardware)
