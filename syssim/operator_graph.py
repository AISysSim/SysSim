"""OperatorGraph IR: graph data structure for operator-level performance modeling.

Provides OperatorType enum, TensorMeta, OperatorNode, and OperatorGraph with
topological sort, critical path analysis, DOT/JSON export, and summary.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class OperatorType(Enum):
    # Math/Compute
    GEMM = "gemm"
    ATTN = "attn"
    RMSNORM = "rmsnorm"
    SILU = "silu"
    MATH = "math"

    # Communication
    COLLECTIVE = "collective"

    # Memory
    MEMORY = "memory"

    # Sync
    BARRIER = "barrier"
    STREAM_SYNC = "stream_sync"


_MATH_TYPES = frozenset({
    OperatorType.GEMM, OperatorType.ATTN, OperatorType.RMSNORM,
    OperatorType.SILU, OperatorType.MATH,
})

_COLLECTIVE_TYPES = frozenset({
    OperatorType.COLLECTIVE,
})

_MEMORY_TYPES = frozenset({
    OperatorType.MEMORY,
})

_SYNC_TYPES = frozenset({
    OperatorType.BARRIER, OperatorType.STREAM_SYNC,
})


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    dtype: str
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype, "device": self.device}


@dataclass
class OperatorNode:
    # Identity
    name: str
    op_type: OperatorType

    # Configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Dependencies
    data_deps: list[str] = field(default_factory=list)
    stream_deps: list[str] = field(default_factory=list)

    # Execution context
    stream_id: int = 0
    device_id: int = 0

    # Tensor metadata
    inputs: list[TensorMeta] = field(default_factory=list)
    outputs: list[TensorMeta] = field(default_factory=list)

    # Performance
    estimated_time_ms: float = 0.0

    # Critical path state (computed by OperatorGraph.compute_critical_path)
    earliest_start: float = 0.0
    earliest_finish: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type.value,
            "config": self.config,
            "data_deps": self.data_deps,
            "stream_deps": self.stream_deps,
            "stream_id": self.stream_id,
            "device_id": self.device_id,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "estimated_time_ms": self.estimated_time_ms,
            "earliest_start": self.earliest_start,
            "earliest_finish": self.earliest_finish,
        }


class OperatorGraph:
    """DAG of OperatorNodes with multi-stream critical path analysis."""

    def __init__(self, name: str = "model"):
        self.name = name
        self.operators: dict[str, OperatorNode] = {}
        self.streams: set[int] = set()
        self._topo_cache: Optional[list[str]] = None

    def __len__(self) -> int:
        return len(self.operators)

    def add_operator(self, node: OperatorNode) -> None:
        if node.name in self.operators:
            raise ValueError(f"Duplicate operator name: {node.name}")
        self.operators[node.name] = node
        self.streams.add(node.stream_id)
        self._topo_cache = None

    def validate(self) -> None:
        """Validate DAG: reference integrity and cycle detection (DFS coloring)."""
        for name, op in self.operators.items():
            for dep in op.data_deps + op.stream_deps:
                if dep not in self.operators:
                    raise ValueError(
                        f"Operator '{name}' depends on non-existent operator '{dep}'"
                    )

        # Cycle detection via DFS coloring
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self.operators}

        def dfs(u: str) -> None:
            color[u] = GRAY
            node = self.operators[u]
            for v in node.data_deps + node.stream_deps:
                if color[v] == GRAY:
                    raise ValueError(f"Cycle detected involving operator '{v}'")
                if color[v] == WHITE:
                    dfs(v)
            color[u] = BLACK

        for name in self.operators:
            if color[name] == WHITE:
                dfs(name)

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm. Result is cached until the graph is modified."""
        if self._topo_cache is not None:
            return self._topo_cache

        in_degree: dict[str, int] = {name: 0 for name in self.operators}
        for op in self.operators.values():
            for dep in op.data_deps + op.stream_deps:
                pass  # deps are predecessors, not successors
        # Build successor list and in-degree
        successors: dict[str, list[str]] = {name: [] for name in self.operators}
        for name, op in self.operators.items():
            for dep in op.data_deps + op.stream_deps:
                if dep not in successors:
                    continue
                in_degree[name] += 1
            # dep -> name means dep is predecessor of name

        # Re-compute: for each edge dep -> name, dep is a predecessor
        in_degree = {name: 0 for name in self.operators}
        for name, op in self.operators.items():
            # Each dep in data_deps/stream_deps is a predecessor of 'name'
            # So 'name' has an incoming edge from each dep
            in_degree[name] = len(set(op.data_deps + op.stream_deps))

        # Successors: if name depends on dep, then dep has name as successor
        successors = {name: [] for name in self.operators}
        for name, op in self.operators.items():
            for dep in set(op.data_deps + op.stream_deps):
                if dep in successors:
                    successors[dep].append(name)

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        result = []
        while queue:
            u = queue.popleft()
            result.append(u)
            for v in successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(result) != len(self.operators):
            raise ValueError("Graph contains a cycle")

        self._topo_cache = result
        return result

    def compute_critical_path(self) -> float:
        """DP on topological order with multi-stream awareness.

        Returns the critical path length (max earliest_finish across all operators).
        """
        if not self.operators:
            return 0.0

        topo = self.topological_sort()
        stream_times: dict[int, float] = {s: 0.0 for s in self.streams}

        for name in topo:
            op = self.operators[name]
            start = stream_times.get(op.stream_id, 0.0)

            # Data and stream dependencies
            for dep_name in op.data_deps + op.stream_deps:
                dep = self.operators[dep_name]
                start = max(start, dep.earliest_finish)

            # Special sync semantics
            if op.op_type == OperatorType.BARRIER:
                start = max(start, max(stream_times.values()) if stream_times else 0.0)
            elif op.op_type == OperatorType.STREAM_SYNC:
                target_stream = op.config.get("target_stream")
                if target_stream is not None and target_stream in stream_times:
                    start = max(start, stream_times[target_stream])

            op.earliest_start = start
            op.earliest_finish = start + op.estimated_time_ms
            stream_times[op.stream_id] = op.earliest_finish

        return max(op.earliest_finish for op in self.operators.values())

    def to_dot(self) -> str:
        """Generate Graphviz DOT representation, color-coded by op type."""
        color_map = {
            **{t: "lightblue" for t in _MATH_TYPES},
            **{t: "lightyellow" for t in _COLLECTIVE_TYPES},
            **{t: "lightgreen" for t in _MEMORY_TYPES},
            **{t: "lightgray" for t in _SYNC_TYPES},
        }

        lines = [f'digraph "{self.name}" {{', "  rankdir=TB;"]
        for name, op in self.operators.items():
            color = color_map.get(op.op_type, "white")
            label = f"{name}\\n{op.op_type.value}\\n{op.estimated_time_ms:.2e}ms"
            lines.append(
                f'  "{name}" [label="{label}", style=filled, fillcolor="{color}"];'
            )
        for name, op in self.operators.items():
            for dep in op.data_deps:
                lines.append(f'  "{dep}" -> "{name}" [style=solid];')
            for dep in op.stream_deps:
                lines.append(f'  "{dep}" -> "{name}" [style=dashed];')
        lines.append("}")
        return "\n".join(lines)

    def to_json(self) -> str:
        data = {
            "name": self.name,
            "operators": [op.to_dict() for op in self.operators.values()],
            "streams": sorted(self.streams),
        }
        return json.dumps(data, indent=2)

    def summary(self) -> str:
        """Return a human-readable summary of op counts, critical path, and total time."""
        counts: dict[str, int] = {}
        total_time = 0.0
        for op in self.operators.values():
            key = op.op_type.value
            counts[key] = counts.get(key, 0) + 1
            total_time += op.estimated_time_ms

        critical_path = self.compute_critical_path()

        lines = [f"OperatorGraph '{self.name}': {len(self.operators)} operators, {len(self.streams)} streams"]
        lines.append("Op counts:")
        for k in sorted(counts):
            lines.append(f"  {k}: {counts[k]}")
        lines.append(f"Total estimated time: {total_time:.6e} ms")
        lines.append(f"Critical path time:   {critical_path:.6e} ms")
        if total_time > 0:
            lines.append(f"Parallelism:          {total_time / critical_path:.2f}x")
        return "\n".join(lines)
