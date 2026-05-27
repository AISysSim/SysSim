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
    MATH = "math"

    # Communication
    COLLECTIVE = "collective"

    # Memory
    MEMORY = "memory"

    # Sync
    BARRIER = "barrier"
    STREAM_SYNC = "stream_sync"


_MATH_TYPES = frozenset({
    OperatorType.GEMM, OperatorType.ATTN, OperatorType.MATH,
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

    # Single dependency list — same-stream FIFO and cross-stream sync both here
    predecessors: list[str] = field(default_factory=list)

    # Execution context
    stream_id: int = 0
    device_id: int = 0

    # Tensor metadata
    inputs: list[TensorMeta] = field(default_factory=list)
    outputs: list[TensorMeta] = field(default_factory=list)

    # Performance
    estimated_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type.value,
            "config": self.config,
            "predecessors": self.predecessors,
            "stream_id": self.stream_id,
            "device_id": self.device_id,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "estimated_time_ms": self.estimated_time_ms,
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
            for dep in op.predecessors:
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
            for v in node.predecessors:
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

        # Re-compute: for each edge dep -> name, dep is a predecessor
        in_degree = {name: 0 for name in self.operators}
        for name, op in self.operators.items():
            # Each dep in predecessors is a predecessor of 'name'
            # So 'name' has an incoming edge from each dep
            in_degree[name] = len(set(op.predecessors))

        # Successors: if name depends on dep, then dep has name as successor
        successors = {name: [] for name in self.operators}
        for name, op in self.operators.items():
            for dep in set(op.predecessors):
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
            for dep_name in op.predecessors:
                dep = self.operators.get(dep_name)
                if dep is None:
                    continue
                style = "solid" if dep.stream_id == op.stream_id else "dashed"
                lines.append(f'  "{dep_name}" -> "{name}" [style={style}];')
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
        """Return a human-readable summary of op counts and total time."""
        counts: dict[str, int] = {}
        total_time = 0.0
        for op in self.operators.values():
            key = op.op_type.value
            counts[key] = counts.get(key, 0) + 1
            total_time += op.estimated_time_ms
        lines = [f"OperatorGraph '{self.name}': {len(self.operators)} operators, {len(self.streams)} streams"]
        lines.append("Op counts:")
        for k in sorted(counts):
            lines.append(f"  {k}: {counts[k]}")
        lines.append(f"Total estimated time: {total_time:.6e} ms")
        return "\n".join(lines)
