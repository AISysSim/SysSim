"""Stream-queue, overlap-aware runtime simulator over an OperatorGraph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..operator_graph import OperatorGraph, OperatorNode, OperatorType


def build_stream_queues(graph: OperatorGraph) -> dict[int, list[OperatorNode]]:
    """Group operators by stream_id, preserving insertion order."""
    queues: dict[int, list[OperatorNode]] = {}
    for op in graph.operators.values():
        queues.setdefault(op.stream_id, []).append(op)
    return queues


@dataclass
class RuntimeResult:
    step_time_ms: float
    per_op_finish_ms: dict[str, float] = field(default_factory=dict)


def simulate_runtime(graph: OperatorGraph) -> RuntimeResult:
    if not graph.operators:
        return RuntimeResult(step_time_ms=0.0)

    queues = build_stream_queues(graph)
    head: dict[int, int] = {s: 0 for s in queues}
    active: dict[str, float] = {}
    completed: dict[str, float] = {}
    t_now = 0.0

    def _head_op(s: int) -> Optional[OperatorNode]:
        i = head[s]
        return queues[s][i] if i < len(queues[s]) else None

    while True:
        any_pending = any(head[s] < len(queues[s]) for s in queues)
        if not any_pending and not active:
            break

        for s in queues:
            op = _head_op(s)
            if op is None or op.name in active or op.name in completed:
                continue
            if not all(d in completed for d in op.predecessors):
                continue
            if op.op_type == OperatorType.BARRIER:
                others_drained = all(
                    head[s2] >= len(queues[s2]) for s2 in queues if s2 != s
                )
                if not others_drained:
                    continue
            start = t_now
            for d in op.predecessors:
                start = max(start, completed[d])
            active[op.name] = start + op.estimated_time_ms

        if not active:
            raise RuntimeError("runtime simulator stalled — no eligible op")

        next_t = min(active.values())
        for name, fin in list(active.items()):
            if fin <= next_t + 1e-12:
                completed[name] = fin
                del active[name]
                for s in queues:
                    op = _head_op(s)
                    if op is not None and op.name == name:
                        head[s] += 1
                        break
        t_now = next_t

    step_time = max(completed.values()) if completed else 0.0
    return RuntimeResult(step_time_ms=step_time, per_op_finish_ms=dict(completed))


def phase_breakdown_ms(graph: OperatorGraph) -> dict[str, float]:
    """Sum estimated time by training phase (forward, backward, optimizer, unknown)."""
    out = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0, "unknown": 0.0}
    for op in graph.operators.values():
        ph = op.config.get("phase") if op.config else None
        out[ph if ph in out else "unknown"] += op.estimated_time_ms
    return out


def collective_exposed_ms(graph: OperatorGraph, result: RuntimeResult) -> float:
    """Non-overlapped portion of collective time.

    Computed by re-simulating with all COLLECTIVE ops stripped and subtracting
    the bare step time from the full step time.
    """
    stripped_names = {op.name for op in graph.operators.values()
                      if op.op_type == OperatorType.COLLECTIVE}
    no_comm = OperatorGraph(name=graph.name + "_no_comm")
    for op in graph.operators.values():
        if op.op_type == OperatorType.COLLECTIVE:
            continue
        clone = OperatorNode(
            name=op.name, op_type=op.op_type, config=dict(op.config) if op.config else {},
            predecessors=[d for d in op.predecessors if d not in stripped_names],
            stream_id=op.stream_id, device_id=op.device_id,
            estimated_time_ms=op.estimated_time_ms,
        )
        no_comm.add_operator(clone)
    if not no_comm.operators:
        return result.step_time_ms
    return max(0.0, result.step_time_ms - simulate_runtime(no_comm).step_time_ms)
