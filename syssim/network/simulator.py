"""Event-driven network congestion simulator with max-min fair bandwidth sharing.

This module implements the core simulation engine that takes a DAG of communication
ops and simulates their execution on a given topology with LogGP performance model.

Algorithm:
1. Initialize all ops with remaining_bytes = size
2. Maintain active set of currently transferring ops
3. At each event:
   a. Compute resource usage (how many ops share each resource)
   b. Compute bottleneck bandwidth for each op (min across its path)
   c. Advance time until next event (op completion or new injection)
   d. Drain bytes from active ops proportional to time elapsed
   e. Complete ops that reach zero bytes (add LogGP overhead)
   f. Inject ops whose dependencies are satisfied

Max-min fair sharing:
- If N ops share a resource with bandwidth B, each gets B/N
- Each op's effective bandwidth = min(B_r / n_r) across all resources r on its path
- This models congestion naturally without explicit contention rules

Design choices:
- Global LogGP params (extended in Phase 4 for HierarchicalTopology with layer-specific params)
- Float tolerance for numerical stability (1e-9 seconds, ~1 nanosecond)
- Path caching for performance (topology queries are expensive)
- Use op indices (not Op objects) as keys to avoid hashability issues
"""

import heapq
from dataclasses import dataclass
from typing import Optional

from .dag_builder import Op
from .loggp import LogGPParams
from .topology import Resource, Topology

# Tolerance for floating-point comparisons (1 nanosecond)
FLOAT_TOL = 1e-9


@dataclass
class SimulationResult:
    """Results from network simulation.

    Attributes:
        ops: List of Op objects with start_time and finish_time populated
        makespan: Total execution time in seconds (max finish time)
        per_rank_finish: Dict mapping rank ID to its latest finish time

    Example:
        >>> result = simulate(ops, topo, loggp)
        >>> print(f"Total time: {result.makespan * 1e3:.2f} ms")
        >>> print(f"Rank 0 finished at: {result.per_rank_finish[0] * 1e3:.2f} ms")
    """

    ops: list[Op]
    makespan: float  # seconds
    per_rank_finish: dict[int, float]  # rank -> finish time (seconds)


def simulate(
    ops: list[Op],
    topology: Topology,
    loggp: Optional[LogGPParams] = None,
) -> SimulationResult:
    """Simulate communication DAG on given topology with congestion modeling.

    This function performs event-driven simulation with max-min fair bandwidth
    sharing across all resources. Each op's transfer rate is limited by the
    most congested resource on its path.

    Args:
        ops: List of Op objects forming the communication DAG
        topology: Network topology (resolve_path method)
        loggp: LogGP performance parameters (optional if ops have layer-specific loggp)

    Returns:
        SimulationResult with timing information populated

    Raises:
        ValueError: If ops list is empty or contains cycles

    Example:
        >>> from syssim.network import allreduce, FullyConnectedTopology, LogGPParams, simulate
        >>> topo = FullyConnectedTopology(4, 25e9)
        >>> loggp = LogGPParams(L=1e-6, o=7e-6, G=1/(25e9))
        >>> ops = allreduce([0,1,2,3], 1e9)
        >>> result = simulate(ops, topo, loggp)
        >>> print(f"Makespan: {result.makespan * 1e3:.2f} ms")
    """
    if not ops:
        return SimulationResult(ops=[], makespan=0.0, per_rank_finish={})

    # Initialize op state
    for op in ops:
        op.remaining_bytes = op.size
        op.start_time = 0.0
        op.finish_time = 0.0

    # Populate layer-specific LogGP for HierarchicalTopology
    # Check if topology has get_loggp method (duck typing)
    has_layer_loggp = hasattr(topology, "get_loggp") and callable(getattr(topology, "get_loggp"))

    if has_layer_loggp:
        for op in ops:
            if op.loggp is None:  # Only populate if not already set
                op.loggp = topology.get_loggp(op.src, op.dst)

    # Create op index mapping (op -> index in ops list)
    op_to_idx: dict[int, int] = {id(op): i for i, op in enumerate(ops)}

    # Build path cache for performance
    path_cache: dict[tuple[int, int], list[Resource]] = {}
    for op in ops:
        key = (op.src, op.dst)
        if key not in path_cache:
            path_cache[key] = topology.resolve_path(op.src, op.dst)

    # Track which ops are currently transferring (use indices)
    active: set[int] = set()  # set of op indices

    # Track ops waiting for dependencies (min-heap by earliest ready time)
    # Heap elements: (ready_time, counter, op_idx)
    eligible: list[tuple[float, int, int]] = []
    counter = 0  # Tie-breaker for heap (ensures deterministic ordering)

    # Build dependency tracking (use indices)
    num_unsatisfied: list[int] = [0] * len(ops)  # op_idx -> count of unsatisfied deps
    dependents: dict[int, list[int]] = {}  # op_idx -> list of op_idx that depend on it

    for i, op in enumerate(ops):
        num_unsatisfied[i] = len(op.deps)
        for dep in op.deps:
            dep_idx = op_to_idx[id(dep)]
            if dep_idx not in dependents:
                dependents[dep_idx] = []
            dependents[dep_idx].append(i)

    # Initialize eligible set (ops with no dependencies)
    for i, op in enumerate(ops):
        if num_unsatisfied[i] == 0:
            heapq.heappush(eligible, (0.0, counter, i))
            counter += 1

    # Simulation state
    current_time = 0.0

    # Main event loop
    while active or eligible:
        # Inject ops whose dependencies are satisfied and ready at current_time
        while eligible and eligible[0][0] <= current_time + FLOAT_TOL:
            _, _, op_idx = heapq.heappop(eligible)
            op = ops[op_idx]

            # Determine LogGP params (layer-specific if available, else global)
            op_loggp = op.loggp if op.loggp is not None else loggp
            if op_loggp is None:
                raise ValueError("No LogGP params provided (neither global nor op-specific)")

            # Start transfer (add LogGP overhead at finish time, not start time)
            op.start_time = current_time
            active.add(op_idx)

        if not active:
            # No active ops, jump to next eligible time
            if eligible:
                current_time = eligible[0][0]
                continue
            else:
                # No more work
                break

        # Compute resource usage for max-min fair sharing
        resource_usage: dict[str, int] = {}  # resource name -> active count

        for op_idx in active:
            op = ops[op_idx]
            path = path_cache[(op.src, op.dst)]
            for res in path:
                resource_usage[res.name] = resource_usage.get(res.name, 0) + 1

        # Compute bottleneck bandwidth for each active op
        op_bandwidth: dict[int, float] = {}

        for op_idx in active:
            op = ops[op_idx]
            path = path_cache[(op.src, op.dst)]

            if not path:
                # Zero-byte transfer (src == dst), complete instantly
                op_bandwidth[op_idx] = float("inf")
            else:
                # Effective bandwidth = min(B_r / n_r) across path
                min_bw = float("inf")
                for res in path:
                    n_r = resource_usage[res.name]
                    effective_bw = res.bandwidth / n_r
                    min_bw = min(min_bw, effective_bw)
                op_bandwidth[op_idx] = min_bw

        # Compute time to next event
        time_to_completion = float("inf")

        for op_idx in active:
            op = ops[op_idx]
            if op.remaining_bytes <= FLOAT_TOL:
                # Already drained, will complete at dt=0
                time_to_completion = 0.0
                break

            bw = op_bandwidth[op_idx]
            if bw == float("inf"):
                # Infinite bandwidth (e.g., self-send), completes instantly
                time_to_completion = 0.0
                break
            elif bw > 0:
                time_to_drain = op.remaining_bytes / bw
                time_to_completion = min(time_to_completion, time_to_drain)

        # Time to next injection
        time_to_injection = float("inf")
        if eligible:
            next_inject_time = eligible[0][0]
            time_to_injection = max(0.0, next_inject_time - current_time)

        # Advance time
        dt = min(time_to_completion, time_to_injection)
        if dt == float("inf"):
            # Should not happen if we have active ops
            raise RuntimeError("Simulation stuck: active ops but no progress possible")

        current_time += dt

        # Drain bytes from active ops
        completed = []
        for op_idx in list(active):
            op = ops[op_idx]

            if op.remaining_bytes <= FLOAT_TOL:
                # Already at zero
                completed.append(op_idx)
                continue

            bw = op_bandwidth[op_idx]

            if bw == float("inf"):
                # Infinite bandwidth, complete instantly
                op.remaining_bytes = 0.0
                completed.append(op_idx)
            else:
                drained = bw * dt
                op.remaining_bytes -= drained

                # Check if completed (within tolerance)
                if op.remaining_bytes <= FLOAT_TOL:
                    op.remaining_bytes = 0.0
                    completed.append(op_idx)

        # Complete ops (remove from active, add LogGP overhead, update dependents)
        for op_idx in completed:
            active.discard(op_idx)
            op = ops[op_idx]

            # Determine LogGP params
            op_loggp = op.loggp if op.loggp is not None else loggp

            # Add LogGP overhead to finish time
            op.finish_time = current_time + op_loggp.alpha

            # Update dependents
            if op_idx in dependents:
                for dep_op_idx in dependents[op_idx]:
                    num_unsatisfied[dep_op_idx] -= 1
                    if num_unsatisfied[dep_op_idx] == 0:
                        # All dependencies satisfied, can inject after this op finishes
                        ready_time = op.finish_time
                        heapq.heappush(eligible, (ready_time, counter, dep_op_idx))
                        counter += 1

    # Compute per-rank finish times
    per_rank_finish: dict[int, float] = {}
    for op in ops:
        for rank in [op.src, op.dst]:
            per_rank_finish[rank] = max(per_rank_finish.get(rank, 0.0), op.finish_time)

    # Makespan is max finish time
    makespan = max((op.finish_time for op in ops), default=0.0)

    return SimulationResult(
        ops=ops,
        makespan=makespan,
        per_rank_finish=per_rank_finish,
    )
