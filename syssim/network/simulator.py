"""Event-driven, flow-level network simulator.

Each Op is a point-to-point send with a list of dependency Ops. The simulator
runs an event loop, resolving paths via `topology.resolve_path()` and re-solving
max-min fair rates over the active flow set whenever the topology of in-flight
flows changes (op activates, op completes). Path latency delays a flow's
entry into the active set.

The topology model lives in `syssim.network.topology`; the max-min fair
solver below (`solve_max_min_fair`) is invoked at every event.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Op:
    """A single point-to-point send operation in the communication DAG.

    Attributes:
        src: Source GPU rank
        dst: Destination GPU rank
        size: Message size in bytes
        deps: Ops that must complete before this op can start
        tag: Optional label for debugging / visualisation

    Simulation state (populated by the simulator):
        remaining_bytes: bytes still to transfer
        start_time: when deps were satisfied
        active_time: when the flow's path latency has elapsed and it joins
            the active set
        finish_time: when bytes hit zero
    """
    src: int
    dst: int
    size: float
    deps: list["Op"] = field(default_factory=list)
    tag: str = ""

    # Simulation state
    remaining_bytes: float = 0.0
    start_time: float = 0.0
    active_time: float = 0.0
    finish_time: float = 0.0

    def __repr__(self) -> str:
        d = f", deps={len(self.deps)}" if self.deps else ""
        t = f", tag={self.tag}" if self.tag else ""
        return f"Op({self.src}->{self.dst}, {self.size/1e6:.2f}MB{d}{t})"


def solve_max_min_fair(
    flow_paths: np.ndarray,        # bool (F, L): flow i uses link j
    link_capacities: np.ndarray,   # float (L,)
) -> np.ndarray:                   # float (F,) per-flow rate
    """Iterative max-min fair allocation.

    At each iteration:
      1. share[l] = remaining_cap[l] / unassigned_flow_count[l]
      2. bottleneck link b = argmin(share)
      3. all unassigned flows on b receive rate = share[b]
      4. subtract their demand from remaining_cap of every link on their path
    Repeat until all flows assigned.

    Complexity: O(L * F) per iter; iters bounded by L (each iter removes the
    bottleneck link). Typical: <= 5 iters for DL workloads.
    """
    flow_paths = np.asarray(flow_paths, dtype=bool)
    link_capacities = np.asarray(link_capacities, dtype=float)
    F, L = flow_paths.shape
    rates = np.zeros(F)
    assigned = np.zeros(F, dtype=bool)
    remaining_cap = link_capacities.copy()

    while not assigned.all():
        active = flow_paths & ~assigned[:, None]   # (F, L)
        flow_count = active.sum(axis=0)            # (L,)
        share = np.where(
            flow_count > 0,
            remaining_cap / np.maximum(flow_count, 1),
            np.inf,
        )
        if not np.isfinite(share.min()):
            # Every unassigned flow has an empty path; assign zero.
            rates[~assigned] = 0.0
            break
        b = int(share.argmin())
        rate = float(share[b])
        newly_assigned = active[:, b].copy()
        rates[newly_assigned] = rate
        # Subtract their demand from every link on their paths
        consumed = active[newly_assigned].sum(axis=0) * rate
        remaining_cap -= consumed
        assigned |= newly_assigned

    return rates


from dataclasses import dataclass as _dc
import heapq


@_dc
class SimulationResult:
    """Output of one simulate() run."""
    makespan: float                       # seconds
    finish_times: dict[str, float]        # op.tag -> finish_time


def simulate(ops: list[Op], topology) -> SimulationResult:
    """Event-driven DES with max-min fair bandwidth allocation.

    Each op's path is resolved via topology.resolve_path(). At every event
    (op becomes ready, op activates after path latency, op completes), the
    active set's rates are re-solved.
    """
    # 1) Resolve paths for every op (once)
    op_paths: list[list] = []
    for i, op in enumerate(ops):
        path = topology.resolve_path(src=op.src, dst=op.dst, flow_tag=i)
        op_paths.append(path)
        op.remaining_bytes = op.size

    op_index = {id(op): i for i, op in enumerate(ops)}

    # 2) Build the unique link index
    all_links = {id(link): link for path in op_paths for link in path}
    link_list = list(all_links.values())
    link_idx = {id(l): k for k, l in enumerate(link_list)}
    capacities = np.array([l.capacity_GBps * 1e9 for l in link_list])  # bytes/sec

    # Per-op path as link indices, plus path latency
    op_link_idx = [
        np.array([link_idx[id(l)] for l in p], dtype=int)
        for p in op_paths
    ]
    op_latency = np.array([
        sum(l.latency_us for l in p) * 1e-6
        for p in op_paths
    ])

    # 3) Event-driven loop
    t = 0.0
    finished = set()
    ready: set[int] = set()       # ops with deps satisfied but waiting for path latency
    active: set[int] = set()      # ops currently draining bytes

    def ready_at(i: int) -> float:
        if not ops[i].deps:
            return 0.0
        return max(d.finish_time for d in ops[i].deps)

    def deps_satisfied(i: int) -> bool:
        return all(op_index[id(d)] in finished for d in ops[i].deps)

    # Initial ready set: ops with no deps
    pending = list(range(len(ops)))
    for i in list(pending):
        if not ops[i].deps:
            ops[i].start_time = 0.0
            ops[i].active_time = op_latency[i]
            ready.add(i)
            pending.remove(i)

    while ready or active or pending:
        # Move ops from ready -> active when active_time <= t
        for i in list(ready):
            if ops[i].active_time <= t + 1e-12:
                active.add(i)
                ready.remove(i)

        # If nothing is active but ops are ready (waiting for latency), jump to
        # the earliest active_time. Pending ops can only become ready when an
        # active op completes; if active is empty AND ready is empty, we're done.
        if not active:
            if not ready:
                break
            next_t = min(ops[i].active_time for i in ready)
            if next_t <= t:
                break
            t = next_t
            continue

        # Solve max-min fair on the active set
        F = len(active)
        active_list = list(active)
        L = len(link_list)
        fp = np.zeros((F, L), dtype=bool)
        for row, op_i in enumerate(active_list):
            fp[row, op_link_idx[op_i]] = True
        rates = solve_max_min_fair(fp, capacities)
        rate_of = {op_i: rates[row] for row, op_i in enumerate(active_list)}

        # Determine the next event time: when the next active op finishes
        #   OR when a ready op's active_time is reached
        #   OR when a pending op's deps could be satisfied
        next_active = min(
            (ops[i].remaining_bytes / rate_of[i] if rate_of[i] > 0 else float("inf"))
            for i in active
        )
        next_ready_latency = min(
            (ops[i].active_time - t for i in ready),
            default=float("inf"),
        )
        dt = min(next_active, next_ready_latency)
        if dt <= 0 or not np.isfinite(dt):
            # No further progress possible
            break
        t_next = t + dt

        # Drain bytes from all active ops
        for op_i in active:
            ops[op_i].remaining_bytes -= rate_of[op_i] * dt

        # Complete ops whose bytes <= 0
        for op_i in list(active):
            if ops[op_i].remaining_bytes <= 1e-6:
                ops[op_i].finish_time = t_next
                finished.add(op_i)
                active.remove(op_i)

        t = t_next

        # Move newly-eligible pending -> ready
        for i in list(pending):
            if deps_satisfied(i):
                ops[i].start_time = ready_at(i)
                ops[i].active_time = ops[i].start_time + op_latency[i]
                ready.add(i)
                pending.remove(i)

    makespan = max((ops[i].finish_time for i in range(len(ops))), default=0.0)
    return SimulationResult(
        makespan=makespan,
        finish_times={ops[i].tag: ops[i].finish_time for i in range(len(ops))},
    )
