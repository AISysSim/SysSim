"""Collective communication decomposers.

Each function returns a list of Op objects with explicit data-causality
deps inlined. There is no shared `build_dag` helper: the dep rule
("a round's send from rank r depends on the previous round's op that
wrote to rank r") is a single dict lookup per op.

Other dependencies that the old code expressed explicitly (send-from-r
serialization) are dropped because the simulator's max-min fairness on
per-GPU uplinks handles per-rank contention automatically.
"""

from __future__ import annotations

from .simulator import Op


def allreduce(ranks: list[int], total_bytes: float, tag: str = "") -> list[Op]:
    """Ring all-reduce over `ranks`. Returns 2*(n-1)*n Op objects."""
    n = len(ranks)
    if n < 2:
        return []
    chunk = total_bytes / n
    ops: list[Op] = []
    prev_into: dict[int, Op] = {}
    for step in range(2 * (n - 1)):
        new_into: dict[int, Op] = {}
        for i, src in enumerate(ranks):
            dst = ranks[(i + 1) % n]
            deps = [prev_into[src]] if src in prev_into else []
            op = Op(src=src, dst=dst, size=chunk,
                    deps=deps, tag=f"{tag}_step_{step}")
            new_into[dst] = op
            ops.append(op)
        prev_into = new_into
    return ops


def reduce_scatter(ranks: list[int], total_bytes: float, tag: str = "") -> list[Op]:
    """Ring reduce-scatter over `ranks`. Returns (n-1)*n Op objects."""
    n = len(ranks)
    if n < 2:
        return []
    chunk = total_bytes / n
    ops: list[Op] = []
    prev_into: dict[int, Op] = {}
    for step in range(n - 1):
        new_into: dict[int, Op] = {}
        for i, src in enumerate(ranks):
            dst = ranks[(i + 1) % n]
            deps = [prev_into[src]] if src in prev_into else []
            op = Op(src=src, dst=dst, size=chunk,
                    deps=deps, tag=f"{tag}_step_{step}")
            new_into[dst] = op
            ops.append(op)
        prev_into = new_into
    return ops


def allgather(ranks: list[int], total_bytes: float, tag: str = "") -> list[Op]:
    """Ring all-gather over `ranks`. Returns (n-1)*n Op objects."""
    n = len(ranks)
    if n < 2:
        return []
    chunk = total_bytes / n
    ops: list[Op] = []
    prev_into: dict[int, Op] = {}
    for step in range(n - 1):
        new_into: dict[int, Op] = {}
        for i, src in enumerate(ranks):
            dst = ranks[(i + 1) % n]
            deps = [prev_into[src]] if src in prev_into else []
            op = Op(src=src, dst=dst, size=chunk,
                    deps=deps, tag=f"{tag}_step_{step}")
            new_into[dst] = op
            ops.append(op)
        prev_into = new_into
    return ops


def all_to_all(ranks: list[int], total_bytes: float, tag: str = "") -> list[Op]:
    """All-to-all over `ranks`. Returns n*(n-1) Op objects.

    In an all-to-all every rank sends a distinct chunk to every OTHER rank.
    `total_bytes` is the total bytes a single rank sends to all other ranks
    combined (the per-rank send volume), so each of the n-1 destinations
    receives `total_bytes / (n - 1)`.

    Unlike the ring collectives, all n*(n-1) flows are CONCURRENT and
    INDEPENDENT: there are no data-causality deps between them (deps=[] on
    every Op). A single all-to-all is one round of simultaneous transfers;
    per-link / per-uplink contention is resolved by the simulator's max-min
    fair solver (same rationale stated at the top of this module).
    """
    n = len(ranks)
    if n < 2 or total_bytes <= 0:
        return []
    chunk = total_bytes / (n - 1)
    ops: list[Op] = []
    for src in ranks:
        for dst in ranks:
            if src == dst:
                continue
            ops.append(Op(src=src, dst=dst, size=chunk, deps=[], tag=tag))
    return ops


def broadcast(
    ranks: list[int], total_bytes: float, root: int, tag: str = "",
) -> list[Op]:
    """Binary-tree broadcast from `root` to all other ranks.

    Each rank that has the data sends it to one rank that doesn't, doubling
    the number of holders per round. Total ops = n - 1.
    """
    if root not in ranks:
        raise ValueError(f"root {root} not in ranks {ranks}")
    others = [r for r in ranks if r != root]
    if not others:
        return []
    # holders maps rank → the op that delivered the data to that rank (None for root)
    holders: dict[int, Op | None] = {root: None}
    ops: list[Op] = []
    step = 0
    remaining = list(others)
    while remaining:
        new_holders: dict[int, Op | None] = {}
        senders = list(holders.keys())
        # Pair each sender with one receiver from `remaining`
        for sender in senders:
            if not remaining:
                break
            recv = remaining.pop(0)
            deps = [holders[sender]] if holders[sender] is not None else []
            op = Op(src=sender, dst=recv, size=total_bytes,
                    deps=deps, tag=f"{tag}_step_{step}")
            ops.append(op)
            new_holders[recv] = op
        holders.update(new_holders)
        step += 1
    return ops
