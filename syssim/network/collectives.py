"""Collective communication algorithm builders.

This module provides 8 collective communication primitives that generate
Step sequences (communication patterns). These Steps are then converted to
DAGs by build_dag() for simulation.

All builders follow the pattern:
    collective(ranks, size, ...) -> list[Op]

Design principles:
- Builders specify WHAT (communication pattern), not HOW (resources/timing)
- Each builder returns a complete Op DAG (via build_dag)
- Topology and LogGP parameters are supplied later to simulate()

Collective algorithms:
- allreduce: Ring algorithm (reduce-scatter + allgather)
- broadcast: Binomial tree from root
- reduce: Binomial tree to root
- reduce_scatter: First half of ring allreduce
- allgather: Second half of ring allreduce
- alltoall: Direct with staggered pairings
- scatter: Flat from root (serialized by Rule 2)
- gather: Flat to root (parallel, contention on root)

References:
- "Optimization of Collective Communication Operations in MPICH"
  (Thakur et al., 2005)
- NCCL algorithms documentation
"""

import math

from .dag_builder import Op, Step, build_dag


def allreduce(ranks: list[int], total_size: float, tag_prefix: str = "") -> list[Op]:
    """All-reduce using ring algorithm (reduce-scatter + allgather).

    Algorithm: Ring with 2(P-1) steps
    - Reduce-scatter: P-1 steps, each rank sends chunk_i to next rank
    - Allgather: P-1 steps, each rank sends chunk_i to next rank

    Time complexity: 2(P-1) * (α + M/P * G)
    - α = L + 2*o (LogGP overhead)
    - M = total message size
    - P = number of ranks

    Args:
        ranks: List of participating rank IDs
        total_size: Total data size in bytes (will be chunked into P pieces)
        tag_prefix: Optional prefix for op tags (default: "allreduce")

    Returns:
        List of Op objects forming the allreduce DAG

    Example:
        >>> ops = allreduce([0, 1, 2, 3], total_size=1e9)
        >>> len(ops)
        24  # 4 ranks * (4-1) * 2 phases = 24 ops
        >>> ops[0].size
        250000000.0  # 1e9 / 4 ranks
    """
    P = len(ranks)
    if P < 2:
        raise ValueError(f"allreduce requires at least 2 ranks, got {P}")

    chunk_size = total_size / P
    steps: list[Step] = []

    # Reduce-scatter phase: P-1 steps
    for step in range(P - 1):
        step_ops = []
        for i in range(P):
            src = ranks[i]
            dst = ranks[(i + 1) % P]
            step_ops.append((src, dst, chunk_size))
        steps.append(step_ops)

    # Allgather phase: P-1 steps
    for step in range(P - 1):
        step_ops = []
        for i in range(P):
            src = ranks[i]
            dst = ranks[(i + 1) % P]
            step_ops.append((src, dst, chunk_size))
        steps.append(step_ops)

    return build_dag(steps, tag_prefix or "allreduce")


def broadcast(ranks: list[int], total_size: float, root: int = 0, tag_prefix: str = "") -> list[Op]:
    """Broadcast using binomial tree algorithm.

    Algorithm: Binary tree propagation
    - Step k: ranks with data send to 2^k neighbors
    - Total steps: ⌈log₂ P⌉

    Time complexity: ⌈log₂ P⌉ * (α + M * G)

    Args:
        ranks: List of participating rank IDs
        total_size: Data size in bytes
        root: Root rank that initiates broadcast (must be in ranks)
        tag_prefix: Optional prefix for op tags (default: "broadcast")

    Returns:
        List of Op objects forming the broadcast DAG

    Example:
        >>> ops = broadcast([0, 1, 2, 3], total_size=1e6, root=0)
        >>> len(ops)
        3  # ⌈log₂ 4⌉ = 2 steps, but binomial tree has varying sends per step
        >>> # Step 0: 0->1
        >>> # Step 1: 0->2, 1->3
        >>> ops[0].src, ops[0].dst
        (0, 1)
    """
    P = len(ranks)
    if P < 2:
        raise ValueError(f"broadcast requires at least 2 ranks, got {P}")
    if root not in ranks:
        raise ValueError(f"root {root} not in ranks {ranks}")

    # Reorder ranks so root is at index 0
    root_idx = ranks.index(root)
    reordered = ranks[root_idx:] + ranks[:root_idx]

    steps: list[Step] = []
    num_steps = math.ceil(math.log2(P))

    # Binomial tree: at step k, ranks [0, 2^k) send to [2^k, 2^(k+1))
    for step in range(num_steps):
        step_ops = []
        stride = 1 << step  # 2^step

        for i in range(min(stride, P)):
            dst_idx = i + stride
            if dst_idx < P:
                src = reordered[i]
                dst = reordered[dst_idx]
                step_ops.append((src, dst, total_size))

        if step_ops:
            steps.append(step_ops)

    return build_dag(steps, tag_prefix or "broadcast")


def reduce(ranks: list[int], total_size: float, root: int = 0, tag_prefix: str = "") -> list[Op]:
    """Reduce using binomial tree algorithm (mirror of broadcast).

    Algorithm: Binary tree collection
    - Step k: ranks at positions (i + 2^k) send to positions i, where i is multiple of 2^(k+1)
    - Total steps: ⌈log₂ P⌉

    Time complexity: ⌈log₂ P⌉ * (α + M * G)

    Args:
        ranks: List of participating rank IDs
        total_size: Data size in bytes
        root: Root rank that receives final result (must be in ranks)
        tag_prefix: Optional prefix for op tags (default: "reduce")

    Returns:
        List of Op objects forming the reduce DAG

    Example:
        >>> ops = reduce([0, 1, 2, 3], total_size=1e6, root=0)
        >>> # Step 0: 1->0, 3->2
        >>> # Step 1: 2->0
        >>> ops[-1].dst  # Last op sends to root
        0
    """
    P = len(ranks)
    if P < 2:
        raise ValueError(f"reduce requires at least 2 ranks, got {P}")
    if root not in ranks:
        raise ValueError(f"root {root} not in ranks {ranks}")

    # Reorder ranks so root is at index 0
    root_idx = ranks.index(root)
    reordered = ranks[root_idx:] + ranks[:root_idx]

    steps: list[Step] = []
    num_steps = math.ceil(math.log2(P))

    # Binomial tree reduce: at step k, pairs are separated by distance 2^k
    # Receiver is at positions that are multiples of 2^(k+1)
    for step in range(num_steps):
        step_ops = []
        stride = 1 << step  # 2^step (distance between sender and receiver)
        pair_distance = 1 << (step + 1)  # 2^(step+1) (distance between pairs)

        # Iterate over receiver positions
        i = 0
        while i < P:
            src_idx = i + stride
            if src_idx < P:
                src = reordered[src_idx]
                dst = reordered[i]
                step_ops.append((src, dst, total_size))
            i += pair_distance

        if step_ops:
            steps.append(step_ops)

    return build_dag(steps, tag_prefix or "reduce")


def reduce_scatter(ranks: list[int], total_size: float, tag_prefix: str = "") -> list[Op]:
    """Reduce-scatter using ring algorithm (first half of allreduce).

    Algorithm: Ring with P-1 steps
    - Each rank sends chunk_i to next rank in ring
    - After P-1 steps, each rank has reduced one chunk

    Time complexity: (P-1) * (α + M/P * G)

    Args:
        ranks: List of participating rank IDs
        total_size: Total data size in bytes (will be chunked into P pieces)
        tag_prefix: Optional prefix for op tags (default: "reduce_scatter")

    Returns:
        List of Op objects forming the reduce-scatter DAG

    Example:
        >>> ops = reduce_scatter([0, 1, 2, 3], total_size=1e9)
        >>> len(ops)
        12  # 4 ranks * 3 steps
        >>> ops[0].size
        250000000.0  # 1e9 / 4 ranks
    """
    P = len(ranks)
    if P < 2:
        raise ValueError(f"reduce_scatter requires at least 2 ranks, got {P}")

    chunk_size = total_size / P
    steps: list[Step] = []

    # Reduce-scatter phase: P-1 steps
    for step in range(P - 1):
        step_ops = []
        for i in range(P):
            src = ranks[i]
            dst = ranks[(i + 1) % P]
            step_ops.append((src, dst, chunk_size))
        steps.append(step_ops)

    return build_dag(steps, tag_prefix or "reduce_scatter")


def allgather(ranks: list[int], total_size: float, tag_prefix: str = "") -> list[Op]:
    """Allgather using ring algorithm (second half of allreduce).

    Algorithm: Ring with P-1 steps
    - Each rank sends chunk_i to next rank in ring
    - After P-1 steps, each rank has all chunks

    Time complexity: (P-1) * (α + M/P * G)

    Args:
        ranks: List of participating rank IDs
        total_size: Total data size in bytes (will be chunked into P pieces)
        tag_prefix: Optional prefix for op tags (default: "allgather")

    Returns:
        List of Op objects forming the allgather DAG

    Example:
        >>> ops = allgather([0, 1, 2, 3], total_size=1e9)
        >>> len(ops)
        12  # 4 ranks * 3 steps
    """
    P = len(ranks)
    if P < 2:
        raise ValueError(f"allgather requires at least 2 ranks, got {P}")

    chunk_size = total_size / P
    steps: list[Step] = []

    # Allgather phase: P-1 steps
    for step in range(P - 1):
        step_ops = []
        for i in range(P):
            src = ranks[i]
            dst = ranks[(i + 1) % P]
            step_ops.append((src, dst, chunk_size))
        steps.append(step_ops)

    return build_dag(steps, tag_prefix or "allgather")


def alltoall(ranks: list[int], total_size: float, tag_prefix: str = "") -> list[Op]:
    """All-to-all using direct algorithm with staggered pairings.

    Algorithm: P-1 steps, each rank sends to different neighbor
    - Step k: rank i sends to rank (i+k) % P
    - Each message has size M/P

    Time complexity: (P-1) * (α + M/P * G)

    Args:
        ranks: List of participating rank IDs
        total_size: Total data size in bytes (M/P sent per pair)
        tag_prefix: Optional prefix for op tags (default: "alltoall")

    Returns:
        List of Op objects forming the alltoall DAG

    Example:
        >>> ops = alltoall([0, 1, 2, 3], total_size=1e9)
        >>> len(ops)
        12  # 4 ranks * 3 steps
        >>> ops[0].size
        250000000.0  # 1e9 / 4 ranks
    """
    P = len(ranks)
    if P < 2:
        raise ValueError(f"alltoall requires at least 2 ranks, got {P}")

    chunk_size = total_size / P
    steps: list[Step] = []

    # P-1 steps, staggered pairings
    for step in range(1, P):
        step_ops = []
        for i in range(P):
            src = ranks[i]
            dst = ranks[(i + step) % P]
            step_ops.append((src, dst, chunk_size))
        steps.append(step_ops)

    return build_dag(steps, tag_prefix or "alltoall")


def scatter(ranks: list[int], total_size: float, root: int = 0, tag_prefix: str = "") -> list[Op]:
    """Scatter using flat tree (root sends to all others).

    Algorithm: Root sends chunk_i to rank i in sequence
    - All sends serialized by Rule 2 (send serialization)
    - P-1 sequential sends from root

    Time complexity: (P-1) * (α + M/P * G)
    - Note: sends are serialized, not pipelined

    Args:
        ranks: List of participating rank IDs
        total_size: Total data size in bytes (will be chunked into P pieces)
        root: Root rank that initiates scatter (must be in ranks)
        tag_prefix: Optional prefix for op tags (default: "scatter")

    Returns:
        List of Op objects forming the scatter DAG

    Example:
        >>> ops = scatter([0, 1, 2, 3], total_size=1e9, root=0)
        >>> len(ops)
        3  # root sends to 3 other ranks
        >>> all(op.src == 0 for op in ops)
        True
    """
    P = len(ranks)
    if P < 2:
        raise ValueError(f"scatter requires at least 2 ranks, got {P}")
    if root not in ranks:
        raise ValueError(f"root {root} not in ranks {ranks}")

    chunk_size = total_size / P
    steps: list[Step] = []

    # Root sends to all other ranks (one per step for serialization via Rule 2)
    for rank in ranks:
        if rank != root:
            steps.append([(root, rank, chunk_size)])

    return build_dag(steps, tag_prefix or "scatter")


def gather(ranks: list[int], total_size: float, root: int = 0, tag_prefix: str = "") -> list[Op]:
    """Gather using flat tree (all send to root in parallel).

    Algorithm: All non-root ranks send to root simultaneously
    - All sends in single step (parallel)
    - Contention on root's receive bandwidth handled by simulator

    Time complexity (with contention): (P-1) * (α + M/P * G)
    - Actual time depends on topology (inbound bandwidth to root)

    Args:
        ranks: List of participating rank IDs
        total_size: Total data size in bytes (will be chunked into P pieces)
        root: Root rank that receives all data (must be in ranks)
        tag_prefix: Optional prefix for op tags (default: "gather")

    Returns:
        List of Op objects forming the gather DAG

    Example:
        >>> ops = gather([0, 1, 2, 3], total_size=1e9, root=0)
        >>> len(ops)
        3  # 3 non-root ranks send to root
        >>> all(op.dst == 0 for op in ops)
        True
    """
    P = len(ranks)
    if P < 2:
        raise ValueError(f"gather requires at least 2 ranks, got {P}")
    if root not in ranks:
        raise ValueError(f"root {root} not in ranks {ranks}")

    chunk_size = total_size / P

    # All non-root ranks send to root in single step (parallel)
    step_ops = []
    for rank in ranks:
        if rank != root:
            step_ops.append((rank, root, chunk_size))

    return build_dag([step_ops], tag_prefix or "gather")
