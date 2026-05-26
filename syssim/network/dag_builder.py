"""DAG construction from communication steps with automatic dependency inference.

This module converts high-level communication patterns (list of steps) into a
dependency DAG (list of Ops) suitable for simulation.

Key concepts:
- Step: A list of concurrent sends in a single algorithmic phase
        Format: list[tuple[src, dst, size_bytes]]
- Op: A single point-to-point send with explicit dependencies
- build_dag(): Infers dependencies from 2 physical rules

Dependency rules:
1. Data dependency: Send FROM rank r at step s depends on sends TO rank r at step s-1
                   (rank can't send data it hasn't received)
2. Send serialization: Send FROM rank r depends on previous send FROM rank r
                      (NIC can only send one message at a time)

Design choice: We do NOT add receive-side serialization dependencies because
the congestion engine already handles inbound bandwidth contention. Adding
explicit receive dependencies would be redundant and overly conservative.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .loggp import LogGPParams

# Type alias for communication steps
# Each step is a list of concurrent sends: [(src, dst, size_bytes), ...]
Step = list[tuple[int, int, float]]


@dataclass
class Op:
    """A single point-to-point send operation in the communication DAG.

    Attributes:
        src: Source rank ID
        dst: Destination rank ID
        size: Message size in bytes
        deps: List of Ops that must complete before this Op can start
        tag: Optional label for debugging/visualization (e.g., "allreduce_step_0")
        loggp: Optional layer-specific LogGP params (for HierarchicalTopology)

    Simulation state (populated by simulator):
        remaining_bytes: Bytes left to transfer (starts at size, decrements to 0)
        start_time: Timestamp when transfer begins (seconds)
        finish_time: Timestamp when transfer completes (seconds)

    Example:
        >>> op = Op(src=0, dst=1, size=1e6, tag="allreduce")
        >>> op.size
        1000000.0
        >>> op.remaining_bytes  # Initialized to 0, set by simulator
        0.0
    """

    src: int
    dst: int
    size: float  # bytes
    deps: list["Op"] = field(default_factory=list)
    tag: str = ""
    loggp: Optional["LogGPParams"] = None  # For HierarchicalTopology layer-specific params

    # Simulation state (populated by engine)
    remaining_bytes: float = 0.0
    start_time: float = 0.0
    finish_time: float = 0.0

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        dep_str = f", deps={len(self.deps)}" if self.deps else ""
        tag_str = f", tag={self.tag}" if self.tag else ""
        return f"Op({self.src}->{self.dst}, {self.size / 1e6:.2f}MB{dep_str}{tag_str})"


def build_dag(steps: list[Step], tag_prefix: str = "") -> list[Op]:
    """Build dependency DAG from communication steps using 2 physical rules.

    This function converts a high-level algorithmic description (list of steps)
    into a concrete DAG suitable for simulation. Dependencies are inferred
    automatically from physical constraints.

    Rules:
    1. Data dependency (receive-before-send):
       - Send FROM rank r at step s depends on sends TO rank r at step s-1
       - Example: In reduce-scatter, rank 0 can't send chunk_i at step 1
         until it receives chunk_i from rank 1 at step 0

    2. Send serialization (single-threaded NIC):
       - Send FROM rank r at step s depends on send FROM rank r at step s-1
       - Example: Rank 0 can't start sending message 2 until message 1 is
         in flight (NIC serialization)

    Why these 2 rules are sufficient:
    - Rule 1 enforces causality (can't send data you don't have)
    - Rule 2 enforces sender exclusivity (one send at a time per rank)
    - Receive-side contention handled by simulator (bandwidth sharing)
    - No need for explicit receive-side serialization dependencies

    Args:
        steps: List of communication steps, each step is list[(src, dst, size)]
        tag_prefix: Optional prefix for op tags (e.g., "allreduce", "broadcast")

    Returns:
        List of Op objects with dependencies populated

    Example:
        >>> # 2-rank reduce-scatter (simplified)
        >>> steps = [
        ...     [(0, 1, 100), (1, 0, 100)],  # Step 0: exchange chunks
        ... ]
        >>> ops = build_dag(steps, tag_prefix="reduce_scatter")
        >>> len(ops)
        2
        >>> ops[0].tag
        'reduce_scatter_step_0'
        >>> # No dependencies in first step
        >>> ops[0].deps
        []

    Example with dependencies:
        >>> # 3-rank ring allreduce (reduce-scatter phase)
        >>> steps = [
        ...     [(0, 1, 100), (1, 2, 100), (2, 0, 100)],  # Step 0
        ...     [(0, 1, 100), (1, 2, 100), (2, 0, 100)],  # Step 1
        ... ]
        >>> ops = build_dag(steps)
        >>> # Step 1 ops depend on:
        >>> # - Receives TO same rank at step 0 (Rule 1)
        >>> # - Send FROM same rank at step 0 (Rule 2)
        >>> ops[3].deps  # First op in step 1
        [Op(2->0, ...), Op(0->1, ...)]  # Received from rank 2, sent to rank 1
    """
    all_ops: list[Op] = []

    # Track last send FROM each rank (for Rule 2: send serialization)
    last_send_from: dict[int, Op] = {}

    # Track ops that send TO each rank in previous step (for Rule 1: data dependency)
    prev_sends_to: dict[int, list[Op]] = {}

    for step_idx, step in enumerate(steps):
        # Current step's sends TO each rank (becomes prev_sends_to for next step)
        curr_sends_to: dict[int, list[Op]] = {}

        for src, dst, size in step:
            # Create op with tag
            tag = f"{tag_prefix}_step_{step_idx}" if tag_prefix else f"step_{step_idx}"
            op = Op(src=src, dst=dst, size=size, tag=tag)

            # Rule 1: Data dependency
            # This send FROM src depends on sends TO src in previous step
            # (src can't send data it hasn't received)
            if src in prev_sends_to:
                op.deps.extend(prev_sends_to[src])

            # Rule 2: Send serialization
            # This send FROM src depends on previous send FROM src
            # (NIC can only send one message at a time)
            if src in last_send_from:
                # Avoid duplicate if Rule 1 already added this dependency
                if last_send_from[src] not in op.deps:
                    op.deps.append(last_send_from[src])

            # Update tracking
            last_send_from[src] = op
            if dst not in curr_sends_to:
                curr_sends_to[dst] = []
            curr_sends_to[dst].append(op)

            all_ops.append(op)

        # Move to next step
        prev_sends_to = curr_sends_to

    return all_ops
