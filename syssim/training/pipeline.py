"""Multi-rank graph composition for pipeline-parallel (MPMD) tracing.

When `pipeline_model_parallel_size > 1`, each pipeline stage is traced in its
own process via FakeProcessGroup. This module owns the post-trace step:

  1. Map spawn ranks <-> global ranks <-> pipeline stages.
  2. Compose per-stage operator graphs into one multi-rank graph by
     namespacing stream IDs and wiring cross-stage P2P send/recv as
     predecessor edges.
  3. Compute P2P estimated time via the flow-level network simulator
     against the topology built from HardwareConfig.topology.
"""

from __future__ import annotations


_STREAM_STRIDE = 1000  # max number of local streams per rank


def pp_rank_to_global_rank(pp_rank: int, parallelism) -> int:
    """Return the global rank that represents this PP stage's (tp=0, dp=0, cp=0) slice.

    Megatron's default rank ordering puts PP as the outer dimension, so the
    first rank in each PP stage is `pp_rank * (TP * DP * CP)`.
    """
    block = (
        parallelism.tensor_model_parallel_size
        * parallelism.data_parallel_size
        * parallelism.context_parallel_size
    )
    return pp_rank * block


def rank_to_node(global_rank: int, gpus_per_node: int) -> int:
    """Integer node index for a global rank under uniform gpus_per_node."""
    return global_rank // gpus_per_node


def p2p_time_ms(*, bytes_: int, src_rank: int, dst_rank: int, topology) -> float:
    """Estimated P2P time (ms) via the flow-level network simulator.

    Simulates a single Op(src_rank → dst_rank) against the given topology.
    The topology picks NVLink for same-node and the leaf↔spine path for
    cross-node.
    """
    if bytes_ <= 0:
        return 0.0
    from ..network import Op, simulate
    op = Op(src=int(src_rank), dst=int(dst_rank), size=float(bytes_))
    return simulate([op], topology).makespan * 1000.0


from ..operator_graph import OperatorGraph, OperatorNode, OperatorType


def _rename(name: str, pp_rank: int) -> str:
    return f"r{pp_rank}__{name}"


def compose_multi_rank_graph(
    *,
    per_stage_graphs: dict[int, "OperatorGraph"],
    parallelism,
    hardware,
) -> "OperatorGraph":
    """Combine per-stage operator graphs into one multi-rank graph.

    Steps:
      1. Rename every node to "r{pp_rank}__{name}" so names stay unique.
      2. Re-namespace stream IDs to "pp_rank * STREAM_STRIDE + local_stream".
      3. For each P2P send op on stage A targeting peer rank R (global), locate
         the matching P2P recv on the stage that owns rank R; add the cross-stage
         edge `recv.predecessors += [send.name]`.
      4. Set estimated_time_ms on both endpoints via `p2p_time_ms(...)`.
    """
    out = OperatorGraph(name="multi_rank")

    if hardware.topology is None:
        raise ValueError(
            "HardwareConfig.topology is required: PP P2P times come from the "
            "flow-level network simulator."
        )
    from ..network import build_topology_from_config
    topology = build_topology_from_config(hardware)

    # Pass 1: copy nodes with renamed streams and namespaced predecessors.
    for pp_rank, g in per_stage_graphs.items():
        for op in g.operators.values():
            new = OperatorNode(
                name=_rename(op.name, pp_rank),
                op_type=op.op_type,
                config=dict(op.config) if op.config else {},
                predecessors=[_rename(d, pp_rank) for d in op.predecessors],
                stream_id=pp_rank * _STREAM_STRIDE + op.stream_id,
                device_id=op.device_id,
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                estimated_time_ms=op.estimated_time_ms,
            )
            new.config["pp_rank"] = pp_rank
            out.add_operator(new)

    # Pass 2: pair P2P send (on stage A, peer = global rank in stage B's group)
    # with P2P recv (on stage B, peer = global rank in stage A's group).
    sends: dict[tuple[int, int, int], str] = {}   # (src_pp, dst_pp, tag) -> send op name
    recvs: dict[tuple[int, int, int], str] = {}   # (src_pp, dst_pp, tag) -> recv op name

    block = (
        parallelism.tensor_model_parallel_size
        * parallelism.data_parallel_size
        * parallelism.context_parallel_size
    )

    for pp_rank, g in per_stage_graphs.items():
        for op in g.operators.values():
            if op.config.get("kind") != "p2p":
                continue
            peer_global = op.config["peer_rank"]
            peer_pp = peer_global // block
            tag = op.config["tag"]
            key = (
                (pp_rank, peer_pp, tag) if op.config["direction"] == "send"
                else (peer_pp, pp_rank, tag)
            )
            (sends if op.config["direction"] == "send" else recvs)[key] = _rename(op.name, pp_rank)

    for key, send_name in sends.items():
        recv_name = recvs.get(key)
        if recv_name is None:
            continue  # unmatched send (e.g. last stage cooldown send to nowhere)
        send_op = out.operators[send_name]
        recv_op = out.operators[recv_name]
        # Cross-stage predecessor edge: recv waits on the matching send.
        if send_name not in recv_op.predecessors:
            recv_op.predecessors.append(send_name)
        # Time the P2P transfer on both endpoints.
        src_pp = key[0]
        dst_pp = key[1]
        t = p2p_time_ms(
            bytes_=send_op.config["bytes"],
            src_rank=pp_rank_to_global_rank(src_pp, parallelism),
            dst_rank=pp_rank_to_global_rank(dst_pp, parallelism),
            topology=topology,
        )
        send_op.estimated_time_ms = t
        recv_op.estimated_time_ms = t

    return out


def in_flight_microbatches(pp: int, stage_rank: int, num_microbatches: int) -> int:
    """Number of microbatches whose forward activations are simultaneously
    resident on `stage_rank` at the memory peak, under Megatron's 1F1B schedule.

    - pp == 1: 1 (sequential grad-accumulation frees activations each microbatch).
    - pp > 1 (1F1B): min(pp - stage_rank, num_microbatches) — earlier stages hold more.
    """
    if pp == 1:
        return 1
    return min(pp - stage_rank, num_microbatches)
