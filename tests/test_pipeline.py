import pytest


def test_pp_rank_to_global_rank_pp_only():
    """For PP-only (TP=DP=CP=1), spawn rank == pp_rank == global rank."""
    from syssim.training.pipeline import pp_rank_to_global_rank
    from syssim.training.spec import ParallelismConfig

    p = ParallelismConfig(pp=4)
    for r in range(4):
        assert pp_rank_to_global_rank(r, p) == r


def test_pp_rank_to_global_rank_with_tp_dp():
    """Each PP stage owns a contiguous block of (TP*DP*CP) ranks; stage's rank-0 is what we trace."""
    from syssim.training.pipeline import pp_rank_to_global_rank
    from syssim.training.spec import ParallelismConfig

    p = ParallelismConfig(tp=2, dp=4, cp=1, pp=8)
    assert pp_rank_to_global_rank(0, p) == 0
    assert pp_rank_to_global_rank(1, p) == 8     # 2*4*1
    assert pp_rank_to_global_rank(7, p) == 56


def test_rank_to_node():
    """Map a global rank to its node index given gpus_per_node."""
    from syssim.training.pipeline import rank_to_node
    assert rank_to_node(0, gpus_per_node=8) == 0
    assert rank_to_node(7, gpus_per_node=8) == 0
    assert rank_to_node(8, gpus_per_node=8) == 1
    assert rank_to_node(63, gpus_per_node=8) == 7


def _two_node_topology():
    from syssim.network.topology import build_two_layer_multipath
    return build_two_layer_multipath(
        num_racks=2, nodes_per_rack=1, gpus_per_node=8, num_spines=1,
        per_gpu_bandwidth_GBps=50.0, uplink_bandwidth_GBps=400.0,
        intra_node_bandwidth_GBps=900.0,
    )


def test_p2p_time_intra_node():
    """Same-node P2P routes via NVLink (900 GB/s)."""
    from syssim.training.pipeline import p2p_time_ms
    topology = _two_node_topology()
    # 100 MB between ranks 0 and 1 (same node, gpus_per_node=8) — uses NVLink
    t = p2p_time_ms(bytes_=100_000_000, src_rank=0, dst_rank=1, topology=topology)
    # 100 MB / 900 GB/s ~= 0.111 ms
    assert 0.10 < t < 0.13


def test_p2p_time_inter_node():
    """Cross-node P2P routes via the leaf-spine path (bottlenecked by per-GPU uplink at 50 GB/s)."""
    from syssim.training.pipeline import p2p_time_ms
    topology = _two_node_topology()
    # ranks 7 and 8 are on different nodes (gpus_per_node=8); bottleneck = per-GPU uplink
    t = p2p_time_ms(bytes_=100_000_000, src_rank=7, dst_rank=8, topology=topology)
    # 100 MB / 50 GB/s = 2.0 ms
    assert 1.9 < t < 2.1


def test_p2p_time_zero_bytes():
    from syssim.training.pipeline import p2p_time_ms
    topology = _two_node_topology()
    assert p2p_time_ms(bytes_=0, src_rank=0, dst_rank=1, topology=topology) == 0.0


def test_compose_pairs_p2p_send_recv():
    """One per-rank graph with a send, one with a matching recv -> composed graph wires the edge."""
    from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType
    from syssim.training.pipeline import compose_multi_rank_graph
    from syssim.training.spec import ParallelismConfig, HardwareConfig

    g0 = OperatorGraph(name="stage0")
    g0.add_operator(OperatorNode(
        name="op_0", op_type=OperatorType.GEMM, estimated_time_ms=1.0, stream_id=0,
    ))
    g0.add_operator(OperatorNode(
        name="p2p_send_1",
        op_type=OperatorType.COLLECTIVE,
        config={"kind": "p2p", "direction": "send", "peer_rank": 1, "tag": 0, "bytes": 1_000_000},
        predecessors=["op_0"], stream_id=1,
    ))

    g1 = OperatorGraph(name="stage1")
    g1.add_operator(OperatorNode(
        name="p2p_recv_0",
        op_type=OperatorType.COLLECTIVE,
        config={"kind": "p2p", "direction": "recv", "peer_rank": 0, "tag": 0, "bytes": 1_000_000},
        stream_id=1,
    ))
    g1.add_operator(OperatorNode(
        name="op_1", op_type=OperatorType.GEMM, estimated_time_ms=1.0,
        predecessors=["p2p_recv_0"], stream_id=0,
    ))

    p = ParallelismConfig(pp=2)
    hw = HardwareConfig(
        peak_tflops_mm=1000, peak_tflops_math=1000,
        peak_memory_bandwidth_GBps=3000, gpus_per_node=8,
        topology={"dims": ["fully_connected"], "size": [8],
                  "bandwidth": [900.0], "latency": [1000.0]},
    )
    composed = compose_multi_rank_graph(
        per_stage_graphs={0: g0, 1: g1}, parallelism=p, hardware=hw,
    )
    # Both p2p ops survive, renamed by stage prefix; recv now has send as predecessor.
    send = composed.operators["r0__p2p_send_1"]
    recv = composed.operators["r1__p2p_recv_0"]
    assert "r0__p2p_send_1" in recv.predecessors
    # Both p2p ops have non-zero estimated time (intra-node NVLink: ranks 0 and 1 in node 0).
    assert send.estimated_time_ms > 0
    assert recv.estimated_time_ms > 0


def test_in_flight_pp1_is_one():
    from syssim.training.pipeline import in_flight_microbatches
    assert in_flight_microbatches(pp=1, stage_rank=0, num_microbatches=8) == 1


def test_in_flight_1f1b_decreases_by_stage():
    from syssim.training.pipeline import in_flight_microbatches
    got = [in_flight_microbatches(pp=4, stage_rank=s, num_microbatches=8) for s in range(4)]
    assert got == [4, 3, 2, 1]


def test_in_flight_clamped_to_num_microbatches():
    from syssim.training.pipeline import in_flight_microbatches
    got = [in_flight_microbatches(pp=4, stage_rank=s, num_microbatches=2) for s in range(4)]
    assert got == [2, 2, 2, 1]
