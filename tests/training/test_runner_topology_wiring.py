"""Verify that HardwareConfig.topology routes all collectives through the network simulator."""

import pytest

from syssim.training.runner import _estimate_dp_allreduce_ms, _estimate_collective_ms
from syssim.network.topology import build_two_layer_multipath


def _topology():
    return build_two_layer_multipath(
        num_racks=1, nodes_per_rack=1, gpus_per_node=4, num_spines=1,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )


def test_dp_allreduce_routes_through_simulator():
    """DP all-reduce time comes from simulate() on the ring-decomposed Op DAG."""
    ms = _estimate_dp_allreduce_ms(int(1e9), [0, 1, 2, 3], _topology())
    # Ring AR on 4 ranks, intra-node only: 6 rounds x 4 parallel flows x (1GB/4)/900GBps
    # = 6 x 0.25 GB / 900 GBps ~ 1.667 ms.
    assert ms > 0
    assert ms < 10


def test_dp_comm_distributed_optimizer_is_param_allgather():
    """ZeRO-1 DP comm = a bf16 all-gather of the TP-sharded params (measured on GH200: no
    reduce-scatter on the wire, bf16). Plain DDP = all-reduce of the bf16 gradient."""
    from syssim.training.runner import _estimate_dp_comm_ms, _estimate_collective_ms
    topo = _topology()
    g = int(1e9); spb = int(2e9); ranks = [0, 1, 2, 3]
    zero = _estimate_dp_comm_ms(grad_bytes=g, sharded_param_bytes=spb, dp_ranks=ranks,
                                topology=topo, distributed_optimizer=True)
    ag = _estimate_collective_ms(collective_name="all_gather", total_bytes=spb, ranks=ranks, topology=topo)
    assert abs(zero - ag) < 1e-9, (zero, ag)
    plain = _estimate_dp_comm_ms(grad_bytes=g, sharded_param_bytes=spb, dp_ranks=ranks,
                                 topology=topo, distributed_optimizer=False)
    ar = _estimate_collective_ms(collective_name="all_reduce", total_bytes=g, ranks=ranks, topology=topo)
    assert abs(plain - ar) < 1e-9, (plain, ar)


def test_dp_comm_single_rank_is_zero():
    """No DP communication when the DP group has one member."""
    from syssim.training.runner import _estimate_dp_comm_ms
    assert _estimate_dp_comm_ms(grad_bytes=int(1e9), sharded_param_bytes=int(1e9), dp_ranks=[0],
                                topology=_topology(), distributed_optimizer=True) == 0.0


def test_dp_allreduce_returns_zero_when_p_le_1():
    """Single-rank or zero-byte DP allreduce returns 0 immediately."""
    topology = _topology()
    assert _estimate_dp_allreduce_ms(int(1e9), [0], topology) == 0.0
    assert _estimate_dp_allreduce_ms(0, [0, 1, 2, 3], topology) == 0.0


def test_dp_group_ranks_strided_by_tp():
    """The DP group containing rank 0 is strided by tp_size (Megatron's tp-fastest rank order):
    DP replicas of a tensor-parallel shard are tp_size apart, e.g. tp4 dp2 -> [0, 4]. Using
    list(range(dp)) instead picks the first dp TENSOR-parallel ranks, which share a node."""
    from syssim.training.runner import dp_group_ranks
    assert dp_group_ranks(4, 2) == [0, 4]
    assert dp_group_ranks(4, 4) == [0, 4, 8, 12]
    assert dp_group_ranks(2, 4) == [0, 2, 4, 6]


def test_dp_group_ranks_tp1_is_contiguous():
    """tp=1 (e.g. single-node pure-DP): strided-by-1 == contiguous, unchanged from list(range(dp))."""
    from syssim.training.runner import dp_group_ranks
    assert dp_group_ranks(1, 4) == [0, 1, 2, 3]
    assert dp_group_ranks(1, 1) == [0]


def test_dp_allreduce_multinode_routes_internode_not_intranode():
    """On a 2-node topology (4 GPUs/node), the tp4 dp2 DP group must span nodes, so the DP
    all-reduce is timed over the SLOW inter-node fabric. list(range(dp))=[0,1] would (wrongly)
    pick two same-node ranks and time it over fast intra-node NVLink."""
    from syssim.training.runner import _estimate_dp_allreduce_ms, dp_group_ranks
    from syssim.network import build_dimensional
    topo = build_dimensional(
        dims=["fully_connected", "switch"], size=[4, 2],
        bandwidth=[450.0, 25.0], latency_ns=[12000.0, 3000.0], gpus_per_node=4,
    )
    intra = _estimate_dp_allreduce_ms(int(6e9), [0, 1], topo)               # same node (the bug)
    inter = _estimate_dp_allreduce_ms(int(6e9), dp_group_ranks(4, 2), topo)  # [0,4], spans nodes
    assert topo.gpus[0].node_id != topo.gpus[4].node_id, "ranks 0 and 4 should be on different nodes"
    assert inter > 3 * intra, (intra, inter)


def test_collective_dispatch_all_modeled_collectives():
    """Each modeled collective routes through the simulator and returns positive time."""
    topology = _topology()
    for name in ("all_reduce", "all_gather", "reduce_scatter", "broadcast",
                 "all_gather_into_tensor", "reduce_scatter_tensor"):
        ms = _estimate_collective_ms(
            collective_name=name, total_bytes=int(1e9),
            ranks=[0, 1, 2, 3], topology=topology,
        )
        assert ms > 0, f"{name} returned {ms}"
        assert ms < 20, f"{name} returned {ms} ms"


def test_collective_dispatch_unknown_returns_zero():
    """Unknown collective names (barrier, etc.) return 0."""
    assert _estimate_collective_ms(
        collective_name="barrier", total_bytes=int(1e9),
        ranks=[0, 1, 2, 3], topology=_topology(),
    ) == 0.0


def test_p2p_time_routes_through_simulator():
    """p2p_time_ms simulates a single Op against the topology."""
    from syssim.training.pipeline import p2p_time_ms
    topology = _topology()
    # 100 MB intra-node -> ~0.111 ms on 900 GB/s NVLink
    t = p2p_time_ms(bytes_=100_000_000, src_rank=0, dst_rank=1, topology=topology)
    assert 0.10 < t < 0.13, t


def test_captured_tp_collectives_get_times_filled_in():
    """A captured TP all_reduce with estimated_time_ms=0 contributes to
    collective_total_ms in the simulation report, proving the fill-in pass ran.
    """
    from syssim.training.runner import _simulate_on_hardware
    from syssim.training.trace import Trace
    from syssim.training.spec import (
        ModelConfig, ParallelismConfig, TrainingConfig, HardwareConfig,
    )
    from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType

    g = OperatorGraph(name="manual")
    g.add_operator(OperatorNode(
        name="op_0_gemm", op_type=OperatorType.GEMM,
        estimated_time_ms=1.0, stream_id=0,
    ))
    g.add_operator(OperatorNode(
        name="collective_1_all_reduce",
        op_type=OperatorType.COLLECTIVE,
        config={"collective": "all_reduce", "bytes": int(1e8),
                "group_ranks": [0, 1, 2, 3]},
        predecessors=["op_0_gemm"], stream_id=1,
        # No estimated_time_ms passed -> defaults to 0.0
    ))

    t = Trace(
        graph=g,
        model=ModelConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, num_query_groups=4,
            ffn_hidden_size=128, seq_length=8, max_position_embeddings=8, vocab_size=64,
        ),
        parallelism=ParallelismConfig(tp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
        gpus_per_node=1,
    )
    hw = HardwareConfig(
        peak_tflops_mm=1979.0, peak_tflops_math=989.0,
        peak_memory_bandwidth_GBps=3350.0, gpus_per_node=1,
        topology={"dims": ["switch"], "size": [4], "bandwidth": [200.0], "latency": [1000.0]},
    )
    report = _simulate_on_hardware(t, hw)
    assert report.by_op_type_ms.get("collective", 0.0) > 0


def test_simulate_with_topology_runs_end_to_end(tmp_path):
    """A full simulate() run with HardwareConfig.topology set produces a finite makespan."""
    pytest.importorskip("megatron.core")
    from syssim.training.runner import simulate
    from syssim.training.spec import (
        ModelConfig, ParallelismConfig, TrainingConfig, HardwareConfig,
    )
    report = simulate(
        model=ModelConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, num_query_groups=4,
            ffn_hidden_size=128, seq_length=32, max_position_embeddings=32, vocab_size=128,
        ),
        hardware=HardwareConfig(
            peak_tflops_mm=1979, peak_tflops_math=989,
            peak_memory_bandwidth_GBps=3350, gpus_per_node=1,
            topology={"dims": ["fully_connected"], "size": [1],
                      "bandwidth": [450.0], "latency": [0.0]},
        ),
        parallelism=ParallelismConfig(tp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
        workdir=str(tmp_path),
    )
    assert report.step_time_ms > 0
    assert report.peak_memory_gb > 0


def test_simulate_without_topology_raises():
    """HardwareConfig without `topology` set is rejected by _simulate_on_hardware."""
    from syssim.training.runner import _simulate_on_hardware
    from syssim.training.trace import Trace
    from syssim.training.spec import (
        ModelConfig, ParallelismConfig, TrainingConfig, HardwareConfig,
    )
    from syssim.operator_graph import OperatorGraph

    t = Trace(
        graph=OperatorGraph(name="empty"),
        model=ModelConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, num_query_groups=4,
            ffn_hidden_size=128, seq_length=8, max_position_embeddings=8, vocab_size=64,
        ),
        parallelism=ParallelismConfig(tp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
        gpus_per_node=1,
    )
    hw_no_topology = HardwareConfig(
        peak_tflops_mm=1979.0, peak_tflops_math=989.0,
        peak_memory_bandwidth_GBps=3350.0, gpus_per_node=1,
        topology=None,
    )
    with pytest.raises(ValueError, match="topology"):
        _simulate_on_hardware(t, hw_no_topology)
