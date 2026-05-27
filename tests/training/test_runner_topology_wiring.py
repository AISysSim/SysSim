"""Verify that HardwareConfig.topology routes all collectives through the network simulator."""

import pytest

from syssim.training.runner import _estimate_dp_allreduce_ms, _estimate_collective_ms
from syssim.network.topology import build_two_layer_multipath


def _hw(*, gpus_per_node=8):
    from syssim.training.spec import HardwareConfig
    return HardwareConfig(
        peak_tflops_mm=1979.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_GBps=3350.0,
        gpus_per_node=gpus_per_node,
        topology={"type": "two_layer_multipath", "num_racks": 1, "nodes_per_rack": 1, "num_spines": 1,
                  "intra_node_bandwidth_GBps": 900.0,
                  "per_gpu_bandwidth_GBps": 200.0, "uplink_bandwidth_GBps": 200.0},
    )


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


def test_dp_allreduce_returns_zero_when_p_le_1():
    """Single-rank or zero-byte DP allreduce returns 0 immediately."""
    topology = _topology()
    assert _estimate_dp_allreduce_ms(int(1e9), [0], topology) == 0.0
    assert _estimate_dp_allreduce_ms(0, [0, 1, 2, 3], topology) == 0.0


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
        topology={"type": "two_layer_multipath", "num_racks": 4, "nodes_per_rack": 1, "num_spines": 1,
                  "intra_node_bandwidth_GBps": 900.0,
                  "per_gpu_bandwidth_GBps": 200.0, "uplink_bandwidth_GBps": 200.0},
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
            topology={
                "type": "two_layer_multipath",
                "num_racks": 1, "nodes_per_rack": 1, "num_spines": 1,
                "intra_node_bandwidth_GBps": 900,
                "per_gpu_bandwidth_GBps": 25.0,
                "uplink_bandwidth_GBps": 200.0,
            },
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
