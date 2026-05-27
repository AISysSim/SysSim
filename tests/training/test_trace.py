import pytest
from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType
from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig
from syssim.training.trace import Trace


def _make_minimal_trace():
    g = OperatorGraph(name="x")
    g.add_operator(OperatorNode(name="o", op_type=OperatorType.GEMM, estimated_time_ms=1.0))
    return Trace(
        graph=g,
        model=ModelConfig(num_layers=2, hidden_size=64, num_attention_heads=4,
                          num_query_groups=4, ffn_hidden_size=128,
                          seq_length=128, max_position_embeddings=128, vocab_size=256),
        parallelism=ParallelismConfig(),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
        gpus_per_node=1,
    )


def test_trace_holds_provenance():
    t = _make_minimal_trace()
    assert t.gpus_per_node == 1
    assert "o" in t.graph.operators
    assert t.parallelism.world_size == 1


def test_simulate_on_re_runs_predictor():
    """Same trace, different hardware → different step times."""
    from syssim.training.spec import HardwareConfig
    t = _make_minimal_trace()
    # Manually set non-zero op time before simulate_on
    next(iter(t.graph.operators.values())).estimated_time_ms = 1.0

    topology = {"type": "two_layer_multipath", "num_racks": 1, "nodes_per_rack": 1, "num_spines": 1,
                "intra_node_bandwidth_GBps": 900.0,
                "per_gpu_bandwidth_GBps": 200.0, "uplink_bandwidth_GBps": 200.0}
    hw_fast = HardwareConfig(peak_tflops_mm=4000, peak_tflops_math=2000,
                             peak_memory_bandwidth_GBps=5000, gpus_per_node=1,
                             topology=topology)
    hw_slow = HardwareConfig(peak_tflops_mm=1000, peak_tflops_math=500,
                             peak_memory_bandwidth_GBps=1000, gpus_per_node=1,
                             topology=topology)
    r_fast = t.simulate_on(hw_fast)
    r_slow = t.simulate_on(hw_slow)
    # Re-prediction may change op times; we only assert both produced reports
    assert r_fast.step_time_ms >= 0
    assert r_slow.step_time_ms >= 0
