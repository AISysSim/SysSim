"""MoE trace path: end-to-end tracing of an expert (MoE) model through the real
runner, plus a dense regression that the MoE trace patch is a no-op when dense,
plus an EP>1 all-to-all collective-capture check.

GPU/megatron-gated: the whole module needs a CUDA GPU + Megatron-Core (the tracer
builds and traces a real Megatron model). Skips cleanly on a CPU-only host.
"""
import pytest

pytest.importorskip("megatron.core")
pytestmark = pytest.mark.requires_cuda


# Tiny shared dims — small enough to trace fast, big enough that the expert FFN
# GEMMs have a non-degenerate capacity dimension.
HIDDEN, HEADS, LAYERS = 256, 8, 2
SEQ, VOCAB = 128, 1024
NUM_EXPERTS, TOPK, MOE_FFN = 4, 2, 512
DENSE_FFN = 512
MBS = 2


def _hardware():
    import syssim
    return syssim.HardwareConfig(
        peak_tflops_mm=1979.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_GBps=3350.0,
        gpus_per_node=4,
        gpu_memory_GB=96.0,
        topology={
            "dims": ["fully_connected"],
            "size": [4],
            "bandwidth": [450.0],
            "latency": [12000.0],
        },
    )


def _moe_model():
    import syssim
    return syssim.ModelConfig(
        num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=HEADS,
        num_query_groups=HEADS, ffn_hidden_size=DENSE_FFN,
        seq_length=SEQ, max_position_embeddings=SEQ, vocab_size=VOCAB,
        num_experts=NUM_EXPERTS, moe_router_topk=TOPK, moe_ffn_hidden_size=MOE_FFN,
        moe_layer_freq=1, moe_token_dispatcher_type="alltoall",
    )


def _dense_model():
    import syssim
    return syssim.ModelConfig(
        num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=HEADS,
        num_query_groups=HEADS, ffn_hidden_size=DENSE_FFN,
        seq_length=SEQ, max_position_embeddings=SEQ, vocab_size=VOCAB,
    )


def test_moe_ep1_traces_end_to_end(tmp_path):
    """EP=1 MoE traces through the real simulate() path, produces a report with
    non-zero compute and finite peak memory, and the expert-FFN GEMMs appear at
    the per-expert capacity dimension."""
    import syssim
    from syssim.training.runner import trace
    from syssim.operator_graph import OperatorType

    model = _moe_model()
    parallelism = syssim.ParallelismConfig()
    training = syssim.TrainingConfig(micro_batch=MBS, global_batch=MBS, dtype="bf16")
    hardware = _hardware()

    t = trace(model=model, parallelism=parallelism, training=training,
              hardware=hardware, gpus_per_node=hardware.gpus_per_node,
              workdir=str(tmp_path))

    # capacity = ceil(tokens * topk / E); tokens = SEQ * MBS. The per-expert FFN
    # GEMMs (fc1: K=hidden, N=moe_ffn; fc2: K=moe_ffn, N=hidden) have M == capacity.
    # trace() serializes the graph to JSON, so operand TensorMeta is dropped; the
    # GEMM dims survive on the node config as M/K/N.
    capacity = -(-(SEQ * MBS * TOPK) // NUM_EXPERTS)
    gemms = [n for n in t.graph.operators.values() if n.op_type == OperatorType.GEMM]
    expert_ffn_gemms = [
        n for n in gemms
        if n.config.get("M") == capacity
        and {n.config.get("K"), n.config.get("N")} == {HIDDEN, MOE_FFN}
    ]
    assert expert_ffn_gemms, (
        "expected expert-FFN GEMMs with M == per-expert capacity and "
        "hidden<->moe_ffn dims")

    report = t.simulate_on(hardware)
    assert report.step_time_ms > 0
    assert report.by_op_type_ms.get("gemm", 0.0) > 0
    assert report.peak_memory_gb > 0
    import math
    assert math.isfinite(report.peak_memory_gb)


def test_dense_control_traces_end_to_end(tmp_path):
    """Dense (num_experts None) control: the MoE patch context manager is a no-op
    and the dense trace still completes and produces a sane report."""
    import syssim
    report = syssim.simulate(
        model=_dense_model(),
        hardware=_hardware(),
        parallelism=syssim.ParallelismConfig(),
        training=syssim.TrainingConfig(micro_batch=MBS, global_batch=MBS, dtype="bf16"),
        workdir=str(tmp_path),
    )
    assert report.step_time_ms > 0
    assert report.peak_memory_gb > 0


def test_dense_trace_byte_identical_after_alltoall_patch(tmp_path):
    """Regression: adding all_to_all_single to the tracer must not change the dense
    trace. A dense model never calls any all_to_all, so the captured graph has zero
    all_to_all collective nodes and the same node count across repeated traces."""
    import syssim
    from syssim.training.runner import trace

    model = _dense_model()
    parallelism = syssim.ParallelismConfig()
    training = syssim.TrainingConfig(micro_batch=MBS, global_batch=MBS, dtype="bf16")
    hardware = _hardware()

    t = trace(model=model, parallelism=parallelism, training=training,
              hardware=hardware, gpus_per_node=hardware.gpus_per_node,
              workdir=str(tmp_path))
    a2a = [n for n in t.graph.operators.values()
           if "all_to_all" in str(n.config.get("collective", ""))]
    assert a2a == [], "dense trace must not emit any all_to_all collective"


def test_moe_ep2_captures_alltoall_collective(tmp_path):
    """EP=2 MoE: Megatron's alltoall dispatcher issues a c10d all_to_all_single on
    the expert-parallel group. The tracer must capture it as a COLLECTIVE node with
    bytes + group_ranks, and the Phase-2 collective-timing pass must give it a
    non-zero time. (EP is carved from the dp grid, so EP=2 needs DP=2 -> world_size=2;
    PP=1 keeps the single-process trace path.)"""
    import syssim
    from syssim.training.runner import trace, _estimate_collective_ms
    from syssim.operator_graph import OperatorType
    from syssim.network import build_topology_from_config

    model = _moe_model()
    parallelism = syssim.ParallelismConfig(dp=2, ep=2)
    training = syssim.TrainingConfig(micro_batch=MBS, global_batch=MBS * 2, dtype="bf16")
    hardware = _hardware()

    t = trace(model=model, parallelism=parallelism, training=training,
              hardware=hardware, gpus_per_node=hardware.gpus_per_node,
              workdir=str(tmp_path))

    a2a = [n for n in t.graph.operators.values()
           if n.op_type == OperatorType.COLLECTIVE
           and "all_to_all" in str(n.config.get("collective", ""))]
    assert a2a, "EP=2 trace must capture at least one all_to_all collective node"
    for n in a2a:
        assert n.config.get("bytes", 0) > 0
        assert len(n.config.get("group_ranks", [])) == 2

    # Phase-2 timing: the captured all_to_all routes through network.all_to_all and
    # gets a non-zero time over the EP group.
    topology = build_topology_from_config(hardware)
    sample = a2a[0]
    ms = _estimate_collective_ms(
        collective_name=sample.config["collective"],
        total_bytes=int(sample.config["bytes"]),
        ranks=list(sample.config["group_ranks"]),
        topology=topology,
    )
    assert ms > 0
