"""Integration tests: MemTracker-driven PP/SPMD memory estimate + bottlenecks."""

import os
import socket
import pytest
import torch.distributed as dist


@pytest.fixture(autouse=True)
def _local_dist():
    if not dist.is_initialized():
        s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = str(port)
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _model():
    from syssim.training.spec import ModelConfig
    return ModelConfig(
        num_layers=4, hidden_size=128, num_attention_heads=8, num_query_groups=8,
        ffn_hidden_size=256, seq_length=64, max_position_embeddings=64, vocab_size=256,
    )


def _hw(gpu_memory_GB=80, gpus_per_node=1):
    from syssim.training.spec import HardwareConfig
    return HardwareConfig(
        peak_tflops_mm=1979, peak_tflops_math=989, peak_memory_bandwidth_GBps=3350,
        gpus_per_node=gpus_per_node, gpu_memory_GB=gpu_memory_GB,
        inter_node_bandwidth_GBps=200,
        topology={"type": "simple", "num_nodes": 1,
                  "intra_node_bandwidth_GBps": 900, "inter_node_bandwidth_GBps": 200},
    )


def test_spmd_memory_estimate_populated(tmp_path):
    pytest.importorskip("megatron.core")
    from syssim.training.runner import simulate
    from syssim.training.spec import ParallelismConfig, TrainingConfig
    r = simulate(model=_model(), hardware=_hw(),
                 parallelism=ParallelismConfig(tp=1, dp=1),
                 training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
                 workdir=str(tmp_path))
    assert r.peak_memory_gb > 0
    assert r.bottlenecks is not None
    assert r.bottlenecks.binding_stage == 0
    assert r.bottlenecks.memory_by_type.get("Parameter", 0) > 0
    assert "Activation" in r.bottlenecks.memory_by_type
    assert len(r.bottlenecks.top_ops_by_time) >= 1
    assert r.bottlenecks.dominant_op_type != ""


def test_pp2_earlier_stage_holds_more_activation(tmp_path):
    pytest.importorskip("megatron.core")
    from syssim.training.runner import simulate
    from syssim.training.spec import ParallelismConfig, TrainingConfig
    # global_batch=4 -> 4 microbatches; pp=2 1F1B: stage0 in_flight=2, stage1 in_flight=1.
    # gpus_per_node=2 so world_size=2 (pp=2,tp=1,dp=1) fits one node (intra-node P2P).
    r = simulate(model=_model(),
                 hardware=_hw(gpus_per_node=2),
                 parallelism=ParallelismConfig(tp=1, dp=1, pp=2),
                 training=TrainingConfig(micro_batch=1, global_batch=4, dtype="bf16"),
                 workdir=str(tmp_path))
    assert len(r.pp_stage_memory_gb) == 2
    assert r.pp_stage_memory_gb[0] >= r.pp_stage_memory_gb[1]
    assert r.bottlenecks.binding_stage == 0


def test_oom_flagged_but_step_time_finite(tmp_path):
    pytest.importorskip("megatron.core")
    import math
    from syssim.training.runner import simulate
    from syssim.training.spec import ParallelismConfig, TrainingConfig
    r = simulate(model=_model(), hardware=_hw(gpu_memory_GB=0.001),
                 parallelism=ParallelismConfig(tp=1, dp=1),
                 training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
                 workdir=str(tmp_path))
    assert r.bottlenecks.oom is True
    assert r.bottlenecks.oom_excess_gb > 0
    assert r.step_time_ms > 0 and math.isfinite(r.step_time_ms)
