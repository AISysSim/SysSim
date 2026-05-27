"""PLENA wired as a custom Estimator on the Megatron tracer path.

Requires the PLENA_Simulator submodule (third_party/PLENA_Simulator). Skips
cleanly when it is not checked out (`git submodule update --init`).
"""

import os
import pathlib
import socket
import pytest
import torch.distributed as dist

_PLENA_SETTINGS = (
    pathlib.Path(__file__).resolve().parents[3]
    / "third_party" / "PLENA_Simulator" / "plena_settings.toml"
)
pytestmark = pytest.mark.skipif(
    not _PLENA_SETTINGS.exists(),
    reason="PLENA_Simulator submodule not checked out (git submodule update --init)",
)


@pytest.fixture(autouse=True)
def _local_dist():
    if not dist.is_initialized():
        s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = str(port)
        os.environ["WORLD_SIZE"] = "1"; os.environ["RANK"] = "0"
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def test_simulate_with_plena_estimator(tmp_path):
    pytest.importorskip("megatron.core")
    from syssim.training.runner import simulate
    from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig, HardwareConfig
    from syssim.external.plena import PLENAEstimator, PLENAConfig

    est = PLENAEstimator(PLENAConfig.from_plena_submodule())
    hw = HardwareConfig(
        peak_tflops_mm=1979, peak_tflops_math=989, peak_memory_bandwidth_GBps=3350,
        gpus_per_node=1, gpu_memory_GB=80,
        topology={"type": "simple", "num_nodes": 1,
                  "intra_node_bandwidth_GBps": 900, "inter_node_bandwidth_GBps": 200},
        estimator=est,
    )
    r = simulate(
        model=ModelConfig(num_layers=2, hidden_size=128, num_attention_heads=8,
                          num_query_groups=8, ffn_hidden_size=256, seq_length=64,
                          max_position_embeddings=64, vocab_size=256),
        hardware=hw,
        parallelism=ParallelismConfig(tp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
        workdir=str(tmp_path),
    )
    # PLENA-estimated op times flow through the unchanged tracer/simulator.
    assert r.step_time_ms > 0
