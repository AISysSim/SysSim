import pytest
pytest.importorskip("megatron.core")
from syssim.training.spec import (
    ModelConfig, ParallelismConfig, TrainingConfig, HardwareConfig,
)
from syssim.training.sweep import sweep


def test_sweep_over_tp(tmp_path):
    rows = sweep(
        model=ModelConfig(num_layers=2, hidden_size=64, num_attention_heads=4, num_query_groups=4,
                          ffn_hidden_size=128, seq_length=32, max_position_embeddings=32, vocab_size=128),
        hardware=HardwareConfig(
            peak_tflops_mm=1979, peak_tflops_math=989,
            peak_memory_bandwidth_GBps=3350, gpus_per_node=1,
            topology={"type": "two_layer_multipath", "num_racks": 1, "nodes_per_rack": 1, "num_spines": 1,
                      "intra_node_bandwidth_GBps": 900.0,
                      "per_gpu_bandwidth_GBps": 200.0, "uplink_bandwidth_GBps": 200.0},
        ),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
        over={"parallelism.tp": [1]},
        workdir=str(tmp_path),
    )
    df = rows.to_dataframe()
    assert len(df) == 1
    best = rows.best("step_time_ms")
    assert best is not None
