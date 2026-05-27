import pytest
import torch
import torch.distributed as dist
from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig


@pytest.fixture(autouse=True)
def _local_dist():
    import os, socket
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


def test_build_gpt_model_on_meta(_local_dist):
    pytest.importorskip("megatron.core")
    from megatron.core import parallel_state
    from syssim.training.runner import _build_gpt_model_on_meta
    from syssim.training.sources import resolve_megatron_provider
    m = ModelConfig(num_layers=2, hidden_size=64, num_attention_heads=4, num_query_groups=4,
                    ffn_hidden_size=128, seq_length=128, max_position_embeddings=128, vocab_size=256)
    par = ParallelismConfig(tp=1, dp=1)
    tr  = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    provider = resolve_megatron_provider(m, par, tr)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            create_gloo_process_groups=False,
        )
    try:
        model = _build_gpt_model_on_meta(provider, max_sequence_length=128, vocab_size=256)
        assert next(model.parameters()).is_meta
    finally:
        parallel_state.destroy_model_parallel()


def test_lm_forward_step_returns_scalar_loss():
    pytest.importorskip("megatron.core")
    from syssim.training.runner import make_lm_forward_step, make_lm_data_iterator
    import torch.nn as nn

    class TinyLM(nn.Module):
        def __init__(self, vocab=256, h=8):
            super().__init__()
            self.emb = nn.Embedding(vocab, h)
            self.proj = nn.Linear(h, vocab)
        def forward(self, input_ids, position_ids, attention_mask):
            return self.proj(self.emb(input_ids))

    model = TinyLM().cuda()
    forward_step = make_lm_forward_step(vocab_size=256)
    it = make_lm_data_iterator(vocab_size=256, micro_batch_size=2, seq_length=8)
    logits, loss_func = forward_step(it, model)
    assert logits.ndim == 3  # (batch, seq, vocab)
    loss, info = loss_func(logits)
    assert loss.ndim == 0
    assert "lm_loss" in info


def test_simulate_tp1_dp1_end_to_end(tmp_path):
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
            topology={"type": "two_layer_multipath", "num_racks": 1, "nodes_per_rack": 1, "num_spines": 1,
                      "intra_node_bandwidth_GBps": 900.0,
                      "per_gpu_bandwidth_GBps": 200.0, "uplink_bandwidth_GBps": 200.0},
        ),
        parallelism=ParallelismConfig(tp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
        workdir=str(tmp_path),
    )
    assert report.step_time_ms > 0
    assert report.peak_memory_gb > 0
    # Phase breakdown must attribute real compute to forward and backward
    # (not dump it all into "unknown").
    assert report.forward_ms > 0
    assert report.backward_ms > 0


def test_estimate_memory_returns_quickly():
    import time
    from syssim.training.runner import estimate_memory
    from syssim.training.spec import (
        ModelConfig, ParallelismConfig, TrainingConfig, HardwareConfig,
    )
    m = ModelConfig(num_layers=4, hidden_size=128, num_attention_heads=8, num_query_groups=8,
                    ffn_hidden_size=512, seq_length=128, max_position_embeddings=128, vocab_size=256)
    hw = HardwareConfig(peak_tflops_mm=1979, peak_tflops_math=989,
                        peak_memory_bandwidth_GBps=3350, gpus_per_node=1)
    start = time.time()
    mem = estimate_memory(model=m, hardware=hw,
                          parallelism=ParallelismConfig(),
                          training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"))
    elapsed = time.time() - start
    assert elapsed < 1.0  # no tracing — instant
    assert mem.peak_gb > 0
