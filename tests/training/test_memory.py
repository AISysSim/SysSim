import pytest
from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig
from syssim.training.memory import (
    count_parameters,
    param_bytes_per_rank,
    grad_bytes_per_rank,
    optimizer_state_bytes_per_rank,
)

QWEN3_1P7B = ModelConfig(
    num_layers=28, hidden_size=2048, num_attention_heads=16, num_query_groups=8,
    ffn_hidden_size=6144, seq_length=4096, max_position_embeddings=40960,
    vocab_size=151936, swiglu=True, rope=True, tie_word_embeddings=True,
)
QWEN3_8B = ModelConfig(
    num_layers=36, hidden_size=4096, num_attention_heads=32, num_query_groups=8,
    ffn_hidden_size=12288, seq_length=4096, max_position_embeddings=40960,
    vocab_size=151936, swiglu=True, rope=True, tie_word_embeddings=False,
)


def test_qwen3_1p7b_param_count_within_5pct():
    n = count_parameters(QWEN3_1P7B)
    assert 1.6e9 <= n <= 1.85e9, f"got {n/1e9:.3f}B params"


def test_qwen3_8b_param_count_within_5pct():
    n = count_parameters(QWEN3_8B)
    assert 7.7e9 <= n <= 8.6e9, f"got {n/1e9:.3f}B params"


def test_grad_bytes_equals_param_bytes_bf16():
    par = ParallelismConfig()
    tr = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    assert grad_bytes_per_rank(QWEN3_1P7B, par, tr) == param_bytes_per_rank(QWEN3_1P7B, par, tr)


def test_optim_state_adam_12_bytes_per_param():
    par = ParallelismConfig()
    tr = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    n = count_parameters(QWEN3_1P7B)
    assert optimizer_state_bytes_per_rank(QWEN3_1P7B, par, tr) == 12 * n


def test_param_bytes_shrinks_with_tp():
    par1 = ParallelismConfig(tp=1)
    par4 = ParallelismConfig(tp=4)
    tr = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    b1 = param_bytes_per_rank(QWEN3_8B, par1, tr)
    b4 = param_bytes_per_rank(QWEN3_8B, par4, tr)
    assert b4 < b1
    # TP=4 should bring per-rank weights to ~25% (with norms+embed replicated)
    assert b4 < b1 * 0.45


def test_activation_scales_with_seq_length():
    from syssim.training.memory import activation_bytes_per_rank
    par = ParallelismConfig()
    tr  = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    short = ModelConfig(num_layers=8, hidden_size=1024, num_attention_heads=8,
                        num_query_groups=8, ffn_hidden_size=4096,
                        seq_length=512, max_position_embeddings=512, vocab_size=1024)
    long_ = ModelConfig(**{**short.__dict__, "seq_length": 2048})
    assert activation_bytes_per_rank(long_, par, tr) > activation_bytes_per_rank(short, par, tr) * 3


def test_activation_shrinks_with_sequence_parallel():
    from syssim.training.memory import activation_bytes_per_rank
    cfg = ModelConfig(num_layers=8, hidden_size=1024, num_attention_heads=8,
                      num_query_groups=8, ffn_hidden_size=4096,
                      seq_length=512, max_position_embeddings=512, vocab_size=1024)
    no_sp = ParallelismConfig(tp=4, sp=False)
    sp    = ParallelismConfig(tp=4, sp=True)
    tr    = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    assert activation_bytes_per_rank(cfg, sp, tr) < activation_bytes_per_rank(cfg, no_sp, tr)


def test_peak_memory_sums_components():
    from syssim.training.memory import (
        activation_bytes_per_rank, peak_memory_gb_per_rank, MemoryBreakdown,
    )
    par = ParallelismConfig()
    tr  = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    bd  = peak_memory_gb_per_rank(QWEN3_1P7B, par, tr)
    assert isinstance(bd, MemoryBreakdown)
    assert bd.peak_bytes == (bd.param_bytes + bd.grad_bytes
                             + bd.optimizer_state_bytes + bd.activation_bytes)
