def test_per_pp_stage_memory_first_stage_larger():
    """First PP stage has embedding params; last stage has lm_head when untied; middle stages have neither."""
    from syssim.training.memory import per_pp_stage_peak_memory_gb
    from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig

    m = ModelConfig(
        num_layers=8, hidden_size=4096, num_attention_heads=32, num_query_groups=8,
        ffn_hidden_size=14336, seq_length=2048, max_position_embeddings=2048,
        vocab_size=128_000, tie_word_embeddings=False,
    )
    p = ParallelismConfig(pp=4)
    tr = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    per_stage = per_pp_stage_peak_memory_gb(m, p, tr)

    assert len(per_stage) == 4
    # First stage (embedding) > middle stages
    assert per_stage[0] > per_stage[1]
    # Last stage (lm_head, untied) > middle stages
    assert per_stage[-1] > per_stage[1]
    # Middle two stages are equal
    assert abs(per_stage[1] - per_stage[2]) < 1e-9


def test_per_pp_stage_memory_pp1():
    """PP=1 falls back to a one-element list equal to the existing peak."""
    from syssim.training.memory import per_pp_stage_peak_memory_gb, peak_memory_gb_per_rank
    from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig

    m = ModelConfig(
        num_layers=4, hidden_size=1024, num_attention_heads=16, num_query_groups=8,
        ffn_hidden_size=4096, seq_length=512, max_position_embeddings=512, vocab_size=32_000,
    )
    p = ParallelismConfig()
    tr = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    per_stage = per_pp_stage_peak_memory_gb(m, p, tr)
    full = peak_memory_gb_per_rank(m, p, tr)
    assert len(per_stage) == 1
    assert abs(per_stage[0] - full.peak_gb) < 1e-6
