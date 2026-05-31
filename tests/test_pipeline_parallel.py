"""End-to-end PP tests. These call `simulate(...)` with pp > 1."""

import pytest
from syssim.training import simulate, ParallelismConfig, TrainingConfig


def test_pp2_smoke():
    """PP=2 with a small model produces a finite report with both stages represented."""
    report = simulate(
        model="examples/configs/models/qwen3-1_7b.yaml",
        hardware="examples/configs/hardware/isambard_gh200_4gpu.yaml",
        parallelism=ParallelismConfig(pp=2),
        training=TrainingConfig(micro_batch=1, global_batch=2, dtype="bf16"),
    )
    assert report.step_time_ms > 0
    assert len(report.pp_stage_memory_gb) == 2
    assert len(report.per_pp_rank_step_time_ms) == 2


def test_pp2_collective_includes_p2p():
    """PP=2 trace has non-zero collective time (P2P transfers)."""
    report = simulate(
        model="examples/configs/models/qwen3-1_7b.yaml",
        hardware="examples/configs/hardware/isambard_gh200_4gpu.yaml",
        parallelism=ParallelismConfig(pp=2),
        training=TrainingConfig(micro_batch=1, global_batch=2, dtype="bf16"),
    )
    assert report.collective_total_ms > 0


def test_pp1_regression():
    """PP=1 still matches the single-process SPMD fixture after PP-aware changes."""
    report_pp1 = simulate(
        model="examples/configs/models/qwen3-1_7b.yaml",
        hardware="examples/configs/hardware/isambard_gh200_4gpu.yaml",
        parallelism=ParallelismConfig(),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
    )
    assert len(report_pp1.pp_stage_memory_gb) == 1
    assert report_pp1.peak_memory_gb == report_pp1.pp_stage_memory_gb[0]
    assert report_pp1.per_pp_rank_step_time_ms == [report_pp1.step_time_ms]


@pytest.mark.slow
def test_pp8_llama_70b_per_stage_memory_distribution():
    """PP=8 large-model fixture: first/last stages have heavier memory than middle stages."""
    from syssim.training import estimate_memory
    breakdown = estimate_memory(
        model="examples/configs/models/qwen3-8b_hf.yaml",
        hardware="examples/configs/hardware/isambard_gh200_2node.yaml",
        parallelism=ParallelismConfig(pp=8),
        training=TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    )
    per_stage = breakdown.pp_stage_memory_gb
    assert len(per_stage) == 8
    middle_mean = sum(per_stage[1:-1]) / 6
    assert per_stage[0] > middle_mean
    assert per_stage[-1] > middle_mean
