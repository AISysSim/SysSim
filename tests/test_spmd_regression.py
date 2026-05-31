"""Pin single-process SPMD trace outputs so future changes can't silently regress."""

from syssim.training import simulate, ParallelismConfig, TrainingConfig


def test_tp2_dp4_smoke_does_not_crash():
    """TP=2 DP=4 (world=8) traces in single process and produces a finite report.

    world=8 on the GH200 (4 GPUs/node) spans 2 nodes — the 2-node Isambard fixture.
    """
    report = simulate(
        model="examples/configs/models/qwen3-1_7b.yaml",
        hardware="examples/configs/hardware/isambard_gh200_2node.yaml",
        parallelism=ParallelismConfig(tp=2, dp=4),
        training=TrainingConfig(micro_batch=1, global_batch=4, dtype="bf16"),
    )
    assert report.step_time_ms > 0
    assert report.peak_memory_gb > 0
    # collective work should be non-zero with TP>1
    assert report.collective_total_ms > 0


def test_tp1_dp1_regression_matches_known_fixture():
    """PP=1 TP=1 DP=1 — most basic config, must produce identical output before and after."""
    report = simulate(
        model="examples/configs/models/qwen3-1_7b.yaml",
        hardware="examples/configs/hardware/isambard_gh200_4gpu.yaml",
        parallelism=ParallelismConfig(),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
    )
    assert report.step_time_ms > 0
    assert report.collective_total_ms == 0  # no collectives at TP=DP=1
    assert report.optimizer_ms > 0
