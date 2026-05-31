"""Test SimulationReport fields and per-stage breakdowns."""


def test_simulation_report_has_per_stage_fields_pp1():
    """PP=1 produces single-element per-stage lists, peak_memory_gb = max."""
    from syssim.training import simulate, ParallelismConfig, TrainingConfig
    report = simulate(
        model="examples/configs/models/qwen3-1_7b.yaml",
        hardware="examples/configs/hardware/isambard_gh200_4gpu.yaml",
        parallelism=ParallelismConfig(),
        training=TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
    )
    assert len(report.pp_stage_memory_gb) == 1
    assert report.peak_memory_gb == report.pp_stage_memory_gb[0]
    assert len(report.per_pp_rank_step_time_ms) == 1
    assert report.per_pp_rank_step_time_ms[0] == report.step_time_ms
