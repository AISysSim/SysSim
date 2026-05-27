import pytest
from pathlib import Path

pytest.importorskip("megatron.core")

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_qwen3_1_7b_single_h100(tmp_path):
    import syssim
    report = syssim.simulate(
        model=str(REPO_ROOT / "examples/configs/models/qwen3-1_7b.yaml"),
        hardware=str(REPO_ROOT / "examples/configs/hardware/single_h100.yaml"),
        training=syssim.TrainingConfig(micro_batch=4, global_batch=4, dtype="bf16"),
        workdir=str(tmp_path),
    )
    assert report.step_time_ms > 0
    assert report.peak_memory_gb > 0
    assert report.mfu >= 0


@pytest.mark.slow
def test_qwen3_8b_tp4_dp2_dgx_h100(tmp_path):
    import syssim
    report = syssim.simulate(
        model=str(REPO_ROOT / "examples/configs/models/qwen3-8b.yaml"),
        hardware=str(REPO_ROOT / "examples/configs/hardware/dgx_h100.yaml"),
        parallelism=syssim.ParallelismConfig(tp=4, dp=2, sp=True),
        training=syssim.TrainingConfig(
            micro_batch=1, global_batch=16, dtype="bf16", recompute="selective",
        ),
        workdir=str(tmp_path),
    )
    assert report.step_time_ms > 0
    assert report.peak_memory_gb > 1.0


@pytest.mark.slow
def test_qwen3_8b_hf_matches_yaml_within_2pct(tmp_path):
    pytest.importorskip("megatron.bridge")
    import syssim
    common = dict(
        hardware=str(REPO_ROOT / "examples/configs/hardware/dgx_h100.yaml"),
        parallelism=syssim.ParallelismConfig(tp=4, dp=2, sp=True),
        training=syssim.TrainingConfig(
            micro_batch=1, global_batch=16, dtype="bf16", recompute="selective",
        ),
        workdir=str(tmp_path),
    )
    yaml_report = syssim.simulate(
        model=str(REPO_ROOT / "examples/configs/models/qwen3-8b.yaml"), **common,
    )
    hf_report = syssim.simulate(
        model=str(REPO_ROOT / "examples/configs/models/qwen3-8b_hf.yaml"), **common,
    )
    delta = abs(hf_report.step_time_ms - yaml_report.step_time_ms) / yaml_report.step_time_ms
    assert delta < 0.02, f"HF vs YAML step time delta = {delta*100:.2f}%"
