import pytest
import torch

from syssim.profiling.measure import median_of_reps, measure_gemm


def test_median_of_reps_discards_warmup():
    # 2 warmup (large) + 5 real (around 10) -> median ~10, warmup ignored
    samples = [100.0, 100.0, 9.0, 10.0, 11.0, 10.5, 9.5]
    assert median_of_reps(samples, warmup=2) == pytest.approx(10.0, abs=0.6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_measure_gemm_returns_positive_latency():
    rec = measure_gemm(M=512, K=512, N=512, dtype="bf16", reps=5, warmup=2)
    assert rec["latency_ns"] > 0
    assert rec["M"] == 512
