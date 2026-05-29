import pytest
import torch

from syssim.profiling.measure import (
    measure_attention, measure_norm, measure_elementwise, measure_reduction,
    measure_worklist,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_measure_attention():
    r = measure_attention(B=1, H_q=16, H_kv=8, S=512, D=64, causal=True,
                          dtype="bf16", reps=3, warmup=1)
    assert r["latency_ns"] > 0 and r["B"] == 1 and r["H_kv"] == 8 and r["family"] != ""


def test_measure_norm():
    r = measure_norm(tokens=4096, hidden=2048, op_subtype="rmsnorm",
                     dtype="bf16", reps=3, warmup=1)
    assert r["latency_ns"] > 0 and r["op_subtype"] == "rmsnorm"


def test_measure_elementwise():
    r = measure_elementwise(total_elements=4096 * 2048, op_subtype="gelu",
                            dtype="bf16", reps=3, warmup=1)
    assert r["latency_ns"] > 0


def test_measure_reduction():
    r = measure_reduction(B=1, H=16, S=512, dtype="bf16", reps=3, warmup=1)
    assert r["latency_ns"] > 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >=2 GPUs")
def test_measure_worklist_multigpu():
    items = [{"family": "elementwise", "total_elements": 1024 * 1024,
              "op_subtype": "gelu", "dtype": "bf16"} for _ in range(12)]
    rows = measure_worklist(items, num_workers=4)
    assert len(rows) == 12 and all(r["latency_ns"] > 0 for r in rows)
