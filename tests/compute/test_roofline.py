import torch

from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType
from syssim.compute.predictor.roofline import roofline

aten = torch.ops.aten
HW = HardwareInfo(peak_tflops_mm=1979.0, peak_tflops_math=989.0,
                  peak_memory_bandwidth_gbps=3350.0)


def _mm(m, k, n, dtype=torch.bfloat16):
    a = torch.empty(m, k, dtype=dtype, device="cpu")
    b = torch.empty(k, n, dtype=dtype, device="cpu")
    out = torch.empty(m, n, dtype=dtype, device="cpu")
    return aten.mm, (a, b), {}, out


def test_large_gemm_binds_on_tensor_not_memory():
    fp, args, kw, out = _mm(4096, 4096, 4096)
    b = roofline(fp, args, kw, out, HW, OperatorType.GEMM)
    assert b.tensor_ns > 0
    assert b.sfu_ns == 0.0          # GEMM emits no transcendentals (MVP)
    assert b.roofline_ns == max(b.tensor_ns, b.fma_ns, b.sfu_ns, b.mem_ns, b.launch_ns)
    assert b.roofline_ns == b.tensor_ns  # large GEMM is compute-bound


def test_launch_floor_dominates_tiny_op():
    fp, args, kw, out = _mm(1, 1, 1)
    b = roofline(fp, args, kw, out, HW, OperatorType.GEMM, t_launch_ns=5000.0)
    assert b.roofline_ns == 5000.0       # launch floor wins for a tiny op
    # the bare pipeline demand (no launch) is below the floor
    assert max(b.tensor_ns, b.fma_ns, b.sfu_ns, b.mem_ns) < 5000.0


def test_ignore_op_returns_zero_bound():
    a = torch.empty(8, 8, device="cpu")
    b = roofline(aten.view, (a, [64]), {}, a, HW, OperatorType.MATH)
    assert b.roofline_ns == 0.0


def test_softmax_populates_sfu_term():
    # With the instruction-mix wired, softmax's SFU (transcendental) demand is live.
    x = torch.empty(4096, 4096, dtype=torch.float32, device="cpu")
    b = roofline(aten._softmax, (x, -1, False), {}, x, HW, OperatorType.MATH)
    assert b.sfu_ns > 0.0
    assert b.fma_ns >= 0.0
    assert b.roofline_ns == max(b.tensor_ns, b.fma_ns, b.sfu_ns, b.mem_ns, b.launch_ns)
