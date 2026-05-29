import torch

from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType
from syssim.compute.predictor.roofline import roofline
from syssim.compute.predictor.router import Family
from syssim.compute.predictor import features as F

aten = torch.ops.aten
HW = HardwareInfo(peak_tflops_mm=1979.0, peak_tflops_math=989.0,
                  peak_memory_bandwidth_gbps=3350.0)


def test_schema_version_present():
    assert isinstance(F.SCHEMA_VERSION, str) and F.SCHEMA_VERSION


def test_gemm_feature_row_has_universal_and_gemm_columns():
    a = torch.empty(2048, 4096, dtype=torch.bfloat16, device="cpu")
    b = torch.empty(4096, 1024, dtype=torch.bfloat16, device="cpu")
    out = torch.empty(2048, 1024, dtype=torch.bfloat16, device="cpu")
    bound = roofline(aten.mm, (a, b), {}, out, HW, OperatorType.GEMM)
    row = F.featurize(aten.mm, (a, b), {}, out, Family.GEMM, bound)
    for col in ("log_tensor_ns", "log_mem_ns", "log_anchor_ns",
                "arithmetic_intensity", "dtype", "M", "N", "K",
                "M_mod_128", "N_mod_128", "K_mod_128"):
        assert col in row, col
    assert row["M"] == 2048 and row["K"] == 4096 and row["N"] == 1024
    assert row["dtype"] == "bf16"
    assert row["M_mod_128"] == 0
