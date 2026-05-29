import torch

from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType
from syssim.compute.predictor.roofline import roofline
from syssim.compute.predictor.router import Family
from syssim.compute.predictor import features as F

aten = torch.ops.aten
HW = HardwareInfo(peak_tflops_mm=1979.0, peak_tflops_math=989.0,
                  peak_memory_bandwidth_gbps=3350.0)


def _row(func, args, kwargs, out, family):
    op_type = OperatorType.ATTN if family is Family.ATTENTION else OperatorType.MATH
    b = roofline(func, args, kwargs, out, HW, op_type)
    return F.featurize(func, args, kwargs, out, family, b)


def test_attention_features():
    q = torch.empty(2, 16, 1024, 64, device="cpu")    # B, H_q, S_q, D
    k = torch.empty(2, 8, 1024, 64, device="cpu")      # H_kv=8 -> GQA ratio 2
    v = torch.empty(2, 8, 1024, 64, device="cpu")
    out = torch.empty(2, 16, 1024, 64, device="cpu")
    row = _row(aten._scaled_dot_product_flash_attention, (q, k, v, 0.0, True), {}, out,
               Family.ATTENTION)
    for col in ("B", "H_q", "H_kv", "gqa_ratio", "S_q", "S_kv", "D_head", "is_causal",
                "variant", "S_q_mod_128"):
        assert col in row, col
    assert row["B"] == 2 and row["H_q"] == 16 and row["H_kv"] == 8
    assert row["gqa_ratio"] == 2.0 and row["S_q"] == 1024 and row["D_head"] == 64
    assert row["is_causal"] == 1
    assert row["S_q_mod_128"] == 0


def test_normalization_features():
    x = torch.empty(4096, 2048, device="cpu")
    w = torch.empty(2048, device="cpu"); bias = torch.empty(2048, device="cpu")
    out = torch.empty(4096, 2048, device="cpu")
    row = _row(aten.native_layer_norm, (x, [2048], w, bias, 1e-5), {}, out,
               Family.NORMALIZATION)
    for col in ("outer_dims_product", "norm_dim", "op_subtype", "has_weight", "has_bias"):
        assert col in row, col
    assert row["norm_dim"] == 2048 and row["outer_dims_product"] == 4096
    assert row["has_weight"] == 1 and row["has_bias"] == 1


def test_elementwise_features():
    x = torch.empty(4096, 2048, device="cpu")
    out = torch.empty(4096, 2048, device="cpu")
    row = _row(aten.gelu, (x,), {}, out, Family.ELEMENTWISE)
    for col in ("total_elements", "num_operands", "op_subtype"):
        assert col in row, col
    assert row["total_elements"] == 4096 * 2048
    assert row["num_operands"] == 1
    assert row["op_subtype"] == "gelu"


def test_reduction_features():
    x = torch.empty(2, 16, 1024, 1024, device="cpu")   # attention-scores-like
    out = torch.empty(2, 16, 1024, 1024, device="cpu")
    row = _row(aten._softmax, (x, -1, False), {}, out, Family.REDUCTION)
    for col in ("input_volume", "reduced_axis_size", "num_non_reduced_elements", "op_subtype"):
        assert col in row, col
    assert row["input_volume"] == x.numel()
    assert row["reduced_axis_size"] == 1024            # dim=-1 size
    assert row["num_non_reduced_elements"] == x.numel() // 1024


def test_feature_columns_per_family_distinct():
    for fam in (Family.ATTENTION, Family.NORMALIZATION, Family.ELEMENTWISE, Family.REDUCTION):
        cols = F.feature_columns(fam)
        assert "log_anchor_ns" in cols                 # universal present
        assert len(cols) > 11                          # has family-specific cols
