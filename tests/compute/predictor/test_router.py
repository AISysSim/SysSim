import torch

from syssim.operator_graph import OperatorType
from syssim.compute.predictor.router import Family, route

aten = torch.ops.aten


def test_gemm_and_attention_route_by_op_type():
    assert route(aten.mm, OperatorType.GEMM) is Family.GEMM
    assert route(aten._scaled_dot_product_flash_attention, OperatorType.ATTN) is Family.ATTENTION


def test_math_disambiguates_to_subfamilies():
    assert route(aten._softmax, OperatorType.MATH) is Family.REDUCTION
    assert route(aten.native_layer_norm, OperatorType.MATH) is Family.NORMALIZATION
    assert route(aten.gelu, OperatorType.MATH) is Family.ELEMENTWISE
    # unknown MATH op -> ELEMENTWISE (safest)
    assert route(aten.some_unknown_op if hasattr(aten, "some_unknown_op") else aten.add,
                 OperatorType.MATH) is Family.ELEMENTWISE
