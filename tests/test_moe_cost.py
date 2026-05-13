import pytest
import torch

from syssim import HardwareInfo
from syssim.compute.moe_cost import (
    dtype_nbytes,
    estimate_combine_ms,
    estimate_dispatch_ms,
    estimate_expert_ffn_ms,
    estimate_memory_ms,
    estimate_router_ms,
)


@pytest.fixture
def hw():
    return HardwareInfo(
        peak_tflops_mm=989.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )


class TestMoECostHelpers:
    def test_dtype_nbytes_common_types(self):
        assert dtype_nbytes(torch.float16) == 2
        assert dtype_nbytes(torch.bfloat16) == 2
        assert dtype_nbytes(torch.float32) == 4
        assert dtype_nbytes("nvfp4") == 1

    def test_router_time_positive(self, hw):
        assert estimate_router_ms(128, 2048, 128, 8, hw, torch.bfloat16) > 0.0

    def test_expert_time_positive(self, hw):
        assert estimate_expert_ffn_ms(128, 2048, 768, hw, torch.bfloat16) > 0.0

    def test_expert_time_monotonic_with_tokens(self, hw):
        small = estimate_expert_ffn_ms(32, 2048, 768, hw, torch.bfloat16)
        large = estimate_expert_ffn_ms(64, 2048, 768, hw, torch.bfloat16)
        assert large > small

    def test_zero_work_returns_zero(self, hw):
        assert estimate_router_ms(0, 2048, 128, 8, hw, torch.bfloat16) == 0.0
        assert estimate_dispatch_ms(0, 2048, hw, torch.bfloat16) == 0.0
        assert estimate_combine_ms(0, 2048, hw, torch.bfloat16) == 0.0
        assert estimate_memory_ms(0, hw) == 0.0

    def test_negative_values_are_rejected(self, hw):
        with pytest.raises(ValueError, match="num_assignments"):
            estimate_dispatch_ms(-1, 2048, hw, torch.bfloat16)
