"""Test suite for critical FLOP /2 bug fix.

Verifies that FLOP counts are NOT divided by 2 in the roofline model.
"""

import pytest
import torch

from syssim.compute.compute_cost_predictor import roofline_estimate
from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType

aten = torch.ops.aten


@pytest.fixture
def hw_info():
    """Standard GH200 hardware configuration."""
    return HardwareInfo(
        peak_tflops_mm=535.3,  # TFLOP/s
        peak_tflops_math=535.3,
        peak_memory_bandwidth_gbps=4022.8,  # GB/s
    )


class TestFlopBugFix:
    """Test that FLOP count is NOT divided by 2."""

    def test_gemm_flop_count_not_halved(self, hw_info):
        """GEMM should use full FLOP count (2×M×N×K), not halved."""
        M, N, K = 1024, 1024, 512
        a = torch.randn(M, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        out = torch.randn(M, N, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)

        # Find the math constraint
        math_constraint = None
        for c in result.constraints:
            if c.work_type == "math":
                math_constraint = c
                break

        assert math_constraint is not None, "Should have math constraint"

        # Verify FLOP count is 2×M×N×K (not divided by 2)
        expected_flops = 2 * M * N * K
        assert math_constraint.work_amount == expected_flops, (
            f"FLOP count should be {expected_flops}, got {math_constraint.work_amount}"
        )

        print(f"✓ FLOP count correct: {expected_flops} FLOPs")

    def test_compute_time_uses_correct_flops(self, hw_info):
        """Verify compute time is calculated from correct (non-halved) FLOPs."""
        M, N, K = 512, 512, 512
        a = torch.randn(M, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        out = torch.randn(M, N, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)

        math_constraint = next(c for c in result.constraints if c.work_type == "math")

        # Calculate expected compute time manually
        expected_flops = 2 * M * N * K
        peak_flops = hw_info.get_peak_tflops(OperatorType.GEMM, torch.float32) * 1e12
        expected_time_ns = (expected_flops / peak_flops) * 1e9
        expected_time_ms = expected_time_ns / 1e6

        # Should match (within floating point precision)
        assert abs(math_constraint.time_ms - expected_time_ms) < 1e-9, (
            f"Compute time should be {expected_time_ms:.9f}ms, got {math_constraint.time_ms:.9f}ms"
        )

        print(f"✓ Compute time correct: {math_constraint.time_ms:.9f}ms for {expected_flops} FLOPs")

    def test_small_matrix_correct_flops(self, hw_info):
        """Small matrices should also use correct FLOP count."""
        M, N, K = 8, 8, 8
        a = torch.randn(M, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        out = torch.randn(M, N, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)

        math_constraint = next(c for c in result.constraints if c.work_type == "math")
        expected_flops = 2 * M * N * K
        assert math_constraint.work_amount == expected_flops

        print(f"✓ Small matrix FLOP count correct: {expected_flops}")

    def test_different_dtypes_correct_flops(self, hw_info):
        """Different data types should use same FLOP count formula."""
        M, N, K = 256, 256, 256

        for dtype in [torch.float16, torch.float32]:
            a = torch.randn(M, K, dtype=dtype)
            b = torch.randn(K, N, dtype=dtype)
            out = torch.randn(M, N, dtype=dtype)

            result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)

            math_constraint = next(c for c in result.constraints if c.work_type == "math")
            expected_flops = 2 * M * N * K
            assert math_constraint.work_amount == expected_flops, f"FLOP count for {dtype} should be {expected_flops}"

        print("✓ All dtypes use correct FLOP count")

    def test_flash_attention_mixed_output_dtypes_has_compute_time(self, hw_info):
        """Flash attention should use input dtype when auxiliary outputs differ."""
        q = torch.empty((1, 4, 128, 64), device="meta", dtype=torch.bfloat16)
        k = torch.empty((1, 4, 128, 64), device="meta", dtype=torch.bfloat16)
        v = torch.empty((1, 4, 128, 64), device="meta", dtype=torch.bfloat16)
        out = (
            torch.empty_like(q),
            torch.empty((1, 4, 128), device="meta", dtype=torch.float32),
            torch.empty((2,), device="meta", dtype=torch.int64),
            torch.empty((), device="meta", dtype=torch.int64),
            torch.empty((0,), device="meta", dtype=torch.float32),
        )

        result = roofline_estimate(
            aten._scaled_dot_product_flash_attention,
            (q, k, v, 0.0, False, False),
            {},
            out,
            hw_info,
            OperatorType.ATTN,
        )

        math_constraint = next(c for c in result.constraints if c.work_type == "math")
        assert math_constraint.work_amount > 0
        assert math_constraint.time_ms > 0.0

    def test_roofline_time_doubled_after_fix(self, hw_info):
        """After fixing /2 bug, roofline compute time should be 2x what it was before."""
        M, N, K = 1024, 1024, 1024
        a = torch.randn(M, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        out = torch.randn(M, N, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)

        math_constraint = next(c for c in result.constraints if c.work_type == "math")

        # Compute what the BUGGY version would have produced
        buggy_flops = M * N * K  # Was divided by 2
        correct_flops = 2 * M * N * K

        # Verify we're using the correct FLOPs
        assert math_constraint.work_amount == correct_flops, (
            f"Should use {correct_flops} FLOPs (not buggy {buggy_flops})"
        )

        # Verify compute time matches expected
        peak_flops = hw_info.get_peak_tflops(OperatorType.GEMM, torch.float32) * 1e12
        expected_time_ms = (correct_flops / peak_flops) * 1e9 / 1e6

        assert abs(math_constraint.time_ms - expected_time_ms) < 1e-9, (
            f"Compute time should be {expected_time_ms:.9f}ms"
        )

        print(f"✓ Bug fixed: using {correct_flops} FLOPs (not {buggy_flops})")
        print(f"  Compute time: {math_constraint.time_ms:.6f}ms (2x faster than buggy version)")


class TestRooflineIsAnalyticalCeiling:
    """Verify roofline represents pure hardware limits."""

    def test_roofline_uses_peak_flops(self, hw_info):
        """Roofline should use peak FLOP/s without any derating."""
        M = 2048
        a = torch.randn(M, M, dtype=torch.float32)
        result = roofline_estimate(aten.mm, (a, a), {}, a, hw_info, OperatorType.GEMM)

        math_constraint = next(c for c in result.constraints if c.work_type == "math")

        # Capacity should be exactly peak FLOP/s
        expected_capacity = hw_info.get_peak_tflops(OperatorType.GEMM, torch.float32) * 1e12
        assert math_constraint.capacity == expected_capacity, f"Should use peak capacity {expected_capacity:.2e} FLOP/s"

        print(f"✓ Roofline uses peak FLOP/s: {expected_capacity / 1e12:.1f} TFLOP/s")

    def test_roofline_uses_peak_bandwidth(self, hw_info):
        """Roofline should use peak memory bandwidth without any derating."""
        M = 2048
        a = torch.randn(M, M, dtype=torch.float32)
        result = roofline_estimate(aten.mm, (a, a), {}, a, hw_info, OperatorType.GEMM)

        mem_constraint = next(c for c in result.constraints if c.work_type == "memory")

        # Capacity should be exactly peak bandwidth
        expected_capacity = hw_info.get_peak_memory_bandwidth_gbps()
        assert mem_constraint.capacity == expected_capacity, f"Should use peak bandwidth {expected_capacity:.1f} GB/s"

        print(f"✓ Roofline uses peak bandwidth: {expected_capacity:.1f} GB/s")

    def test_roofline_has_no_overhead(self, hw_info):
        """Roofline should not include kernel launch overhead or other fixed costs."""
        # Tiny operation
        M = 8
        a = torch.randn(M, M, dtype=torch.float32)
        result = roofline_estimate(aten.mm, (a, a), {}, a, hw_info, OperatorType.GEMM)

        # For tiny ops, roofline time should be very small (nanoseconds)
        # If kernel launch overhead were included, it would be ~7μs
        t_roofline_us = result.t_roofline_ms * 1000

        # Should be much less than 7μs (no launch overhead in analytical model)
        assert t_roofline_us < 1.0, (
            f"Analytical roofline should not include launch overhead: "
            f"{t_roofline_us:.3f}μs (should be < 1μs for tiny op)"
        )

        print(f"✓ No overhead in analytical model: {t_roofline_us:.6f}μs for tiny op")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
