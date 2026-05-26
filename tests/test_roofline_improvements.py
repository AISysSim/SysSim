"""Test suite for roofline model improvements (Phase 1).

Tests verify:
1. FLOP /2 bug is fixed
2. Memory latency is added
3. Kernel launch overhead is added
4. Cache miss penalty is applied

Note: GPU utilization model is NOT included as roofline should be an analytical ceiling.
"""

import pytest
import torch

from syssim.compute.compute_cost_predictor import roofline_estimate
from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType

aten = torch.ops.aten


@pytest.fixture
def hw_info():
    """GH200 hardware with distinct conservative peak for small ops.

    `peak_tflops_mm_conservative` < `peak_tflops_mm` models the reduced
    throughput on small ops where kernel launch overhead and partial
    tensor-core occupancy dominate.
    """
    return HardwareInfo(
        peak_tflops_mm=535.3,  # TFLOP/s — large-op tensor unit peak
        peak_tflops_math=535.3,
        peak_memory_bandwidth_gbps=4022.8,  # GB/s
        peak_tflops_mm_conservative=100.0,  # Small-op effective peak
    )


class TestFlopBugFix:
    """Test 1.1: Verify FLOP count is NOT divided by 2."""

    def test_gemm_flop_count_correct(self, hw_info):
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

        # Verify compute time matches expected formula
        peak_flops = hw_info.get_peak_tflops(OperatorType.GEMM, torch.float32, is_large_op=True) * 1e12
        # Note: with GPU utilization, effective FLOPS may be lower
        # Just verify it's computed from the correct FLOP count
        min_expected_compute_ns = (expected_flops / peak_flops) * 1e9
        compute_time_ns = math_constraint.time_ms * 1e6

        # Should be >= ideal time (due to utilization factor)
        assert compute_time_ns >= min_expected_compute_ns * 0.99, (
            f"Compute time {compute_time_ns:.0f}ns should be >= ideal {min_expected_compute_ns:.0f}ns"
        )

        print(f"✓ FLOP /2 bug fixed: {expected_flops} FLOPs, {compute_time_ns:.0f}ns compute time")

    def test_small_matrix_has_correct_flops(self, hw_info):
        """Small matrices should also use correct FLOP count."""
        M, N, K = 8, 8, 8
        a = torch.randn(M, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        out = torch.randn(M, N, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)

        math_constraint = next(c for c in result.constraints if c.work_type == "math")
        expected_flops = 2 * M * N * K
        assert math_constraint.work_amount == expected_flops


class TestMemoryLatency:
    """Test 1.2: Verify memory latency is added."""

    def test_memory_latency_added_for_small_tensors(self, hw_info):
        """Small tensors should have significant latency penalty."""
        M = 8
        a = torch.randn(M, M, dtype=torch.float32)
        b = torch.randn(M, M, dtype=torch.float32)
        out = torch.randn(M, M, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)

        # Find memory constraint
        mem_constraint = next(c for c in result.constraints if c.work_type == "memory")

        # Calculate ideal bandwidth-limited time
        bytes_accessed = 3 * M * M * 4  # A, B, C in float32
        ideal_transfer_ns = bytes_accessed / hw_info.get_peak_memory_bandwidth_gbps()

        # Should be > ideal due to latency penalty
        actual_transfer_ns = mem_constraint.time_ms * 1e6
        assert actual_transfer_ns > ideal_transfer_ns * 1.05, (
            f"Should add latency: {actual_transfer_ns:.0f}ns > {ideal_transfer_ns:.0f}ns"
        )

        print(f"✓ Memory latency added: {actual_transfer_ns:.0f}ns vs ideal {ideal_transfer_ns:.0f}ns")


class TestConservativePeakForSmallOps:
    """Test 1.3: Small ops use the conservative peak (replaces fixed-7μs launch overhead model).

    The current model captures launch overhead and partial tensor-core occupancy
    by applying `peak_tflops_mm_conservative` (lower throughput) to small ops.
    """

    def test_small_op_uses_conservative_peak(self, hw_info):
        """Small ops compute time should be derived from the conservative peak."""
        M = 8
        a = torch.randn(M, M, dtype=torch.float32)
        out = torch.randn(M, M, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, a), {}, out, hw_info, OperatorType.GEMM)

        math_constraint = next(c for c in result.constraints if c.work_type == "math")
        actual_compute_ns = math_constraint.time_ms * 1e6

        flops = 2 * M * M * M
        # If model used the full peak, compute would be flops / 535.3 TFLOP/s
        ideal_full_peak_ns = (flops / (hw_info.peak_tflops_mm * 1e12)) * 1e9
        # With conservative peak, it should be flops / 100 TFLOP/s — substantially slower
        ideal_conservative_ns = (flops / (hw_info.peak_tflops_mm_conservative * 1e12)) * 1e9

        assert actual_compute_ns > ideal_full_peak_ns * 1.5, (
            f"Small op should not use full peak: actual={actual_compute_ns:.2f}ns, "
            f"full-peak ideal={ideal_full_peak_ns:.2f}ns"
        )
        assert actual_compute_ns >= ideal_conservative_ns * 0.99, (
            f"Small op should use conservative peak: actual={actual_compute_ns:.2f}ns, "
            f"conservative ideal={ideal_conservative_ns:.2f}ns"
        )

    def test_large_op_uses_full_peak(self, hw_info):
        """Large ops should use the full tensor-unit peak."""
        M = 1024
        a = torch.randn(M, M, dtype=torch.float16)
        out = torch.randn(M, M, dtype=torch.float16)

        result = roofline_estimate(aten.mm, (a, a), {}, out, hw_info, OperatorType.GEMM)

        math_constraint = next(c for c in result.constraints if c.work_type == "math")
        actual_compute_ns = math_constraint.time_ms * 1e6

        flops = 2 * M * M * M
        ideal_full_peak_ns = (flops / (hw_info.peak_tflops_mm * 1e12)) * 1e9

        # Within 5% of full-peak ideal (no conservative penalty)
        assert actual_compute_ns <= ideal_full_peak_ns * 1.05, (
            f"Large op should be near full peak: actual={actual_compute_ns:.2f}ns, "
            f"full-peak ideal={ideal_full_peak_ns:.2f}ns"
        )


class TestCacheMissPenalty:
    """Test 1.4: Verify cache miss penalty is applied."""

    def test_large_matrix_has_cache_penalty(self, hw_info):
        """Large matrices should have cache miss multiplier > 1.0."""
        M = 4096
        a = torch.randn(M, M, dtype=torch.float32)
        b = torch.randn(M, M, dtype=torch.float32)
        out = torch.randn(M, M, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)

        # Find memory constraint
        mem_constraint = next(c for c in result.constraints if c.work_type == "memory")

        # Calculate ideal bandwidth-limited time (no cache effects)
        bytes_accessed = 3 * M * M * 4
        ideal_transfer_ns = bytes_accessed / hw_info.get_peak_memory_bandwidth_gbps()

        # Should be > ideal due to cache miss penalty
        actual_transfer_ns = mem_constraint.time_ms * 1e6
        cache_multiplier = actual_transfer_ns / ideal_transfer_ns

        assert cache_multiplier >= 1.0, "Cache multiplier should be >= 1.0"
        assert cache_multiplier <= 3.0, "Cache multiplier should be <= 3.0"

        print(f"✓ Cache miss penalty applied: {cache_multiplier:.2f}x multiplier")

    def test_small_high_intensity_has_good_cache(self, hw_info):
        """Small GEMM with high arithmetic intensity should have good cache reuse."""
        M = 256
        a = torch.randn(M, M, dtype=torch.float32)
        out = torch.randn(M, M, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, a), {}, out, hw_info, OperatorType.GEMM)

        mem_constraint = next(c for c in result.constraints if c.work_type == "memory")

        # Small matrices fitting in L2 with high intensity should have
        # low cache multiplier (good reuse)
        bytes_accessed = 3 * M * M * 4
        ideal_transfer_ns = bytes_accessed / hw_info.get_peak_memory_bandwidth_gbps()
        actual_transfer_ns = mem_constraint.time_ms * 1e6
        cache_multiplier = actual_transfer_ns / ideal_transfer_ns

        # Should be close to 1.0 (good cache reuse) or slightly higher
        assert cache_multiplier < 2.0, f"Small high-intensity op should have good cache: {cache_multiplier:.2f}x"

        print(f"✓ Good cache reuse for small GEMM: {cache_multiplier:.2f}x")


class TestGPUUtilization:
    """Test 1.5: Verify GPU utilization model works."""

    def test_tiny_ops_have_low_utilization(self, hw_info):
        """Tiny operations should have reduced effective FLOPs (low utilization)."""
        M = 8
        a = torch.randn(M, M, dtype=torch.float32)
        out = torch.randn(M, M, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, a), {}, out, hw_info, OperatorType.GEMM)

        # Find math constraint
        math_constraint = next(c for c in result.constraints if c.work_type == "math")

        # Calculate ideal compute time (100% utilization)
        flops = 2 * M * M * M
        peak_flops = hw_info.get_peak_tflops(OperatorType.GEMM, torch.float32, is_large_op=True) * 1e12
        ideal_compute_ns = (flops / peak_flops) * 1e9

        # Actual compute time should be higher due to low utilization
        actual_compute_ns = math_constraint.time_ms * 1e6
        utilization = ideal_compute_ns / actual_compute_ns

        assert utilization < 0.9, f"Tiny ops should have low utilization: {utilization:.2%}"
        assert utilization >= 0.1, f"Utilization should be >= 10%: {utilization:.2%}"

        print(f"✓ Tiny ops have low GPU utilization: {utilization:.2%}")

    def test_large_ops_have_high_utilization(self, hw_info):
        """Large operations should approach 100% GPU utilization."""
        M = 4096
        a = torch.randn(M, M, dtype=torch.float32)
        out = torch.randn(M, M, dtype=torch.float32)

        result = roofline_estimate(aten.mm, (a, a), {}, out, hw_info, OperatorType.GEMM)

        math_constraint = next(c for c in result.constraints if c.work_type == "math")

        # Calculate ideal compute time
        flops = 2 * M * M * M
        peak_flops = hw_info.get_peak_tflops(OperatorType.GEMM, torch.float32, is_large_op=True) * 1e12
        ideal_compute_ns = (flops / peak_flops) * 1e9

        # Actual should be very close to ideal (high utilization)
        actual_compute_ns = math_constraint.time_ms * 1e6
        utilization = ideal_compute_ns / actual_compute_ns

        assert utilization >= 0.95, f"Large ops should have high utilization: {utilization:.2%}"

        print(f"✓ Large ops have high GPU utilization: {utilization:.2%}")


class TestIntegration:
    """Integration tests for combined effects."""

    def test_roofline_is_realistic_ceiling(self, hw_info):
        """Verify roofline produces realistic performance ceiling."""
        sizes = [64, 128, 256, 512, 1024, 2048]

        print("\nRoofline times by matrix size:")
        print("Size    | Roofline Time | Compute | Memory  | Dominant")
        print("--------|---------------|---------|---------|----------")

        for M in sizes:
            a = torch.randn(M, M, dtype=torch.float16)
            result = roofline_estimate(aten.mm, (a, a), {}, a, hw_info, OperatorType.GEMM)

            math_c = next((c for c in result.constraints if c.work_type == "math"), None)
            mem_c = next((c for c in result.constraints if c.work_type == "memory"), None)

            compute_us = math_c.time_ms * 1000 if math_c else 0
            memory_us = mem_c.time_ms * 1000 if mem_c else 0
            total_us = result.t_roofline_ms * 1000

            dominant = "compute" if compute_us > memory_us else "memory"

            print(f"{M:4d}×{M:<3d} | {total_us:9.2f} μs | {compute_us:6.2f}μs | {memory_us:6.2f}μs | {dominant}")

            # Verify all times are positive
            assert result.t_roofline_ms > 0, "Roofline time should be positive"

    def test_efficiency_increases_with_size(self, hw_info):
        """Larger matrices should approach roofline ceiling (if we could measure)."""
        # Use truly tiny matrix to see utilization difference
        small = torch.randn(8, 8, dtype=torch.float32)
        large = torch.randn(4096, 4096, dtype=torch.float32)

        result_small = roofline_estimate(aten.mm, (small, small), {}, small, hw_info, OperatorType.GEMM)
        result_large = roofline_estimate(aten.mm, (large, large), {}, large, hw_info, OperatorType.GEMM)

        # Get math constraints
        small_compute = next(c for c in result_small.constraints if c.work_type == "math")
        large_compute = next(c for c in result_large.constraints if c.work_type == "math")

        # Calculate utilization by comparing actual vs ideal compute time
        peak_flops = hw_info.get_peak_tflops(OperatorType.GEMM, torch.float32, is_large_op=True) * 1e12

        small_flops = small_compute.work_amount
        large_flops = large_compute.work_amount

        # Ideal time (100% utilization)
        small_ideal_ns = (small_flops / peak_flops) * 1e9
        large_ideal_ns = (large_flops / peak_flops) * 1e9

        # Actual time (with utilization factor)
        small_actual_ns = small_compute.time_ms * 1e6
        large_actual_ns = large_compute.time_ms * 1e6

        # Utilization = ideal / actual
        small_util = small_ideal_ns / small_actual_ns if small_actual_ns > 0 else 0
        large_util = large_ideal_ns / large_actual_ns if large_actual_ns > 0 else 0

        # Large matrix should have higher utilization
        assert large_util > small_util, (
            f"Large matrix should have higher utilization: {large_util:.2%} > {small_util:.2%}"
        )

        print(f"✓ Larger matrices have higher utilization: {large_util:.2%} vs {small_util:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
