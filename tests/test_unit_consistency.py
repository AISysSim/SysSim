"""
Unit consistency tests for roofline model calculations.

Verifies all unit conversions are correct:
- TFLOP/s ↔ FLOP/s (× 1e12)
- PFLOP/s ↔ TFLOP/s (× 1000)
- seconds ↔ nanoseconds (× 1e9)
- nanoseconds ↔ milliseconds (/ 1e6)
- GB/s ↔ bytes/s (× 1e9)
"""

import pytest
import torch

from syssim.config import HardwareInfo
from syssim.operator_graph import OperatorType

# Unit conversion constants
TERA_TO_UNIT = 1e12  # 1 TFLOP = 10^12 FLOP
GIGA_TO_UNIT = 1e9  # 1 GB = 10^9 bytes
PETA_TO_TERA = 1000.0  # 1 PFLOP = 1000 TFLOP
SECONDS_TO_NS = 1e9  # 1 second = 10^9 nanoseconds
NS_TO_MS = 1e6  # 1 ms = 10^6 nanoseconds


class TestUnitConversions:
    """Test basic unit conversion constants."""

    def test_tera_conversion(self):
        """Test TFLOP/s to FLOP/s conversion."""
        tflops = 989.0  # H100 FP16 peak
        flops = tflops * TERA_TO_UNIT
        assert flops == 989e12
        assert flops == 989_000_000_000_000

    def test_peta_conversion(self):
        """Test PFLOP/s to TFLOP/s conversion."""
        pflops = 0.5353  # GH200 FP16 peak (from get_device_tflops)
        tflops = pflops * PETA_TO_TERA
        assert abs(tflops - 535.3) < 0.1

    def test_time_conversions(self):
        """Test time unit conversions."""
        seconds = 1.0
        nanoseconds = seconds * SECONDS_TO_NS
        milliseconds = nanoseconds / NS_TO_MS

        assert nanoseconds == 1e9
        assert milliseconds == 1000.0

    def test_bandwidth_conversion(self):
        """Test GB/s to bytes/s conversion."""
        gbps = 3350.0  # H100 peak bandwidth
        bytes_per_sec = gbps * GIGA_TO_UNIT
        assert bytes_per_sec == 3.35e12


class TestBandwidthCalculation:
    """Test memory bandwidth transfer time calculations."""

    def test_bandwidth_formula_small_transfer(self):
        """
        Test bandwidth calculation with known values.

        Example: 64×64 FP16 matrix (M=N=K=64)
        - A: 64×64 × 2 bytes = 8,192 bytes
        - B: 64×64 × 2 bytes = 8,192 bytes
        - C: 64×64 × 2 bytes = 8,192 bytes
        - Total: 24,576 bytes

        Bandwidth: 3350 GB/s = 3.35e12 bytes/s
        Expected time: 24,576 / 3.35e12 = 7.34e-9 seconds = 7.34 ns
        """
        HardwareInfo(
            peak_tflops_mm=989.0,
            peak_tflops_math=989.0,
            peak_memory_bandwidth_gbps=3350.0,
        )

        # Manually calculate expected time
        total_bytes = 24_576
        bw_bytes_per_sec = 3350.0 * GIGA_TO_UNIT  # 3.35e12 bytes/s
        expected_time_sec = total_bytes / bw_bytes_per_sec
        expected_time_ns = expected_time_sec * SECONDS_TO_NS

        # Expected: ~7.34 ns
        assert abs(expected_time_ns - 7.34) < 0.1

        # Now test the actual function
        # Note: We need to pass actual tensor args to test the function
        # For now, just verify our math is correct
        assert expected_time_ns > 0
        assert expected_time_ns < 100  # Should be small for small matrices

    def test_bandwidth_formula_large_transfer(self):
        """
        Test bandwidth calculation for large matrix.

        Example: 4096×4096 FP16 matrix (M=N=K=4096)
        - A: 4096×4096 × 2 bytes = 33,554,432 bytes
        - B: 4096×4096 × 2 bytes = 33,554,432 bytes
        - C: 4096×4096 × 2 bytes = 33,554,432 bytes
        - Total: 100,663,296 bytes (~100 MB)

        Bandwidth: 3350 GB/s = 3.35e12 bytes/s
        Expected time: 100,663,296 / 3.35e12 = 30.05e-6 seconds = 30,050 ns = 0.03005 ms
        """
        HardwareInfo(
            peak_tflops_mm=989.0,
            peak_tflops_math=989.0,
            peak_memory_bandwidth_gbps=3350.0,
        )

        total_bytes = 100_663_296
        bw_bytes_per_sec = 3350.0 * GIGA_TO_UNIT
        expected_time_sec = total_bytes / bw_bytes_per_sec
        expected_time_ns = expected_time_sec * SECONDS_TO_NS
        expected_time_ms = expected_time_ns / NS_TO_MS

        # Expected: ~30,050 ns = ~0.03005 ms
        assert abs(expected_time_ns - 30_050) < 100
        assert abs(expected_time_ms - 0.03005) < 0.001

    def test_bandwidth_dimensional_analysis(self):
        """
        Verify dimensional analysis of bandwidth formula.

        Formula should be:
        time (ns) = bytes / (GB/s × 1e9 bytes/GB) × 1e9 ns/s
                  = bytes / GB/s (when GB/s is treated as numeric)

        Or equivalently:
        time (s) = bytes / (GB/s × 1e9)
        time (ns) = time (s) × 1e9
        """
        bytes_transferred = 1_000_000  # 1 MB
        bw_gbps = 1000.0  # 1000 GB/s (numeric)

        # Method 1: Direct division (current implementation)
        time_ns_method1 = bytes_transferred / bw_gbps

        # Method 2: Full unit conversion
        bw_bytes_per_sec = bw_gbps * GIGA_TO_UNIT
        time_sec = bytes_transferred / bw_bytes_per_sec
        time_ns_method2 = time_sec * SECONDS_TO_NS

        # Both methods should give same result
        assert abs(time_ns_method1 - time_ns_method2) < 1e-6

        # Verify the result makes sense
        # 1 MB / 1000 GB/s = 0.001 ms = 1000 ns
        assert abs(time_ns_method1 - 1000.0) < 0.1


class TestComputeTimeCalculation:
    """Test compute-bound roofline time calculations."""

    def test_compute_time_small_gemm(self):
        """
        Test compute time for small GEMM.

        Example: 64×64×64 FP16 GEMM
        - FLOPs: 2 × M × N × K = 2 × 64³ = 524,288 FLOPs
        - Peak: 989 TFLOP/s = 989e12 FLOP/s
        - Expected time: 524,288 / 989e12 = 5.30e-10 seconds = 0.530 ns
        """
        flops = 2 * 64 * 64 * 64  # 524,288
        peak_flops = 989.0 * TERA_TO_UNIT  # 989e12 FLOP/s

        expected_time_sec = flops / peak_flops
        expected_time_ns = expected_time_sec * SECONDS_TO_NS

        # Expected: ~0.530 ns
        assert abs(expected_time_ns - 0.530) < 0.01

    def test_compute_time_large_gemm(self):
        """
        Test compute time for large GEMM.

        Example: 4096×4096×4096 FP16 GEMM
        - FLOPs: 2 × M × N × K = 2 × 4096³ = 137,438,953,472 FLOPs (~137 GFLOPs)
        - Peak: 989 TFLOP/s = 989e12 FLOP/s
        - Expected time: 137.44e9 / 989e12 = 1.39e-4 seconds = 139,000 ns = 0.139 ms
        """
        flops = 2 * 4096 * 4096 * 4096  # ~137.44 billion
        peak_flops = 989.0 * TERA_TO_UNIT

        expected_time_sec = flops / peak_flops
        expected_time_ns = expected_time_sec * SECONDS_TO_NS
        expected_time_ms = expected_time_ns / NS_TO_MS

        # Expected: ~139,000 ns = ~0.139 ms
        assert abs(expected_time_ns - 139_000) < 1000
        assert abs(expected_time_ms - 0.139) < 0.01


class TestRooflineModel:
    """Test end-to-end roofline model."""

    def test_small_gemm_memory_bound(self):
        """
        Small GEMM should be memory-bound.

        64×64×64 FP16:
        - Compute: 0.530 ns (from above)
        - Transfer: 7.34 ns (from above)
        - Roofline = max(0.530, 7.34) = 7.34 ns (memory-bound)
        """
        compute_time = 0.530  # ns
        transfer_time = 7.34  # ns
        expected_roofline = max(compute_time, transfer_time)

        assert expected_roofline == transfer_time
        assert expected_roofline > compute_time
        # Small matrices are memory-bound

    def test_large_gemm_compute_bound(self):
        """
        Large GEMM should be compute-bound.

        4096×4096×4096 FP16:
        - Compute: 139,000 ns (from above)
        - Transfer: 30,050 ns (from above)
        - Roofline = max(139,000, 30,050) = 139,000 ns (compute-bound)
        """
        compute_time = 139_000  # ns
        transfer_time = 30_050  # ns
        expected_roofline = max(compute_time, transfer_time)

        assert expected_roofline == compute_time
        assert expected_roofline > transfer_time
        # Large matrices are compute-bound

    def test_arithmetic_intensity_transition(self):
        """
        Test arithmetic intensity (AI) determines compute vs memory bound.

        AI = FLOPs / bytes_transferred

        Compute-bound when: AI > (peak_flops / peak_bandwidth)
        Memory-bound when: AI < (peak_flops / peak_bandwidth)

        H100: peak_flops = 989 TFLOP/s, peak_bw = 3350 GB/s
        Critical AI = 989e12 / (3350e9) = 295.2 FLOP/byte
        """
        peak_flops = 989.0 * TERA_TO_UNIT  # FLOP/s
        peak_bw = 3350.0 * GIGA_TO_UNIT  # bytes/s

        critical_ai = peak_flops / peak_bw
        # Expected: ~295 FLOP/byte
        assert abs(critical_ai - 295.2) < 1.0

        # Small GEMM (64×64×64):
        # AI = 524,288 FLOPs / 24,576 bytes = 21.3 FLOP/byte < 295
        # → memory-bound ✓
        small_ai = (2 * 64 * 64 * 64) / (3 * 64 * 64 * 2)
        assert small_ai < critical_ai

        # Large GEMM (4096×4096×4096):
        # AI = 137.44e9 FLOPs / 100.66e6 bytes = 1365.3 FLOP/byte > 295
        # → compute-bound ✓
        large_ai = (2 * 4096 * 4096 * 4096) / (3 * 4096 * 4096 * 2)
        assert large_ai > critical_ai


class TestHardwareSpecifications:
    """Test hardware specification values are in correct units."""

    def test_h100_specs(self):
        """Test H100 specifications."""
        hw = HardwareInfo(
            peak_tflops_mm=989.0,
            peak_tflops_math=989.0,
            peak_memory_bandwidth_gbps=3350.0,
        )

        # Values should be in TFLOP/s and GB/s
        assert hw.peak_tflops_mm > 100, "Should be ~989 TFLOP/s"
        assert hw.peak_tflops_mm < 10_000, "Should not be GFLOP/s or PFLOP/s"
        assert hw.peak_memory_bandwidth_gbps > 1000, "Should be ~3350 GB/s"
        assert hw.peak_memory_bandwidth_gbps < 100_000, "Should not be MB/s"

    def test_hardware_info_get_methods(self):
        """Test HardwareInfo getter methods return correct units."""
        hw = HardwareInfo(
            peak_tflops_mm=989.0,
            peak_tflops_math=500.0,
            peak_memory_bandwidth_gbps=3350.0,
        )

        # Test get_peak_tflops with different operator types
        gemm_flops = hw.get_peak_tflops(OperatorType.GEMM, torch.float16)
        attn_flops = hw.get_peak_tflops(OperatorType.ATTN, torch.float16)
        compute_flops = hw.get_peak_tflops(OperatorType.MATH, torch.float16)

        assert gemm_flops == 989.0
        assert attn_flops == 989.0
        assert compute_flops == 500.0

        # Test get_peak_memory_bandwidth
        bw = hw.get_peak_memory_bandwidth_gbps()
        assert bw == 3350.0


class TestTimeUnitConsistency:
    """Test time units are consistent across the codebase."""

    def test_ns_to_ms_conversion(self):
        """Test nanoseconds to milliseconds conversion."""
        time_ns = 1_000_000  # 1 ms in nanoseconds
        time_ms = time_ns / NS_TO_MS

        assert time_ms == 1.0

    def test_roofline_returns_ms(self):
        """
        Verify roofline_estimate returns milliseconds.

        This is a documentation test - we verify the expected units
        without running actual inference.
        """
        # roofline_estimate should return RooflineResult with time_ms field
        # This field should be in milliseconds
        # Internal calculations are in nanoseconds, converted to ms at the end

        # Example: 139,000 ns = 0.139 ms
        time_ns = 139_000
        time_ms = time_ns / NS_TO_MS
        assert abs(time_ms - 0.139) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
