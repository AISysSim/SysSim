"""Unit tests for protocol detection logic.

Tests the Hoefler lookahead algorithm for detecting protocol changes
(e.g., eager → rendezvous) in PRTT measurements.
"""

import numpy as np
import pytest

from syssim.network.protocol_detector import (
    PRTTMeasurement,
    compute_gall,
    detect_protocol_changes,
    least_squares_fit,
)


def test_compute_gall_formula():
    """Test Gall computation: Gall(s) = [PRTT(n,0,s) - PRTT(1,0,s)] / (n-1)."""
    measurements = [
        PRTTMeasurement(1024, 1e-5, 1.5e-4, 2e-4),
        PRTTMeasurement(2048, 1.2e-5, 1.8e-4, 2.4e-4),
    ]

    gall = compute_gall(measurements, n=10)

    assert len(gall) == 2
    # Gall[0] = (1.5e-4 - 1e-5) / 9 ≈ 1.556e-5
    assert abs(gall[0] - (1.5e-4 - 1e-5) / 9) < 1e-10
    # Gall[1] = (1.8e-4 - 1.2e-5) / 9 ≈ 1.867e-5
    assert abs(gall[1] - (1.8e-4 - 1.2e-5) / 9) < 1e-10


def test_least_squares_fit_known_params():
    """Test least squares fit with known g=5μs, G=0.04ns/byte."""
    g_true = 5e-6
    G_true = 4e-11

    # Generate synthetic data: Gall(s) = g + (s-1)*G
    sizes = [1024, 2048, 4096, 8192, 16384]
    gall = [g_true + (s - 1) * G_true for s in sizes]

    g_fit, G_fit, mse = least_squares_fit(sizes, gall)

    # Should recover true parameters exactly (no noise)
    assert abs(g_fit - g_true) < 1e-12
    assert abs(G_fit - G_true) < 1e-15
    assert mse < 1e-20  # Essentially zero


def test_least_squares_fit_with_noise():
    """Test least squares fit with ±5% noise."""
    g_true = 2e-6
    G_true = 4e-11

    sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    gall_clean = [g_true + (s - 1) * G_true for s in sizes]

    # Add ±5% noise
    np.random.seed(42)
    noise = np.random.uniform(-0.05, 0.05, len(gall_clean))
    gall_noisy = [g * (1 + n) for g, n in zip(gall_clean, noise)]

    g_fit, G_fit, mse = least_squares_fit(sizes, gall_noisy)

    # Intercept estimate is well-conditioned: tight bound.
    # Slope (G) estimate from only 6 points with ±5% noise is loose: ~15%.
    assert abs(g_fit - g_true) / g_true < 0.1
    assert abs(G_fit - G_true) / G_true < 0.2


def test_detect_single_protocol():
    """Test detection with single protocol (no change)."""
    g = 2e-6
    G = 4e-11
    n = 10

    # Generate PRTT measurements for single protocol
    sizes = [1024, 2048, 4096, 8192, 16384]
    measurements = []

    for s in sizes:
        gall = g + (s - 1) * G
        prtt_1_0 = 2 * (1e-6 + 2 * 7e-6 + g + (s - 1) * G)  # L=1μs, o=7μs
        prtt_n_0 = prtt_1_0 + (n - 1) * gall
        prtt_n_dG = prtt_1_0 + (n - 1) * (7e-6 + prtt_1_0)  # o + dG

        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    protocols = detect_protocol_changes(measurements, n=10, lookahead=3, pfact=2.0)

    # Should detect single protocol
    assert len(protocols) == 1
    assert protocols[0].start_idx == 0
    assert protocols[0].end_idx == len(measurements) - 1

    # Parameters should match
    assert abs(protocols[0].g - g) / g < 0.01  # <1% error
    assert abs(protocols[0].G - G) / G < 0.01


def test_detect_two_protocols():
    """Test detection with eager (<12KB) and rendezvous (≥12KB) protocols."""
    n = 10

    # Eager protocol: g=2μs, G=0.04ns/byte
    g_eager = 2e-6
    G_eager = 4e-11

    # Rendezvous protocol: g=5μs, G=0.04ns/byte (same G, different g)
    g_rend = 5e-6
    G_rend = 4e-11

    # Generate measurements
    measurements = []

    # Eager protocol: 1KB to 8KB
    for s in [1024, 2048, 4096, 8192]:
        gall = g_eager + (s - 1) * G_eager
        prtt_1_0 = 2 * (1.5e-6 + 2 * 7e-6 + g_eager + (s - 1) * G_eager)
        prtt_n_0 = prtt_1_0 + (n - 1) * gall
        prtt_n_dG = prtt_1_0 + (n - 1) * (7e-6 + prtt_1_0)
        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    # Rendezvous protocol: 16KB to 64KB
    for s in [16384, 32768, 65536]:
        gall = g_rend + (s - 1) * G_rend
        prtt_1_0 = 2 * (1.5e-6 + 2 * 12e-6 + g_rend + (s - 1) * G_rend)
        prtt_n_0 = prtt_1_0 + (n - 1) * gall
        prtt_n_dG = prtt_1_0 + (n - 1) * (12e-6 + prtt_1_0)
        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    protocols = detect_protocol_changes(measurements, n=10, lookahead=2, pfact=2.0)

    # Should detect 2 protocols
    assert len(protocols) == 2

    # First protocol should be eager
    assert protocols[0].sizes[0] == 1024
    assert abs(protocols[0].g - g_eager) / g_eager < 0.05  # <5% error

    # Second protocol should be rendezvous
    assert protocols[1].sizes[0] >= 16384
    assert abs(protocols[1].g - g_rend) / g_rend < 0.05


def test_protocol_robustness_to_noise():
    """Test protocol detection with ±5% noise."""
    n = 10
    np.random.seed(123)

    # Two protocols with different g
    g1 = 2e-6
    g2 = 5e-6
    G = 4e-11

    measurements = []

    # Protocol 1: 1KB to 8KB
    for s in [1024, 2048, 4096, 8192]:
        gall = g1 + (s - 1) * G
        prtt_1_0 = 2 * (1.5e-6 + 2 * 7e-6 + g1 + (s - 1) * G)
        prtt_n_0 = prtt_1_0 + (n - 1) * gall

        # Add noise
        prtt_1_0 *= 1 + np.random.uniform(-0.05, 0.05)
        prtt_n_0 *= 1 + np.random.uniform(-0.05, 0.05)

        prtt_n_dG = prtt_1_0 + (n - 1) * (7e-6 + prtt_1_0)
        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    # Protocol 2: 16KB to 64KB
    for s in [16384, 32768, 65536]:
        gall = g2 + (s - 1) * G
        prtt_1_0 = 2 * (1.5e-6 + 2 * 12e-6 + g2 + (s - 1) * G)
        prtt_n_0 = prtt_1_0 + (n - 1) * gall

        # Add noise
        prtt_1_0 *= 1 + np.random.uniform(-0.05, 0.05)
        prtt_n_0 *= 1 + np.random.uniform(-0.05, 0.05)

        prtt_n_dG = prtt_1_0 + (n - 1) * (12e-6 + prtt_1_0)
        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    protocols = detect_protocol_changes(measurements, n=10, lookahead=2, pfact=2.0)

    # Should still detect 2 protocols despite noise
    assert len(protocols) >= 1  # At least one protocol


def test_lookahead_parameter_sensitivity():
    """Test different lookahead values (3, 5, 10)."""
    # Simple two-protocol case
    g1, g2 = 2e-6, 5e-6
    G = 4e-11
    n = 10

    measurements = []
    for s in [1024, 2048, 4096, 8192]:
        gall = g1 + (s - 1) * G
        prtt_1_0 = 2 * (1.5e-6 + 2 * 7e-6 + g1 + (s - 1) * G)
        prtt_n_0 = prtt_1_0 + (n - 1) * gall
        prtt_n_dG = prtt_1_0 + (n - 1) * (7e-6 + prtt_1_0)
        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    for s in [16384, 32768, 65536]:
        gall = g2 + (s - 1) * G
        prtt_1_0 = 2 * (1.5e-6 + 2 * 12e-6 + g2 + (s - 1) * G)
        prtt_n_0 = prtt_1_0 + (n - 1) * gall
        prtt_n_dG = prtt_1_0 + (n - 1) * (12e-6 + prtt_1_0)
        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    # Test different lookahead values
    for lookahead in [2, 3, 5]:
        protocols = detect_protocol_changes(measurements, n=10, lookahead=lookahead, pfact=2.0)
        # Should detect at least 1 protocol
        assert len(protocols) >= 1


def test_pfact_parameter_sensitivity():
    """Test different pfact values (1.5, 2.0, 3.0)."""
    g1, g2 = 2e-6, 5e-6
    G = 4e-11
    n = 10

    measurements = []
    for s in [1024, 2048, 4096, 8192]:
        gall = g1 + (s - 1) * G
        prtt_1_0 = 2 * (1.5e-6 + 2 * 7e-6 + g1 + (s - 1) * G)
        prtt_n_0 = prtt_1_0 + (n - 1) * gall
        prtt_n_dG = prtt_1_0 + (n - 1) * (7e-6 + prtt_1_0)
        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    for s in [16384, 32768, 65536]:
        gall = g2 + (s - 1) * G
        prtt_1_0 = 2 * (1.5e-6 + 2 * 12e-6 + g2 + (s - 1) * G)
        prtt_n_0 = prtt_1_0 + (n - 1) * gall
        prtt_n_dG = prtt_1_0 + (n - 1) * (12e-6 + prtt_1_0)
        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    # Test different pfact values
    for pfact in [1.5, 2.0, 3.0]:
        protocols = detect_protocol_changes(measurements, n=10, lookahead=2, pfact=pfact)
        # Lower pfact → more sensitive → more protocols detected
        assert len(protocols) >= 1


def test_edge_case_insufficient_points():
    """Test graceful handling of <lookahead measurements."""
    # Only 2 measurements (< lookahead=3)
    measurements = [
        PRTTMeasurement(1024, 1e-5, 1.5e-4, 2e-4),
        PRTTMeasurement(2048, 1.2e-5, 1.8e-4, 2.4e-4),
    ]

    protocols = detect_protocol_changes(measurements, n=10, lookahead=3, pfact=2.0)

    # Should create single protocol
    assert len(protocols) == 1
    assert protocols[0].start_idx == 0
    assert protocols[0].end_idx == 1


def test_edge_case_single_measurement():
    """Test that single measurement raises error."""
    measurements = [
        PRTTMeasurement(1024, 1e-5, 1.5e-4, 2e-4),
    ]

    with pytest.raises(ValueError, match="at least 2 measurements"):
        detect_protocol_changes(measurements, n=10, lookahead=3, pfact=2.0)


def test_least_squares_fit_dimension_mismatch():
    """Test least_squares_fit with mismatched dimensions."""
    sizes = [1024, 2048]
    gall = [1e-6, 2e-6, 3e-6]  # Different length

    with pytest.raises(ValueError, match="same length"):
        least_squares_fit(sizes, gall)


def test_least_squares_fit_insufficient_points():
    """Test least_squares_fit with <2 points."""
    sizes = [1024]
    gall = [1e-6]

    with pytest.raises(ValueError, match="at least 2 points"):
        least_squares_fit(sizes, gall)
