"""End-to-end integration tests for LogGP profiling.

Tests the complete workflow:
1. Generate synthetic PRTT measurements
2. Detect protocols
3. Extract LogGP parameters
4. Save to JSON
5. Load from JSON
6. Use in simulation
"""

import json
import tempfile
from pathlib import Path

import pytest

from syssim.network import (
    FullyConnectedTopology,
    LogGPParams,
    allreduce,
    simulate,
)
from syssim.network.model_loader import (
    get_protocol_for_size,
    load_all_protocols,
    load_loggp_params,
)
from syssim.network.profiler import (
    ProfilingResult,
    extract_loggp_parameters,
    save_profiling_result,
)
from syssim.network.protocol_detector import (
    PRTTMeasurement,
    detect_protocol_changes,
)


def generate_synthetic_prtt(L, o, g, G, sizes, n=10):
    """Generate PRTT measurements from known LogGP parameters.

    This creates ground truth data for testing parameter extraction.

    Args:
        L, o, g, G: LogGP parameters
        sizes: List of message sizes
        n: Number of iterations

    Returns:
        List of PRTTMeasurement objects
    """
    measurements = []

    for s in sizes:
        # Gall = g + (s-1)*G
        gall = g + (s - 1) * G

        # PRTT(1, 0, s) = 2 * (L + 2*o + g + (s-1)*G)
        prtt_1_0 = 2 * (L + 2 * o + g + (s - 1) * G)

        # PRTT(n, 0, s) = PRTT(1, 0, s) + (n-1)*Gall
        prtt_n_0 = prtt_1_0 + (n - 1) * gall

        # dG = PRTT(1, 0, s)
        dG = prtt_1_0

        # PRTT(n, dG, s) = PRTT(1, 0, s) + (n-1)*(o + dG)
        prtt_n_dG = prtt_1_0 + (n - 1) * (o + dG)

        measurements.append(PRTTMeasurement(s, prtt_1_0, prtt_n_0, prtt_n_dG))

    return measurements


def test_extract_loggp_from_synthetic():
    """Test parameter extraction from synthetic PRTT data."""
    # Ground truth parameters
    L_true = 1.5e-6
    o_true = 7e-6
    g_true = 2e-6
    G_true = 4e-11

    sizes = [1024, 2048, 4096, 8192, 16384]
    measurements = generate_synthetic_prtt(L_true, o_true, g_true, G_true, sizes, n=10)

    # Detect protocols (should be single protocol)
    protocols = detect_protocol_changes(measurements, n=10, lookahead=3, pfact=2.0)
    assert len(protocols) == 1

    # Extract parameters
    L, o, g, G = extract_loggp_parameters(measurements, protocols[0], n=10)

    # Should match ground truth within 1%
    assert abs(L - L_true) / L_true < 0.01
    assert abs(o - o_true) / o_true < 0.01
    assert abs(g - g_true) / g_true < 0.01
    assert abs(G - G_true) / G_true < 0.01


def test_parameter_extraction_accuracy():
    """Test extraction accuracy across multiple parameter sets."""
    test_cases = [
        # (L, o, g, G)
        (1e-6, 5e-6, 2e-6, 4e-11),  # NVLink-like
        (5e-6, 10e-6, 5e-6, 8e-11),  # InfiniBand-like
        (2e-6, 8e-6, 3e-6, 6e-11),  # Custom
    ]

    for L_true, o_true, g_true, G_true in test_cases:
        sizes = [1024, 2048, 4096, 8192]
        measurements = generate_synthetic_prtt(L_true, o_true, g_true, G_true, sizes, n=10)

        protocols = detect_protocol_changes(measurements, n=10, lookahead=3, pfact=2.0)
        L, o, g, G = extract_loggp_parameters(measurements, protocols[0], n=10)

        # <10% error allowed
        assert abs(L - L_true) / L_true < 0.1
        assert abs(o - o_true) / o_true < 0.1
        assert abs(g - g_true) / g_true < 0.1
        assert abs(G - G_true) / G_true < 0.1


def test_json_serialization_roundtrip():
    """Test saving and loading ProfilingResult to/from JSON."""
    result = ProfilingResult(
        topology="nvlink",
        protocols=[
            {"size_range": [1, 12288], "L": 1.5e-6, "o": 7e-6, "g": 2e-6, "G": 4e-11},
            {"size_range": [12289, 65536], "L": 1.5e-6, "o": 12e-6, "g": 5e-6, "G": 4e-11},
        ],
        primary={"L": 1.5e-6, "o": 7e-6, "g": 2e-6, "G": 4e-11},
        metadata={"timestamp": "2026-02-14T12:00:00", "num_protocols": 2},
    )

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    try:
        save_profiling_result(result, temp_path)

        # Load back
        with open(temp_path) as f:
            loaded = json.load(f)

        # Verify structure
        assert loaded["topology"] == "nvlink"
        assert len(loaded["protocols"]) == 2
        assert loaded["primary"]["L"] == 1.5e-6
        assert loaded["metadata"]["num_protocols"] == 2

    finally:
        temp_path.unlink()


def test_load_loggp_params_from_file():
    """Test loading LogGP parameters from JSON file."""
    # Create temporary JSON file
    data = {
        "topology": "nvlink",
        "protocols": [{"size_range": [1, 65536], "L": 1.5e-6, "o": 7e-6, "g": 2e-6, "G": 4e-11}],
        "primary": {"L": 1.5e-6, "o": 7e-6, "g": 2e-6, "G": 4e-11},
        "metadata": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        # Load parameters
        loggp = load_loggp_params(temp_path)

        # Verify
        assert loggp.L == 1.5e-6
        assert loggp.o == 7e-6
        assert loggp.g == 2e-6
        assert loggp.G == 4e-11

    finally:
        temp_path.unlink()


def test_load_all_protocols():
    """Test loading all protocol ranges."""
    data = {
        "topology": "nvlink",
        "protocols": [
            {"size_range": [1, 12288], "L": 1.5e-6, "o": 7e-6, "g": 2e-6, "G": 4e-11},
            {"size_range": [12289, 65536], "L": 1.5e-6, "o": 12e-6, "g": 5e-6, "G": 4e-11},
        ],
        "primary": {"L": 1.5e-6, "o": 7e-6, "g": 2e-6, "G": 4e-11},
        "metadata": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        protocols = load_all_protocols(temp_path)

        # Should have 2 protocols
        assert len(protocols) == 2

        # First protocol
        (min_size, max_size), params = protocols[0]
        assert min_size == 1
        assert max_size == 12288
        assert params.g == 2e-6

        # Second protocol
        (min_size, max_size), params = protocols[1]
        assert min_size == 12289
        assert max_size == 65536
        assert params.g == 5e-6

    finally:
        temp_path.unlink()


def test_get_protocol_for_size():
    """Test selecting protocol based on message size."""
    protocols = [
        ((1, 12288), LogGPParams(L=1.5e-6, o=7e-6, G=4e-11, g=2e-6)),
        ((12289, 65536), LogGPParams(L=1.5e-6, o=12e-6, G=4e-11, g=5e-6)),
    ]

    # Small message: should use first protocol
    params_small = get_protocol_for_size(protocols, 1024)
    assert params_small.g == 2e-6

    # Large message: should use second protocol
    params_large = get_protocol_for_size(protocols, 32768)
    assert params_large.g == 5e-6

    # Boundary: should use first protocol
    params_boundary = get_protocol_for_size(protocols, 12288)
    assert params_boundary.g == 2e-6

    # Out of range: should raise
    with pytest.raises(ValueError, match="No protocol found"):
        get_protocol_for_size(protocols, 1000000)


def test_hoefler_table2_validation():
    """Validate against Hoefler's Table 2 (synthetic example).

    This test uses synthetic PRTT values that match the LogGP model
    to verify the extraction formulas are correct.
    """
    # Example from conceptual Hoefler Table 2
    # L=1μs, o=5μs, g=2μs, G=0.04ns/byte
    L = 1e-6
    o = 5e-6
    g = 2e-6
    G = 4e-11

    sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    measurements = generate_synthetic_prtt(L, o, g, G, sizes, n=10)

    # Detect and extract
    protocols = detect_protocol_changes(measurements, n=10, lookahead=3, pfact=2.0)
    L_extracted, o_extracted, g_extracted, G_extracted = extract_loggp_parameters(measurements, protocols[0], n=10)

    # Should match exactly (no noise)
    assert abs(L_extracted - L) < 1e-12
    assert abs(o_extracted - o) < 1e-12
    assert abs(g_extracted - g) < 1e-12
    assert abs(G_extracted - G) < 1e-15


def test_profiled_params_in_simulation():
    """Test using profiled parameters in network simulation."""
    # Create synthetic profiled parameters
    loggp = LogGPParams(L=1.5e-6, o=7e-6, G=4e-11, g=2e-6)

    # Create topology
    topo = FullyConnectedTopology(num_ranks=8, link_bandwidth=1.0 / loggp.G)

    # Simulate allreduce
    ops = allreduce(list(range(8)), 1e6)  # 1 MB allreduce
    result = simulate(ops, topo, loggp)

    # Should complete without errors
    assert result.makespan > 0


def test_backward_compatibility_g_optional():
    """Test that g parameter is optional (defaults to 0.0)."""
    # JSON without g field
    data = {
        "topology": "simple",
        "protocols": [
            {
                "size_range": [1, 65536],
                "L": 1e-6,
                "o": 5e-6,
                "G": 4e-11,
                # No g field
            }
        ],
        "primary": {
            "L": 1e-6,
            "o": 5e-6,
            "G": 4e-11,
            # No g field
        },
        "metadata": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        loggp = load_loggp_params(temp_path)

        # g should default to 0.0
        assert loggp.g == 0.0
        assert loggp.L == 1e-6
        assert loggp.o == 5e-6
        assert loggp.G == 4e-11

    finally:
        temp_path.unlink()


def test_topology_bandwidth_consistency():
    """Test that topology.get_bandwidth() matches 1/G from profiling."""
    # Create LogGP params
    L = 1.5e-6
    o = 7e-6
    g = 2e-6
    G = 4e-11  # 1/G = 25 GB/s

    LogGPParams(L=L, o=o, G=G, g=g)

    # Create topology with matching bandwidth
    expected_bw = 1.0 / G
    topo = FullyConnectedTopology(num_ranks=4, link_bandwidth=expected_bw)

    # Check bandwidth matches
    actual_bw = topo.get_bandwidth(0, 1)
    assert abs(actual_bw - expected_bw) < 1e-6


def test_load_missing_file_error():
    """Test that loading nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="LogGP model not found"):
        load_loggp_params("nonexistent_topology")


def test_load_malformed_json_error():
    """Test that malformed JSON raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{invalid json")
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_loggp_params(temp_path)
    finally:
        temp_path.unlink()


def test_load_missing_primary_field_error():
    """Test that JSON without 'primary' field raises ValueError."""
    data = {
        "topology": "test",
        "protocols": [],
        # Missing "primary" field
        "metadata": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Missing 'primary' field"):
            load_loggp_params(temp_path)
    finally:
        temp_path.unlink()
