"""Unit tests for NCCL backend (GPU-to-GPU profiling).

These tests require:
- 2+ CUDA-capable GPUs
- torch.distributed initialized with NCCL backend
- Must be run via distributed launcher: torchrun --nproc_per_node=2 -m pytest tests/test_nccl_backend.py

Tests are skipped if:
- CUDA unavailable
- <2 GPUs available
- torch.distributed not available
"""

import sys

import pytest

# Check if torch.distributed is available
try:
    import torch
    import torch.distributed as dist

    HAS_TORCH_DISTRIBUTED = True
except ImportError:
    HAS_TORCH_DISTRIBUTED = False

# Skip all tests if requirements not met
pytestmark = pytest.mark.skipif(
    not HAS_TORCH_DISTRIBUTED or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason=(
        "Requires PyTorch with CUDA and 2+ GPUs, "
        "run via: torchrun --nproc_per_node=2 -m pytest tests/test_nccl_backend.py"
    ),
)


@pytest.fixture(scope="module")
def nccl_backend():
    """Initialize NCCL backend for testing.

    This fixture initializes torch.distributed if not already initialized.
    """
    from syssim.network.profiler import NCCLBackend

    # Check if already initialized (via torchrun)
    if not dist.is_initialized():
        # Initialize for testing (assumes single node, local testing)
        dist.init_process_group(backend="nccl", init_method="env://")

    backend = NCCLBackend()

    yield backend

    # Cleanup
    backend.cleanup()


def test_nccl_backend_initialization(nccl_backend):
    """Test NCCLBackend initializes correctly."""
    assert nccl_backend.rank in [0, 1]  # Should be rank 0 or 1
    assert nccl_backend.world_size >= 2


def test_nccl_backend_ping_pong(nccl_backend):
    """Test basic ping-pong communication."""
    # Single ping-pong, no delay, 1KB message
    elapsed = nccl_backend.ping_pong(n=1, delay=0.0, size=1024)

    if nccl_backend.is_client():
        # Client should measure non-zero time
        assert elapsed > 0
        assert elapsed < 1.0  # Should be <1 second for 1KB

    if nccl_backend.is_server():
        # Server returns 0.0
        assert elapsed == 0.0


def test_nccl_backend_linearity_n(nccl_backend):
    """Test PRTT(2n) ≈ 2*PRTT(n) - PRTT(1)."""
    size = 4096  # 4KB message

    # Measure PRTT for different n
    elapsed_1 = nccl_backend.ping_pong(n=1, delay=0.0, size=size)
    elapsed_10 = nccl_backend.ping_pong(n=10, delay=0.0, size=size)
    elapsed_20 = nccl_backend.ping_pong(n=20, delay=0.0, size=size)

    if nccl_backend.is_client():
        # PRTT(20) ≈ 2*PRTT(10) - PRTT(1)
        # Allow 30% tolerance due to GPU scheduling variance
        expected = 2 * elapsed_10 - elapsed_1
        assert abs(elapsed_20 - expected) / expected < 0.3


def test_nccl_backend_linearity_size(nccl_backend):
    """Test PRTT increases with message size."""
    sizes = [1024, 4096, 16384]  # 1KB, 4KB, 16KB

    times = []
    for size in sizes:
        elapsed = nccl_backend.ping_pong(n=1, delay=0.0, size=size)
        times.append(elapsed)

    if nccl_backend.is_client():
        # Larger messages should take more time
        assert times[1] >= times[0]  # 4KB >= 1KB
        assert times[2] >= times[1]  # 16KB >= 4KB


def test_nccl_backend_delay_effect(nccl_backend):
    """Test PRTT(n, d) > PRTT(n, 0) by approximately n*d."""
    size = 2048
    n = 5
    delay = 0.001  # 1ms delay

    elapsed_no_delay = nccl_backend.ping_pong(n=n, delay=0.0, size=size)
    elapsed_with_delay = nccl_backend.ping_pong(n=n, delay=delay, size=size)

    if nccl_backend.is_client():
        # With delay should be longer by approximately n*delay
        delta = elapsed_with_delay - elapsed_no_delay
        expected_delta = n * delay

        # Allow 50% tolerance due to sleep() accuracy
        assert abs(delta - expected_delta) / expected_delta < 0.5


def test_measure_prtt_statistical(nccl_backend):
    """Test measure_prtt with multiple runs returns consistent median."""
    from syssim.network.profiler import measure_prtt

    # Measure with 5 runs
    median_time = measure_prtt(nccl_backend, n=10, delay=0.0, size=4096, num_runs=5)

    if nccl_backend.is_client():
        assert median_time > 0
        # Should be reasonable (<1 second for 10 iterations of 4KB)
        assert median_time < 1.0


def test_nccl_backend_multiple_runs(nccl_backend):
    """Test that multiple runs give consistent results."""
    size = 2048
    n = 10

    times = []
    for _ in range(3):
        elapsed = nccl_backend.ping_pong(n=n, delay=0.0, size=size)
        times.append(elapsed)

    if nccl_backend.is_client():
        # Coefficient of variation should be <30%
        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
        cv = std / mean

        assert cv < 0.3  # <30% variance


def test_sweep_message_sizes(nccl_backend):
    """Test sweep_message_sizes returns expected structure."""
    from syssim.network.profiler import sweep_message_sizes

    # Small sweep for testing
    measurements = sweep_message_sizes(nccl_backend, min_size=1024, max_size=8192, n=5, num_runs=3)

    if nccl_backend.is_client():
        # Should have measurements for sizes: 1024, 2048, 4096, 8192
        assert len(measurements) == 4

        # Each measurement should have valid PRTT values
        for m in measurements:
            assert m.size in [1024, 2048, 4096, 8192]
            assert m.prtt_1_0 > 0
            assert m.prtt_n_0 > 0
            assert m.prtt_n_dG > 0

            # PRTT(n, 0) should be > PRTT(1, 0)
            assert m.prtt_n_0 > m.prtt_1_0


# Helper to run tests
if __name__ == "__main__":
    # Print instructions if run directly
    if not dist.is_initialized():
        print("NCCL backend tests require torch.distributed initialization.")
        print("Run via: torchrun --nproc_per_node=2 -m pytest tests/test_nccl_backend.py")
        sys.exit(1)

    pytest.main([__file__, "-v"])
