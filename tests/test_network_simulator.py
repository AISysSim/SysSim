"""Unit tests for network simulation engine.

Tests cover:
- Single message transfer
- Bandwidth sharing (contention)
- LogGP overhead
- Dependency ordering
- Edge cases (empty, zero-size messages)
"""

import pytest

from syssim.network import (
    FullyConnectedTopology,
    LogGPParams,
    Op,
    Resource,
    Topology,
    simulate,
)


class TestSimulateBasic:
    """Test basic simulation functionality."""

    def test_empty_ops(self):
        """Empty ops list returns zero makespan."""
        topo = FullyConnectedTopology(2, 1e9)
        loggp = LogGPParams(L=0, o=0, G=1 / 1e9)

        result = simulate([], topo, loggp)

        assert result.makespan == 0.0
        assert result.ops == []
        assert result.per_rank_finish == {}

    def test_single_message(self):
        """Single message transfers at full bandwidth."""
        # Setup: 1 GB/s link, 1 MB message
        topo = FullyConnectedTopology(2, 1e9)
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)

        # Create single op
        op = Op(src=0, dst=1, size=1e6)
        result = simulate([op], topo, loggp)

        # Expected time: (1e6 bytes) / (1e9 bytes/s) + alpha
        # = 1e-3 s + (1e-6 + 2*5e-6) = 1e-3 + 1.1e-5 s
        expected_time = 1e-3 + 1.1e-5

        assert abs(result.makespan - expected_time) < 1e-9
        assert abs(op.finish_time - expected_time) < 1e-9
        assert op.start_time == 0.0

    def test_loggp_overhead(self):
        """LogGP overhead (alpha = L + 2*o) added to finish time."""
        topo = FullyConnectedTopology(2, 1e9)
        loggp = LogGPParams(L=10e-6, o=5e-6, G=1 / 1e9)

        # Small message (10 KB) -> transfer time negligible
        op = Op(src=0, dst=1, size=1e4)
        result = simulate([op], topo, loggp)

        # Expected: 1e4 / 1e9 + (10e-6 + 2*5e-6) = 1e-5 + 2e-5 = 3e-5
        expected_time = 1e-5 + 2e-5

        assert abs(result.makespan - expected_time) < 1e-9

    def test_zero_size_message(self):
        """Zero-size message completes with only LogGP overhead."""
        topo = FullyConnectedTopology(2, 1e9)
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)

        op = Op(src=0, dst=1, size=0.0)
        result = simulate([op], topo, loggp)

        # Expected: only alpha = 1e-6 + 2*5e-6 = 1.1e-5
        expected_time = 1.1e-5

        assert abs(result.makespan - expected_time) < 1e-9


class TestSimulateBandwidthSharing:
    """Test max-min fair bandwidth sharing."""

    def test_two_concurrent_messages_shared_link(self):
        """Two concurrent messages on shared link get half bandwidth each.

        Setup: Simple topology where 0->2 and 1->2 share the same link to rank 2.
        """

        # Custom topology: all messages to rank 2 share one resource
        class SharedLinkTopology(Topology):
            def __init__(self):
                self.link_to_2 = Resource("link_to_2", 1e9)  # 1 GB/s
                self.other_link = Resource("other", 1e9)

            def resolve_path(self, src, dst):
                if dst == 2:
                    return [self.link_to_2]
                else:
                    return [self.other_link]

            def all_resources(self):
                return [self.link_to_2, self.other_link]

        topo = SharedLinkTopology()
        loggp = LogGPParams(L=0, o=0, G=1 / 1e9)

        # Two concurrent messages to rank 2
        op1 = Op(src=0, dst=2, size=1e6)  # 1 MB
        op2 = Op(src=1, dst=2, size=1e6)  # 1 MB

        result = simulate([op1, op2], topo, loggp)

        # Expected: each gets 0.5 GB/s, so 1 MB takes 2 ms
        expected_time = 1e6 / 0.5e9  # 2e-3 seconds

        assert abs(result.makespan - expected_time) < 1e-9
        assert abs(op1.finish_time - expected_time) < 1e-9
        assert abs(op2.finish_time - expected_time) < 1e-9

    def test_two_messages_different_links(self):
        """Two concurrent messages on different links don't interfere."""
        topo = FullyConnectedTopology(4, 1e9)
        loggp = LogGPParams(L=0, o=0, G=1 / 1e9)

        # Different pairs -> different resources
        op1 = Op(src=0, dst=1, size=1e6)
        op2 = Op(src=2, dst=3, size=1e6)

        result = simulate([op1, op2], topo, loggp)

        # Expected: each gets full bandwidth, 1 ms
        expected_time = 1e6 / 1e9  # 1e-3 seconds

        assert abs(result.makespan - expected_time) < 1e-9
        assert abs(op1.finish_time - expected_time) < 1e-9
        assert abs(op2.finish_time - expected_time) < 1e-9


class TestSimulateDependencies:
    """Test dependency ordering."""

    def test_serial_dependency(self):
        """Op with dependency starts after dependency finishes."""
        topo = FullyConnectedTopology(3, 1e9)
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)

        # Create dependency chain: op1 -> op2
        op1 = Op(src=0, dst=1, size=1e6)  # 1 MB
        op2 = Op(src=1, dst=2, size=1e6, deps=[op1])  # 1 MB, depends on op1

        result = simulate([op1, op2], topo, loggp)

        # op1 finishes at: 1e-3 + 1.1e-5
        # op2 starts at: op1.finish_time
        # op2 finishes at: op1.finish_time + 1e-3 + 1.1e-5

        transfer_time = 1e6 / 1e9
        alpha = 1.1e-5

        assert abs(op1.finish_time - (transfer_time + alpha)) < 1e-9
        assert op2.start_time >= op1.finish_time - 1e-9
        assert abs(op2.finish_time - (2 * transfer_time + 2 * alpha)) < 1e-9
        assert abs(result.makespan - (2 * transfer_time + 2 * alpha)) < 1e-9

    def test_multiple_dependencies(self):
        """Op waits for all dependencies to complete."""
        topo = FullyConnectedTopology(4, 1e9)
        loggp = LogGPParams(L=0, o=0, G=1 / 1e9)

        # Create diamond dependency:
        #   op1 (0->1, 1 MB)
        #   op2 (0->2, 2 MB)
        #   op3 (1->3, depends on op1 and op2)

        op1 = Op(src=0, dst=1, size=1e6)
        op2 = Op(src=0, dst=2, size=2e6)
        op3 = Op(src=1, dst=3, size=1e6, deps=[op1, op2])

        result = simulate([op1, op2, op3], topo, loggp)

        # op1 finishes at 1 ms
        # op2 finishes at 2 ms (slower)
        # op3 starts at 2 ms (waits for op2)

        assert abs(op1.finish_time - 1e-3) < 1e-9
        assert abs(op2.finish_time - 2e-3) < 1e-9
        assert op3.start_time >= op2.finish_time - 1e-9
        assert abs(result.makespan - 3e-3) < 1e-9


class TestSimulateResultFields:
    """Test SimulationResult fields."""

    def test_per_rank_finish(self):
        """per_rank_finish tracks latest finish time per rank."""
        topo = FullyConnectedTopology(4, 1e9)
        loggp = LogGPParams(L=0, o=0, G=1 / 1e9)

        # Rank 0 sends twice
        op1 = Op(src=0, dst=1, size=1e6)  # finishes at 1 ms
        op2 = Op(src=0, dst=2, size=2e6, deps=[op1])  # finishes at 3 ms

        result = simulate([op1, op2], topo, loggp)

        # Rank 0 participates in both (as src), latest is 3 ms
        assert abs(result.per_rank_finish[0] - 3e-3) < 1e-9

        # Rank 1 receives at 1 ms
        assert abs(result.per_rank_finish[1] - 1e-3) < 1e-9

        # Rank 2 receives at 3 ms
        assert abs(result.per_rank_finish[2] - 3e-3) < 1e-9

    def test_makespan_is_max_finish(self):
        """Makespan equals max finish time across all ops."""
        topo = FullyConnectedTopology(4, 1e9)
        loggp = LogGPParams(L=0, o=0, G=1 / 1e9)

        ops = [
            Op(src=0, dst=1, size=1e6),  # 1 ms
            Op(src=2, dst=3, size=3e6),  # 3 ms
        ]

        result = simulate(ops, topo, loggp)

        assert abs(result.makespan - 3e-3) < 1e-9


class TestSimulateEdgeCases:
    """Test edge cases and error handling."""

    def test_self_send(self):
        """Message from rank to itself completes instantly."""
        topo = FullyConnectedTopology(2, 1e9)
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)

        # src == dst
        op = Op(src=0, dst=0, size=1e6)
        result = simulate([op], topo, loggp)

        # Path is empty, transfer is instant, only LogGP overhead
        expected_time = loggp.alpha

        assert abs(result.makespan - expected_time) < 1e-9

    def test_large_message(self):
        """Large message transfer works correctly."""
        topo = FullyConnectedTopology(2, 100e9)  # 100 GB/s
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 100e9)

        # 10 GB message
        op = Op(src=0, dst=1, size=10e9)
        result = simulate([op], topo, loggp)

        # Expected: 10e9 / 100e9 + alpha = 0.1 + 1.1e-5
        expected_time = 0.1 + 1.1e-5

        assert abs(result.makespan - expected_time) < 1e-6

    def test_no_loggp_provided(self):
        """Simulate raises error if no LogGP params provided."""
        topo = FullyConnectedTopology(2, 1e9)

        op = Op(src=0, dst=1, size=1e6)

        with pytest.raises(ValueError, match="No LogGP params"):
            simulate([op], topo, loggp=None)

    def test_op_specific_loggp(self):
        """Op-specific LogGP params override global params."""
        topo = FullyConnectedTopology(2, 1e9)
        global_loggp = LogGPParams(L=10e-6, o=10e-6, G=1 / 1e9)
        op_loggp = LogGPParams(L=1e-6, o=1e-6, G=1 / 1e9)

        # Op with specific LogGP
        op = Op(src=0, dst=1, size=1e6)
        op.loggp = op_loggp

        result = simulate([op], topo, global_loggp)

        # Should use op_loggp (alpha = 3e-6), not global_loggp (alpha = 30e-6)
        expected_time = 1e6 / 1e9 + 3e-6

        assert abs(result.makespan - expected_time) < 1e-9
