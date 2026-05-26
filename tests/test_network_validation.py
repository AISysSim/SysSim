"""Validation tests comparing simulation against analytical formulas.

These tests verify that the simulator produces results matching closed-form
performance formulas for collective algorithms on FullyConnectedTopology
(where there's no contention).

Target: <1e-5 relative error for all collectives (tolerance accounts for
floating-point rounding in the (m-1)*G term of LogGP formula)
"""

from syssim.network import (
    FullyConnectedTopology,
    LogGPParams,
    allgather,
    allreduce,
    alltoall,
    broadcast,
    gather,
    reduce,
    reduce_scatter,
    scatter,
    simulate,
)
from syssim.network.validation import (
    validate_allgather,
    validate_allreduce,
    validate_alltoall,
    validate_broadcast,
    validate_gather,
    validate_reduce,
    validate_reduce_scatter,
    validate_scatter,
)


class TestAnalyticalValidation:
    """Test simulation results against analytical formulas."""

    def test_allreduce_4_ranks(self):
        """AllReduce on 4 ranks matches analytical formula."""
        num_ranks = 4
        total_size = 1e6  # 1 MB
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)
        topo = FullyConnectedTopology(num_ranks, 1e9)

        ops = allreduce(list(range(num_ranks)), total_size)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_allreduce(num_ranks, total_size, loggp, result.makespan)

        assert is_valid, f"Error {error:.2e} exceeds tolerance (analytical={analytical:.6e})"

    def test_allreduce_8_ranks(self):
        """AllReduce on 8 ranks matches analytical formula."""
        num_ranks = 8
        total_size = 1e9  # 1 GB
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 100e9)
        topo = FullyConnectedTopology(num_ranks, 100e9)

        ops = allreduce(list(range(num_ranks)), total_size)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_allreduce(num_ranks, total_size, loggp, result.makespan)

        assert is_valid, f"Error {error:.2e} exceeds tolerance"
        assert error < 1e-5

    def test_broadcast_4_ranks(self):
        """Broadcast on 4 ranks matches analytical formula."""
        num_ranks = 4
        total_size = 1e6  # 1 MB
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)
        topo = FullyConnectedTopology(num_ranks, 1e9)

        ops = broadcast(list(range(num_ranks)), total_size, root=0)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_broadcast(num_ranks, total_size, loggp, result.makespan)

        assert is_valid, f"Error {error:.2e} exceeds tolerance"

    def test_broadcast_8_ranks(self):
        """Broadcast on 8 ranks matches analytical formula."""
        num_ranks = 8
        total_size = 1e9
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 100e9)
        topo = FullyConnectedTopology(num_ranks, 100e9)

        ops = broadcast(list(range(num_ranks)), total_size, root=0)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_broadcast(num_ranks, total_size, loggp, result.makespan)

        assert is_valid
        assert error < 1e-5

    def test_reduce_4_ranks(self):
        """Reduce on 4 ranks matches analytical formula."""
        num_ranks = 4
        total_size = 1e6
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)
        topo = FullyConnectedTopology(num_ranks, 1e9)

        ops = reduce(list(range(num_ranks)), total_size, root=0)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_reduce(num_ranks, total_size, loggp, result.makespan)

        assert is_valid
        assert error < 1e-5

    def test_reduce_scatter_4_ranks(self):
        """ReduceScatter on 4 ranks matches analytical formula."""
        num_ranks = 4
        total_size = 1e6
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)
        topo = FullyConnectedTopology(num_ranks, 1e9)

        ops = reduce_scatter(list(range(num_ranks)), total_size)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_reduce_scatter(num_ranks, total_size, loggp, result.makespan)

        assert is_valid
        assert error < 1e-5

    def test_allgather_4_ranks(self):
        """AllGather on 4 ranks matches analytical formula."""
        num_ranks = 4
        total_size = 1e6
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)
        topo = FullyConnectedTopology(num_ranks, 1e9)

        ops = allgather(list(range(num_ranks)), total_size)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_allgather(num_ranks, total_size, loggp, result.makespan)

        assert is_valid
        assert error < 1e-5

    def test_alltoall_4_ranks(self):
        """AlltoAll on 4 ranks matches analytical formula."""
        num_ranks = 4
        total_size = 1e6
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)
        topo = FullyConnectedTopology(num_ranks, 1e9)

        ops = alltoall(list(range(num_ranks)), total_size)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_alltoall(num_ranks, total_size, loggp, result.makespan)

        assert is_valid
        assert error < 1e-5

    def test_scatter_4_ranks(self):
        """Scatter on 4 ranks matches analytical formula."""
        num_ranks = 4
        total_size = 1e6
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)
        topo = FullyConnectedTopology(num_ranks, 1e9)

        ops = scatter(list(range(num_ranks)), total_size, root=0)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_scatter(num_ranks, total_size, loggp, result.makespan)

        assert is_valid
        assert error < 1e-5

    def test_gather_4_ranks(self):
        """Gather on 4 ranks matches analytical formula."""
        num_ranks = 4
        total_size = 1e6
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 1e9)
        topo = FullyConnectedTopology(num_ranks, 1e9)

        ops = gather(list(range(num_ranks)), total_size, root=0)
        result = simulate(ops, topo, loggp)

        is_valid, analytical, error = validate_gather(num_ranks, total_size, loggp, 1e9, result.makespan)

        assert is_valid
        assert error < 1e-5

    def test_all_collectives_match(self):
        """All collectives match analytical formulas on larger problem."""
        num_ranks = 16
        total_size = 1e9  # 1 GB
        loggp = LogGPParams(L=2e-6, o=10e-6, G=1 / 25e9)
        topo = FullyConnectedTopology(num_ranks, 25e9)

        # Test each collective
        collectives = [
            (allreduce, validate_allreduce, "AllReduce"),
            (broadcast, validate_broadcast, "Broadcast"),
            (reduce, validate_reduce, "Reduce"),
            (reduce_scatter, validate_reduce_scatter, "ReduceScatter"),
            (allgather, validate_allgather, "AllGather"),
            (alltoall, validate_alltoall, "AlltoAll"),
            (scatter, validate_scatter, "Scatter"),
        ]

        for collective_fn, validate_fn, name in collectives:
            # Build and simulate
            if name in ["Broadcast", "Reduce", "Scatter"]:
                ops = collective_fn(list(range(num_ranks)), total_size, root=0)
            else:
                ops = collective_fn(list(range(num_ranks)), total_size)

            result = simulate(ops, topo, loggp)

            # Validate
            is_valid, analytical, error = validate_fn(num_ranks, total_size, loggp, result.makespan)

            assert is_valid, f"{name} error {error:.2e} exceeds tolerance"
            assert error < 1e-5, f"{name} relative error too high"

        # Test gather separately (different signature)
        ops = gather(list(range(num_ranks)), total_size, root=0)
        result = simulate(ops, topo, loggp)
        is_valid, analytical, error = validate_gather(num_ranks, total_size, loggp, 25e9, result.makespan)
        assert is_valid and error < 1e-5


class TestContrast:
    """Test that contention DOES affect performance (sanity check)."""

    def test_switch_slower_than_fully_connected(self):
        """Collectives on SwitchTopology are slower due to fabric contention."""
        from syssim.network import SwitchTopology

        num_ranks = 8
        total_size = 1e9
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / 25e9)

        # Fully connected (no contention)
        topo_fc = FullyConnectedTopology(num_ranks, 25e9)
        ops_fc = allreduce(list(range(num_ranks)), total_size)
        result_fc = simulate(ops_fc, topo_fc, loggp)

        # Switch with limited fabric bandwidth (creates contention)
        # Ring allreduce has 8 concurrent messages per step, so fabric bandwidth
        # of 100e9 means each message gets ~12.5e9 (half the link bandwidth)
        topo_switch = SwitchTopology(num_ranks, 25e9, 100e9)
        ops_switch = allreduce(list(range(num_ranks)), total_size)
        result_switch = simulate(ops_switch, topo_switch, loggp)

        # Switch should be slower due to contention
        assert result_switch.makespan > result_fc.makespan

    def test_hierarchical_intra_vs_inter(self):
        """Inter-node communication is slower than intra-node on HierarchicalTopology."""
        from syssim.network import HierarchicalTopology

        loggp_nvlink = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        loggp_ib = LogGPParams(L=5e-6, o=10e-6, G=1 / (25e9))

        topo = HierarchicalTopology(
            num_nodes=4,
            gpus_per_node=8,
            nvlink_bandwidth=25e9,
            nvlink_count=12,
            ib_bandwidth=25e9,
            loggp_nvlink=loggp_nvlink,
            loggp_ib=loggp_ib,
        )

        total_size = 1e9

        # Intra-node allreduce (ranks 0-7, all on node 0)
        ops_intra = allreduce(list(range(8)), total_size)
        result_intra = simulate(ops_intra, topo)

        # Inter-node allreduce (ranks 0-15, spans nodes 0-1)
        ops_inter = allreduce(list(range(16)), total_size)
        result_inter = simulate(ops_inter, topo)

        # Inter-node should be slower (more ranks + slower IB links)
        assert result_inter.makespan > result_intra.makespan
