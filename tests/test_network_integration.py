"""Integration tests for multi-collective composition and RLHF scenarios.

These tests verify that multiple collectives can be composed and simulated
together, and demonstrate real-world use cases like multi-node RLHF training.
"""

from syssim import (
    HierarchicalTopology,
    LogGPParams,
    # Config
    NetworkParams,
    NVLinkMeshTopology,
    # Network simulator
    allreduce,
    broadcast,
    simulate,
)


class TestMultiCollectiveComposition:
    """Test composing multiple collectives in a single simulation."""

    def test_sequential_collectives(self):
        """Sequential collectives work correctly with dependencies."""
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        topo = NVLinkMeshTopology(8, 25e9, 12)

        # First: allreduce
        ar_ops = allreduce([0, 1, 2, 3, 4, 5, 6, 7], 1e9, "ar")

        # Second: broadcast (depends on allreduce)
        bc_ops = broadcast([0, 1, 2, 3, 4, 5, 6, 7], 500e6, root=0, tag_prefix="bc")

        # Add dependency: All broadcast ops depend on ALL allreduce ops
        # (Conservative: ensures broadcast starts after allreduce finishes)
        for bc_op in bc_ops:
            if not bc_op.deps:  # Only add to ops with no existing deps
                bc_op.deps.extend(ar_ops[-8:])  # Last 8 ops (last step of allreduce)

        # Simulate together
        all_ops = ar_ops + bc_ops
        simulate(all_ops, topo, loggp)

        # Broadcast should start after allreduce finishes
        ar_finish = max(op.finish_time for op in ar_ops)
        bc_start = min(op.start_time for op in bc_ops)

        assert bc_start >= ar_finish - 1e-9  # Allow floating-point tolerance

    def test_concurrent_collectives_with_overlap(self):
        """Concurrent collectives on overlapping ranks show contention."""
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        topo = NVLinkMeshTopology(8, 25e9, 12)

        # Actor allreduce (ranks 0-5)
        actor_ar = allreduce([0, 1, 2, 3, 4, 5], 2e9, "actor")

        # Critic allreduce (ranks 2-7) - overlaps with actor on ranks 2-5
        critic_ar = allreduce([2, 3, 4, 5, 6, 7], 1e9, "critic")

        # Simulate concurrently (no dependencies)
        all_ops = actor_ar + critic_ar
        result = simulate(all_ops, topo, loggp)

        # Both should complete (no deadlock)
        assert result.makespan > 0

        # Overlapping ranks (2-5) show contention
        # Their finish time should be later than non-overlapping ranks
        rank_finish = result.per_rank_finish

        # Ranks 2-5 participate in both, should finish last
        overlap_finish = max(rank_finish[i] for i in [2, 3, 4, 5])

        # Ranks 0,1 only in actor, ranks 6,7 only in critic
        nonoverlap_finish = max(rank_finish[i] for i in [0, 1, 6, 7])

        # Overlapping ranks should take longer (due to double work)
        # This is a qualitative check
        assert overlap_finish >= nonoverlap_finish

    def test_three_collectives_pipeline(self):
        """Three collectives in a pipeline."""
        loggp = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        topo = NVLinkMeshTopology(8, 25e9, 12)

        # Stage 1: AllReduce
        ar_ops = allreduce([0, 1, 2, 3, 4, 5, 6, 7], 1e9, "ar")

        # Stage 2: Broadcast (depends on AllReduce)
        bc_ops = broadcast([0, 1, 2, 3, 4, 5, 6, 7], 500e6, root=0, tag_prefix="bc")
        for op in bc_ops:
            if not op.deps:
                op.deps.extend(ar_ops[-8:])  # Last step of allreduce

        # Stage 3: Second AllReduce (depends on Broadcast)
        ar2_ops = allreduce([0, 1, 2, 3, 4, 5, 6, 7], 1e9, "ar2")
        for op in ar2_ops:
            if not op.deps:
                op.deps.extend(bc_ops[-1:])  # Last op of broadcast

        # Simulate pipeline
        all_ops = ar_ops + bc_ops + ar2_ops
        simulate(all_ops, topo, loggp)

        # Verify ordering
        ar_finish = max(op.finish_time for op in ar_ops)
        bc_start = min(op.start_time for op in bc_ops)
        bc_finish = max(op.finish_time for op in bc_ops)
        ar2_start = min(op.start_time for op in ar2_ops)

        assert bc_start >= ar_finish - 1e-9
        assert ar2_start >= bc_finish - 1e-9


class TestMultiNodeRLHF:
    """Test multi-node RLHF scenarios with HierarchicalTopology."""

    def test_single_node_rlhf(self):
        """RLHF on single DGX node (8 GPUs)."""
        # Single-node: NVLink mesh
        loggp_nvlink = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        loggp_ib = LogGPParams(L=5e-6, o=10e-6, G=1 / (25e9))  # Unused on single node

        topo = HierarchicalTopology(
            num_nodes=1,
            gpus_per_node=8,
            nvlink_bandwidth=25e9,
            nvlink_count=12,
            ib_bandwidth=25e9,
            loggp_nvlink=loggp_nvlink,
            loggp_ib=loggp_ib,
        )

        # Actor model: ranks 0-3
        actor_ar = allreduce([0, 1, 2, 3], 2e9, "actor")

        # Critic model: ranks 4-7
        critic_ar = allreduce([4, 5, 6, 7], 1e9, "critic")

        # Simulate (actor and critic run concurrently)
        all_ops = actor_ar + critic_ar
        result = simulate(all_ops, topo)

        # Both should complete
        assert result.makespan > 0

        # All communication is intra-node (NVLink)
        for op in all_ops:
            path = topo.resolve_path(op.src, op.dst)
            if path:  # Non-empty path
                assert "nvlink" in path[0].name

    def test_multi_node_rlhf_separate_models(self):
        """RLHF with actor and critic on separate nodes."""
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

        # Actor: nodes 0-1 (ranks 0-15)
        actor_ranks = list(range(16))
        actor_ar = allreduce(actor_ranks, 4e9, "actor")

        # Critic: nodes 2-3 (ranks 16-31)
        critic_ranks = list(range(16, 32))
        critic_ar = allreduce(critic_ranks, 2e9, "critic")

        # Simulate (no overlap, no contention between models)
        all_ops = actor_ar + critic_ar
        result = simulate(all_ops, topo)

        # Both complete
        assert result.makespan > 0

        # Actor uses inter-node communication
        actor_inter_node = any(
            topo._rank_to_node(op.src) != topo._rank_to_node(op.dst) for op in actor_ar if op.src != op.dst
        )
        assert actor_inter_node  # Should have some inter-node ops

    def test_multi_node_rlhf_overlapping_nodes(self):
        """RLHF with actor and critic overlapping on node 1 (contention)."""
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

        # Actor: nodes 0-1 (ranks 0-15)
        actor_ranks = list(range(16))
        actor_ar = allreduce(actor_ranks, 4e9, "actor")

        # Critic: nodes 1-2 (ranks 8-23) - overlaps with actor on node 1
        critic_ranks = list(range(8, 24))
        critic_ar = allreduce(critic_ranks, 2e9, "critic")

        # Simulate (actor and critic run concurrently)
        all_ops = actor_ar + critic_ar
        result = simulate(all_ops, topo)

        # Node 1 ranks (8-15) participate in both, should show contention
        node1_finish = max(result.per_rank_finish[i] for i in range(8, 16))

        # Node 0 ranks (0-7) only in actor
        node0_finish = max(result.per_rank_finish[i] for i in range(0, 8))

        # Node 1 should take longer due to double work
        assert node1_finish >= node0_finish

    def test_network_params_integration(self):
        """NetworkParams can be created and used with HardwareInfo."""
        net_params = NetworkParams(
            num_nodes=2,
            gpus_per_node=8,
            nvlink_bandwidth=25e9,
            nvlink_count=12,
            ib_bandwidth=25e9,
            loggp_nvlink_L=1e-6,
            loggp_nvlink_o=5e-6,
            loggp_ib_L=5e-6,
            loggp_ib_o=10e-6,
        )

        # Can be attached to HardwareInfo (for future integration)
        from syssim import HardwareInfo

        hw = HardwareInfo(
            peak_tflops_mm=989.0,
            peak_tflops_math=989.0,
            peak_memory_bandwidth_gbps=3350.0,
            network=net_params,
        )

        assert hw.network.num_nodes == 2
        assert hw.network.gpus_per_node == 8
