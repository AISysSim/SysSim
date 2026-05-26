"""Unit tests for collective communication builders.

Tests cover:
- Correct number of ops generated
- Correct message sizes
- Correct root handling (broadcast, reduce, scatter, gather)
- Step structure and dependencies
- Error handling (invalid inputs)
"""

import pytest

from syssim.network import (
    allgather,
    allreduce,
    alltoall,
    broadcast,
    gather,
    reduce,
    reduce_scatter,
    scatter,
)


class TestAllreduce:
    """Test ring allreduce algorithm."""

    def test_allreduce_4_ranks(self):
        """4-rank allreduce generates correct number of ops."""
        ranks = [0, 1, 2, 3]
        ops = allreduce(ranks, total_size=1e9)

        # Ring: 2(P-1) = 2*3 = 6 steps, P=4 sends per step
        assert len(ops) == 4 * 6

    def test_allreduce_chunk_size(self):
        """Message size is total_size / P."""
        ranks = [0, 1, 2, 3]
        ops = allreduce(ranks, total_size=1e9)

        expected_chunk = 1e9 / 4
        for op in ops:
            assert op.size == expected_chunk

    def test_allreduce_tag(self):
        """Tags include prefix and step index."""
        ops = allreduce([0, 1], total_size=1e6, tag_prefix="test")

        assert all("test_step_" in op.tag for op in ops)

    def test_allreduce_single_rank(self):
        """Allreduce rejects single rank."""
        with pytest.raises(ValueError, match="at least 2 ranks"):
            allreduce([0], total_size=1e6)


class TestBroadcast:
    """Test binomial tree broadcast."""

    def test_broadcast_4_ranks(self):
        """4-rank broadcast has correct structure."""
        ranks = [0, 1, 2, 3]
        ops = broadcast(ranks, total_size=1e6, root=0)

        # Binomial tree: ⌈log₂ 4⌉ = 2 steps
        # Step 0: 0->1 (1 op)
        # Step 1: 0->2, 1->3 (2 ops)
        # Total: 3 ops
        assert len(ops) == 3

    def test_broadcast_message_size(self):
        """All messages have full total_size (not chunked)."""
        ops = broadcast([0, 1, 2, 3], total_size=1e6, root=0)

        for op in ops:
            assert op.size == 1e6

    def test_broadcast_root_sends_first(self):
        """Root rank sends in first step."""
        ops = broadcast([0, 1, 2, 3], total_size=1e6, root=0)

        # First op should be from root
        assert ops[0].src == 0

    def test_broadcast_non_zero_root(self):
        """Broadcast works with non-zero root."""
        ranks = [0, 1, 2, 3]
        ops = broadcast(ranks, total_size=1e6, root=2)

        # First op should be from root
        assert ops[0].src == 2

    def test_broadcast_invalid_root(self):
        """Broadcast rejects root not in ranks."""
        with pytest.raises(ValueError, match="root .* not in ranks"):
            broadcast([0, 1, 2], total_size=1e6, root=5)

    def test_broadcast_single_rank(self):
        """Broadcast rejects single rank."""
        with pytest.raises(ValueError, match="at least 2 ranks"):
            broadcast([0], total_size=1e6)


class TestReduce:
    """Test binomial tree reduce."""

    def test_reduce_4_ranks(self):
        """4-rank reduce has correct structure."""
        ranks = [0, 1, 2, 3]
        ops = reduce(ranks, total_size=1e6, root=0)

        # Binomial tree: ⌈log₂ 4⌉ = 2 steps
        # Step 0: 1->0, 3->2 (2 ops)
        # Step 1: 2->0 (1 op)
        # Total: 3 ops
        assert len(ops) == 3

    def test_reduce_final_send_to_root(self):
        """Last op sends to root."""
        ops = reduce([0, 1, 2, 3], total_size=1e6, root=0)

        # Last op should send to root
        assert ops[-1].dst == 0

    def test_reduce_non_zero_root(self):
        """Reduce works with non-zero root."""
        ops = reduce([0, 1, 2, 3], total_size=1e6, root=2)

        # Last op should send to root
        assert ops[-1].dst == 2

    def test_reduce_invalid_root(self):
        """Reduce rejects root not in ranks."""
        with pytest.raises(ValueError, match="root .* not in ranks"):
            reduce([0, 1, 2], total_size=1e6, root=5)


class TestReduceScatter:
    """Test ring reduce-scatter."""

    def test_reduce_scatter_4_ranks(self):
        """4-rank reduce-scatter generates correct number of ops."""
        ranks = [0, 1, 2, 3]
        ops = reduce_scatter(ranks, total_size=1e9)

        # Ring: (P-1) = 3 steps, P=4 sends per step
        assert len(ops) == 4 * 3

    def test_reduce_scatter_chunk_size(self):
        """Message size is total_size / P."""
        ops = reduce_scatter([0, 1, 2, 3], total_size=1e9)

        expected_chunk = 1e9 / 4
        for op in ops:
            assert op.size == expected_chunk


class TestAllgather:
    """Test ring allgather."""

    def test_allgather_4_ranks(self):
        """4-rank allgather generates correct number of ops."""
        ranks = [0, 1, 2, 3]
        ops = allgather(ranks, total_size=1e9)

        # Ring: (P-1) = 3 steps, P=4 sends per step
        assert len(ops) == 4 * 3

    def test_allgather_chunk_size(self):
        """Message size is total_size / P."""
        ops = allgather([0, 1, 2, 3], total_size=1e9)

        expected_chunk = 1e9 / 4
        for op in ops:
            assert op.size == expected_chunk


class TestAlltoAll:
    """Test direct alltoall with staggered pairings."""

    def test_alltoall_4_ranks(self):
        """4-rank alltoall generates correct number of ops."""
        ranks = [0, 1, 2, 3]
        ops = alltoall(ranks, total_size=1e9)

        # P-1 = 3 steps, P=4 sends per step
        assert len(ops) == 4 * 3

    def test_alltoall_chunk_size(self):
        """Message size is total_size / P."""
        ops = alltoall([0, 1, 2, 3], total_size=1e9)

        expected_chunk = 1e9 / 4
        for op in ops:
            assert op.size == expected_chunk

    def test_alltoall_no_self_sends(self):
        """No rank sends to itself."""
        ops = alltoall([0, 1, 2, 3], total_size=1e9)

        for op in ops:
            assert op.src != op.dst


class TestScatter:
    """Test flat tree scatter."""

    def test_scatter_4_ranks(self):
        """4-rank scatter generates P-1 ops."""
        ranks = [0, 1, 2, 3]
        ops = scatter(ranks, total_size=1e9, root=0)

        # Root sends to P-1 other ranks
        assert len(ops) == 3

    def test_scatter_all_from_root(self):
        """All ops send from root."""
        ops = scatter([0, 1, 2, 3], total_size=1e9, root=0)

        for op in ops:
            assert op.src == 0

    def test_scatter_chunk_size(self):
        """Message size is total_size / P."""
        ops = scatter([0, 1, 2, 3], total_size=1e9, root=0)

        expected_chunk = 1e9 / 4
        for op in ops:
            assert op.size == expected_chunk

    def test_scatter_non_zero_root(self):
        """Scatter works with non-zero root."""
        ops = scatter([0, 1, 2, 3], total_size=1e9, root=2)

        for op in ops:
            assert op.src == 2

    def test_scatter_serialization(self):
        """Scatter ops are serialized (each in separate step)."""
        ops = scatter([0, 1, 2, 3], total_size=1e9, root=0)

        # Each op (except first) should depend on previous op from same src
        for i in range(1, len(ops)):
            assert ops[i - 1] in ops[i].deps


class TestGather:
    """Test flat tree gather."""

    def test_gather_4_ranks(self):
        """4-rank gather generates P-1 ops."""
        ranks = [0, 1, 2, 3]
        ops = gather(ranks, total_size=1e9, root=0)

        # P-1 non-root ranks send to root
        assert len(ops) == 3

    def test_gather_all_to_root(self):
        """All ops send to root."""
        ops = gather([0, 1, 2, 3], total_size=1e9, root=0)

        for op in ops:
            assert op.dst == 0

    def test_gather_chunk_size(self):
        """Message size is total_size / P."""
        ops = gather([0, 1, 2, 3], total_size=1e9, root=0)

        expected_chunk = 1e9 / 4
        for op in ops:
            assert op.size == expected_chunk

    def test_gather_non_zero_root(self):
        """Gather works with non-zero root."""
        ops = gather([0, 1, 2, 3], total_size=1e9, root=2)

        for op in ops:
            assert op.dst == 2

    def test_gather_parallel(self):
        """Gather ops are parallel (no dependencies in single step)."""
        ops = gather([0, 1, 2, 3], total_size=1e9, root=0)

        # All ops should have no dependencies (parallel)
        for op in ops:
            assert op.deps == []


class TestCollectivesEdgeCases:
    """Test edge cases and error handling across collectives."""

    def test_2_ranks(self):
        """All collectives work with 2 ranks."""
        ranks = [0, 1]
        size = 1e6

        # Should not raise
        allreduce(ranks, size)
        broadcast(ranks, size, root=0)
        reduce(ranks, size, root=0)
        reduce_scatter(ranks, size)
        allgather(ranks, size)
        alltoall(ranks, size)
        scatter(ranks, size, root=0)
        gather(ranks, size, root=0)

    def test_8_ranks(self):
        """All collectives work with 8 ranks."""
        ranks = list(range(8))
        size = 1e9

        # Should not raise
        ops_ar = allreduce(ranks, size)
        ops_bc = broadcast(ranks, size, root=0)
        ops_red = reduce(ranks, size, root=0)
        ops_rs = reduce_scatter(ranks, size)
        ops_ag = allgather(ranks, size)
        ops_a2a = alltoall(ranks, size)
        ops_scat = scatter(ranks, size, root=0)
        ops_gath = gather(ranks, size, root=0)

        # Verify op counts
        assert len(ops_ar) == 8 * 2 * (8 - 1)  # 2(P-1) steps, P ops per step
        assert len(ops_bc) == 7  # Binomial tree: 7 sends for 8 ranks
        assert len(ops_red) == 7  # Binomial tree: 7 receives for 8 ranks
        assert len(ops_rs) == 8 * (8 - 1)  # (P-1) steps, P ops per step
        assert len(ops_ag) == 8 * (8 - 1)  # (P-1) steps, P ops per step
        assert len(ops_a2a) == 8 * (8 - 1)  # (P-1) steps, P ops per step
        assert len(ops_scat) == 7  # Root sends to P-1 others
        assert len(ops_gath) == 7  # P-1 send to root

    def test_non_contiguous_ranks(self):
        """Collectives work with non-contiguous rank IDs."""
        ranks = [5, 10, 15, 20]
        size = 1e6

        # Should not raise
        ops = allreduce(ranks, size)
        assert len(ops) == 4 * 2 * (4 - 1)

        # Check that actual rank IDs are used
        all_srcs = {op.src for op in ops}
        all_dsts = {op.dst for op in ops}
        assert all_srcs == {5, 10, 15, 20}
        assert all_dsts == {5, 10, 15, 20}
