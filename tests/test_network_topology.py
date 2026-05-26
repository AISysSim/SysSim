"""Unit tests for network topology implementations.

Tests cover:
- Resource validation
- FullyConnectedTopology path resolution and resource uniqueness
- (Additional topologies added in Phase 4)
"""

import pytest

from syssim.network import (
    FullyConnectedTopology,
    HierarchicalTopology,
    LogGPParams,
    NVLinkMeshTopology,
    Resource,
    RingTopology,
    SwitchTopology,
)


class TestResource:
    """Test Resource dataclass validation and immutability."""

    def test_resource_creation(self):
        """Resource can be created with valid parameters."""
        r = Resource("test_link", 25e9)
        assert r.name == "test_link"
        assert r.bandwidth == 25e9

    def test_resource_immutable(self):
        """Resource is frozen (immutable)."""
        r = Resource("test", 1e9)
        with pytest.raises(Exception):  # FrozenInstanceError
            r.bandwidth = 2e9

    def test_resource_negative_bandwidth(self):
        """Resource rejects negative bandwidth."""
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            Resource("bad", -1.0)

    def test_resource_zero_bandwidth(self):
        """Resource rejects zero bandwidth."""
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            Resource("bad", 0.0)


class TestFullyConnectedTopology:
    """Test FullyConnectedTopology path resolution and properties."""

    def test_creation(self):
        """Topology can be created with valid parameters."""
        topo = FullyConnectedTopology(num_ranks=4, link_bandwidth=25e9)
        assert topo.num_ranks == 4
        assert topo.link_bandwidth == 25e9

    def test_creation_single_rank(self):
        """Topology rejects single rank (need at least 2 for communication)."""
        with pytest.raises(ValueError, match="num_ranks must be >= 2"):
            FullyConnectedTopology(num_ranks=1, link_bandwidth=25e9)

    def test_creation_negative_bandwidth(self):
        """Topology rejects negative bandwidth."""
        with pytest.raises(ValueError, match="link_bandwidth must be positive"):
            FullyConnectedTopology(num_ranks=4, link_bandwidth=-1.0)

    def test_resolve_path_same_rank(self):
        """Path from rank to itself is empty."""
        topo = FullyConnectedTopology(4, 25e9)
        path = topo.resolve_path(0, 0)
        assert path == []

    def test_resolve_path_different_ranks(self):
        """Path between different ranks returns single dedicated link."""
        topo = FullyConnectedTopology(4, 25e9)
        path = topo.resolve_path(0, 3)

        assert len(path) == 1
        assert path[0].name == "link_0->3"
        assert path[0].bandwidth == 25e9

    def test_resolve_path_out_of_range_src(self):
        """resolve_path rejects out-of-range src."""
        topo = FullyConnectedTopology(4, 25e9)
        with pytest.raises(ValueError, match="src .* out of range"):
            topo.resolve_path(5, 0)

    def test_resolve_path_out_of_range_dst(self):
        """resolve_path rejects out-of-range dst."""
        topo = FullyConnectedTopology(4, 25e9)
        with pytest.raises(ValueError, match="dst .* out of range"):
            topo.resolve_path(0, 5)

    def test_path_uniqueness(self):
        """Different (src, dst) pairs have unique resources (no contention)."""
        topo = FullyConnectedTopology(4, 25e9)

        path_01 = topo.resolve_path(0, 1)
        path_02 = topo.resolve_path(0, 2)
        path_23 = topo.resolve_path(2, 3)

        # All paths should use different Resource objects
        assert path_01[0].name != path_02[0].name
        assert path_01[0].name != path_23[0].name
        assert path_02[0].name != path_23[0].name

        # Resource objects should be different (no sharing)
        assert path_01[0] is not path_02[0]
        assert path_01[0] is not path_23[0]

    def test_path_directionality(self):
        """Forward and reverse paths use different resources."""
        topo = FullyConnectedTopology(4, 25e9)

        path_01 = topo.resolve_path(0, 1)
        path_10 = topo.resolve_path(1, 0)

        assert path_01[0].name == "link_0->1"
        assert path_10[0].name == "link_1->0"
        assert path_01[0] is not path_10[0]

    def test_all_resources_count(self):
        """all_resources returns P*(P-1) resources."""
        topo = FullyConnectedTopology(num_ranks=4, link_bandwidth=25e9)
        resources = topo.all_resources()

        # 4 ranks: 4 * 3 = 12 directional links
        assert len(resources) == 12

    def test_all_resources_uniqueness(self):
        """All resources are unique objects."""
        topo = FullyConnectedTopology(4, 25e9)
        resources = topo.all_resources()

        # Check all names are unique
        names = [r.name for r in resources]
        assert len(names) == len(set(names))

        # Check all are different objects
        for i, r1 in enumerate(resources):
            for j, r2 in enumerate(resources):
                if i != j:
                    assert r1 is not r2

    def test_all_resources_bandwidth(self):
        """All resources have correct bandwidth."""
        topo = FullyConnectedTopology(4, 100e9)
        resources = topo.all_resources()

        for r in resources:
            assert r.bandwidth == 100e9

    def test_path_consistency(self):
        """Calling resolve_path multiple times returns same Resource object."""
        topo = FullyConnectedTopology(4, 25e9)

        path1 = topo.resolve_path(0, 2)
        path2 = topo.resolve_path(0, 2)

        # Should return same Resource object (not just equal, but identical)
        assert path1[0] is path2[0]


class TestRingTopology:
    """Test RingTopology path resolution and shortest-path routing."""

    def test_creation(self):
        """Ring topology can be created with valid parameters."""
        topo = RingTopology(num_ranks=8, link_bandwidth=25e9)
        assert topo.num_ranks == 8
        assert topo.link_bandwidth == 25e9

    def test_forward_path(self):
        """Short forward path uses consecutive hops."""
        topo = RingTopology(8, 25e9)
        path = topo.resolve_path(0, 2)

        # Should go 0->1->2 (2 hops)
        assert len(path) == 2
        assert "ring_0->1" in path[0].name
        assert "ring_1->2" in path[1].name

    def test_backward_path(self):
        """Long forward path uses backward shortcut."""
        topo = RingTopology(8, 25e9)
        path = topo.resolve_path(0, 7)

        # Backward (0->7) is shorter than forward (0->1->...->7)
        # Forward would be 7 hops, backward is 1 hop
        assert len(path) == 1
        assert "ring_0->7" in path[0].name

    def test_halfway_point(self):
        """Path to opposite side has equal forward/backward distance."""
        topo = RingTopology(8, 25e9)
        path = topo.resolve_path(0, 4)

        # Both directions are 4 hops, algorithm picks forward
        assert len(path) == 4

    def test_same_rank(self):
        """Path to self is empty."""
        topo = RingTopology(8, 25e9)
        assert topo.resolve_path(0, 0) == []

    def test_all_resources_count(self):
        """Ring has 2*P directional resources (P physical links, 2 directions)."""
        topo = RingTopology(4, 25e9)
        resources = topo.all_resources()

        # 4 physical links, 2 directions each = 8 resources
        assert len(resources) == 8


class TestSwitchTopology:
    """Test SwitchTopology with shared switch fabric."""

    def test_creation(self):
        """Switch topology can be created."""
        topo = SwitchTopology(num_ranks=8, link_bandwidth=25e9, switch_bandwidth=200e9)
        assert topo.num_ranks == 8
        assert topo.link_bandwidth == 25e9
        assert topo.switch_bandwidth == 200e9

    def test_path_structure(self):
        """Path goes uplink -> switch -> downlink."""
        topo = SwitchTopology(8, 25e9, 200e9)
        path = topo.resolve_path(0, 3)

        assert len(path) == 3
        assert "uplink_0" in path[0].name
        assert "switch_fabric" in path[1].name
        assert "downlink_3" in path[2].name

    def test_switch_fabric_shared(self):
        """All paths share same switch fabric resource."""
        topo = SwitchTopology(8, 25e9, 200e9)

        path1 = topo.resolve_path(0, 1)
        path2 = topo.resolve_path(2, 3)

        # Middle resource (switch fabric) should be identical object
        assert path1[1] is path2[1]
        assert path1[1].name == "switch_fabric"

    def test_uplinks_not_shared(self):
        """Different sources use different uplinks."""
        topo = SwitchTopology(8, 25e9, 200e9)

        path1 = topo.resolve_path(0, 5)
        path2 = topo.resolve_path(1, 5)

        # Uplinks are different
        assert path1[0] is not path2[0]

    def test_all_resources_count(self):
        """Switch has P uplinks + 1 fabric + P downlinks."""
        topo = SwitchTopology(4, 25e9, 200e9)
        resources = topo.all_resources()

        # 4 uplinks + 1 switch + 4 downlinks = 9
        assert len(resources) == 9


class TestNVLinkMeshTopology:
    """Test NVLinkMeshTopology with aggregated bandwidth."""

    def test_creation(self):
        """NVLink mesh can be created."""
        topo = NVLinkMeshTopology(8, nvlink_bandwidth=25e9, links_per_pair=12)
        assert topo.num_gpus == 8
        assert topo.nvlink_bandwidth == 25e9
        assert topo.links_per_pair == 12

    def test_direct_path(self):
        """Path is single direct link."""
        topo = NVLinkMeshTopology(8, 25e9, 12)
        path = topo.resolve_path(0, 3)

        assert len(path) == 1
        assert "nvlink_0->3" in path[0].name

    def test_aggregated_bandwidth(self):
        """Bandwidth is aggregated across multiple links."""
        topo = NVLinkMeshTopology(8, 25e9, 12)
        path = topo.resolve_path(0, 3)

        # 12 links * 25 GB/s = 300 GB/s
        assert path[0].bandwidth == 300e9

    def test_no_cross_pair_contention(self):
        """Different pairs use different resources."""
        topo = NVLinkMeshTopology(8, 25e9, 12)

        path1 = topo.resolve_path(0, 1)
        path2 = topo.resolve_path(2, 3)

        # Should be different resource objects
        assert path1[0] is not path2[0]

    def test_all_resources_count(self):
        """Mesh has P*(P-1) directional resources."""
        topo = NVLinkMeshTopology(4, 25e9, 12)
        resources = topo.all_resources()

        # 4 * 3 = 12
        assert len(resources) == 12


class TestHierarchicalTopology:
    """Test HierarchicalTopology with NVLink + InfiniBand layers."""

    def test_creation(self):
        """Hierarchical topology can be created."""
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

        assert topo.num_nodes == 4
        assert topo.gpus_per_node == 8
        assert topo.total_ranks == 32

    def test_intra_node_path(self):
        """Intra-node path uses NVLink mesh."""
        loggp_nvlink = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        loggp_ib = LogGPParams(L=5e-6, o=10e-6, G=1 / (25e9))

        topo = HierarchicalTopology(4, 8, 25e9, 12, 25e9, loggp_nvlink, loggp_ib)

        # Rank 0 (node 0, GPU 0) -> Rank 7 (node 0, GPU 7)
        path = topo.resolve_path(0, 7)

        # Should be single NVLink
        assert len(path) == 1
        assert "nvlink" in path[0].name

    def test_inter_node_path(self):
        """Inter-node path uses InfiniBand."""
        loggp_nvlink = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        loggp_ib = LogGPParams(L=5e-6, o=10e-6, G=1 / (25e9))

        topo = HierarchicalTopology(4, 8, 25e9, 12, 25e9, loggp_nvlink, loggp_ib)

        # Rank 0 (node 0) -> Rank 8 (node 1)
        path = topo.resolve_path(0, 8)

        # Should be uplink -> fabric -> downlink
        assert len(path) == 3
        assert "ib_uplink_0" in path[0].name
        assert "ib_fabric" in path[1].name
        assert "ib_downlink_1" in path[2].name

    def test_get_loggp_intra_node(self):
        """get_loggp returns NVLink params for intra-node."""
        loggp_nvlink = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        loggp_ib = LogGPParams(L=5e-6, o=10e-6, G=1 / (25e9))

        topo = HierarchicalTopology(4, 8, 25e9, 12, 25e9, loggp_nvlink, loggp_ib)

        params = topo.get_loggp(0, 7)  # Same node

        assert params.L == 1e-6
        assert params.o == 5e-6

    def test_get_loggp_inter_node(self):
        """get_loggp returns InfiniBand params for inter-node."""
        loggp_nvlink = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        loggp_ib = LogGPParams(L=5e-6, o=10e-6, G=1 / (25e9))

        topo = HierarchicalTopology(4, 8, 25e9, 12, 25e9, loggp_nvlink, loggp_ib)

        params = topo.get_loggp(0, 8)  # Different nodes

        assert params.L == 5e-6
        assert params.o == 10e-6

    def test_rank_mapping(self):
        """Rank mapping works correctly."""
        loggp_nvlink = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        loggp_ib = LogGPParams(L=5e-6, o=10e-6, G=1 / (25e9))

        topo = HierarchicalTopology(4, 8, 25e9, 12, 25e9, loggp_nvlink, loggp_ib)

        # Rank 0: node 0, local GPU 0
        assert topo._rank_to_node(0) == 0
        assert topo._rank_to_local(0) == 0

        # Rank 8: node 1, local GPU 0
        assert topo._rank_to_node(8) == 1
        assert topo._rank_to_local(8) == 0

        # Rank 15: node 1, local GPU 7
        assert topo._rank_to_node(15) == 1
        assert topo._rank_to_local(15) == 7

        # Rank 31: node 3, local GPU 7
        assert topo._rank_to_node(31) == 3
        assert topo._rank_to_local(31) == 7

    def test_all_resources(self):
        """all_resources includes both NVLink and InfiniBand."""
        loggp_nvlink = LogGPParams(L=1e-6, o=5e-6, G=1 / (25e9 * 12))
        loggp_ib = LogGPParams(L=5e-6, o=10e-6, G=1 / (25e9))

        topo = HierarchicalTopology(4, 8, 25e9, 12, 25e9, loggp_nvlink, loggp_ib)

        resources = topo.all_resources()

        # NVLink: 4 nodes * 8 GPUs * 7 others = 4 * 56 = 224
        # IB: 4 uplinks + 1 fabric + 4 downlinks = 9
        # Total: 233
        assert len(resources) == 224 + 9
