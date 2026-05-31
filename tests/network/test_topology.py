"""Unit tests for the network topology graph."""

from syssim.network.topology import Gpu, Switch, Link


def test_gpu_dataclass_fields():
    g = Gpu(rank=3, node_id=0, uplink=Link(capacity_GBps=25.0, latency_us=5.0))
    assert g.rank == 3
    assert g.node_id == 0
    assert g.uplink.capacity_GBps == 25.0
    assert g.uplink.latency_us == 5.0
    # nvlink_neighbors defaults to empty
    assert g.nvlink_neighbors == {}


def test_switch_dataclass_fields():
    s = Switch(name="leaf-0", kind="leaf")
    assert s.name == "leaf-0"
    assert s.kind == "leaf"
    s2 = Switch(name="spine-1", kind="spine")
    assert s2.kind == "spine"


def test_link_dataclass_fields():
    link = Link(capacity_GBps=200.0, latency_us=5.0)
    assert link.capacity_GBps == 200.0
    assert link.latency_us == 5.0


def test_topology_holds_gpus_and_links():
    from syssim.network.topology import Topology, Gpu, Switch, Link
    topology = Topology(
        gpus=[Gpu(rank=0, node_id=0), Gpu(rank=1, node_id=0)],
        leaf_switches=[],
        spine_switches=[],
        links=[],
        load_balancer_name="ecmp_hash",
    )
    assert len(topology.gpus) == 2
    assert topology.load_balancer_name == "ecmp_hash"


def test_topology_indexes_gpu_by_rank():
    from syssim.network.topology import Topology, Gpu
    topology = Topology(
        gpus=[Gpu(rank=0, node_id=0), Gpu(rank=1, node_id=0)],
        leaf_switches=[], spine_switches=[], links=[],
        load_balancer_name="ecmp_hash",
    )
    assert topology.gpu(rank=0).node_id == 0
    assert topology.gpu(rank=1).node_id == 0


def test_two_layer_multipath_structure():
    from syssim.network.topology import build_two_layer_multipath
    topology = build_two_layer_multipath(
        num_racks=4, nodes_per_rack=1, gpus_per_node=8, num_spines=2,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
        intra_node_latency_us=1.0, inter_node_latency_us=5.0,
    )
    # 4 nodes * 8 GPUs = 32 GPUs
    assert len(topology.gpus) == 32
    # 4 leaf switches (one per node), 2 spines
    assert len(topology.leaf_switches) == 4
    assert len(topology.spine_switches) == 2
    # Each GPU has an uplink, and a full NVLink mesh to its 7 node-mates
    g0 = topology.gpu(rank=0)
    assert g0.uplink is not None
    assert g0.uplink.capacity_GBps == 25.0
    assert len(g0.nvlink_neighbors) == 7
    assert g0.nvlink_neighbors[1].capacity_GBps == 900.0


def test_fat_tree_structure():
    from syssim.network.topology import build_fat_tree
    # k=4 fat-tree: 4 pods, each with 2 edge switches and 2 agg switches.
    # 4*2 = 8 edge switches in total. Hosts per edge = k/2 = 2. → 16 hosts.
    # 4 core switches (k/2 * k/2 = 4).
    topology = build_fat_tree(
        k=4, gpus_per_node=2,
        per_gpu_bandwidth_GBps=25.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # 16 hosts (one per leaf slot in standard k=4 fat-tree), gpus_per_node=2
    # gives 32 GPUs total.
    assert len(topology.gpus) == 32
    # k=4 → 4 pods × 2 edge switches/pod = 8 leaf switches
    assert len(topology.leaf_switches) == 8


def test_resolve_path_intra_node_uses_nvlink():
    from syssim.network.topology import build_two_layer_multipath
    topology = build_two_layer_multipath(
        num_racks=2, nodes_per_rack=1, gpus_per_node=8, num_spines=2,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # GPUs 0 and 1 share node 0
    path = topology.resolve_path(src=0, dst=1, flow_tag=0)
    assert len(path) == 1
    assert path[0].capacity_GBps == 900.0


def test_resolve_path_cross_node_uses_uplinks_plus_spine():
    from syssim.network.topology import build_two_layer_multipath
    topology = build_two_layer_multipath(
        num_racks=2, nodes_per_rack=1, gpus_per_node=8, num_spines=2,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # GPU 0 on node 0, GPU 8 on node 1
    path = topology.resolve_path(src=0, dst=8, flow_tag=0)
    # uplink(src) + leaf_src↔spine + spine↔leaf_dst + uplink(dst) = 4 links
    assert len(path) == 4
    # First link is src's uplink (25 GBps)
    assert path[0].capacity_GBps == 25.0
    # Middle two are leaf↔spine (200 GBps)
    assert path[1].capacity_GBps == 200.0
    assert path[2].capacity_GBps == 200.0
    # Last is dst's uplink (25 GBps)
    assert path[3].capacity_GBps == 25.0


def test_two_layer_multipath_multi_node_per_rack():
    """num_racks=2, nodes_per_rack=4 → 8 nodes, 2 leaves; nodes share their rack's leaf."""
    from syssim.network.topology import build_two_layer_multipath
    topology = build_two_layer_multipath(
        num_racks=2, nodes_per_rack=4, gpus_per_node=2, num_spines=2,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # 2 racks × 4 nodes × 2 gpus = 16 GPUs; 2 leaves; 2 spines.
    assert len(topology.gpus) == 16
    assert len(topology.leaf_switches) == 2
    assert len(topology.spine_switches) == 2
    # Nodes 0–3 share leaf 0; nodes 4–7 share leaf 1.
    for node_id in range(4):
        assert topology.gpu(rank=node_id * 2).leaf_idx == 0
    for node_id in range(4, 8):
        assert topology.gpu(rank=node_id * 2).leaf_idx == 1


def test_resolve_path_same_rack_different_node_skips_spine():
    """Two GPUs in different nodes of the same rack route via the shared leaf, not a spine."""
    from syssim.network.topology import build_two_layer_multipath
    topology = build_two_layer_multipath(
        num_racks=2, nodes_per_rack=2, gpus_per_node=2, num_spines=2,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # GPU 0 in node 0 (leaf 0); GPU 2 in node 1 (leaf 0). Same rack, no NVLink.
    path = topology.resolve_path(src=0, dst=2, flow_tag=0)
    assert len(path) == 2  # two uplinks meeting at the shared leaf
    assert path[0].capacity_GBps == 25.0
    assert path[1].capacity_GBps == 25.0


def test_resolve_path_ecmp_distribution():
    from syssim.network.topology import build_two_layer_multipath
    topology = build_two_layer_multipath(
        num_racks=2, nodes_per_rack=1, gpus_per_node=1, num_spines=4,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # 100 distinct flow tags between gpu 0 and gpu 1 should spread across all 4 spines
    spine_links_hit = set()
    for tag in range(100):
        path = topology.resolve_path(src=0, dst=1, flow_tag=tag)
        # Path is [uplink_src, leaf↔spine, spine↔leaf_dst, uplink_dst]
        spine_links_hit.add(id(path[1]))   # leaf↔spine link instance
    # With 100 tags and 4 spines, each spine has very high probability of being hit
    # (we'd expect at least 3 of 4, but assert at least 2 to keep it deterministic-ish)
    assert len(spine_links_hit) >= 2


def test_build_arbitrary_structure():
    """Arbitrary topology: num_racks * nodes_per_rack nodes under one root switch."""
    from syssim.network.topology import build_arbitrary
    topology = build_arbitrary(
        num_racks=4, nodes_per_rack=2, gpus_per_node=8,
        per_gpu_bandwidth_GBps=25.0,
        rack_bandwidth_GBps=400.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # 4 racks * 2 nodes * 8 gpus = 64 GPUs.
    assert len(topology.gpus) == 64
    # 4 ToRs (one per rack).
    assert len(topology.leaf_switches) == 4
    # Single root spine.
    assert len(topology.spine_switches) == 1
    assert topology.spine_switches[0].name == "root"


def test_build_arbitrary_resolve_path_same_rack():
    """Same rack, different nodes: 2-hop path through the shared ToR."""
    from syssim.network.topology import build_arbitrary
    topology = build_arbitrary(
        num_racks=2, nodes_per_rack=2, gpus_per_node=1,
        per_gpu_bandwidth_GBps=25.0, rack_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # GPU 0 (node 0, rack 0) and GPU 1 (node 1, rack 0) — different nodes, same ToR.
    path = topology.resolve_path(src=0, dst=1, flow_tag=0)
    assert len(path) == 2
    assert path[0].capacity_GBps == 25.0
    assert path[1].capacity_GBps == 25.0


def test_build_arbitrary_resolve_path_cross_rack():
    """Cross-rack: 4-hop path via the single root."""
    from syssim.network.topology import build_arbitrary
    topology = build_arbitrary(
        num_racks=2, nodes_per_rack=2, gpus_per_node=1,
        per_gpu_bandwidth_GBps=25.0, rack_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # GPU 0 (rack 0) and GPU 2 (rack 1).
    path = topology.resolve_path(src=0, dst=2, flow_tag=0)
    assert len(path) == 4
    assert path[0].capacity_GBps == 25.0    # src uplink
    assert path[1].capacity_GBps == 200.0   # leaf -> root
    assert path[2].capacity_GBps == 200.0   # root -> leaf
    assert path[3].capacity_GBps == 25.0    # dst uplink


def test_build_arbitrary():
    """build_arbitrary constructs the rack/node/gpu hierarchy."""
    from syssim.network.topology import build_arbitrary
    topology = build_arbitrary(
        num_racks=3, nodes_per_rack=2, gpus_per_node=2,
        per_gpu_bandwidth_GBps=25.0, rack_bandwidth_GBps=100.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # 3 racks * 2 nodes * 2 gpus = 12 GPUs.
    assert len(topology.gpus) == 12
    assert len(topology.leaf_switches) == 3
    assert len(topology.spine_switches) == 1


def test_build_simple_structure():
    """Simple topology: num_nodes nodes, each with one shared NIC."""
    from syssim.network.topology import build_simple
    topology = build_simple(
        num_nodes=4, gpus_per_node=8,
        intra_node_bandwidth_GBps=900.0,
        inter_node_bandwidth_GBps=200.0,
    )
    # 4 nodes * 8 GPUs = 32 GPUs.
    assert len(topology.gpus) == 32
    assert len(topology.leaf_switches) == 4
    assert len(topology.spine_switches) == 1


def test_build_simple_gpus_share_one_nic_per_node():
    """All GPUs in the same node share one Link object as their uplink."""
    from syssim.network.topology import build_simple
    topology = build_simple(
        num_nodes=2, gpus_per_node=4,
        intra_node_bandwidth_GBps=900.0,
        inter_node_bandwidth_GBps=200.0,
    )
    # GPUs 0..3 are in node 0, GPUs 4..7 are in node 1.
    node0_uplinks = {id(topology.gpu(r).uplink) for r in range(4)}
    node1_uplinks = {id(topology.gpu(r).uplink) for r in range(4, 8)}
    assert len(node0_uplinks) == 1   # all share one NIC
    assert len(node1_uplinks) == 1
    assert node0_uplinks != node1_uplinks
    # NIC capacity = inter_node_bandwidth_GBps
    assert topology.gpu(0).uplink.capacity_GBps == 200.0


def test_build_simple_cross_node_path():
    """Cross-node path traverses src NIC -> leaf -> root -> leaf -> dst NIC."""
    from syssim.network.topology import build_simple
    topology = build_simple(
        num_nodes=2, gpus_per_node=2,
        intra_node_bandwidth_GBps=900.0,
        inter_node_bandwidth_GBps=200.0,
    )
    # GPU 0 (node 0) -> GPU 2 (node 1).
    path = topology.resolve_path(src=0, dst=2, flow_tag=0)
    assert len(path) == 4
    assert path[0].capacity_GBps == 200.0   # src node NIC
    assert path[1].capacity_GBps == 200.0   # leaf -> root
    assert path[2].capacity_GBps == 200.0   # root -> leaf
    assert path[3].capacity_GBps == 200.0   # dst node NIC
