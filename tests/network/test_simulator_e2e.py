"""End-to-end smoke tests for the rewritten simulator + topology."""

from syssim.network.simulator import Op, simulate
from syssim.network.topology import build_two_layer_multipath


def test_single_op_completion_time():
    """One flow, 1 GB, no contention. Time = bytes / bandwidth."""
    topology = build_two_layer_multipath(
        num_racks=1, nodes_per_rack=1, gpus_per_node=2, num_spines=1,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # Intra-node flow between GPU 0 and GPU 1: uses one NVLink at 900 GB/s
    op = Op(src=0, dst=1, size=1e9, tag="t")
    result = simulate([op], topology)
    # 1 GB / 900 GB/s = ~1.11 ms; 1.111e-3 s
    expected = 1e9 / (900 * 1e9)
    assert abs(result.makespan - expected) / expected < 0.05


def test_two_flows_sharing_uplink_halve():
    """Two cross-node flows from the same node share its leaf uplink."""
    topology = build_two_layer_multipath(
        num_racks=2, nodes_per_rack=1, gpus_per_node=2, num_spines=1,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    # GPU 0 -> GPU 2 (cross-node), GPU 1 -> GPU 3 (cross-node, different src GPU
    # so different per-GPU uplink - but BOTH still share the leaf<->spine link)
    op_a = Op(src=0, dst=2, size=1e9, tag="a")
    op_b = Op(src=1, dst=3, size=1e9, tag="b")
    result = simulate([op_a, op_b], topology)
    # leaf<->spine is 200 GB/s, shared by both flows -> each gets 100 GB/s.
    # Per-GPU uplinks are 25 GB/s each, NOT shared -> each flow capped at 25.
    # So each flow's bottleneck is its uplink at 25 GB/s.
    # Completion: 1 GB / 25 GB/s = 40 ms.
    expected = 1e9 / (25 * 1e9)
    assert abs(result.makespan - expected) / expected < 0.05


def test_serial_dependency_chain():
    """An op with a dep waits for the dep to complete before transferring."""
    topology = build_two_layer_multipath(
        num_racks=1, nodes_per_rack=1, gpus_per_node=2, num_spines=1,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    op_a = Op(src=0, dst=1, size=1e9, tag="a")
    op_b = Op(src=1, dst=0, size=1e9, tag="b", deps=[op_a])
    result = simulate([op_a, op_b], topology)
    # Each op: 1 GB / 900 GB/s = ~1.11 ms; serial -> ~2.22 ms.
    single = 1e9 / (900 * 1e9)
    assert abs(result.makespan - 2 * single) / (2 * single) < 0.05
