from syssim.network.collectives import all_to_all
from syssim.network.simulator import simulate
from syssim.network.topology import build_simple
from syssim.training.runner import _estimate_collective_ms


def test_all_to_all_op_count_and_no_deps():
    ranks = [0, 1, 2, 3]
    total_bytes = 3e9
    ops = all_to_all(ranks, total_bytes=total_bytes, tag="a2a")
    # Every rank sends a distinct chunk to every OTHER rank: n*(n-1) ops.
    assert len(ops) == 4 * 3 == 12
    # All flows are concurrent and independent — no data-causality deps.
    assert all(o.deps == [] for o in ops)
    # Per-flow size = total per-rank send volume / (n - 1).
    assert all(o.size == total_bytes / 3 for o in ops)
    # The full set of ordered (src, dst) pairs with src != dst, each once.
    pairs = [(o.src, o.dst) for o in ops]
    expected = [(i, j) for i in ranks for j in ranks if i != j]
    assert sorted(pairs) == sorted(expected)
    # No self-sends.
    assert all(o.src != o.dst for o in ops)


def test_all_to_all_degenerate_cases():
    assert all_to_all([0], total_bytes=1e9) == []
    assert all_to_all([], total_bytes=1e9) == []
    assert all_to_all([0, 1, 2, 3], total_bytes=0) == []
    assert all_to_all([0, 1, 2, 3], total_bytes=-5.0) == []


def test_all_to_all_topology_sensitivity():
    ranks = [0, 1, 2, 3]
    total_bytes = 4e9
    # High intra-node bandwidth: all 4 ranks in one node (NVLink mesh).
    fast = build_simple(
        num_nodes=1, gpus_per_node=4,
        intra_node_bandwidth_GBps=900.0,
        inter_node_bandwidth_GBps=200.0,
    )
    # Low inter-node bandwidth: each rank on its own node, traffic on slow NICs.
    slow = build_simple(
        num_nodes=4, gpus_per_node=1,
        intra_node_bandwidth_GBps=900.0,
        inter_node_bandwidth_GBps=25.0,
    )
    fast_makespan = simulate(all_to_all(ranks, total_bytes, tag="a2a"), fast).makespan
    slow_makespan = simulate(all_to_all(ranks, total_bytes, tag="a2a"), slow).makespan
    import math
    assert fast_makespan > 0 and math.isfinite(fast_makespan)
    assert slow_makespan > 0 and math.isfinite(slow_makespan)
    assert slow_makespan > fast_makespan


def test_all_to_all_dispatch_routing():
    topology = build_simple(
        num_nodes=4, gpus_per_node=1,
        intra_node_bandwidth_GBps=900.0,
        inter_node_bandwidth_GBps=25.0,
    )
    ms = _estimate_collective_ms(
        collective_name="all_to_all", total_bytes=4e9,
        ranks=[0, 1, 2, 3], topology=topology,
    )
    assert ms > 0
    # An unmodeled collective name returns 0.0.
    assert _estimate_collective_ms(
        collective_name="barrier", total_bytes=4e9,
        ranks=[0, 1, 2, 3], topology=topology,
    ) == 0.0
