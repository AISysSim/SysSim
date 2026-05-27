from syssim.network.collectives import allreduce
from syssim.network.simulator import simulate
from syssim.network.topology import build_two_layer_multipath


def test_allreduce_op_count_and_dependency_structure():
    ranks = [0, 1, 2, 3]
    ops = allreduce(ranks, total_bytes=4e9, tag="ar")
    # Ring all-reduce: 2 * (n-1) steps, n flows each → 2*3*4 = 24
    assert len(ops) == 24
    # First step (step 0) has no deps
    step0 = [o for o in ops if o.tag.endswith("_step_0")]
    assert len(step0) == 4
    assert all(not o.deps for o in step0)
    # Subsequent steps each have one data-causality dep
    for o in ops:
        if not o.tag.endswith("_step_0"):
            assert len(o.deps) == 1


def test_allreduce_runs_to_completion_on_intra_node():
    topology = build_two_layer_multipath(
        num_racks=1, nodes_per_rack=1, gpus_per_node=4, num_spines=1,
        per_gpu_bandwidth_GBps=25.0, uplink_bandwidth_GBps=200.0,
        intra_node_bandwidth_GBps=900.0,
    )
    ops = allreduce(ranks=[0, 1, 2, 3], total_bytes=4e9, tag="ar")
    result = simulate(ops, topology)
    assert result.makespan > 0


def test_allgather_op_count():
    from syssim.network.collectives import allgather
    ranks = [0, 1, 2, 3]
    ops = allgather(ranks, total_bytes=4e9, tag="ag")
    # Ring all-gather: (n-1) steps, n flows each → 3 * 4 = 12
    assert len(ops) == 12


def test_allgather_first_step_has_no_deps():
    from syssim.network.collectives import allgather
    ops = allgather(ranks=[0, 1, 2, 3], total_bytes=4e9, tag="ag")
    step0 = [o for o in ops if o.tag.endswith("_step_0")]
    assert len(step0) == 4
    assert all(not o.deps for o in step0)


def test_reduce_scatter_op_count():
    from syssim.network.collectives import reduce_scatter
    ops = reduce_scatter(ranks=[0, 1, 2, 3], total_bytes=4e9, tag="rs")
    # Ring reduce-scatter: (n-1) steps, n flows each → 12
    assert len(ops) == 12


def test_broadcast_op_count_pow2():
    from syssim.network.collectives import broadcast
    # 8 ranks → 7 sends in a balanced binary tree (one per non-root)
    ops = broadcast(ranks=[0, 1, 2, 3, 4, 5, 6, 7], total_bytes=1e9,
                    root=0, tag="bc")
    assert len(ops) == 7


def test_broadcast_root_has_no_dep():
    from syssim.network.collectives import broadcast
    ops = broadcast(ranks=[0, 1, 2, 3], total_bytes=1e9, root=0, tag="bc")
    # First send (from root) has no deps; later sends depend on whichever
    # send delivered the data to the new sender.
    first = [o for o in ops if o.src == 0 and not o.deps]
    assert len(first) >= 1
