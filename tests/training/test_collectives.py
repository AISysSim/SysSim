import pytest
from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType
from syssim.training.collectives import inject_dp_gradient_allreduce


def test_inject_dp_allreduce_adds_collective():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="bwd", op_type=OperatorType.GEMM,
                                config={"phase": "backward"}, estimated_time_ms=1.0))
    inject_dp_gradient_allreduce(
        g, total_grad_bytes=1024*1024, dp_ranks=[0,1,2,3],
        last_backward_op_name="bwd", estimated_time_ms=0.5,
    )
    cols = [op for op in g.operators.values() if op.op_type == OperatorType.COLLECTIVE]
    assert len(cols) == 1
    assert cols[0].config["bytes"] == 1024*1024
    assert "bwd" in cols[0].predecessors
    assert cols[0].stream_id == 1


def test_inject_dp_allreduce_noop_when_dp1():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="bwd", op_type=OperatorType.GEMM, estimated_time_ms=1.0))
    inject_dp_gradient_allreduce(
        g, total_grad_bytes=1024, dp_ranks=[0],
        last_backward_op_name="bwd", estimated_time_ms=0.5,
    )
    assert all(op.op_type != OperatorType.COLLECTIVE for op in g.operators.values())


from syssim.training.collectives import inject_optimizer_step


def test_inject_optimizer_step():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="dp_ar", op_type=OperatorType.COLLECTIVE,
                                estimated_time_ms=0.5))
    inject_optimizer_step(
        g, bytes_moved=50_000_000, peak_memory_bandwidth_GBps=3350.0,
        last_op_name="dp_ar",
    )
    optim = [op for op in g.operators.values() if op.config.get("phase") == "optimizer"]
    assert len(optim) == 1
    # Fused Adam is bandwidth-bound: 50 MB moved / 3350 GB/s ≈ 0.01493 ms
    assert optim[0].estimated_time_ms == pytest.approx(50_000_000 / 3350e9 * 1000, rel=1e-6)
    assert "dp_ar" in optim[0].predecessors
