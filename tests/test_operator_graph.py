import pytest
from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType


def test_operator_node_uses_predecessors_field():
    n = OperatorNode(name="a", op_type=OperatorType.GEMM, predecessors=["x", "y"])
    assert n.predecessors == ["x", "y"]
    # The old fields should NOT exist on the dataclass
    assert not hasattr(n, "data_deps")
    assert not hasattr(n, "stream_deps")
    assert not hasattr(n, "earliest_start")
    assert not hasattr(n, "earliest_finish")


def test_operator_graph_has_no_compute_critical_path():
    g = OperatorGraph()
    assert not hasattr(g, "compute_critical_path")
