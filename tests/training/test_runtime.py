"""Tests for the runtime simulator."""

import pytest
from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType
from syssim.training.runtime import build_stream_queues, simulate_runtime, RuntimeResult, phase_breakdown_ms, collective_exposed_ms


def test_build_stream_queues_groups_by_stream_id():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="a", op_type=OperatorType.GEMM, stream_id=0, estimated_time_ms=1.0))
    g.add_operator(OperatorNode(name="b", op_type=OperatorType.GEMM, stream_id=0, estimated_time_ms=2.0))
    g.add_operator(OperatorNode(name="c", op_type=OperatorType.COLLECTIVE, stream_id=1, estimated_time_ms=3.0))
    queues = build_stream_queues(g)
    assert sorted(queues) == [0, 1]
    assert [op.name for op in queues[0]] == ["a", "b"]
    assert [op.name for op in queues[1]] == ["c"]


def test_simulate_runtime_empty_graph():
    res = simulate_runtime(OperatorGraph())
    assert isinstance(res, RuntimeResult)
    assert res.step_time_ms == 0.0


def test_simulate_runtime_single_stream_sum():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="a", op_type=OperatorType.GEMM,
                                stream_id=0, estimated_time_ms=2.0))
    g.add_operator(OperatorNode(name="b", op_type=OperatorType.GEMM,
                                stream_id=0, estimated_time_ms=3.0,
                                predecessors=["a"]))
    assert simulate_runtime(g).step_time_ms == pytest.approx(5.0)


def test_simulate_runtime_two_streams_overlap():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="compute", op_type=OperatorType.GEMM,
                                stream_id=0, estimated_time_ms=10.0))
    g.add_operator(OperatorNode(name="comm", op_type=OperatorType.COLLECTIVE,
                                stream_id=1, estimated_time_ms=4.0))
    assert simulate_runtime(g).step_time_ms == pytest.approx(10.0)


def test_simulate_runtime_stream_dep_serializes():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="comm", op_type=OperatorType.COLLECTIVE,
                                stream_id=1, estimated_time_ms=4.0))
    g.add_operator(OperatorNode(name="compute", op_type=OperatorType.GEMM,
                                stream_id=0, estimated_time_ms=10.0,
                                predecessors=["comm"]))
    assert simulate_runtime(g).step_time_ms == pytest.approx(14.0)


def test_simulate_runtime_barrier_waits_for_all_streams():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="s0a", op_type=OperatorType.GEMM,
                                stream_id=0, estimated_time_ms=2.0))
    g.add_operator(OperatorNode(name="s1a", op_type=OperatorType.COLLECTIVE,
                                stream_id=1, estimated_time_ms=5.0))
    g.add_operator(OperatorNode(name="bar", op_type=OperatorType.BARRIER,
                                stream_id=0, estimated_time_ms=0.0,
                                predecessors=["s0a"]))
    g.add_operator(OperatorNode(name="s0b", op_type=OperatorType.GEMM,
                                stream_id=0, estimated_time_ms=1.0,
                                predecessors=["bar"]))
    assert simulate_runtime(g).step_time_ms == pytest.approx(6.0)


def test_phase_breakdown_groups_by_config_phase():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="f1", op_type=OperatorType.GEMM,
                                estimated_time_ms=1.0, config={"phase": "forward"}))
    g.add_operator(OperatorNode(name="f2", op_type=OperatorType.ATTN,
                                estimated_time_ms=2.0, config={"phase": "forward"}))
    g.add_operator(OperatorNode(name="b1", op_type=OperatorType.GEMM,
                                estimated_time_ms=3.0, config={"phase": "backward"}))
    g.add_operator(OperatorNode(name="o1", op_type=OperatorType.MATH,
                                estimated_time_ms=0.2, config={"phase": "optimizer"}))
    bd = phase_breakdown_ms(g)
    assert bd["forward"] == pytest.approx(3.0)
    assert bd["backward"] == pytest.approx(3.0)
    assert bd["optimizer"] == pytest.approx(0.2)


def test_collective_exposed_zero_when_overlapped():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="compute", op_type=OperatorType.GEMM,
                                stream_id=0, estimated_time_ms=10.0))
    g.add_operator(OperatorNode(name="comm", op_type=OperatorType.COLLECTIVE,
                                stream_id=1, estimated_time_ms=4.0))
    res = simulate_runtime(g)
    assert collective_exposed_ms(g, res) == pytest.approx(0.0, abs=1e-9)


def test_collective_exposed_equals_comm_when_serialized():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="comm", op_type=OperatorType.COLLECTIVE,
                                stream_id=1, estimated_time_ms=4.0))
    g.add_operator(OperatorNode(name="compute", op_type=OperatorType.GEMM,
                                stream_id=0, estimated_time_ms=10.0,
                                predecessors=["comm"]))
    res = simulate_runtime(g)
    assert collective_exposed_ms(g, res) == pytest.approx(4.0)
