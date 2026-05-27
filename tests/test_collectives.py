"""Tests for syssim.training.collectives helpers."""


def test_last_backward_op_per_pp_stage():
    from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType
    from syssim.training.collectives import last_backward_op_per_pp_stage

    g = OperatorGraph(name="test")
    # Stage 0: two compute ops on stream 0
    g.add_operator(OperatorNode(
        name="r0__op_a", op_type=OperatorType.GEMM,
        config={"pp_rank": 0}, stream_id=0,
    ))
    g.add_operator(OperatorNode(
        name="r0__op_b", op_type=OperatorType.GEMM,
        config={"pp_rank": 0}, stream_id=0,
    ))
    # Stage 0: a collective on stream 1 (should NOT be picked)
    g.add_operator(OperatorNode(
        name="r0__coll", op_type=OperatorType.COLLECTIVE,
        config={"pp_rank": 0, "collective": "all_reduce"}, stream_id=1,
    ))
    # Stage 1: one compute op on stream 1000 (1*1000 + 0)
    g.add_operator(OperatorNode(
        name="r1__op_a", op_type=OperatorType.GEMM,
        config={"pp_rank": 1}, stream_id=1000,
    ))

    last = last_backward_op_per_pp_stage(g)
    assert last == {0: "r0__op_b", 1: "r1__op_a"}
