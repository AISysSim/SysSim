# tests/training/test_tracer_capture.py
"""Tests for the new Megatron-shaped OperatorGraphTracer.trace signature."""
import pytest
import torch
import torch.nn as nn
from syssim.tracer import OperatorGraphTracer
from syssim.operator_graph import OperatorGraph, OperatorType


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tracer requires CUDA")
def test_tracer_runs_forward_backward_func():
    """The new tracer should call forward_backward_func(...) inside fake CUDA contexts."""
    tracer = OperatorGraphTracer()
    model = nn.Linear(16, 16)
    invocations = []

    def fake_forward_step(data_iterator, model):
        batch = next(data_iterator)
        out = model(batch)
        return out.sum(), {}

    def fake_data_iterator():
        while True:
            yield torch.randn(2, 16, device="cuda")

    def fake_forward_backward_func(
        *, forward_step_func, data_iterator, model,
        num_microbatches, seq_length, micro_batch_size, forward_only,
    ):
        invocations.append({
            "num_microbatches": num_microbatches,
            "seq_length": seq_length,
            "micro_batch_size": micro_batch_size,
            "forward_only": forward_only,
        })
        for _ in range(num_microbatches):
            loss, _ = forward_step_func(data_iterator, model)
            if not forward_only:
                loss.backward()

    graph = tracer.trace(
        model=model,
        forward_backward_func=fake_forward_backward_func,
        forward_step_func=fake_forward_step,
        data_iterator=fake_data_iterator(),
        num_microbatches=2,
        seq_length=16,
        micro_batch_size=2,
    )
    assert isinstance(graph, OperatorGraph)
    assert len(graph.operators) > 0
    assert invocations == [{
        "num_microbatches": 2, "seq_length": 16,
        "micro_batch_size": 2, "forward_only": False,
    }]
    # 2 microbatches → at least 2 mat-mul ops in the graph
    gemm_ops = [op for op in graph.operators.values()
                if op.op_type == OperatorType.GEMM]
    assert len(gemm_ops) >= 2


import torch.distributed as dist
from syssim.tracer import _dist_noop_context


@pytest.fixture(autouse=True)
def _cleanup_dist():
    """Ensure process group is destroyed after each test."""
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _fake_init_local_dist():
    import os, socket
    s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)


def test_dist_noop_context_records_sync_collective_with_post_sync():
    """Sync all_reduce → COLLECTIVE on stream 1 + STREAM_SYNC on caller stream."""
    if not dist.is_available():
        return
    _fake_init_local_dist()
    g = OperatorGraph(name="trace")
    last_op = {}
    with _dist_noop_context(graph=g, last_op_on_stream=last_op):
        t = torch.zeros(1024, dtype=torch.float32)
        dist.all_reduce(t)         # async_op=False (default) — sync semantics
    collectives = [op for op in g.operators.values()
                   if op.op_type == OperatorType.COLLECTIVE]
    syncs = [op for op in g.operators.values()
             if op.op_type == OperatorType.STREAM_SYNC]
    assert len(collectives) == 1
    assert len(syncs) == 1
    assert collectives[0].config["collective"] == "all_reduce"
    assert collectives[0].config["bytes"] == 1024 * 4
    assert collectives[0].name in syncs[0].predecessors


def test_dist_noop_context_async_collective_no_caller_sync_until_wait():
    """Async all_reduce → COLLECTIVE only; STREAM_SYNC appears on .wait()."""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        _fake_init_local_dist()
    g = OperatorGraph(name="trace")
    last_op = {}
    with _dist_noop_context(graph=g, last_op_on_stream=last_op):
        t = torch.zeros(64, dtype=torch.float32)
        handle = dist.all_reduce(t, async_op=True)
        syncs_after_dispatch = [op for op in g.operators.values()
                                if op.op_type == OperatorType.STREAM_SYNC]
        assert syncs_after_dispatch == []   # no caller-side sync at dispatch time
        handle.wait()                       # this emits the STREAM_SYNC
    syncs = [op for op in g.operators.values() if op.op_type == OperatorType.STREAM_SYNC]
    assert len(syncs) == 1


def test_dist_noop_context_no_graph_is_pure_noop():
    """No graph passed → no nodes recorded anywhere; existing call sites stay safe."""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        _fake_init_local_dist()
    with _dist_noop_context():
        pass
