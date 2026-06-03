"""Inject derived operators (DP gradient all-reduce, optimizer step) into a traced graph."""

from __future__ import annotations

from ..operator_graph import OperatorGraph, OperatorNode, OperatorType


def inject_dp_gradient_allreduce(
    graph: OperatorGraph,
    total_grad_bytes: int,
    dp_ranks: list[int],
    last_backward_op_name: str,
    estimated_time_ms: float,
) -> None:
    if len(dp_ranks) <= 1:
        return
    idx = len(graph.operators)
    graph.add_operator(OperatorNode(
        name=f"dp_allreduce_{idx}",
        op_type=OperatorType.COLLECTIVE,
        config={
            "collective": "all_reduce",
            "bytes": int(total_grad_bytes),
            "group_ranks": list(dp_ranks),
            "phase": "backward",
        },
        predecessors=[last_backward_op_name],
        stream_id=1,
        estimated_time_ms=estimated_time_ms,
    ))


def inject_optimizer_step(
    graph: OperatorGraph,
    bytes_moved: int,
    peak_memory_bandwidth_GBps: float,
    last_op_name: str,
    bandwidth_efficiency: float = 1.0,
) -> None:
    """Inject one MATH op modeling the (fused, memory-bound) Adam parameter update.

    Real Megatron fuses Adam into a single multi_tensor_apply kernel, so it is bandwidth-bound:
    time = bytes_moved / (peak_bandwidth * bandwidth_efficiency). The caller computes bytes_moved
    from the mixed-precision state traffic (fp32 master+m+v read+write, fp32 grad read, bf16 param
    write). This is NOT traced because the fused kernel has no FakeTensor impl and decomposes into
    ~100x-heavier per-param ops. `bandwidth_efficiency` (default 1.0 = spec peak) is the realized
    fraction of peak HBM bandwidth for this optimizer-update phase, which also absorbs the
    mixed-precision grad/master plumbing copies that share the phase; the caller reads it from the
    calibration manifest (a measured device property), not the hardware YAML.
    """
    bytes_moved = int(bytes_moved)
    eff_bw = peak_memory_bandwidth_GBps * bandwidth_efficiency
    time_ms = bytes_moved / (eff_bw * 1e9) * 1000.0 if eff_bw > 0 else 0.0
    idx = len(graph.operators)
    graph.add_operator(OperatorNode(
        name=f"optimizer_step_{idx}",
        op_type=OperatorType.MATH,
        config={"phase": "optimizer", "bytes_moved": bytes_moved},
        predecessors=[last_op_name],
        stream_id=0,
        estimated_time_ms=time_ms,
    ))


def last_backward_op_per_pp_stage(graph: "OperatorGraph") -> dict[int, str]:
    """Return mapping pp_rank -> name of the latest backward-ish op on that stage.

    Heuristic: any op carrying config["pp_rank"] is associated with that stage.
    Within a stage, the last op (highest insertion order) on stream 0 (compute)
    is the last backward op.

    NOTE: the stream offset per stage is pp_rank * 1000, mirroring _STREAM_STRIDE
    in pipeline.py. Hardcoded here to avoid a circular import.
    """
    per_stage_last: dict[int, str] = {}
    for op in graph.operators.values():
        pp_rank = op.config.get("pp_rank")
        if pp_rank is None:
            continue
        # Compute stream for this stage is pp_rank * 1000 + 0
        if (op.stream_id - pp_rank * 1000) != 0:
            continue
        per_stage_last[pp_rank] = op.name
    return per_stage_last
