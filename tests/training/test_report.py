import pytest
from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType
from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig
from syssim.training.report import (
    SimulationReport, aggregate_by_op_type, compute_efficiency,
    compute_model_flops_budget, ModelFlopsBudget,
)


def test_simulation_report_holds_provenance():
    par = ParallelismConfig()
    tr  = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    r = SimulationReport(
        step_time_ms=10.0, forward_ms=4.0, backward_ms=5.0, optimizer_ms=0.5,
        collective_total_ms=2.0, collective_exposed_ms=0.5,
        by_op_type_ms={"gemm": 8.0},
        model_flops_per_step=1_000_000, achieved_tflops=0.1, mfu=0.5, hfu=0.5,
        param_bytes=10, grad_bytes=10, optimizer_state_bytes=30, activation_bytes=50,
        peak_memory_gb=1e-7,
        model=None, parallelism=par, training=tr, hardware=None,
    )
    assert r.step_time_ms == 10.0


def test_aggregate_by_op_type():
    g = OperatorGraph()
    g.add_operator(OperatorNode(name="a", op_type=OperatorType.GEMM, estimated_time_ms=2.0))
    g.add_operator(OperatorNode(name="b", op_type=OperatorType.GEMM, estimated_time_ms=3.0))
    g.add_operator(OperatorNode(name="c", op_type=OperatorType.ATTN, estimated_time_ms=1.5))
    g.add_operator(OperatorNode(name="d", op_type=OperatorType.COLLECTIVE, estimated_time_ms=0.8))
    out = aggregate_by_op_type(g)
    assert out["gemm"] == 5.0
    assert out["attn"] == 1.5
    assert out["collective"] == 0.8


def test_compute_efficiency():
    budget = ModelFlopsBudget(model_flops_per_step=2.0e12, hardware_flops_per_step=2.5e12)
    eff = compute_efficiency(budget=budget, step_time_ms=10.0, world_size=1, peak_tflops_mm=1979.0)
    assert eff["achieved_tflops"] == pytest.approx(200.0, rel=1e-6)
    assert eff["mfu"] == pytest.approx(200.0 / 1979.0, rel=1e-6)
    assert eff["hfu"] > eff["mfu"]


def test_compute_model_flops_budget_recompute_increases_hw_only():
    m = ModelConfig(num_layers=2, hidden_size=128, num_attention_heads=4, num_query_groups=4,
                    ffn_hidden_size=512, seq_length=256, max_position_embeddings=256, vocab_size=1024)
    par = ParallelismConfig()
    tr_none = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    tr_full = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16", recompute="full")
    b_none = compute_model_flops_budget(m, par, tr_none)
    b_full = compute_model_flops_budget(m, par, tr_full)
    assert b_full.model_flops_per_step == b_none.model_flops_per_step
    assert b_full.hardware_flops_per_step > b_none.hardware_flops_per_step
