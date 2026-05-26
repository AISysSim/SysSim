import torch.nn as nn


def test_estimate_training_performance_api_returns_memory_and_wall_time(monkeypatch):
    from syssim import estimate_training_performance
    from syssim.operator_graph import OperatorGraph, OperatorNode, OperatorType

    class FakeTracer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def trace(self, model, example_inputs, forward_backward=False, loss_fn=None):
            assert forward_backward is True
            graph = OperatorGraph("fake")
            graph.add_operator(
                OperatorNode(
                    name="op_0",
                    op_type=OperatorType.GEMM,
                    estimated_time_ms=2.5,
                )
            )
            return graph

    class FakePLENAEstimator:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr("syssim.api.OperatorGraphTracer", FakeTracer)
    monkeypatch.setattr("syssim.compute.plena_backend.PLENAEstimator", FakePLENAEstimator)

    model = nn.Linear(4, 4, bias=False)
    batch = (nn.Parameter(model.weight.detach().new_zeros(2, 4)),)

    result = estimate_training_performance(
        model=model,
        example_inputs=batch,
        accelerator="plena",
        accelerator_config=object(),
        batch_size=2,
        seq_len=4,
        dtype="bf16",
        optimizer="adamw",
        num_parameters_override=100,
    )

    assert result.wall_time_ms == 2.5
    assert result.graph is not None
    assert result.memory.parameter_gb == 100 * 2 / 1e9
    assert result.memory.gradient_gb == result.memory.parameter_gb
    assert result.memory.optimizer_state_gb == result.memory.parameter_gb * 4
    assert result.memory.total_model_state_gb == (
        result.memory.parameter_gb
        + result.memory.gradient_gb
        + result.memory.optimizer_state_gb
    )
