import pytest
import torch

from syssim import (
    HardwareInfo,
    MoEModelSpec,
    MoERuntimeConfig,
    OperatorType,
    SimulatorConfig,
    build_moe_operator_graph,
)


@pytest.fixture
def hw():
    return HardwareInfo(
        peak_tflops_mm=989.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )


@pytest.fixture
def config(hw):
    return SimulatorConfig(hw_info=hw)


def _spec(**kwargs):
    values = dict(
        num_layers=2,
        hidden_size=256,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        vocab_size=1000,
        name="tiny_moe",
    )
    values.update(kwargs)
    return MoEModelSpec(**values)


def _runtime(**kwargs):
    values = dict(batch_size=1, seq_len=8, dtype=torch.bfloat16)
    values.update(kwargs)
    return MoERuntimeConfig(**values)


class TestMoESpecValidation:
    def test_valid_spec_and_runtime(self):
        spec = _spec()
        runtime = _runtime()
        spec.validate()
        runtime.validate(spec)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("num_layers", 0, "num_layers"),
            ("hidden_size", 0, "hidden_size"),
            ("intermediate_size", 0, "intermediate_size"),
            ("num_experts", 0, "num_experts"),
            ("top_k", 0, "top_k"),
            ("decoder_sparse_step", 0, "decoder_sparse_step"),
            ("first_sparse_layer", -1, "first_sparse_layer"),
        ],
    )
    def test_rejects_invalid_spec_fields(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            _spec(**{field: value}).validate()

    def test_rejects_top_k_gt_num_experts(self):
        with pytest.raises(ValueError, match="top_k"):
            _spec(num_experts=2, top_k=3).validate()

    def test_sparse_layer_indices_respect_first_sparse_layer(self):
        spec = _spec(num_layers=6, first_sparse_layer=1, decoder_sparse_step=2)
        assert spec.sparse_layer_indices() == (1, 3, 5)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("batch_size", 0, "batch_size"),
            ("seq_len", 0, "seq_len"),
            ("expert_parallel_size", 0, "expert_parallel_size"),
            ("capacity_factor", 0.0, "capacity_factor"),
        ],
    )
    def test_rejects_invalid_runtime_fields(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            _runtime(**{field: value}).validate(_spec())

    def test_rejects_ep_gt_num_experts(self):
        with pytest.raises(ValueError, match="expert_parallel_size"):
            _runtime(expert_parallel_size=5).validate(_spec(num_experts=4))

    def test_rejects_tokens_per_expert_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            _runtime(tokens_per_expert=(8, 8)).validate(_spec(num_experts=4))

    def test_rejects_tokens_per_expert_sum_mismatch(self):
        with pytest.raises(ValueError, match="sum"):
            _runtime(tokens_per_expert=(1, 1, 1, 1)).validate(_spec(num_experts=4))


class TestMoEGraphBuilder:
    def test_single_layer_stage_chain(self, config):
        spec = _spec(num_layers=1)
        graph = build_moe_operator_graph(spec, _runtime(), config)

        assert len(graph.operators) == 4
        assert "layer_000_moe_router" in graph.operators
        assert graph.operators["layer_000_moe_dispatch"].data_deps == ["layer_000_moe_router"]
        assert graph.operators["layer_000_moe_expert"].data_deps == ["layer_000_moe_dispatch"]
        assert graph.operators["layer_000_moe_combine"].data_deps == ["layer_000_moe_expert"]
        assert graph.compute_critical_path() > 0.0

    def test_multilayer_chain(self, config):
        spec = _spec(num_layers=2)
        graph = build_moe_operator_graph(spec, _runtime(), config)

        assert len(graph.operators) == len(spec.sparse_layer_indices()) * 4
        assert graph.operators["layer_001_moe_router"].data_deps == ["layer_000_moe_combine"]

        summary = graph.summary()
        assert "moe_router: 2" in summary
        assert "moe_dispatch: 2" in summary
        assert "moe_expert: 2" in summary
        assert "moe_combine: 2" in summary

    def test_node_count_respects_sparse_step(self, config):
        spec = _spec(num_layers=6, first_sparse_layer=1, decoder_sparse_step=2)
        graph = build_moe_operator_graph(spec, _runtime(), config)
        assert len(graph.operators) == 3 * 4
        assert "layer_001_moe_router" in graph.operators
        assert "layer_003_moe_router" in graph.operators
        assert "layer_005_moe_router" in graph.operators

    def test_node_config_contains_common_fields(self, config):
        graph = build_moe_operator_graph(_spec(), _runtime(), config)
        node_config = graph.operators["layer_000_moe_router"].config
        for key in (
            "layer_idx",
            "num_tokens",
            "num_assignments",
            "hidden_size",
            "num_experts",
            "top_k",
            "dtype",
        ):
            assert key in node_config

    def test_large_seq_len_does_not_allocate_tokens(self, config):
        runtime = _runtime(batch_size=8, seq_len=8192)
        graph = build_moe_operator_graph(_spec(num_layers=1), runtime, config)
        assert graph.operators["layer_000_moe_router"].config["num_tokens"] == 65536

    def test_explicit_tokens_per_expert_changes_expert_time(self, config):
        spec = _spec(num_layers=1)
        uniform = build_moe_operator_graph(
            spec,
            _runtime(tokens_per_expert=(4, 4, 4, 4)),
            config,
        )
        imbalanced = build_moe_operator_graph(
            spec,
            _runtime(tokens_per_expert=(16, 0, 0, 0)),
            config,
        )
        assert (
            imbalanced.operators["layer_000_moe_expert"].estimated_time_ms
            > uniform.operators["layer_000_moe_expert"].estimated_time_ms
        )


class TestMoEExpertParallel:
    def test_ep_adds_two_collectives_per_layer(self, config):
        spec = _spec(num_layers=1)
        runtime = _runtime(expert_parallel_size=2)
        graph = build_moe_operator_graph(spec, runtime, config)

        collectives = [
            op for op in graph.operators.values()
            if op.op_type == OperatorType.COLLECTIVE
        ]
        assert len(collectives) == 2
        assert graph.operators["layer_000_moe_alltoall_dispatch"].estimated_time_ms > 0.0
        assert graph.operators["layer_000_moe_alltoall_combine"].estimated_time_ms > 0.0
        assert graph.operators["layer_000_moe_alltoall_dispatch"].data_deps == ["layer_000_moe_dispatch"]
        assert graph.operators["layer_000_moe_expert"].data_deps == ["layer_000_moe_alltoall_dispatch"]
        assert graph.operators["layer_000_moe_alltoall_combine"].data_deps == ["layer_000_moe_expert"]
        assert graph.operators["layer_000_moe_combine"].data_deps == ["layer_000_moe_alltoall_combine"]

    def test_rejects_topology_without_loggp(self, config):
        with pytest.raises(ValueError, match="topology and loggp"):
            build_moe_operator_graph(
                _spec(),
                _runtime(expert_parallel_size=2),
                config,
                topology=object(),
            )

    def test_rejects_loggp_without_topology(self, config):
        with pytest.raises(ValueError, match="topology and loggp"):
            build_moe_operator_graph(
                _spec(),
                _runtime(expert_parallel_size=2),
                config,
                loggp=object(),
            )
