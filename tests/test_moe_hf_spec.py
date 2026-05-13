from types import SimpleNamespace

import pytest
import torch

from syssim import (
    ExecutionMode,
    HardwareInfo,
    MoERuntimeConfig,
    OperatorType,
    SimulatorConfig,
    extract_hf_moe_spec,
    trace_hf_moe_model_for_inference,
    trace_hf_moe_model_for_training,
)


@pytest.fixture
def config():
    hw = HardwareInfo(
        peak_tflops_mm=989.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )
    return SimulatorConfig(hw_info=hw)


def _fake_qwen3_config(**overrides):
    values = dict(
        num_hidden_layers=2,
        hidden_size=256,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=1000,
        decoder_sparse_step=1,
        first_sparse_layer=0,
        model_type="qwen3_moe",
        torch_dtype=torch.bfloat16,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestHFMoESpecExtraction:
    def test_extracts_fake_qwen3_config(self):
        spec = extract_hf_moe_spec(_fake_qwen3_config(), name="qwen3-test")
        assert spec.name == "qwen3-test"
        assert spec.num_layers == 2
        assert spec.hidden_size == 256
        assert spec.intermediate_size == 128
        assert spec.num_experts == 4
        assert spec.top_k == 2
        assert spec.vocab_size == 1000

    def test_extracts_from_model_wrapper(self):
        model = SimpleNamespace(config=_fake_qwen3_config())
        spec = extract_hf_moe_spec(model)
        assert spec.name == "qwen3_moe"
        assert spec.sparse_layer_indices() == (0, 1)

    def test_missing_fields_error_names_all_fields(self):
        config = SimpleNamespace(hidden_size=256)
        with pytest.raises(ValueError) as exc:
            extract_hf_moe_spec(config)
        message = str(exc.value)
        assert "num_hidden_layers" in message
        assert "moe_intermediate_size" in message
        assert "num_experts" in message
        assert "num_experts_per_tok" in message


class TestMoEPublicApi:
    def test_training_runtime_inferred_from_input_ids(self, config):
        model = SimpleNamespace(config=_fake_qwen3_config(), dtype=torch.bfloat16)
        input_ids = torch.randint(0, 1000, (2, 8))
        graph = trace_hf_moe_model_for_training(model, {"input_ids": input_ids}, config)

        router = graph.operators["layer_000_moe_router"]
        assert router.config["num_tokens"] == 16
        assert router.config["mode"] == ExecutionMode.TRAINING.value
        assert OperatorType.MOE_EXPERT in {op.op_type for op in graph.operators.values()}

    def test_explicit_runtime_is_used(self, config):
        model = SimpleNamespace(config=_fake_qwen3_config(), dtype=torch.float16)
        runtime = MoERuntimeConfig(batch_size=1, seq_len=4, dtype=torch.float16)
        graph = trace_hf_moe_model_for_training(model, {}, config, runtime=runtime)
        assert graph.operators["layer_000_moe_router"].config["num_tokens"] == 4
        assert graph.operators["layer_000_moe_router"].config["dtype"] == "float16"

    def test_inference_modes(self, config):
        model = SimpleNamespace(config=_fake_qwen3_config(), dtype=torch.bfloat16)
        input_ids = torch.randint(0, 1000, (1, 8))
        prefill = trace_hf_moe_model_for_inference(
            model,
            {"input_ids": input_ids},
            config,
            mode="prefill",
        )
        decode = trace_hf_moe_model_for_inference(
            model,
            {"input_ids": input_ids[:, :1]},
            config,
            mode="decode",
        )
        assert prefill.operators["layer_000_moe_router"].config["mode"] == "prefill"
        assert decode.operators["layer_000_moe_router"].config["mode"] == "decode"

    def test_invalid_mode_raises(self, config):
        model = SimpleNamespace(config=_fake_qwen3_config())
        with pytest.raises(ValueError, match="mode"):
            trace_hf_moe_model_for_inference(model, {}, config, mode="bad")

    def test_missing_input_ids_requires_runtime(self, config):
        model = SimpleNamespace(config=_fake_qwen3_config())
        with pytest.raises(ValueError, match="input_ids"):
            trace_hf_moe_model_for_training(model, {}, config)

    def test_missing_input_shape_requires_runtime(self, config):
        model = SimpleNamespace(config=_fake_qwen3_config())
        with pytest.raises(ValueError, match="shape"):
            trace_hf_moe_model_for_training(model, {"input_ids": torch.tensor([1, 2])}, config)
