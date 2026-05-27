import pytest
from syssim.training.sources import HFModel, CustomModel


def test_hfmodel_constructs_with_id():
    m = HFModel("Qwen/Qwen3-8B")
    assert m.huggingface == "Qwen/Qwen3-8B"
    assert m.overrides == {}


def test_hfmodel_with_overrides():
    m = HFModel("Qwen/Qwen3-8B", overrides={"seq_length": 8192})
    assert m.overrides["seq_length"] == 8192


def test_custom_model_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="CustomModel"):
        CustomModel()


def test_resolve_provider_from_yaml_model():
    pytest.importorskip("megatron.core")
    from syssim.training.spec import ModelConfig, ParallelismConfig, TrainingConfig
    from syssim.training.sources import resolve_megatron_provider
    model = ModelConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, num_query_groups=4,
        ffn_hidden_size=128, seq_length=128, max_position_embeddings=128, vocab_size=256,
    )
    par = ParallelismConfig(tp=1, dp=1)
    tr = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    provider = resolve_megatron_provider(model, par, tr)
    assert provider.num_layers == 2
    assert provider.hidden_size == 64
    assert provider.tensor_model_parallel_size == 1
    assert provider.bf16 is True


def test_resolve_provider_from_hf_model_qwen3_06b():
    """Real HF resolution: Qwen/Qwen3-0.6B (small, downloads only config.json)."""
    pytest.importorskip("megatron.bridge")
    pytest.importorskip("transformers")
    from syssim.training.sources import HFModel, resolve_megatron_provider
    from syssim.training.spec import ParallelismConfig, TrainingConfig
    hf = HFModel("Qwen/Qwen3-0.6B")
    par = ParallelismConfig(tp=1, dp=1)
    tr = TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    provider = resolve_megatron_provider(hf, par, tr)
    # Qwen3-0.6B has 28 layers, 1024 hidden
    assert provider.num_layers == 28
    assert provider.hidden_size == 1024
