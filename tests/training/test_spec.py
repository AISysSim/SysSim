def test_training_package_importable():
    import syssim.training
    assert syssim.training is not None


import pytest
from syssim.training.spec import ModelConfig


def test_model_config_required_fields():
    cfg = ModelConfig(
        num_layers=28, hidden_size=2048,
        num_attention_heads=16, num_query_groups=8,
        ffn_hidden_size=6144, seq_length=4096,
        max_position_embeddings=40960, vocab_size=151936,
    )
    assert cfg.num_layers == 28
    assert cfg.swiglu is True              # default
    assert cfg.tie_word_embeddings is False
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.huggingface is None
    assert cfg.overrides == {}


def test_model_config_huggingface_branch():
    cfg = ModelConfig(huggingface="Qwen/Qwen3-8B")
    assert cfg.huggingface == "Qwen/Qwen3-8B"
    # Architecture fields default to None when HF source
    assert cfg.num_layers is None


from syssim.training.spec import ParallelismConfig


def test_parallelism_short_kwargs():
    p = ParallelismConfig(tp=4, dp=2, sp=True, cp=1)
    assert p.tensor_model_parallel_size == 4
    assert p.data_parallel_size == 2
    assert p.sequence_parallel is True
    assert p.context_parallel_size == 1
    assert p.world_size == 8


def test_parallelism_megatron_kwargs_also_work():
    p = ParallelismConfig(
        tensor_model_parallel_size=2, data_parallel_size=4,
    )
    assert p.world_size == 8


def test_parallelism_rejects_invalid_sizes():
    with pytest.raises(ValueError, match=">= 1"):
        ParallelismConfig(tp=0)


from syssim.training.spec import TrainingConfig


def test_training_config_dtype_string():
    t = TrainingConfig(micro_batch=1, global_batch=16, dtype="bf16")
    assert t.bf16 is True
    assert t.fp16 is False
    assert t.fp8 is False
    assert t.recompute_granularity is None


def test_training_config_explicit_flags():
    t = TrainingConfig(
        micro_batch_size=2, global_batch_size=8,
        fp16=True, bf16=False,
    )
    assert t.fp16 is True
    assert t.micro_batch_size == 2
    assert t.global_batch_size == 8


def test_training_config_rejects_no_dtype():
    with pytest.raises(ValueError, match="exactly one"):
        TrainingConfig(micro_batch=1, global_batch=1, bf16=False)


def test_training_config_recompute_validation():
    with pytest.raises(ValueError, match="recompute_granularity"):
        TrainingConfig(
            micro_batch=1, global_batch=1, recompute="invalid",
        )


from syssim.training.spec import HardwareConfig


def test_hardware_config_required_fields():
    hw = HardwareConfig(
        peak_tflops_mm=1979,
        peak_tflops_math=989,
        peak_memory_bandwidth_GBps=3350,
        gpus_per_node=1,
    )
    assert hw.peak_tflops_mm == 1979
    assert hw.gpus_per_node == 1
    assert hw.peak_tflops_mm_fp8 is None
    assert hw.inter_node_bandwidth_GBps is None    # absent for single-node
    assert hw.topology is None                     # absent until set


def test_hardware_config_full_node():
    hw = HardwareConfig(
        peak_tflops_mm=1979, peak_tflops_math=989,
        peak_memory_bandwidth_GBps=3350, peak_tflops_mm_fp8=3958,
        gpus_per_node=8,
        inter_node_bandwidth_GBps=200, inter_node_latency_us=5,
        topology={"type": "two_layer_multipath", "num_racks": 1, "nodes_per_rack": 1, "num_spines": 1,
                  "intra_node_bandwidth_GBps": 900.0,
                  "per_gpu_bandwidth_GBps": 200.0, "uplink_bandwidth_GBps": 1600.0},
    )
    assert hw.inter_node_bandwidth_GBps == 200
    assert hw.topology["intra_node_bandwidth_GBps"] == 900.0


def test_hardware_config_negative_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        HardwareConfig(
            peak_tflops_mm=-1, peak_tflops_math=989,
            peak_memory_bandwidth_GBps=3350, gpus_per_node=1,
        )


import textwrap
from syssim.training.spec import load_model_yaml


def test_load_model_yaml_megatron_fields(tmp_path):
    yaml_text = textwrap.dedent("""
        num_layers: 28
        hidden_size: 2048
        num_attention_heads: 16
        num_query_groups: 8
        ffn_hidden_size: 6144
        seq_length: 4096
        max_position_embeddings: 40960
        vocab_size: 151936
        swiglu: true
        rope: true
        rope_theta: 1000000.0
        tie_word_embeddings: true
        rms_norm_eps: 1.0e-6
    """).strip()
    p = tmp_path / "qwen3-1_7b.yaml"
    p.write_text(yaml_text)
    cfg = load_model_yaml(str(p))
    assert cfg.num_layers == 28
    assert cfg.tie_word_embeddings is True
    assert cfg.huggingface is None


def test_load_model_yaml_rejects_non_model_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("num_layers: 1\nparallelism:\n  tp: 4\n")
    with pytest.raises(ValueError, match="disallowed key"):
        load_model_yaml(str(p))


def test_load_model_yaml_rejects_neither_branch(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("{}\n")
    with pytest.raises(ValueError, match="huggingface"):
        load_model_yaml(str(p))


from syssim.training.spec import load_hardware_yaml


def test_load_hardware_yaml_single_gpu(tmp_path):
    p = tmp_path / "single_h100.yaml"
    p.write_text(textwrap.dedent("""
        peak_tflops_mm: 1979
        peak_tflops_math: 989
        peak_memory_bandwidth_GBps: 3350
        peak_tflops_mm_fp8: 3958
        gpus_per_node: 1
    """).strip())
    hw = load_hardware_yaml(str(p))
    assert hw.gpus_per_node == 1


def test_load_hardware_yaml_dgx_node(tmp_path):
    p = tmp_path / "dgx_h100.yaml"
    p.write_text(textwrap.dedent("""
        peak_tflops_mm: 1979
        peak_tflops_math: 989
        peak_memory_bandwidth_GBps: 3350
        peak_tflops_mm_fp8: 3958
        gpus_per_node: 8
        inter_node_bandwidth_GBps: 200
        inter_node_latency_us: 5
        topology:
          type: two_layer_multipath
          num_racks: 1
          nodes_per_rack: 1
          num_spines: 1
          intra_node_bandwidth_GBps: 900
          intra_node_latency_us: 1
          per_gpu_bandwidth_GBps: 200
          uplink_bandwidth_GBps: 1600
    """).strip())
    hw = load_hardware_yaml(str(p))
    assert hw.inter_node_bandwidth_GBps == 200
    assert hw.topology["intra_node_bandwidth_GBps"] == 900


def test_load_hardware_yaml_rejects_disallowed_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("peak_tflops_mm: 1\nparallelism:\n  tp: 4\n")
    with pytest.raises(ValueError, match="disallowed key"):
        load_hardware_yaml(str(p))


from syssim.training.spec import derive_num_nodes


def test_derive_num_nodes_single_node():
    par = ParallelismConfig(tp=4, dp=2)              # world_size = 8
    hw = HardwareConfig(
        peak_tflops_mm=1979, peak_tflops_math=989,
        peak_memory_bandwidth_GBps=3350, gpus_per_node=8,
    )
    assert derive_num_nodes(par, hw) == 1


def test_derive_num_nodes_multi_node():
    par = ParallelismConfig(tp=4, dp=4)              # world_size = 16
    hw = HardwareConfig(
        peak_tflops_mm=1979, peak_tflops_math=989,
        peak_memory_bandwidth_GBps=3350, gpus_per_node=8,
        inter_node_bandwidth_GBps=200,
    )
    assert derive_num_nodes(par, hw) == 2


def test_derive_num_nodes_indivisible_raises():
    par = ParallelismConfig(tp=4, dp=3)              # world_size = 12
    hw = HardwareConfig(
        peak_tflops_mm=1979, peak_tflops_math=989,
        peak_memory_bandwidth_GBps=3350, gpus_per_node=8,
    )
    with pytest.raises(ValueError, match="not divisible"):
        derive_num_nodes(par, hw)


def test_derive_num_nodes_multi_node_requires_inter_node_bw():
    par = ParallelismConfig(tp=4, dp=4)
    hw = HardwareConfig(
        peak_tflops_mm=1979, peak_tflops_math=989,
        peak_memory_bandwidth_GBps=3350, gpus_per_node=8,
    )
    with pytest.raises(ValueError, match="inter_node_bandwidth_GBps"):
        derive_num_nodes(par, hw)


def test_top_level_exports_core():
    """Core public surface: simulate, trace, Trace, model/parallelism/training/hardware configs."""
    import syssim
    for name in (
        "simulate", "trace", "Trace",
        "HFModel", "CustomModel", "SimulationReport",
        "ModelConfig", "ParallelismConfig", "TrainingConfig", "HardwareConfig",
    ):
        assert hasattr(syssim, name), name


def test_top_level_exports_estimate_and_sweep():
    """`estimate_memory` and `sweep` are also exported at the top level."""
    import syssim
    for name in ("estimate_memory", "sweep"):
        assert hasattr(syssim, name), name


def test_apply_overrides_dotted_keys():
    from syssim.training.spec import apply_overrides
    data = {
        "parallelism": {"tp": 1, "dp": 1},
        "training": {"micro_batch": 1, "global_batch": 16, "bf16": False},
    }
    out = apply_overrides(data, ["parallelism.tp=4", "training.bf16=true"])
    assert out["parallelism"]["tp"] == 4
    assert out["training"]["bf16"] is True


def test_apply_overrides_rejects_missing_path():
    from syssim.training.spec import apply_overrides
    with pytest.raises(KeyError, match="missing"):
        apply_overrides({"training": {}}, ["training.missing=1"])
