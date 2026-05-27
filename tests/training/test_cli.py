import subprocess
import sys
import textwrap
import pytest


def _model_yaml(tmp_path):
    p = tmp_path / "model.yaml"
    p.write_text(textwrap.dedent("""
        num_layers: 2
        hidden_size: 64
        num_attention_heads: 4
        num_query_groups: 4
        ffn_hidden_size: 128
        seq_length: 32
        max_position_embeddings: 32
        vocab_size: 128
    """).strip())
    return str(p)


def _hardware_yaml(tmp_path):
    p = tmp_path / "hw.yaml"
    p.write_text(textwrap.dedent("""
        peak_tflops_mm: 1979
        peak_tflops_math: 989
        peak_memory_bandwidth_GBps: 3350
        gpus_per_node: 1
        topology:
          type: two_layer_multipath
          num_racks: 1
          nodes_per_rack: 1
          num_spines: 1
          intra_node_bandwidth_GBps: 900
          per_gpu_bandwidth_GBps: 200
          uplink_bandwidth_GBps: 200
    """).strip())
    return str(p)


def test_cli_run_prints_step_time(tmp_path):
    pytest.importorskip("megatron.core")
    r = subprocess.run(
        [sys.executable, "-m", "syssim", "run",
         _model_yaml(tmp_path), "--hardware", _hardware_yaml(tmp_path),
         "--micro-batch", "1", "--global-batch", "1"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "step_time_ms" in r.stdout


def test_cli_memory(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "syssim", "memory",
         _model_yaml(tmp_path), "--hardware", _hardware_yaml(tmp_path),
         "--micro-batch", "1", "--global-batch", "1"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "peak_memory_gb" in r.stdout


def test_cli_summary(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "syssim", "summary",
         _model_yaml(tmp_path), "--hardware", _hardware_yaml(tmp_path),
         "--micro-batch", "1", "--global-batch", "1"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "world_size" in r.stdout
