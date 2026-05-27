"""End-to-end YAML → Topology smoke test."""

import pytest
import yaml
from syssim.training.spec import HardwareConfig
from syssim.network.topology import build_topology_from_config


def test_topology_yaml_two_layer_multipath(tmp_path):
    cfg = """
peak_tflops_mm: 1979
peak_tflops_math: 989
peak_memory_bandwidth_GBps: 3350
gpus_per_node: 8
topology:
  type: two_layer_multipath
  num_racks: 4
  nodes_per_rack: 1
  num_spines: 2
  intra_node_bandwidth_GBps: 900
  intra_node_latency_us: 1
  per_gpu_bandwidth_GBps: 25
  uplink_bandwidth_GBps: 200
  inter_node_latency_us: 5
"""
    f = tmp_path / "hw.yaml"
    f.write_text(cfg)
    hw = HardwareConfig(**yaml.safe_load(cfg))
    topology = build_topology_from_config(hw)
    assert len(topology.gpus) == 32        # 4 nodes × 8 gpus
    assert len(topology.leaf_switches) == 4
    assert len(topology.spine_switches) == 2


def test_topology_yaml_load_balancer_default():
    """Omitting `load_balancer` in topology block uses ecmp_hash."""
    hw_dict = {
        "peak_tflops_mm": 1979,
        "peak_tflops_math": 989,
        "peak_memory_bandwidth_GBps": 3350,
        "gpus_per_node": 8,
        "topology": {
            "type": "two_layer_multipath",
            "num_racks": 2, "nodes_per_rack": 1,
            "num_spines": 2,
            "intra_node_bandwidth_GBps": 900,
            "per_gpu_bandwidth_GBps": 25.0,
            "uplink_bandwidth_GBps": 200.0,
        },
    }
    hw = HardwareConfig(**hw_dict)
    topology = build_topology_from_config(hw)
    assert topology.load_balancer_name == "ecmp_hash"


def test_topology_yaml_missing_intra_node_bandwidth_raises():
    """topology block must specify intra_node_bandwidth_GBps."""
    hw_dict = {
        "peak_tflops_mm": 1979,
        "peak_tflops_math": 989,
        "peak_memory_bandwidth_GBps": 3350,
        "gpus_per_node": 8,
        "topology": {
            "type": "two_layer_multipath",
            "num_racks": 2, "nodes_per_rack": 1,
            "num_spines": 2,
            "per_gpu_bandwidth_GBps": 25.0,
            "uplink_bandwidth_GBps": 200.0,
        },
    }
    hw = HardwareConfig(**hw_dict)
    with pytest.raises(ValueError, match="intra_node_bandwidth_GBps"):
        build_topology_from_config(hw)
