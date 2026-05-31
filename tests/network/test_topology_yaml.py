"""End-to-end YAML -> Topology smoke test for the per-dimension network format."""

import pytest
import yaml
from syssim.training.spec import HardwareConfig
from syssim.network.topology import build_topology_from_config


def test_topology_yaml_switch_dimension():
    cfg = """
peak_tflops_mm: 1979
peak_tflops_math: 989
peak_memory_bandwidth_GBps: 3350
gpus_per_node: 4
topology:
  dims:      [ switch ]
  size:      [ 4 ]
  bandwidth: [ 900 ]
  latency:   [ 2000 ]
"""
    hw = HardwareConfig(**yaml.safe_load(cfg))
    topo = build_topology_from_config(hw)
    assert len(topo.gpus) == 4
    assert len(topo.leaf_switches) == 1        # one switch for the dimension
    assert topo.resolve_path(0, 1, 0)          # a route exists


def test_topology_yaml_two_dimensions():
    """A 2-D topology (intra-node NVLink mesh + an inter-group switch) builds 8 GPUs."""
    hw = HardwareConfig(**{
        "peak_tflops_mm": 1979, "peak_tflops_math": 989, "peak_memory_bandwidth_GBps": 3350,
        "gpus_per_node": 4,
        "topology": {
            "dims": ["fully_connected", "switch"], "size": [4, 2],
            "bandwidth": [900, 200], "latency": [1000, 5000],
        },
    })
    topo = build_topology_from_config(hw)
    assert len(topo.gpus) == 8
    assert topo.load_balancer_name == "ecmp_hash"   # default
    assert topo.resolve_path(0, 5, 0)               # cross-dimension route exists


def test_topology_yaml_requires_dimensional_format():
    """A topology block without the per-dimension `dims` key is rejected."""
    hw = HardwareConfig(**{
        "peak_tflops_mm": 1979, "peak_tflops_math": 989, "peak_memory_bandwidth_GBps": 3350,
        "gpus_per_node": 4,
        "topology": {"type": "simple", "num_nodes": 1, "intra_node_bandwidth_GBps": 900},
    })
    with pytest.raises(ValueError, match="dims"):
        build_topology_from_config(hw)
