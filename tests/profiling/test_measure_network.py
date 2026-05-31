"""Unit tests for the pure topology-derivation in `syssim profile --network` (CPU-only)."""

from syssim.profiling.measure_network import derive_topology, emit_topology


def test_derive_topology_single_node():
    """Intra-node only: fully_connected dim; bandwidth = busBW * (gpus-1); latency = floor/6 hops."""
    meas = {"intra": {"busBW_GBps": 100.0, "latency_floor_us": 72.0}}
    topo = derive_topology(meas, gpus_per_node=4, nodes=1)
    assert topo["dims"] == ["fully_connected"]
    assert topo["size"] == [4]
    assert topo["bandwidth"] == [300]          # 100 * (4-1)
    assert topo["latency"] == [12000]          # 72us*1000 / (2*3) = 72000/6


def test_derive_topology_multi_node():
    """Adds an inter-node `switch` dim (degree 1, so bandwidth = busBW; latency = floor/(2*(nodes-1)))."""
    meas = {"intra": {"busBW_GBps": 100.0, "latency_floor_us": 72.0},
            "inter": {"busBW_GBps": 92.0, "latency_floor_us": 27.0}}
    topo = derive_topology(meas, gpus_per_node=4, nodes=2)
    assert topo["dims"] == ["fully_connected", "switch"]
    assert topo["size"] == [4, 2]
    assert topo["bandwidth"] == [300, 92]      # inter: switch degree 1 -> 92*1
    assert topo["latency"] == [12000, 13500]   # inter: 27us*1000 / (2*(2-1)) = 13500


def test_derived_topology_builds_a_valid_topology():
    """The derived block round-trips through the real topology builder."""
    from syssim.training.spec import HardwareConfig
    from syssim.network.topology import build_topology_from_config
    meas = {"intra": {"busBW_GBps": 100.0, "latency_floor_us": 72.0},
            "inter": {"busBW_GBps": 92.0, "latency_floor_us": 27.0}}
    topo = derive_topology(meas, gpus_per_node=4, nodes=2)
    hw = HardwareConfig(peak_tflops_mm=1979, peak_tflops_math=989,
                        peak_memory_bandwidth_GBps=3350, gpus_per_node=4,
                        inter_node_bandwidth_GBps=100, topology=topo)
    built = build_topology_from_config(hw)
    assert len(built.gpus) == 8


def test_emit_topology_writes_hardware_yaml(tmp_path):
    """--hardware writes the derived topology block into the YAML in place."""
    import yaml
    hw_path = tmp_path / "hw.yaml"
    hw_path.write_text(yaml.safe_dump({
        "peak_tflops_mm": 1979, "peak_tflops_math": 989, "peak_memory_bandwidth_GBps": 3350,
        "gpus_per_node": 4, "topology": {"dims": ["fully_connected"], "size": [4],
                                         "bandwidth": [450], "latency": [12000]}}))
    meas = {"intra": {"busBW_GBps": 100.0, "latency_floor_us": 72.0},
            "inter": {"busBW_GBps": 92.0, "latency_floor_us": 27.0}}
    topo = derive_topology(meas, gpus_per_node=4, nodes=2)
    block = emit_topology(topo, meas, out_dir=str(tmp_path), hardware_path=str(hw_path))
    assert "topology:" in block and "switch" in block
    written = yaml.safe_load(hw_path.read_text())
    assert written["topology"]["bandwidth"] == [300, 92]
    assert (tmp_path / "network.json").exists()
