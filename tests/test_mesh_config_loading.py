"""Integration tests for mesh-based config loading.

Tests that config files are parsed correctly and validated.
"""

import json
from pathlib import Path

import pytest

from syssim.network.profiler import load_hierarchy_config


class TestMeshConfigLoading:
    """Test loading mesh-based hierarchy configs."""

    def test_load_perlmutter_config(self):
        """Test loading Perlmutter mesh config."""
        config_path = Path("examples/configs/perlmutter_mesh.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)

        # Check top-level fields
        assert config.topology_name == "perlmutter"

        # Check mesh
        mesh = config.get_device_mesh()
        assert mesh.shape == (4, 4)
        assert mesh.dimension_names == ["node", "gpu"]
        assert mesh.total_ranks == 16

        # Layers are auto-generated from mesh dimensions
        layers = config.get_auto_layers()
        assert set(layers.keys()) == {"node", "gpu"}
        assert layers["gpu"].topology_type == "nvlink"
        assert layers["node"].topology_type == "slingshot"

    def test_load_dgx_config(self):
        """Test loading DGX A100 mesh config."""
        config_path = Path("examples/configs/dgx_a100_mesh.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)

        mesh = config.get_device_mesh()
        # mesh shape and total_ranks reflect the config — just check basic invariants
        assert len(mesh.shape) == 2
        assert mesh.total_ranks == mesh.shape[0] * mesh.shape[1]

        layers = config.get_auto_layers()
        assert len(layers) == 2

    def test_load_3d_mesh_config(self):
        """Test loading 3D hierarchical mesh config."""
        config_path = Path("examples/configs/3d_mesh_example.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)

        mesh = config.get_device_mesh()
        assert len(mesh.shape) == 3

        layers = config.get_auto_layers()
        assert len(layers) == 3

    def test_get_rank_pairs_from_auto_layer(self):
        """Auto-generated layer should produce valid rank pairs."""
        config_path = Path("examples/configs/perlmutter_mesh.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)
        mesh = config.get_device_mesh()
        layers = config.get_auto_layers()

        # Layer for varying "gpu" within node 0
        gpu_layer = layers["gpu"]
        pairs = gpu_layer.get_rank_pairs(mesh)
        assert len(pairs) >= 1
        src, dst = pairs[0]
        assert 0 <= src < mesh.total_ranks
        assert 0 <= dst < mesh.total_ranks
        assert src != dst

    def test_get_all_ranks_from_auto_layer(self):
        """Auto-generated layer should produce a list of ranks."""
        config_path = Path("examples/configs/perlmutter_mesh.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)
        mesh = config.get_device_mesh()
        layers = config.get_auto_layers()

        # Varying "gpu" with node fixed at 0 → first row of mesh
        ranks = layers["gpu"].get_all_ranks(mesh)
        assert len(ranks) == mesh.shape[1]  # all GPUs in node 0


class TestMeshConfigValidation:
    """Test config validation catches errors."""

    def test_missing_mesh_field(self, tmp_path):
        """Missing top-level mesh field should be caught."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(json.dumps({"topology_name": "test", "profiling_params": {}}))

        with pytest.raises(ValueError, match="Missing 'mesh' field"):
            load_hierarchy_config(config_file)

    def test_missing_topology_types_field(self, tmp_path):
        """Missing mesh.topology_types should be caught."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(
            json.dumps(
                {
                    "topology_name": "test",
                    "mesh": {"shape": [2, 2], "dimension_names": ["a", "b"]},
                    "profiling_params": {},
                }
            )
        )

        with pytest.raises(ValueError, match="mesh must contain 'topology_types'"):
            load_hierarchy_config(config_file)

    def test_missing_dimension_names_field(self, tmp_path):
        """Missing mesh.dimension_names should be caught."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(
            json.dumps(
                {
                    "topology_name": "test",
                    "mesh": {"shape": [2, 2], "topology_types": ["nvlink", "nvlink"]},
                    "profiling_params": {},
                }
            )
        )

        with pytest.raises(ValueError, match="mesh must contain 'dimension_names'"):
            load_hierarchy_config(config_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
