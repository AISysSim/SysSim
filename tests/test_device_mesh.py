"""Unit tests for DeviceMesh class.

Tests mesh coordinate mapping, rank enumeration, and validation.
"""

import pytest

from syssim.network.device_mesh import DeviceMesh


class TestDeviceMeshBasic:
    """Basic DeviceMesh functionality tests."""

    def test_2d_mesh_creation(self):
        """Test creating a simple 2D mesh."""
        mesh = DeviceMesh(shape=[4, 4], dimension_names=["node", "gpu_in_node"])

        assert mesh.total_ranks == 16
        assert mesh.shape == (4, 4)
        assert mesh.dimension_names == ["node", "gpu_in_node"]
        assert mesh.ranks_order == "C"  # Default

    def test_3d_mesh_creation(self):
        """Test creating a 3D mesh (rack, node, gpu)."""
        mesh = DeviceMesh(shape=[2, 4, 4], dimension_names=["rack", "node_in_rack", "gpu_in_node"])

        assert mesh.total_ranks == 32
        assert mesh.shape == (2, 4, 4)
        assert len(mesh.dimension_names) == 3

    def test_shape_dimension_names_mismatch(self):
        """Test that shape and dimension_names must have same length."""
        with pytest.raises(ValueError, match="must have same length"):
            DeviceMesh(
                shape=[4, 4],
                dimension_names=["node"],  # Only 1 name, but shape is 2D
            )

    def test_invalid_ranks_order(self):
        """Test that ranks_order must be 'C' or 'F'."""
        with pytest.raises(ValueError, match="must be 'C'.*or 'F'"):
            DeviceMesh(
                shape=[4, 4],
                dimension_names=["node", "gpu"],
                ranks_order="X",  # Invalid
            )

    def test_zero_dimension_size(self):
        """Test that dimension sizes must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            DeviceMesh(
                shape=[4, 0],  # Invalid: 0 GPUs
                dimension_names=["node", "gpu"],
            )

    def test_duplicate_dimension_names(self):
        """Test that dimension names must be unique."""
        with pytest.raises(ValueError, match="must be unique"):
            DeviceMesh(
                shape=[4, 4],
                dimension_names=["node", "node"],  # Duplicate
            )


class TestDeviceMeshCoordinates:
    """Test coordinate ↔ rank mapping."""

    def test_rank_at_2d(self):
        """Test rank_at() for 2D mesh (row-major)."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        # Row-major: rank = node*4 + gpu
        assert mesh.rank_at([0, 0]) == 0
        assert mesh.rank_at([0, 1]) == 1
        assert mesh.rank_at([0, 3]) == 3
        assert mesh.rank_at([1, 0]) == 4
        assert mesh.rank_at([1, 2]) == 6
        assert mesh.rank_at([3, 3]) == 15

    def test_coords_of_2d(self):
        """Test coords_of() for 2D mesh (row-major)."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        assert mesh.coords_of(0) == [0, 0]
        assert mesh.coords_of(1) == [0, 1]
        assert mesh.coords_of(4) == [1, 0]
        assert mesh.coords_of(6) == [1, 2]
        assert mesh.coords_of(15) == [3, 3]

    def test_rank_at_coords_of_inverse(self):
        """Test that rank_at() and coords_of() are inverses."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        for rank in range(mesh.total_ranks):
            coords = mesh.coords_of(rank)
            reconstructed_rank = mesh.rank_at(coords)
            assert reconstructed_rank == rank

    def test_3d_mesh_coordinates(self):
        """Test 3D mesh coordinate mapping."""
        mesh = DeviceMesh([2, 4, 4], ["rack", "node", "gpu"])

        # Row-major: rank = rack*16 + node*4 + gpu
        assert mesh.rank_at([0, 0, 0]) == 0
        assert mesh.rank_at([0, 1, 2]) == 6
        assert mesh.rank_at([1, 0, 0]) == 16
        assert mesh.rank_at([1, 3, 3]) == 31

        assert mesh.coords_of(0) == [0, 0, 0]
        assert mesh.coords_of(6) == [0, 1, 2]
        assert mesh.coords_of(16) == [1, 0, 0]
        assert mesh.coords_of(31) == [1, 3, 3]

    def test_column_major_order(self):
        """Test column-major (Fortran) ordering."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"], ranks_order="F")

        # Column-major: rank = node + gpu*4
        assert mesh.rank_at([0, 0]) == 0
        assert mesh.rank_at([1, 0]) == 1
        assert mesh.rank_at([0, 1]) == 4
        assert mesh.rank_at([3, 3]) == 15

    def test_rank_at_out_of_bounds(self):
        """Test rank_at() with out-of-bounds coordinates."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        with pytest.raises(ValueError, match="out of bounds"):
            mesh.rank_at([4, 0])  # node=4 is out of bounds

        with pytest.raises(ValueError, match="out of bounds"):
            mesh.rank_at([0, 5])  # gpu=5 is out of bounds

    def test_coords_of_out_of_bounds(self):
        """Test coords_of() with out-of-bounds rank."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        with pytest.raises(ValueError, match="out of bounds"):
            mesh.coords_of(16)  # Only 0-15 valid

        with pytest.raises(ValueError, match="out of bounds"):
            mesh.coords_of(-1)


class TestDeviceMeshSlices:
    """Test ranks_in_slice() for mesh subsetting."""

    def test_ranks_in_slice_intra_node(self):
        """Test getting all GPUs on a single node."""
        mesh = DeviceMesh([4, 4], ["node", "gpu_in_node"])

        # All GPUs on node 0
        ranks = mesh.ranks_in_slice({"node": 0}, ["gpu_in_node"])
        assert ranks == [0, 1, 2, 3]

        # All GPUs on node 2
        ranks = mesh.ranks_in_slice({"node": 2}, ["gpu_in_node"])
        assert ranks == [8, 9, 10, 11]

    def test_ranks_in_slice_inter_node(self):
        """Test getting GPU 0 on all nodes."""
        mesh = DeviceMesh([4, 4], ["node", "gpu_in_node"])

        # GPU 0 on all nodes
        ranks = mesh.ranks_in_slice({"gpu_in_node": 0}, ["node"])
        assert ranks == [0, 4, 8, 12]

        # GPU 2 on all nodes
        ranks = mesh.ranks_in_slice({"gpu_in_node": 2}, ["node"])
        assert ranks == [2, 6, 10, 14]

    def test_ranks_in_slice_3d_mesh(self):
        """Test slicing a 3D mesh."""
        mesh = DeviceMesh([2, 4, 4], ["rack", "node", "gpu"])

        # All GPUs on rack 0, node 1
        ranks = mesh.ranks_in_slice({"rack": 0, "node": 1}, ["gpu"])
        assert ranks == [4, 5, 6, 7]

        # Node 0 on all racks, GPU 0
        ranks = mesh.ranks_in_slice({"node": 0, "gpu": 0}, ["rack"])
        assert ranks == [0, 16]

        # All nodes in rack 1, GPU 0
        ranks = mesh.ranks_in_slice({"rack": 1, "gpu": 0}, ["node"])
        assert ranks == [16, 20, 24, 28]

    def test_ranks_in_slice_invalid_dimension(self):
        """Test ranks_in_slice() with invalid dimension name."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        with pytest.raises(ValueError, match="not in mesh.dimension_names"):
            mesh.ranks_in_slice({"invalid_dim": 0}, ["gpu"])

        with pytest.raises(ValueError, match="not in mesh.dimension_names"):
            mesh.ranks_in_slice({"node": 0}, ["invalid_dim"])

    def test_ranks_in_slice_non_int_value(self):
        """Test ranks_in_slice() with non-int fix_dims value."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        with pytest.raises(ValueError, match="must be int"):
            mesh.ranks_in_slice({"node": "*"}, ["gpu"])  # Wildcard not supported

        with pytest.raises(ValueError, match="must be int"):
            mesh.ranks_in_slice({"node": 0.5}, ["gpu"])  # Float not supported

    def test_ranks_in_slice_out_of_bounds_value(self):
        """Test ranks_in_slice() with out-of-bounds fix_dims value."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        with pytest.raises(ValueError, match="out of bounds"):
            mesh.ranks_in_slice({"node": 4}, ["gpu"])  # node must be 0-3


class TestDeviceMeshRepresentativePairs:
    """Test get_representative_pairs() for profiling."""

    def test_get_representative_pairs_intra_node(self):
        """Test getting rank pairs for intra-node profiling."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        # Intra-node: node=0, vary GPU (1 pair)
        pairs = mesh.get_representative_pairs({"node": 0}, ["gpu"], num_pairs=1)
        assert pairs == [(0, 1)]  # GPU 0→1 on node 0

        # Intra-node: node=0, vary GPU (2 pairs)
        pairs = mesh.get_representative_pairs({"node": 0}, ["gpu"], num_pairs=2)
        assert pairs == [(0, 1), (0, 2)]  # GPU 0→1, 0→2 on node 0

    def test_get_representative_pairs_inter_node(self):
        """Test getting rank pairs for inter-node profiling."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        # Inter-node: gpu=0, vary node (1 pair)
        pairs = mesh.get_representative_pairs({"gpu": 0}, ["node"], num_pairs=1)
        assert pairs == [(0, 4)]  # Node 0→1, GPU 0

        # Inter-node: gpu=0, vary node (2 pairs)
        pairs = mesh.get_representative_pairs({"gpu": 0}, ["node"], num_pairs=2)
        assert pairs == [(0, 4), (0, 8)]  # Node 0→1, 0→2, GPU 0

    def test_get_representative_pairs_3d_mesh(self):
        """Test representative pairs for 3D mesh."""
        mesh = DeviceMesh([2, 4, 4], ["rack", "node", "gpu"])

        # Inter-rack: node=0, gpu=0, vary rack
        pairs = mesh.get_representative_pairs({"node": 0, "gpu": 0}, ["rack"], num_pairs=1)
        assert pairs == [(0, 16)]

        # Inter-node same rack: rack=0, gpu=0, vary node
        pairs = mesh.get_representative_pairs({"rack": 0, "gpu": 0}, ["node"], num_pairs=2)
        assert pairs == [(0, 4), (0, 8)]

    def test_get_representative_pairs_insufficient_ranks(self):
        """Test get_representative_pairs() with <2 ranks in slice."""
        mesh = DeviceMesh([2, 4], ["node", "gpu"])

        # Fix both dimensions → only 1 rank
        with pytest.raises(ValueError, match="<2 ranks"):
            mesh.get_representative_pairs({"node": 0, "gpu": 0}, [], num_pairs=1)

    def test_get_representative_pairs_limit_by_slice_size(self):
        """Test that num_pairs is limited by slice size."""
        mesh = DeviceMesh([2, 4], ["node", "gpu"])

        # Slice has 4 ranks, request 10 pairs → only get 3
        pairs = mesh.get_representative_pairs({"node": 0}, ["gpu"], num_pairs=10)
        assert len(pairs) == 3  # (0,1), (0,2), (0,3)
        assert pairs == [(0, 1), (0, 2), (0, 3)]


class TestDeviceMeshValidation:
    """Test validate_dimension_scope()."""

    def test_validate_dimension_scope_overlap(self):
        """Test that fix_dims and vary_dims cannot overlap."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        with pytest.raises(ValueError, match="cannot overlap"):
            mesh.validate_dimension_scope(
                {"node": 0, "gpu": 0},  # Both fixed
                ["gpu"],  # GPU also in vary_dims → overlap
            )

    def test_validate_dimension_scope_invalid_dimension(self):
        """Test that invalid dimensions are caught."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        with pytest.raises(ValueError, match="not in mesh"):
            mesh.validate_dimension_scope({"invalid_dim": 0}, ["gpu"])

    def test_validate_dimension_scope_partial_coverage(self):
        """Test that partial coverage is allowed (missing dims = implicit varying)."""
        mesh = DeviceMesh([2, 4, 4], ["rack", "node", "gpu"])

        # Only specify rack=0, leave node and gpu unspecified
        # This should NOT raise an error (missing dims are allowed)
        mesh.validate_dimension_scope({"rack": 0}, ["node"])  # gpu is implicitly varying


class TestDeviceMeshRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__() output."""
        mesh = DeviceMesh([4, 4], ["node", "gpu"])

        repr_str = repr(mesh)
        assert "DeviceMesh" in repr_str
        assert "shape=[4, 4]" in repr_str
        assert "dimension_names=['node', 'gpu']" in repr_str
        assert "total_ranks=16" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
