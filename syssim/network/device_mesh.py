"""Device mesh abstraction for hierarchical profiling.

Provides logical multi-dimensional device layout for intuitive network layer specification.
Enables automatic rank enumeration from mesh topology rather than manual rank lists.

Example:
    >>> # 4 nodes × 4 GPUs/node mesh
    >>> mesh = DeviceMesh(shape=[4, 4], dimension_names=["node", "gpu_in_node"])
    >>> mesh.rank_at([0, 2])  # Node 0, GPU 2 → rank 2
    2
    >>> mesh.coords_of(2)  # Reverse lookup
    [0, 2]
    >>> # Get all GPUs on node 0
    >>> mesh.ranks_in_slice({"node": 0}, ["gpu_in_node"])
    [0, 1, 2, 3]
    >>> # Get GPU 0 on all nodes
    >>> mesh.ranks_in_slice({"gpu_in_node": 0}, ["node"])
    [0, 4, 8, 12]
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DeviceMesh:
    """Logical device mesh for hierarchical profiling.

    Maps logical coordinates (node, GPU, rack, etc.) to global ranks.
    Enables intuitive specification of network layers via mesh dimensions.

    Attributes:
        shape: Mesh dimensions (e.g., [4 nodes, 4 GPUs/node] → shape=[4,4])
        dimension_names: Names for each dimension (e.g., ["node", "gpu"])
        topology_types: Topology type for each dimension (e.g., ["infiniband", "nvlink"])
        ranks_order: 'C' (row-major, default) or 'F' (column-major)

    Example:
        >>> mesh = DeviceMesh(
        ...     shape=[2, 4],
        ...     dimension_names=["node", "gpu"],
        ...     topology_types=["infiniband", "nvlink"]
        ... )
        >>> mesh.rank_at([0, 2])  # Node 0, GPU 2
        2
        >>> mesh.total_ranks
        8
    """

    shape: Tuple[int, ...]
    dimension_names: List[str]
    topology_types: List[str] = None
    ranks_order: str = "C"  # 'C' (row-major) or 'F' (column-major)

    def __post_init__(self):
        """Validate mesh parameters and compute derived attributes."""
        # Convert shape to tuple if it's a list
        if isinstance(self.shape, list):
            object.__setattr__(self, "shape", tuple(self.shape))

        if len(self.shape) != len(self.dimension_names):
            raise ValueError(f"shape {self.shape} and dimension_names {self.dimension_names} must have same length")

        # Validate topology_types if provided
        if self.topology_types is not None:
            if len(self.topology_types) != len(self.dimension_names):
                raise ValueError(
                    f"topology_types {self.topology_types} must have same length as "
                    f"dimension_names {self.dimension_names}"
                )

        if self.ranks_order not in ["C", "F"]:
            raise ValueError(f"ranks_order must be 'C' (row-major) or 'F' (column-major), got '{self.ranks_order}'")

        # Validate shape values
        for i, dim_size in enumerate(self.shape):
            if dim_size <= 0:
                raise ValueError(f"shape[{i}] ('{self.dimension_names[i]}') must be positive, got {dim_size}")

        # Validate dimension names are unique
        if len(self.dimension_names) != len(set(self.dimension_names)):
            raise ValueError(f"dimension_names must be unique, got {self.dimension_names}")

        self.total_ranks = int(np.prod(self.shape))

    def rank_at(self, coords: List[int]) -> int:
        """Convert mesh coordinates to global rank.

        Args:
            coords: Mesh coordinates (e.g., [node_idx, gpu_idx])

        Returns:
            Global rank (0 to total_ranks-1)

        Raises:
            ValueError: If coords length mismatches shape or out of bounds

        Example:
            >>> mesh = DeviceMesh([4, 4], ["node", "gpu"])
            >>> mesh.rank_at([1, 2])  # Node 1, GPU 2 → rank 6 (row-major)
            6
        """
        if len(coords) != len(self.shape):
            raise ValueError(f"coords {coords} length must match shape {self.shape}")

        # Validate bounds
        for i, (coord, dim_size) in enumerate(zip(coords, self.shape)):
            if not (0 <= coord < dim_size):
                raise ValueError(f"coords[{i}] ('{self.dimension_names[i]}') = {coord} out of bounds [0, {dim_size})")

        return int(np.ravel_multi_index(coords, self.shape, order=self.ranks_order))

    def coords_of(self, rank: int) -> List[int]:
        """Convert global rank to mesh coordinates.

        Args:
            rank: Global rank (0 to total_ranks-1)

        Returns:
            Mesh coordinates [node_idx, gpu_idx, ...]

        Raises:
            ValueError: If rank out of bounds

        Example:
            >>> mesh = DeviceMesh([4, 4], ["node", "gpu"])
            >>> mesh.coords_of(6)  # Rank 6 → [1, 2] (node 1, GPU 2)
            [1, 2]
        """
        if not (0 <= rank < self.total_ranks):
            raise ValueError(f"rank {rank} out of bounds [0, {self.total_ranks})")

        return list(np.unravel_index(rank, self.shape, order=self.ranks_order))

    def ranks_in_slice(self, fix_dims: Dict[str, int], vary_dims: List[str]) -> List[int]:
        """Get all ranks where fix_dims are constant, vary_dims change.

        Args:
            fix_dims: Dimensions to hold constant (explicit int values)
            vary_dims: Dimensions that vary across the slice

        Returns:
            List of ranks in the mesh slice (sorted)

        Raises:
            ValueError: If dimension names invalid or fix_dims have non-int values

        Example:
            >>> mesh = DeviceMesh([4, 4], ["node", "gpu"])
            >>> # All GPUs on node 0
            >>> mesh.ranks_in_slice({"node": 0}, ["gpu"])
            [0, 1, 2, 3]
            >>> # GPU 0 on all nodes
            >>> mesh.ranks_in_slice({"gpu": 0}, ["node"])
            [0, 4, 8, 12]

        Note:
            Wildcards (e.g., "*") are NOT supported. All fix_dims must be explicit integers.
        """
        # Validate fix_dims keys and values
        for dim_name, dim_value in fix_dims.items():
            if dim_name not in self.dimension_names:
                raise ValueError(f"fix_dims contains '{dim_name}' not in mesh.dimension_names {self.dimension_names}")

            if not isinstance(dim_value, int):
                raise ValueError(
                    f"fix_dims['{dim_name}'] must be int, got {type(dim_value).__name__}. Wildcards not supported."
                )

            # Validate bounds
            dim_idx = self.dimension_names.index(dim_name)
            if not (0 <= dim_value < self.shape[dim_idx]):
                raise ValueError(f"fix_dims['{dim_name}'] = {dim_value} out of bounds [0, {self.shape[dim_idx]})")

        # Validate vary_dims
        for dim_name in vary_dims:
            if dim_name not in self.dimension_names:
                raise ValueError(f"vary_dims contains '{dim_name}' not in mesh.dimension_names {self.dimension_names}")

        # Iterate over all mesh coords, filter by constraints
        ranks = []
        for idx in range(self.total_ranks):
            coords = self.coords_of(idx)
            match = True

            # Check fix_dims constraints (all must be exact matches)
            for dim_name, dim_value in fix_dims.items():
                dim_idx = self.dimension_names.index(dim_name)
                if coords[dim_idx] != dim_value:
                    match = False
                    break

            if match:
                ranks.append(idx)

        return sorted(ranks)

    def get_representative_pairs(
        self, fix_dims: Dict[str, int], vary_dims: List[str], num_pairs: int = 1
    ) -> List[Tuple[int, int]]:
        """Get representative rank pairs from a mesh slice.

        Returns pairs where vary_dims differ, fix_dims are constant.
        Selects diagonal pairs for efficiency (avoid all-to-all).

        Args:
            fix_dims: Dimensions to hold constant (explicit int values)
            vary_dims: Dimensions that vary
            num_pairs: Number of pairs to return (default 1)

        Returns:
            List of (src, dst) rank pairs

        Raises:
            ValueError: If slice has <2 ranks or fix_dims are invalid

        Example:
            >>> mesh = DeviceMesh([4, 4], ["node", "gpu"])
            >>> # Intra-node profiling: node=0, vary GPU
            >>> mesh.get_representative_pairs({"node": 0}, ["gpu"], num_pairs=2)
            [(0, 1), (0, 2)]  # GPU 0→1, 0→2 on node 0
            >>> # Inter-node profiling: gpu=0, vary node
            >>> mesh.get_representative_pairs({"gpu": 0}, ["node"], num_pairs=2)
            [(0, 4), (0, 8)]  # Node 0→1, 0→2 (GPU 0)
        """
        ranks = self.ranks_in_slice(fix_dims, vary_dims)

        if len(ranks) < 2:
            raise ValueError(
                f"Slice has <2 ranks, cannot form pairs: fix_dims={fix_dims}, vary_dims={vary_dims}, ranks={ranks}"
            )

        # Return diagonal pairs: (0,1), (0,2), ..., (0, num_pairs)
        pairs = []
        for i in range(1, min(num_pairs + 1, len(ranks))):
            pairs.append((ranks[0], ranks[i]))

        return pairs

    def validate_dimension_scope(self, fix_dims: Dict[str, int], vary_dims: List[str]) -> None:
        """Validate that fix_dims and vary_dims are consistent with mesh.

        Args:
            fix_dims: Dimensions to hold constant
            vary_dims: Dimensions that vary

        Raises:
            ValueError: If dimensions are invalid or incompatible
        """
        # Check for overlap
        overlap = set(fix_dims.keys()) & set(vary_dims)
        if overlap:
            raise ValueError(f"fix_dims and vary_dims cannot overlap, found: {overlap}")

        # Check coverage (all dimensions should be either fixed or varying)
        all_dims = set(fix_dims.keys()) | set(vary_dims)
        mesh_dims = set(self.dimension_names)

        mesh_dims - all_dims
        extra = all_dims - mesh_dims

        if extra:
            raise ValueError(
                f"Scope contains dimensions not in mesh: {extra}. Valid dimensions: {self.dimension_names}"
            )

        # Note: missing dimensions are allowed (they become implicit varying dimensions)

    def __repr__(self) -> str:
        """String representation of mesh."""
        return (
            f"DeviceMesh(shape={list(self.shape)}, "
            f"dimension_names={self.dimension_names}, "
            f"total_ranks={self.total_ranks})"
        )
