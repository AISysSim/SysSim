"""Network topology abstractions for communication simulation.

This module provides topology models that map (src, dst) pairs to network resources.
The key abstraction is `resolve_path(src, dst) -> list[Resource]`, which returns
the sequence of shared network resources traversed by a message.

Topologies:
- FullyConnectedTopology: Dedicated link per pair (no contention, validation baseline)
- RingTopology: Bidirectional ring with shortest-path routing (added in Phase 4)
- SwitchTopology: Star topology with shared switch (added in Phase 4)
- NVLinkMeshTopology: Fully-connected NVLink mesh (added in Phase 4)
- HierarchicalTopology: Multi-node with NVLink intra-node + InfiniBand inter-node (added in Phase 4)

Design principles:
- Resources are directional (full-duplex link = 2 Resource objects)
- Resources are immutable (frozen dataclass) for safe sharing
- Topology.resolve_path() is the only required method for simulation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Resource:
    """A directional network resource with limited bandwidth.

    Resources represent physical or logical network components that can be shared
    by multiple concurrent messages. Examples:
    - NIC send buffer (one per rank)
    - Network link (directional, e.g., "switch->rank3")
    - Switch fabric (shared by all traffic through switch)

    Attributes:
        name: Unique identifier for this resource
        bandwidth: Maximum bandwidth in bytes/second

    Example:
        >>> link = Resource("nvlink_0->1", 25e9)  # 25 GB/s NVLink
        >>> link.name
        'nvlink_0->1'
        >>> link.bandwidth
        25000000000.0
    """

    name: str
    bandwidth: float  # bytes/second

    def __post_init__(self):
        """Validate resource parameters."""
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {self.bandwidth}")


class Topology(ABC):
    """Abstract base class for network topologies.

    A topology maps (src, dst) communication pairs to sequences of shared network
    resources. The simulator uses this to model bandwidth contention.

    Implementing a new topology requires:
    1. Override resolve_path(src, dst) to return resource sequence
    2. Override all_resources() to return all resources in topology
    3. Optionally override get_loggp(src, dst) for layer-specific LogGP params
    4. Optionally override get_bandwidth(src, dst) for validation/debugging
    """

    @abstractmethod
    def resolve_path(self, src: int, dst: int) -> list[Resource]:
        """Resolve the sequence of resources traversed from src to dst.

        Args:
            src: Source rank ID
            dst: Destination rank ID

        Returns:
            List of Resource objects in traversal order. Empty list if src == dst.

        Example:
            >>> topo = FullyConnectedTopology(4, 25e9)
            >>> path = topo.resolve_path(0, 3)
            >>> [r.name for r in path]
            ['link_0->3']
        """
        pass

    @abstractmethod
    def all_resources(self) -> list[Resource]:
        """Return all resources in the topology.

        Used by simulator to initialize resource usage tracking.

        Returns:
            List of all Resource objects
        """
        pass

    def get_bandwidth(self, src: int, dst: int) -> float:
        """Get effective bandwidth from src to dst.

        Default implementation returns minimum bandwidth across path.
        Topologies can override for specific behavior.

        Args:
            src: Source rank ID
            dst: Destination rank ID

        Returns:
            Effective bandwidth in bytes/second. Returns float('inf') if src == dst.

        Example:
            >>> topo = FullyConnectedTopology(4, 25e9)
            >>> topo.get_bandwidth(0, 3)
            25000000000.0
        """
        if src == dst:
            return float("inf")

        path = self.resolve_path(src, dst)
        if not path:
            return float("inf")

        return min(res.bandwidth for res in path)


class FullyConnectedTopology(Topology):
    """Fully-connected topology with dedicated link per (src, dst) pair.

    This topology has NO contention between different (src, dst) pairs, making it
    ideal for validating collective algorithms against analytical formulas.

    Each directional pair (src, dst) has its own dedicated resource, so concurrent
    messages on different pairs do not interfere.

    Attributes:
        num_ranks: Number of ranks in the topology
        link_bandwidth: Bandwidth per link in bytes/second

    Example:
        >>> topo = FullyConnectedTopology(num_ranks=4, link_bandwidth=25e9)
        >>> path = topo.resolve_path(0, 3)
        >>> path[0].name
        'link_0->3'
        >>> path[0].bandwidth
        25000000000.0
        >>>
        >>> # No shared resources between different pairs
        >>> path_01 = topo.resolve_path(0, 1)
        >>> path_23 = topo.resolve_path(2, 3)
        >>> path_01[0].name == path_23[0].name
        False
    """

    def __init__(self, num_ranks: int, link_bandwidth: float):
        """Initialize fully-connected topology.

        Args:
            num_ranks: Number of ranks (must be >= 2)
            link_bandwidth: Bandwidth per link in bytes/second (must be > 0)

        Raises:
            ValueError: If num_ranks < 2 or link_bandwidth <= 0
        """
        if num_ranks < 2:
            raise ValueError(f"num_ranks must be >= 2, got {num_ranks}")
        if link_bandwidth <= 0:
            raise ValueError(f"link_bandwidth must be positive, got {link_bandwidth}")

        self.num_ranks = num_ranks
        self.link_bandwidth = link_bandwidth

        # Pre-create all resources (num_ranks * (num_ranks - 1) total)
        self._resources: dict[tuple[int, int], Resource] = {}
        for src in range(num_ranks):
            for dst in range(num_ranks):
                if src != dst:
                    self._resources[(src, dst)] = Resource(
                        name=f"link_{src}->{dst}",
                        bandwidth=link_bandwidth,
                    )

    def resolve_path(self, src: int, dst: int) -> list[Resource]:
        """Resolve path from src to dst (single dedicated link).

        Args:
            src: Source rank (0 <= src < num_ranks)
            dst: Destination rank (0 <= dst < num_ranks)

        Returns:
            [Resource(f"link_{src}->{dst}")] if src != dst, else []

        Raises:
            ValueError: If src or dst out of range
        """
        if not (0 <= src < self.num_ranks):
            raise ValueError(f"src {src} out of range [0, {self.num_ranks})")
        if not (0 <= dst < self.num_ranks):
            raise ValueError(f"dst {dst} out of range [0, {self.num_ranks})")

        if src == dst:
            return []

        return [self._resources[(src, dst)]]

    def all_resources(self) -> list[Resource]:
        """Return all P*(P-1) directional links.

        Returns:
            List of all Resource objects
        """
        return list(self._resources.values())


class RingTopology(Topology):
    """Bidirectional ring topology with shortest-path routing.

    Ranks are arranged in a ring: 0 - 1 - 2 - ... - (P-1) - 0
    Each link is bidirectional (two directional resources per physical link).
    Messages use shortest path (forward or backward around ring).

    Attributes:
        num_ranks: Number of ranks in the ring
        link_bandwidth: Bandwidth per directional link in bytes/second

    Example:
        >>> topo = RingTopology(num_ranks=8, link_bandwidth=25e9)
        >>> path = topo.resolve_path(0, 2)  # Forward: 0->1, 1->2
        >>> len(path)
        2
        >>> path = topo.resolve_path(0, 7)  # Backward (shorter): 0->7
        >>> len(path)
        1
    """

    def __init__(self, num_ranks: int, link_bandwidth: float):
        """Initialize ring topology.

        Args:
            num_ranks: Number of ranks (must be >= 2)
            link_bandwidth: Bandwidth per directional link in bytes/second

        Raises:
            ValueError: If num_ranks < 2 or link_bandwidth <= 0
        """
        if num_ranks < 2:
            raise ValueError(f"num_ranks must be >= 2, got {num_ranks}")
        if link_bandwidth <= 0:
            raise ValueError(f"link_bandwidth must be positive, got {link_bandwidth}")

        self.num_ranks = num_ranks
        self.link_bandwidth = link_bandwidth

        # Create bidirectional links
        self._resources: dict[tuple[int, int], Resource] = {}
        for i in range(num_ranks):
            next_rank = (i + 1) % num_ranks

            # Forward link: i -> next
            self._resources[(i, next_rank)] = Resource(
                name=f"ring_{i}->{next_rank}",
                bandwidth=link_bandwidth,
            )

            # Backward link: next -> i
            self._resources[(next_rank, i)] = Resource(
                name=f"ring_{next_rank}->{i}",
                bandwidth=link_bandwidth,
            )

    def resolve_path(self, src: int, dst: int) -> list[Resource]:
        """Resolve shortest path from src to dst on ring.

        Args:
            src: Source rank (0 <= src < num_ranks)
            dst: Destination rank (0 <= dst < num_ranks)

        Returns:
            List of resources on shortest path (forward or backward)

        Raises:
            ValueError: If src or dst out of range
        """
        if not (0 <= src < self.num_ranks):
            raise ValueError(f"src {src} out of range [0, {self.num_ranks})")
        if not (0 <= dst < self.num_ranks):
            raise ValueError(f"dst {dst} out of range [0, {self.num_ranks})")

        if src == dst:
            return []

        # Compute forward and backward distances
        forward_dist = (dst - src) % self.num_ranks
        backward_dist = (src - dst) % self.num_ranks

        # Use shortest path
        if forward_dist <= backward_dist:
            # Forward path
            path = []
            current = src
            for _ in range(forward_dist):
                next_rank = (current + 1) % self.num_ranks
                path.append(self._resources[(current, next_rank)])
                current = next_rank
            return path
        else:
            # Backward path
            path = []
            current = src
            for _ in range(backward_dist):
                prev_rank = (current - 1) % self.num_ranks
                path.append(self._resources[(current, prev_rank)])
                current = prev_rank
            return path

    def all_resources(self) -> list[Resource]:
        """Return all 2*P directional links (P physical links, 2 directions each).

        Returns:
            List of all Resource objects
        """
        return list(self._resources.values())


class SwitchTopology(Topology):
    """Star topology with shared central switch.

    All ranks connect to a central switch. Messages go through:
    - Uplink from src to switch
    - Switch fabric (shared by all traffic)
    - Downlink from switch to dst

    This models ToR (Top-of-Rack) switches with bisection bandwidth limits.

    Attributes:
        num_ranks: Number of ranks connected to switch
        link_bandwidth: Bandwidth per NIC link in bytes/second
        switch_bandwidth: Switch fabric bandwidth in bytes/second

    Example:
        >>> topo = SwitchTopology(num_ranks=8, link_bandwidth=25e9, switch_bandwidth=200e9)
        >>> path = topo.resolve_path(0, 3)
        >>> [r.name for r in path]
        ['uplink_0', 'switch_fabric', 'downlink_3']
    """

    def __init__(self, num_ranks: int, link_bandwidth: float, switch_bandwidth: float):
        """Initialize switch topology.

        Args:
            num_ranks: Number of ranks (must be >= 2)
            link_bandwidth: Bandwidth per NIC link in bytes/second
            switch_bandwidth: Switch fabric bandwidth in bytes/second

        Raises:
            ValueError: If num_ranks < 2 or bandwidths <= 0
        """
        if num_ranks < 2:
            raise ValueError(f"num_ranks must be >= 2, got {num_ranks}")
        if link_bandwidth <= 0:
            raise ValueError(f"link_bandwidth must be positive, got {link_bandwidth}")
        if switch_bandwidth <= 0:
            raise ValueError(f"switch_bandwidth must be positive, got {switch_bandwidth}")

        self.num_ranks = num_ranks
        self.link_bandwidth = link_bandwidth
        self.switch_bandwidth = switch_bandwidth

        # Create uplinks (rank -> switch)
        self.uplinks = [Resource(f"uplink_{i}", link_bandwidth) for i in range(num_ranks)]

        # Create switch fabric (shared by all)
        self.switch_fabric = Resource("switch_fabric", switch_bandwidth)

        # Create downlinks (switch -> rank)
        self.downlinks = [Resource(f"downlink_{i}", link_bandwidth) for i in range(num_ranks)]

    def resolve_path(self, src: int, dst: int) -> list[Resource]:
        """Resolve path from src to dst through switch.

        Args:
            src: Source rank (0 <= src < num_ranks)
            dst: Destination rank (0 <= dst < num_ranks)

        Returns:
            [uplink[src], switch_fabric, downlink[dst]] if src != dst, else []

        Raises:
            ValueError: If src or dst out of range
        """
        if not (0 <= src < self.num_ranks):
            raise ValueError(f"src {src} out of range [0, {self.num_ranks})")
        if not (0 <= dst < self.num_ranks):
            raise ValueError(f"dst {dst} out of range [0, {self.num_ranks})")

        if src == dst:
            return []

        return [
            self.uplinks[src],
            self.switch_fabric,
            self.downlinks[dst],
        ]

    def all_resources(self) -> list[Resource]:
        """Return all uplinks + switch fabric + downlinks.

        Returns:
            List of all Resource objects
        """
        return self.uplinks + [self.switch_fabric] + self.downlinks


class NVLinkMeshTopology(Topology):
    """Fully-connected NVLink mesh (models DGX A100/H100 intra-node).

    Each GPU pair has dedicated NVLink connections (no contention between pairs).
    Multiple NVLinks per pair are aggregated into single bidirectional resource.

    Attributes:
        num_gpus: Number of GPUs in the mesh
        nvlink_bandwidth: Bandwidth per NVLink in bytes/second
        links_per_pair: Number of NVLinks between each GPU pair

    Example:
        >>> # DGX A100: 8 GPUs, 12 NVLinks @ 25 GB/s each
        >>> topo = NVLinkMeshTopology(8, nvlink_bandwidth=25e9, links_per_pair=12)
        >>> path = topo.resolve_path(0, 3)
        >>> path[0].bandwidth
        300000000000.0  # 12 * 25 GB/s
    """

    def __init__(self, num_gpus: int, nvlink_bandwidth: float, links_per_pair: int):
        """Initialize NVLink mesh topology.

        Args:
            num_gpus: Number of GPUs (must be >= 2)
            nvlink_bandwidth: Bandwidth per NVLink in bytes/second
            links_per_pair: Number of NVLinks between each GPU pair

        Raises:
            ValueError: If num_gpus < 2, bandwidth <= 0, or links_per_pair < 1
        """
        if num_gpus < 2:
            raise ValueError(f"num_gpus must be >= 2, got {num_gpus}")
        if nvlink_bandwidth <= 0:
            raise ValueError(f"nvlink_bandwidth must be positive, got {nvlink_bandwidth}")
        if links_per_pair < 1:
            raise ValueError(f"links_per_pair must be >= 1, got {links_per_pair}")

        self.num_gpus = num_gpus
        self.nvlink_bandwidth = nvlink_bandwidth
        self.links_per_pair = links_per_pair

        # Aggregate bandwidth per direction
        aggregate_bw = nvlink_bandwidth * links_per_pair

        # Create bidirectional links for all pairs
        self._resources: dict[tuple[int, int], Resource] = {}
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst:
                    self._resources[(src, dst)] = Resource(
                        name=f"nvlink_{src}->{dst}",
                        bandwidth=aggregate_bw,
                    )

    def resolve_path(self, src: int, dst: int) -> list[Resource]:
        """Resolve direct NVLink path from src to dst.

        Args:
            src: Source GPU (0 <= src < num_gpus)
            dst: Destination GPU (0 <= dst < num_gpus)

        Returns:
            [nvlink_src->dst] if src != dst, else []

        Raises:
            ValueError: If src or dst out of range
        """
        if not (0 <= src < self.num_gpus):
            raise ValueError(f"src {src} out of range [0, {self.num_gpus})")
        if not (0 <= dst < self.num_gpus):
            raise ValueError(f"dst {dst} out of range [0, {self.num_gpus})")

        if src == dst:
            return []

        return [self._resources[(src, dst)]]

    def all_resources(self) -> list[Resource]:
        """Return all P*(P-1) directional NVLink resources.

        Returns:
            List of all Resource objects
        """
        return list(self._resources.values())

    def get_bandwidth(self, src: int, dst: int) -> float:
        """Get aggregate NVLink bandwidth from src to dst.

        Args:
            src: Source GPU
            dst: Destination GPU

        Returns:
            Aggregate bandwidth (nvlink_bandwidth * links_per_pair)
        """
        if src == dst:
            return float("inf")

        return self.nvlink_bandwidth * self.links_per_pair


class HierarchicalTopology(Topology):
    """Multi-node topology with NVLink intra-node + InfiniBand inter-node.

    Critical for multi-node RLHF simulations. Models clusters where:
    - Intra-node: GPUs within same node use NVLink mesh (high bandwidth, low latency)
    - Inter-node: GPUs on different nodes use InfiniBand (lower bandwidth, higher latency)

    Layer-specific LogGP parameters:
    - Each layer (NVLink, InfiniBand) has its own LogGP params
    - Simulator must call get_loggp(src, dst) to get correct params for path

    Attributes:
        num_nodes: Number of nodes in cluster
        gpus_per_node: GPUs per node (e.g., 8 for DGX)
        nvlink_bandwidth: NVLink bandwidth per link (bytes/second)
        nvlink_count: NVLinks per GPU pair
        ib_bandwidth: InfiniBand bandwidth per NIC (bytes/second)
        loggp_nvlink: LogGP parameters for intra-node (NVLink)
        loggp_ib: LogGP parameters for inter-node (InfiniBand)

    Example:
        >>> from syssim.network import LogGPParams
        >>> topo = HierarchicalTopology(
        ...     num_nodes=4,
        ...     gpus_per_node=8,
        ...     nvlink_bandwidth=25e9,
        ...     nvlink_count=12,
        ...     ib_bandwidth=25e9,
        ...     loggp_nvlink=LogGPParams(L=1e-6, o=5e-6, G=1/(25e9*12)),
        ...     loggp_ib=LogGPParams(L=5e-6, o=10e-6, G=1/(25e9)),
        ... )
        >>> # Intra-node: rank 0 (node 0, GPU 0) -> rank 7 (node 0, GPU 7)
        >>> path_intra = topo.resolve_path(0, 7)
        >>> topo.get_loggp(0, 7).L  # NVLink latency
        1e-06
        >>> # Inter-node: rank 0 (node 0) -> rank 8 (node 1)
        >>> path_inter = topo.resolve_path(0, 8)
        >>> topo.get_loggp(0, 8).L  # InfiniBand latency
        5e-06
    """

    def __init__(
        self,
        num_nodes: int,
        gpus_per_node: int,
        nvlink_bandwidth: float,
        nvlink_count: int,
        ib_bandwidth: float,
        loggp_nvlink,  # LogGPParams
        loggp_ib,  # LogGPParams
    ):
        """Initialize hierarchical topology.

        Args:
            num_nodes: Number of nodes (must be >= 1)
            gpus_per_node: GPUs per node (must be >= 1)
            nvlink_bandwidth: NVLink bandwidth per link (bytes/second)
            nvlink_count: NVLinks per GPU pair (must be >= 1)
            ib_bandwidth: InfiniBand bandwidth per NIC (bytes/second)
            loggp_nvlink: LogGP params for intra-node communication
            loggp_ib: LogGP params for inter-node communication

        Raises:
            ValueError: If parameters are invalid
        """
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
        if gpus_per_node < 1:
            raise ValueError(f"gpus_per_node must be >= 1, got {gpus_per_node}")
        if nvlink_bandwidth <= 0:
            raise ValueError(f"nvlink_bandwidth must be positive, got {nvlink_bandwidth}")
        if nvlink_count < 1:
            raise ValueError(f"nvlink_count must be >= 1, got {nvlink_count}")
        if ib_bandwidth <= 0:
            raise ValueError(f"ib_bandwidth must be positive, got {ib_bandwidth}")

        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.total_ranks = num_nodes * gpus_per_node
        self.loggp_nvlink = loggp_nvlink
        self.loggp_ib = loggp_ib

        # Create NVLink meshes (one per node)
        self.nvlink_meshes = [
            NVLinkMeshTopology(gpus_per_node, nvlink_bandwidth, nvlink_count) for _ in range(num_nodes)
        ]

        # Create InfiniBand resources (one uplink + one downlink per node)
        self.ib_uplinks = [Resource(f"ib_uplink_{node}", ib_bandwidth) for node in range(num_nodes)]

        self.ib_downlinks = [Resource(f"ib_downlink_{node}", ib_bandwidth) for node in range(num_nodes)]

        # Shared IB fabric
        self.ib_fabric = Resource("ib_fabric", ib_bandwidth * num_nodes)

    @classmethod
    def from_profiled_model(
        cls,
        model_path,  # str or Path
        num_ranks: int,
        ranks_per_node: int,
        nvlink_count: int = 12,
    ):
        """Create HierarchicalTopology from profiled hierarchical model.

        This factory method loads a hierarchical LogGP model (with intra_node_nvlink
        and inter_node_ib layers) and creates a HierarchicalTopology with the
        profiled parameters.

        Args:
            model_path: Path to hierarchical LogGP JSON (e.g., "perlmutter_loggp.json")
                        or topology name (e.g., "perlmutter")
            num_ranks: Total number of ranks
            ranks_per_node: Ranks per node (for intra-node detection)
            nvlink_count: NVLinks per GPU pair (default: 12)

        Returns:
            HierarchicalTopology with profiled LogGP parameters

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model doesn't have expected layers

        Example:
            >>> topo = HierarchicalTopology.from_profiled_model(
            ...     "data/network_models/perlmutter_loggp.json",
            ...     num_ranks=32,
            ...     ranks_per_node=4
            ... )
            >>> # Or with topology name auto-resolution
            >>> topo = HierarchicalTopology.from_profiled_model(
            ...     "perlmutter",
            ...     num_ranks=32,
            ...     ranks_per_node=4
            ... )
        """
        from .model_loader import load_hierarchical_loggp

        # Load hierarchical params
        params = load_hierarchical_loggp(model_path)

        # Expected layer names (validate)
        expected_layers = ["intra_node_nvlink", "inter_node_ib"]
        for layer in expected_layers:
            if layer not in params:
                raise ValueError(
                    f"Expected layer '{layer}' not found in model. Available layers: {list(params.keys())}"
                )

        # Extract LogGP parameters
        loggp_nvlink = params["intra_node_nvlink"]
        loggp_ib = params["inter_node_ib"]

        # Extract bandwidths from G parameter (BW = 1/G)
        nvlink_bw_per_link = 1.0 / loggp_nvlink.G / nvlink_count  # Total BW divided by link count
        ib_bw = 1.0 / loggp_ib.G

        # Calculate topology parameters
        num_nodes = num_ranks // ranks_per_node
        if num_ranks % ranks_per_node != 0:
            raise ValueError(f"num_ranks ({num_ranks}) must be divisible by ranks_per_node ({ranks_per_node})")

        # Create topology with profiled parameters
        return cls(
            num_nodes=num_nodes,
            gpus_per_node=ranks_per_node,
            nvlink_bandwidth=nvlink_bw_per_link,
            nvlink_count=nvlink_count,
            ib_bandwidth=ib_bw,
            loggp_nvlink=loggp_nvlink,
            loggp_ib=loggp_ib,
        )

    def _rank_to_node(self, rank: int) -> int:
        """Map global rank to node ID."""
        return rank // self.gpus_per_node

    def _rank_to_local(self, rank: int) -> int:
        """Map global rank to local GPU ID within node."""
        return rank % self.gpus_per_node

    def resolve_path(self, src: int, dst: int) -> list[Resource]:
        """Resolve path based on whether src and dst are on same node.

        Args:
            src: Source rank (0 <= src < total_ranks)
            dst: Destination rank (0 <= dst < total_ranks)

        Returns:
            NVLink path if same node, InfiniBand path if different nodes

        Raises:
            ValueError: If src or dst out of range
        """
        if not (0 <= src < self.total_ranks):
            raise ValueError(f"src {src} out of range [0, {self.total_ranks})")
        if not (0 <= dst < self.total_ranks):
            raise ValueError(f"dst {dst} out of range [0, {self.total_ranks})")

        if src == dst:
            return []

        src_node = self._rank_to_node(src)
        dst_node = self._rank_to_node(dst)

        if src_node == dst_node:
            # Intra-node: use NVLink mesh
            src_local = self._rank_to_local(src)
            dst_local = self._rank_to_local(dst)
            return self.nvlink_meshes[src_node].resolve_path(src_local, dst_local)
        else:
            # Inter-node: uplink[src_node] → fabric → downlink[dst_node]
            return [
                self.ib_uplinks[src_node],
                self.ib_fabric,
                self.ib_downlinks[dst_node],
            ]

    def get_loggp(self, src: int, dst: int):
        """Return appropriate LogGP params based on path type.

        Args:
            src: Source rank
            dst: Destination rank

        Returns:
            loggp_nvlink if same node, loggp_ib if different nodes

        Note:
            Simulator must call this to get layer-specific LogGP params.
            If using global LogGP in simulate(), this method is ignored.
        """
        src_node = self._rank_to_node(src)
        dst_node = self._rank_to_node(dst)

        if src_node == dst_node:
            return self.loggp_nvlink
        else:
            return self.loggp_ib

    def all_resources(self) -> list[Resource]:
        """Return all resources across all layers.

        Returns:
            List of all NVLink + InfiniBand resources
        """
        resources = []

        # NVLink resources from all meshes
        for mesh in self.nvlink_meshes:
            resources.extend(mesh.all_resources())

        # InfiniBand resources
        resources.extend(self.ib_uplinks)
        resources.append(self.ib_fabric)
        resources.extend(self.ib_downlinks)

        return resources

    def get_bandwidth(self, src: int, dst: int) -> float:
        """Get effective bandwidth from src to dst (NVLink or InfiniBand).

        Args:
            src: Source rank
            dst: Destination rank

        Returns:
            NVLink bandwidth if same node, InfiniBand bandwidth if different nodes
        """
        if src == dst:
            return float("inf")

        src_node = self._rank_to_node(src)
        dst_node = self._rank_to_node(dst)

        if src_node == dst_node:
            # Intra-node: use NVLink mesh bandwidth
            return self.nvlink_meshes[0].get_bandwidth(0, 1)  # All pairs have same bandwidth
        else:
            # Inter-node: use IB bandwidth (bottlenecked by NIC)
            return self.ib_uplinks[0].bandwidth
