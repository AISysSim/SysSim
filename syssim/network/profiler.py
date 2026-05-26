"""LogGP parameter profiling via NCCL ping-pong benchmarks.

This module provides a CLI tool for automatically calibrating LogGP parameters
using the PRTT (Parametrized Round Trip Time) method from Hoefler et al. (2009).

The profiler:
1. Executes ping-pong microbenchmarks across message sizes (1B to 64KB)
2. Detects protocol changes (eager vs rendezvous) via least-squares deviation
3. Extracts hardware-specific L, o, g, G parameters automatically
4. Saves parameters to data/network_models/{topology}_loggp.json

Usage:
    # Profile NVLink on single node with 2 GPUs
    torchrun --nproc_per_node=2 -m syssim.network.profiler \\
        --topology nvlink \\
        --max-size 65536 \\
        --num-runs 10

    # Profile InfiniBand on 2 nodes
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=node0 \\
        -m syssim.network.profiler \\
        --topology infiniband

References:
    - "LogGP in Theory and Practice" (Hoefler et al., 2009)
"""

import argparse
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .protocol_detector import (
    ProtocolRange,
    PRTTMeasurement,
    detect_protocol_changes,
)

if TYPE_CHECKING:
    from .device_mesh import DeviceMesh


@dataclass
class ProfilingResult:
    """Complete profiling result with all detected protocols.

    Attributes:
        topology: Topology type (e.g., "nvlink", "infiniband")
        protocols: List of protocol dictionaries with size_range, L, o, g, G
        primary: Primary protocol parameters (usually first/smallest protocol)
        metadata: Profiling metadata (timestamp, hardware, num_protocols, etc.)
    """

    topology: str
    protocols: List[Dict[str, Any]]
    primary: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class LayerConfig:
    """Configuration for a single network layer (mesh-based only).

    Mesh is REQUIRED. No backward compatibility with explicit rank lists.

    Attributes:
        topology_type: Network type (nvlink, infiniband, slingshot, ethernet, custom)
        scope: REQUIRED mesh scope specification {"vary_dims": [...], "fix_dims": {...}}
        description: Human-readable description of this layer
        expected_bandwidth_gbs: Optional bandwidth hint for validation
    """

    topology_type: str
    scope: Dict[str, Any]  # REQUIRED: {"vary_dims": [...], "fix_dims": {...}}
    description: str = ""
    expected_bandwidth_gbs: Optional[float] = None

    def get_rank_pairs(self, mesh: "DeviceMesh") -> List[Tuple[int, int]]:
        """Get rank pairs for profiling from mesh and scope.

        Args:
            mesh: DeviceMesh object (required)

        Returns:
            List of (src, dst) rank pairs

        Raises:
            ValueError: If scope is invalid or mesh is incompatible
        """
        from .device_mesh import DeviceMesh

        if not isinstance(mesh, DeviceMesh):
            raise ValueError(f"mesh argument must be DeviceMesh instance, got {type(mesh).__name__}")

        vary_dims = self.scope.get("vary_dims", [])
        fix_dims = self.scope.get("fix_dims", {})
        num_pairs = self.scope.get("num_pairs", 1)

        # Validate: fix_dims must have explicit int values (no wildcards)
        for dim_name, dim_value in fix_dims.items():
            if not isinstance(dim_value, int):
                raise ValueError(
                    f"fix_dims['{dim_name}'] must be int, "
                    f"got {type(dim_value).__name__}. Wildcards not supported in this version."
                )

        # Validate: vary_dims must exist in mesh
        for dim_name in vary_dims:
            if dim_name not in mesh.dimension_names:
                raise ValueError(f"vary_dims contains '{dim_name}' not in mesh.dimension_names {mesh.dimension_names}")

        return mesh.get_representative_pairs(fix_dims, vary_dims, num_pairs)

    def get_all_ranks(self, mesh: "DeviceMesh") -> List[int]:
        """Get all ranks participating in this layer.

        Args:
            mesh: DeviceMesh object

        Returns:
            List of all ranks in the layer's scope
        """

        vary_dims = self.scope.get("vary_dims", [])
        fix_dims = self.scope.get("fix_dims", {})

        return mesh.ranks_in_slice(fix_dims, vary_dims)


@dataclass
class HierarchyConfig:
    """Hierarchical topology profiling configuration (mesh-based only).

    Mesh is REQUIRED with topology_types. Layers are auto-generated.

    Attributes:
        topology_name: Name of the hierarchical topology
        mesh: REQUIRED mesh specification with topology_types
        profiling_params: Profiling parameters (min_size, max_size, num_runs, etc.)
    """

    topology_name: str
    mesh: Dict[str, Any]  # REQUIRED: {"shape": [...], "dimension_names": [...], "topology_types": [...]}
    profiling_params: Dict[str, Any]

    def get_device_mesh(self) -> "DeviceMesh":
        """Parse mesh dict into DeviceMesh object.

        Returns:
            DeviceMesh instance

        Raises:
            ValueError: If mesh dict is invalid
        """
        from .device_mesh import DeviceMesh

        if "shape" not in self.mesh:
            raise ValueError("mesh must contain 'shape' field")

        if "dimension_names" not in self.mesh:
            raise ValueError("mesh must contain 'dimension_names' field")

        if "topology_types" not in self.mesh:
            raise ValueError("mesh must contain 'topology_types' field")

        return DeviceMesh(
            shape=tuple(self.mesh["shape"]),
            dimension_names=self.mesh["dimension_names"],
            topology_types=self.mesh["topology_types"],
            ranks_order=self.mesh.get("ranks_order", "C"),
        )

    def get_auto_layers(self) -> Dict[str, LayerConfig]:
        """Auto-generate layers from mesh dimensions.

        Creates one layer per dimension, varying that dimension while fixing
        all others to 0.

        Returns:
            Dict mapping dimension name to LayerConfig
        """
        mesh = self.get_device_mesh()
        layers = {}

        # Create one layer per dimension
        for dim_idx, dim_name in enumerate(mesh.dimension_names):
            topology_type = mesh.topology_types[dim_idx]

            # Build scope: vary this dimension, fix all others to 0
            vary_dims = [dim_name]
            fix_dims = {other_dim: 0 for i, other_dim in enumerate(mesh.dimension_names) if i != dim_idx}

            layer = LayerConfig(topology_type=topology_type, scope={"vary_dims": vary_dims, "fix_dims": fix_dims})

            layers[dim_name] = layer

        return layers

    def validate(self):
        """Validate mesh config.

        Raises:
            ValueError: If mesh config is invalid
        """
        mesh = self.get_device_mesh()

        # Validate auto-generated layers
        layers = self.get_auto_layers()
        for layer_name, layer in layers.items():
            try:
                layer.get_rank_pairs(mesh)
            except ValueError as e:
                raise ValueError(f"Auto-generated layer '{layer_name}' validation failed: {e}")


@dataclass
class LayerProfilingResult:
    """Profiling result for a single network layer.

    Attributes:
        name: Layer name
        topology_type: Network type
        ranks: Ranks participating in this layer
        protocols: List of protocol dictionaries with size_range, L, o, g, G
        primary: Primary protocol parameters
        metadata: Layer-specific metadata
    """

    name: str
    topology_type: str
    ranks: List[int]
    protocols: List[Dict[str, Any]]
    primary: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class HierarchicalProfilingResult:
    """Complete hierarchical profiling result with all layers.

    Attributes:
        topology_name: Name of the hierarchical topology
        description: Human-readable description
        layers: Dict mapping layer name to LayerProfilingResult
        metadata: Global metadata (timestamp, total_profiling_time_s, etc.)
    """

    topology_name: str
    description: str
    layers: Dict[str, LayerProfilingResult]
    metadata: Dict[str, Any]


class CommBackend(ABC):
    """Abstract communication backend for PRTT measurements."""

    @abstractmethod
    def ping_pong(self, n: int, delay: float, size: int, peer_rank: Optional[int] = None) -> float:
        """Execute n ping-pongs with optional delay between iterations.

        Args:
            n: Number of ping-pong iterations
            delay: Delay between iterations (seconds)
            size: Message size (bytes)
            peer_rank: Peer rank for communication (default: auto-select based on backend)

        Returns:
            Elapsed time in seconds (client only, server returns 0.0)
        """
        pass

    @abstractmethod
    def is_client(self) -> bool:
        """Return True if measurement client (rank 0)."""
        pass

    @abstractmethod
    def is_server(self) -> bool:
        """Return True if echo server (rank 1)."""
        pass

    @abstractmethod
    def barrier(self):
        """Synchronize all processes."""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup backend resources."""
        pass


class NCCLBackend(CommBackend):
    """NCCL backend for GPU-to-GPU communication profiling.

    Uses torch.distributed with NCCL backend to measure actual GPU-to-GPU
    communication overhead (kernel launch, PCIe/NVLink/InfiniBand).

    Requires:
        - 2+ GPUs
        - torch.distributed initialized with NCCL backend
        - CUDA-capable PyTorch build
    """

    def __init__(self):
        """Initialize NCCL backend.

        Initializes torch.distributed if not already initialized (when run via torchrun).
        """
        try:
            import torch
            import torch.distributed as dist
        except ImportError:
            raise ImportError("NCCLBackend requires PyTorch with distributed support")

        if not torch.cuda.is_available():
            raise RuntimeError("NCCLBackend requires CUDA")

        if not dist.is_initialized():
            # Initialize torch.distributed (torchrun sets env vars but doesn't call init)
            try:
                dist.init_process_group(backend="nccl")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize torch.distributed: {e}\n"
                    "Run via: torchrun --nproc_per_node=2 -m syssim.network.profiler ..."
                )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        if self.world_size < 2:
            raise ValueError(f"NCCLBackend requires at least 2 ranks, got {self.world_size}")

        # Use local rank for device selection (supports multi-node)
        # In multi-node setup: rank 0 on node 0 uses GPU 0, rank 1 on node 1 uses GPU 0
        # In single-node setup: rank 0 uses GPU 0, rank 1 uses GPU 1
        import os

        local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        self.torch = torch
        self.dist = dist

    def ping_pong(self, n: int, delay: float, size: int, peer_rank: Optional[int] = None) -> float:
        """Execute n ping-pongs via NCCL send/recv.

        Args:
            n: Number of iterations
            delay: Delay between iterations (seconds)
            size: Message size (bytes)
            peer_rank: Peer rank for communication (default: 1 if rank==0, 0 if rank==1)

        Returns:
            Elapsed time in seconds (client only), 0.0 for server and idle ranks
        """
        # Auto-select peer rank if not provided
        if peer_rank is None:
            if self.rank == 0:
                peer_rank = 1
            elif self.rank == 1:
                peer_rank = 0
            else:
                # Idle rank
                return 0.0

        # Determine client/server role: lower rank = client
        is_client = self.rank < peer_rank
        is_server = self.rank > peer_rank

        if self.rank == peer_rank:
            raise ValueError(f"peer_rank ({peer_rank}) must be different from self.rank ({self.rank})")

        # Create GPU tensor buffer
        buf = self.torch.zeros(size, dtype=self.torch.uint8, device=self.device)

        if is_client:
            # Client: send to peer_rank, receive from peer_rank
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)

            self.torch.cuda.synchronize()
            start_event.record()

            for i in range(n):
                self.dist.send(buf, dst=peer_rank)
                self.dist.recv(buf, src=peer_rank)

                if delay > 0:
                    # CUDA synchronize before delay to ensure communication completed
                    self.torch.cuda.synchronize()
                    time.sleep(delay)

            end_event.record()
            self.torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            return elapsed_ms / 1000.0  # Convert ms to seconds

        elif is_server:
            # Server: receive from peer_rank, send back to peer_rank
            for i in range(n):
                self.dist.recv(buf, src=peer_rank)
                self.dist.send(buf, dst=peer_rank)

            return 0.0
        else:
            # Idle rank (not involved in this communication)
            return 0.0

    def is_client(self) -> bool:
        """Return True if rank 0 (measurement client)."""
        return self.rank == 0

    def is_server(self) -> bool:
        """Return True if rank 1 (echo server)."""
        return self.rank == 1

    def barrier(self):
        """Synchronize all ranks."""
        self.dist.barrier()

    def cleanup(self):
        """Cleanup NCCL backend."""
        if self.dist.is_initialized():
            self.dist.destroy_process_group()


def measure_prtt(
    backend: CommBackend, n: int, delay: float, size: int, num_runs: int = 10, peer_rank: Optional[int] = None
) -> float:
    """Measure PRTT with statistical sampling.

    Args:
        backend: Communication backend
        n: Number of ping-pong iterations per run
        delay: Delay between iterations (seconds)
        size: Message size (bytes)
        num_runs: Number of measurement runs (for statistical stability)
        peer_rank: Peer rank for communication (optional, backend-dependent)

    Returns:
        Median PRTT time in seconds (client only, server returns 0.0)
    """
    # Determine if this rank is involved
    is_involved = (peer_rank is None) or (backend.rank in [0, 1, peer_rank])

    if not is_involved:
        # This rank not involved, just wait at barriers
        for _ in range(num_runs):
            backend.barrier()
        return 0.0

    # Determine client/server role
    if peer_rank is None:
        is_client_rank = backend.is_client()
    else:
        # For hierarchical: client is lower rank
        is_client_rank = backend.rank < peer_rank

    if is_client_rank:
        times = []
        for _ in range(num_runs):
            backend.barrier()
            elapsed = backend.ping_pong(n, delay, size, peer_rank=peer_rank)
            times.append(elapsed)

        # Use median for robustness to outliers
        return float(np.median(times))
    else:
        # Server executes ping-pongs but doesn't measure
        for _ in range(num_runs):
            backend.barrier()
            backend.ping_pong(n, delay, size, peer_rank=peer_rank)
        return 0.0


def sweep_message_sizes(
    backend: CommBackend,
    min_size: int = 1,
    max_size: int = 65536,
    n: int = 10,
    num_runs: int = 10,
    peer_rank: Optional[int] = None,
) -> List[PRTTMeasurement]:
    """Sweep message sizes and measure PRTT(1,0,s), PRTT(n,0,s), PRTT(n,dG,s).

    Args:
        backend: Communication backend
        min_size: Minimum message size (bytes)
        max_size: Maximum message size (bytes)
        n: Number of iterations for PRTT(n,...)
        num_runs: Number of runs per measurement
        peer_rank: Peer rank for communication (optional, backend-dependent)

    Returns:
        List of PRTTMeasurement objects (client only, server returns empty list)
    """
    # Exponential sweep: powers of 2 from min_size to max_size
    sizes = []
    size = min_size
    while size <= max_size:
        sizes.append(size)
        size *= 2

    # Ensure max_size is included if not already
    if sizes[-1] < max_size:
        sizes.append(max_size)

    measurements = []

    # Determine if this rank is client
    if peer_rank is None:
        is_client_rank = backend.is_client()
    else:
        is_client_rank = backend.rank < peer_rank

    if is_client_rank:
        print(f"Measuring PRTT for {len(sizes)} message sizes: {sizes[0]} to {sizes[-1]} bytes")
        print("Progress: ", end="", flush=True)

    for i, size in enumerate(sizes):
        # Measure PRTT(1, 0, size)
        prtt_1_0 = measure_prtt(backend, n=1, delay=0.0, size=size, num_runs=num_runs, peer_rank=peer_rank)

        # Measure PRTT(n, 0, size)
        prtt_n_0 = measure_prtt(backend, n=n, delay=0.0, size=size, num_runs=num_runs, peer_rank=peer_rank)

        # Compute dG = PRTT(1, 0, size) for delay
        dG = prtt_1_0

        # Measure PRTT(n, dG, size)
        prtt_n_dG = measure_prtt(backend, n=n, delay=dG, size=size, num_runs=num_runs, peer_rank=peer_rank)

        if is_client_rank:
            measurements.append(PRTTMeasurement(size=size, prtt_1_0=prtt_1_0, prtt_n_0=prtt_n_0, prtt_n_dG=prtt_n_dG))

            # Progress indicator
            if (i + 1) % max(1, len(sizes) // 10) == 0 or i == len(sizes) - 1:
                print(f"{i + 1}/{len(sizes)} ", end="", flush=True)

    if is_client_rank:
        print()  # Newline after progress

    return measurements


def extract_loggp_parameters(
    measurements: List[PRTTMeasurement], protocol: ProtocolRange, n: int = 10
) -> Tuple[float, float, float, float]:
    """Extract L, o, g, G from PRTT measurements for a single protocol.

    Uses Hoefler's PRTT extraction formulas:
    1. g, G already fitted from protocol detection
    2. Compute os = [PRTT(n, dG, s) - PRTT(1, 0, s)] / (n - 1) - dG
    3. Compute o = median(os)
    4. Compute L from PRTT(1, 0, s) = 2*(L + 2*o + g + (s-1)*G)

    Args:
        measurements: All PRTT measurements
        protocol: Protocol range with fitted g, G
        n: Number of iterations used

    Returns:
        Tuple of (L, o, g, G) in seconds/seconds/seconds/seconds per byte
    """
    # Extract measurements for this protocol
    protocol_measurements = measurements[protocol.start_idx : protocol.end_idx + 1]

    if not protocol_measurements:
        raise ValueError("No measurements in protocol range")

    # Use fitted g, G from protocol detection
    g = protocol.g
    G = protocol.G

    # Compute o from each measurement
    os = []
    for m in protocol_measurements:
        dG = m.prtt_1_0  # Delay used in PRTT(n, dG, s)
        o_s = (m.prtt_n_dG - m.prtt_1_0) / (n - 1) - dG
        os.append(o_s)

    # Use median for robustness
    o = float(np.median(os))

    # Compute L from PRTT(1, 0, s) = 2 * (L + 2*o + g + (s-1)*G)
    Ls = []
    for m in protocol_measurements:
        L_s = (m.prtt_1_0 / 2.0) - 2 * o - g - (m.size - 1) * G
        Ls.append(L_s)

    # Use median for robustness
    L = float(np.median(Ls))

    # Ensure non-negative parameters
    L = max(L, 0.0)
    o = max(o, 0.0)
    g = max(g, 0.0)
    G = max(G, 0.0)

    return L, o, g, G


def save_profiling_result(result: ProfilingResult, output_path: Path):
    """Save profiling result to JSON file.

    Args:
        result: ProfilingResult to save
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)


def load_hierarchy_config(path: Union[str, Path]) -> HierarchyConfig:
    """Load hierarchical topology configuration from JSON file (mesh-based only).

    Args:
        path: Path to hierarchy config JSON

    Returns:
        HierarchyConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If JSON is malformed or missing required fields
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Hierarchy config not found: {path}")

    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {e}")

    # Validate required top-level fields (only topology_name and mesh)
    required_fields = ["topology_name", "mesh"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing '{field}' field in {path}. Required fields: {required_fields}")

    # Load profiling_params from default file if not provided
    if "profiling_params" not in data:
        # Try to load default profiling params
        default_params_path = (
            Path(__file__).parent.parent.parent / "examples" / "configs" / "default_profiling_params.json"
        )
        if default_params_path.exists():
            with open(default_params_path) as f:
                data["profiling_params"] = json.load(f)
        else:
            # Fallback to hardcoded defaults
            data["profiling_params"] = {
                "min_size": 4096,
                "max_size": 2147483648,
                "num_runs": 10,
                "lookahead": 5,
                "pfact": 3.0,
            }

    # Validate mesh structure
    mesh_data = data["mesh"]
    mesh_required = ["shape", "dimension_names", "topology_types"]
    for field in mesh_required:
        if field not in mesh_data:
            raise ValueError(f"mesh must contain '{field}' field in {path}")

    config = HierarchyConfig(
        topology_name=data["topology_name"], mesh=mesh_data, profiling_params=data["profiling_params"]
    )

    # Validate config (checks mesh/layer consistency)
    config.validate()

    return config


def save_hierarchical_result(result: HierarchicalProfilingResult, output_path: Path):
    """Save hierarchical profiling result to JSON file.

    Args:
        result: HierarchicalProfilingResult to save
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict structure matching the expected JSON format
    data = {
        "topology_name": result.topology_name,
        "description": result.description,
        "layers": {},
        "metadata": result.metadata,
    }

    # Convert each LayerProfilingResult to dict
    for layer_name, layer_result in result.layers.items():
        data["layers"][layer_name] = {
            "topology_type": layer_result.topology_type,
            "ranks": layer_result.ranks,
            "protocols": layer_result.protocols,
            "primary": layer_result.primary,
            "metadata": layer_result.metadata,
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def profile_single_layer(
    layer_name: str, layer: LayerConfig, backend: NCCLBackend, mesh: "DeviceMesh", profiling_params: Dict[str, Any]
) -> Optional[LayerProfilingResult]:
    """Profile a single network layer (mesh-based).

    Args:
        layer_name: Name of the layer (from config dict key)
        layer: Layer configuration (with mesh scope)
        backend: NCCL backend
        mesh: DeviceMesh for rank derivation
        profiling_params: Profiling parameters (min_size, max_size, num_runs, etc.)

    Returns:
        LayerProfilingResult if this rank involved, None otherwise
    """

    # Get all ranks participating in this layer from mesh
    all_ranks = layer.get_all_ranks(mesh)

    # Check if this rank participates in this layer
    if backend.rank not in all_ranks:
        # Not involved in this layer, wait at barrier
        if backend.rank == 0:
            print(f"Rank {backend.rank} skipping layer '{layer_name}' (not in ranks {all_ranks})")
        backend.barrier()
        return None

    # Get rank pairs for this layer from mesh
    rank_pairs = layer.get_rank_pairs(mesh)

    # Find if this rank is in any rank pair
    active_pair = None
    for src, dst in rank_pairs:
        if backend.rank in [src, dst]:
            active_pair = (src, dst)
            break

    if active_pair is None:
        # Rank is in layer but not in any rank_pair - wait at barrier
        if backend.rank == 0:
            print(f"Rank {backend.rank} in layer '{layer_name}' but not in any rank pair, waiting...")
        backend.barrier()
        return None

    # Determine if this rank is client or server
    src, dst = active_pair
    is_client = backend.rank == min(src, dst)

    if backend.rank == 0 or is_client:
        # Print mesh-aware description
        scope_desc = f"vary={layer.scope.get('vary_dims', [])}, fix={layer.scope.get('fix_dims', {})}"
        print(f"Profiling layer '{layer_name}' ({layer.topology_type}) with pair {active_pair}")
        print(f"  Mesh scope: {scope_desc}")

    # Profile this layer using active_pair
    measurements = sweep_message_sizes(
        backend,
        min_size=profiling_params.get("min_size", 1),
        max_size=profiling_params.get("max_size", 65536),
        n=10,
        num_runs=profiling_params.get("num_runs", 10),
        peer_rank=dst if backend.rank == src else src,
    )

    # Detect protocol changes (client only)
    if is_client and measurements:
        protocols = detect_protocol_changes(
            measurements, n=10, lookahead=profiling_params.get("lookahead", 3), pfact=profiling_params.get("pfact", 2.0)
        )

        print(f"\nDetected {len(protocols)} protocol(s) for layer '{layer_name}':")

        # Extract LogGP parameters for each protocol
        protocol_dicts = []
        for i, protocol in enumerate(protocols):
            L, o, g, G = extract_loggp_parameters(measurements, protocol, n=10)

            size_min = protocol.sizes[0]
            size_max = protocol.sizes[-1]

            protocol_dict = {"size_range": [size_min, size_max], "L": L, "o": o, "g": g, "G": G}
            protocol_dicts.append(protocol_dict)

            print(f"  Protocol {i}: {size_min}-{size_max} bytes")
            print(f"    L={L * 1e6:.2f}μs, o={o * 1e6:.2f}μs, g={g * 1e6:.2f}μs, G={G * 1e9:.2f}ns/byte")

        # Primary protocol is first
        primary = {
            "L": protocol_dicts[0]["L"],
            "o": protocol_dicts[0]["o"],
            "g": protocol_dicts[0]["g"],
            "G": protocol_dicts[0]["G"],
        }

        # Build metadata (include mesh scope info)
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "rank_pairs_used": [list(active_pair)],
            "median_bandwidth_gbs": 1.0 / primary["G"] / 1e9 if primary["G"] > 0 else 0.0,
            "mesh_scope": {"vary_dims": layer.scope.get("vary_dims", []), "fix_dims": layer.scope.get("fix_dims", {})},
        }

        # Check against expected bandwidth if provided
        if layer.expected_bandwidth_gbs is not None:
            actual_bw = metadata["median_bandwidth_gbs"]
            expected_bw = layer.expected_bandwidth_gbs
            if abs(actual_bw - expected_bw) / expected_bw > 2.0:  # More than 2x difference
                print(
                    f"  WARNING: Measured bandwidth {actual_bw:.1f} GB/s differs from "
                    f"expected {expected_bw:.1f} GB/s by >2x"
                )

        result = LayerProfilingResult(
            name=layer_name,
            topology_type=layer.topology_type,
            ranks=all_ranks,  # All ranks from mesh slice
            protocols=protocol_dicts,
            primary=primary,
            metadata=metadata,
        )

        # Synchronize before next layer
        backend.barrier()

        return result
    else:
        # Server or idle rank: wait at barrier
        backend.barrier()
        return None


def profile_hierarchy(config: HierarchyConfig, backend: NCCLBackend) -> Optional[HierarchicalProfilingResult]:
    """Profile all layers in hierarchy sequentially (mesh-based).

    Args:
        config: Hierarchical topology configuration (with mesh)
        backend: NCCL backend

    Returns:
        HierarchicalProfilingResult (rank 0 only), None for other ranks
    """
    start_time = time.time()
    layer_results: Dict[str, LayerProfilingResult] = {}

    # Get device mesh and auto-generate layers
    mesh = config.get_device_mesh()
    layers = config.get_auto_layers()

    if backend.rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Profiling hierarchical topology: {config.topology_name}")
        print(f"Device mesh: {mesh}")
        print(f"Auto-generated layers: {len(layers)}")
        for dim_name, layer in layers.items():
            print(f"  {dim_name}: {layer.topology_type}")
        print(f"{'=' * 60}\n")

    for layer_name, layer in layers.items():
        result = profile_single_layer(layer_name, layer, backend, mesh, config.profiling_params)

        if result is not None:
            layer_results[layer_name] = result

    # Aggregate results (rank 0 collects)
    if backend.rank == 0:
        end_time = time.time()
        total_time = end_time - start_time

        # Build global metadata (include mesh info)
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_profiling_time_s": total_time,
            "num_layers": len(layer_results),
            "profiling_params": config.profiling_params,
            "mesh": {
                "shape": list(mesh.shape),
                "dimension_names": mesh.dimension_names,
                "total_ranks": mesh.total_ranks,
                "ranks_order": mesh.ranks_order,
            },
        }

        # Try to get hardware info
        try:
            import torch

            if torch.cuda.is_available():
                metadata["gpu"] = torch.cuda.get_device_name(0)
                metadata["num_gpus"] = torch.cuda.device_count()
        except Exception:
            pass

        return HierarchicalProfilingResult(
            topology_name=config.topology_name, description=config.description, layers=layer_results, metadata=metadata
        )
    else:
        return None


def run_profiling(args: argparse.Namespace, backend: CommBackend) -> ProfilingResult:
    """Run complete profiling workflow.

    Args:
        args: Command-line arguments
        backend: Communication backend

    Returns:
        ProfilingResult with all detected protocols
    """
    # Sweep message sizes
    measurements = sweep_message_sizes(
        backend,
        min_size=args.min_size,
        max_size=args.max_size,
        n=10,  # Fixed n=10 for PRTT measurements
        num_runs=args.num_runs,
    )

    if not backend.is_client():
        # Server doesn't process results
        return None

    # Detect protocol changes
    protocols = detect_protocol_changes(measurements, n=10, lookahead=args.lookahead, pfact=args.pfact)

    print(f"\nDetected {len(protocols)} protocol(s):")

    # Extract LogGP parameters for each protocol
    protocol_dicts = []
    for i, protocol in enumerate(protocols):
        L, o, g, G = extract_loggp_parameters(measurements, protocol, n=10)

        # Determine size range
        size_min = protocol.sizes[0]
        size_max = protocol.sizes[-1]

        protocol_dict = {"size_range": [size_min, size_max], "L": L, "o": o, "g": g, "G": G}
        protocol_dicts.append(protocol_dict)

        print(f"  Protocol {i}: {size_min}-{size_max} bytes")
        print(f"    L={L * 1e6:.2f}μs, o={o * 1e6:.2f}μs, g={g * 1e6:.2f}μs, G={G * 1e9:.2f}ns/byte")

    # Primary protocol is first (smallest messages, usually eager)
    primary = {
        "L": protocol_dicts[0]["L"],
        "o": protocol_dicts[0]["o"],
        "g": protocol_dicts[0]["g"],
        "G": protocol_dicts[0]["G"],
    }

    # Build metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "num_protocols": len(protocols),
        "min_size": args.min_size,
        "max_size": args.max_size,
        "num_runs": args.num_runs,
        "lookahead": args.lookahead,
        "pfact": args.pfact,
    }

    # Try to get hardware info
    try:
        import torch

        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
            metadata["num_gpus"] = torch.cuda.device_count()
    except Exception:
        pass

    return ProfilingResult(topology=args.topology, protocols=protocol_dicts, primary=primary, metadata=metadata)


def main():
    """CLI entry point for LogGP profiler."""
    parser = argparse.ArgumentParser(
        description="Profile LogGP parameters via NCCL ping-pong benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile NVLink on single node with 2 GPUs
  torchrun --nproc_per_node=2 -m syssim.network.profiler --topology nvlink

  # Profile InfiniBand on 2 nodes
  torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=node0 \\
      -m syssim.network.profiler --topology infiniband

  # Custom output path
  torchrun --nproc_per_node=2 -m syssim.network.profiler \\
      --topology custom --output data/network_models/custom_nvlink.json

  # SLURM cluster
  srun --nodes=2 --gpus-per-node=2 python -m syssim.network.profiler --topology infiniband
        """,
    )

    parser.add_argument(
        "--topology", required=False, help="Topology type for single-layer profiling (e.g., nvlink, infiniband, custom)"
    )
    parser.add_argument(
        "--hierarchy-config",
        type=str,
        default=None,
        help="Path to hierarchy config JSON for multi-layer profiling (overrides --topology)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path (default: data/network_models/{topology}_loggp.json)"
    )
    parser.add_argument("--min-size", type=int, default=1, help="Minimum message size in bytes (default: 1)")
    parser.add_argument("--max-size", type=int, default=65536, help="Maximum message size in bytes (default: 65536)")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of measurement runs per size (default: 10)")
    parser.add_argument("--lookahead", type=int, default=3, help="Protocol detection lookahead window (default: 3)")
    parser.add_argument("--pfact", type=float, default=2.0, help="Protocol detection sensitivity factor (default: 2.0)")

    args = parser.parse_args()

    # Check mutual exclusivity
    if args.hierarchy_config and args.topology:
        print("Error: Cannot specify both --hierarchy-config and --topology", file=sys.stderr)
        sys.exit(1)

    if not args.hierarchy_config and not args.topology:
        print("Error: Must specify either --hierarchy-config or --topology", file=sys.stderr)
        sys.exit(1)

    # Initialize NCCL backend
    try:
        backend = NCCLBackend()
    except Exception as e:
        print(f"Error: Failed to initialize NCCL backend: {e}", file=sys.stderr)
        print("\nMake sure to run via torchrun:", file=sys.stderr)
        print("  torchrun --nproc_per_node=2 -m syssim.network.profiler --topology nvlink", file=sys.stderr)
        sys.exit(1)

    # Run profiling
    try:
        if args.hierarchy_config:
            # Hierarchical profiling
            config = load_hierarchy_config(args.hierarchy_config)
            result = profile_hierarchy(config, backend)

            if backend.rank == 0 and result is not None:
                output_path = args.output
                if output_path is None:
                    # Auto-resolve from topology name
                    project_root = Path(__file__).parent.parent.parent
                    output_path = project_root / "data" / "network_models" / f"{config.topology_name}_loggp.json"
                else:
                    output_path = Path(output_path)

                save_hierarchical_result(result, output_path)
                print(f"\nSaved hierarchical LogGP parameters to {output_path}")
                print(f"Total profiling time: {result.metadata['total_profiling_time_s']:.1f} seconds")
        else:
            # Single-layer profiling (backward compatible)
            result = run_profiling(args, backend)

            if backend.is_client() and result is not None:
                output_path = args.output
                if output_path is None:
                    # Auto-resolve from topology name
                    project_root = Path(__file__).parent.parent.parent
                    output_path = project_root / "data" / "network_models" / f"{args.topology}_loggp.json"
                else:
                    output_path = Path(output_path)

                save_profiling_result(result, output_path)
                print(f"\nSaved LogGP parameters to {output_path}")
    except Exception as e:
        if backend.rank == 0:
            print(f"Error during profiling: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        backend.cleanup()


if __name__ == "__main__":
    main()
