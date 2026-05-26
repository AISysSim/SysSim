"""Network communication simulator for multi-model RLHF workloads.

This module provides composable collective communication primitives with automatic
dependency inference and contention-aware simulation.

Key features:
- 8 collective algorithms (allreduce, broadcast, reduce, etc.)
- 5 topology models (fully-connected, ring, switch, NVLink mesh, hierarchical)
- Automatic dependency inference from 2 physical rules
- Event-driven congestion simulation with max-min fair bandwidth sharing
- Multi-node support with layer-specific LogGP parameters

Example usage:
    >>> from syssim.network import allreduce, FullyConnectedTopology, LogGPParams, simulate
    >>>
    >>> # Define topology and performance model
    >>> topo = FullyConnectedTopology(num_ranks=8, link_bandwidth=25e9)
    >>> loggp = LogGPParams(L=1e-6, o=7e-6, G=1/(25e9))
    >>>
    >>> # Build and simulate allreduce
    >>> ops = allreduce(ranks=list(range(8)), total_size=1e9)
    >>> result = simulate(ops, topo, loggp)
    >>> print(f"Makespan: {result.makespan * 1e3:.2f} ms")

See Also:
    - netsim.md: Design document and architecture overview
    - examples/: Example notebooks and scripts
"""

# Core abstractions
# Collectives
from .collectives import (
    allgather,
    allreduce,
    alltoall,
    broadcast,
    gather,
    reduce,
    reduce_scatter,
    scatter,
)
from .dag_builder import Op, Step, build_dag
from .loggp import LogGPParams

# Model loader (LogGP profiler)
from .model_loader import (
    get_layer_params,
    get_protocol_for_size,
    is_hierarchical_model,
    load_all_protocols,
    load_hierarchical_loggp,
    load_loggp_params,
)

# Simulation engine
from .simulator import SimulationResult, simulate
from .topology import (
    FullyConnectedTopology,
    HierarchicalTopology,
    NVLinkMeshTopology,
    Resource,
    RingTopology,
    SwitchTopology,
    Topology,
)

# Additional topologies (added in Phase 4)
# from .topology import (
#     RingTopology, SwitchTopology, NVLinkMeshTopology, HierarchicalTopology
# )

__all__ = [
    # LogGP model
    "LogGPParams",
    # Topology
    "Topology",
    "Resource",
    "FullyConnectedTopology",
    "RingTopology",
    "SwitchTopology",
    "NVLinkMeshTopology",
    "HierarchicalTopology",
    # DAG construction
    "Op",
    "Step",
    "build_dag",
    # Collectives
    "allreduce",
    "broadcast",
    "reduce",
    "reduce_scatter",
    "allgather",
    "alltoall",
    "scatter",
    "gather",
    # Simulation
    "simulate",
    "SimulationResult",
    # Model loader
    "load_loggp_params",
    "load_all_protocols",
    "get_protocol_for_size",
    "load_hierarchical_loggp",
    "is_hierarchical_model",
    "get_layer_params",
]
