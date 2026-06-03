"""Network communication simulator for distributed ML training.

This module models flow-level network behavior with max-min fair bandwidth
allocation over a multipath topology. There is no LogGP profiling step —
bandwidth and latency come directly from the hardware spec.
"""

from .simulator import Op, simulate, SimulationResult, solve_max_min_fair
from .topology import (
    Gpu, Switch, Link, Topology,
    build_simple, build_arbitrary, build_two_layer_multipath, build_fat_tree,
    build_dimensional, build_topology_from_config,
)
from .collectives import allreduce, allgather, reduce_scatter, broadcast, all_to_all
from .load_balancer import (
    LoadBalancer, EcmpHash, get_load_balancer, register_load_balancer,
)

__all__ = [
    "Op", "simulate", "SimulationResult", "solve_max_min_fair",
    "Gpu", "Switch", "Link", "Topology",
    "build_simple", "build_arbitrary",
    "build_two_layer_multipath", "build_fat_tree",
    "build_dimensional", "build_topology_from_config",
    "allreduce",
    "allgather",
    "reduce_scatter",
    "broadcast",
    "all_to_all",
    "LoadBalancer", "EcmpHash", "get_load_balancer", "register_load_balancer",
]
