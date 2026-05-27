"""Routing policies for the inter-node topology.

A load balancer picks one of the `num_spines` available paths between two
endpoints. The default is `ecmp_hash`, which hashes (src, dst, flow_tag)
deterministically. Users can register additional policies via
`register_load_balancer(name, instance)`.
"""

from __future__ import annotations

import hashlib


class LoadBalancer:
    """Base class. Subclasses implement `choose_spine`."""

    def choose_spine(self, *, src: int, dst: int, flow_tag: int,
                     num_spines: int) -> int:
        raise NotImplementedError


class EcmpHash(LoadBalancer):
    """Deterministic hash of (src, dst, flow_tag) modulo num_spines."""

    def choose_spine(self, *, src: int, dst: int, flow_tag: int,
                     num_spines: int) -> int:
        h = hashlib.blake2b(
            f"{src}-{dst}-{flow_tag}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(h, "little") % num_spines


_REGISTRY: dict[str, LoadBalancer] = {"ecmp_hash": EcmpHash()}


def register_load_balancer(name: str, instance: LoadBalancer) -> None:
    """Register a new routing policy under `name`."""
    _REGISTRY[name] = instance


def get_load_balancer(name: str) -> LoadBalancer:
    """Look up a registered policy by name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(name)
    return _REGISTRY[name]
