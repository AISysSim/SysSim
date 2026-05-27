"""Unit tests for the load-balancer registry and ECMP-hash policy."""

import pytest
from syssim.network.load_balancer import (
    LoadBalancer, get_load_balancer, register_load_balancer, EcmpHash,
)


def test_default_ecmp_hash_is_registered():
    lb = get_load_balancer("ecmp_hash")
    assert isinstance(lb, EcmpHash)


def test_ecmp_hash_is_deterministic():
    lb = get_load_balancer("ecmp_hash")
    # Same (src, dst, tag) → same spine
    a = lb.choose_spine(src=0, dst=8, flow_tag=42, num_spines=4)
    b = lb.choose_spine(src=0, dst=8, flow_tag=42, num_spines=4)
    assert a == b
    assert 0 <= a < 4


def test_ecmp_hash_varies_with_tag():
    lb = get_load_balancer("ecmp_hash")
    spines = {lb.choose_spine(src=0, dst=8, flow_tag=t, num_spines=4)
              for t in range(100)}
    # With 100 distinct tags and 4 spines, very likely to hit every spine
    assert spines == {0, 1, 2, 3}


def test_unknown_load_balancer_raises():
    with pytest.raises(KeyError, match="round_robin"):
        get_load_balancer("round_robin")


def test_register_custom_load_balancer():
    class Always0(LoadBalancer):
        def choose_spine(self, *, src, dst, flow_tag, num_spines):
            return 0
    register_load_balancer("always_0", Always0())
    lb = get_load_balancer("always_0")
    assert lb.choose_spine(src=1, dst=2, flow_tag=99, num_spines=8) == 0
