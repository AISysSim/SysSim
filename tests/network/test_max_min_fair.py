"""Unit tests for the pure max-min fair solver."""

import numpy as np
from syssim.network.simulator import solve_max_min_fair


def test_single_flow_gets_narrowest_link():
    # 1 flow, 2 links with capacities [10, 5], flow uses both
    flow_paths = np.array([[True, True]])
    link_caps = np.array([10.0, 5.0])
    rates = solve_max_min_fair(flow_paths, link_caps)
    assert rates.shape == (1,)
    assert abs(rates[0] - 5.0) < 1e-9


def test_two_disjoint_flows_get_full_capacity():
    flow_paths = np.array([[True, False], [False, True]])
    link_caps = np.array([10.0, 8.0])
    rates = solve_max_min_fair(flow_paths, link_caps)
    assert abs(rates[0] - 10.0) < 1e-9
    assert abs(rates[1] - 8.0) < 1e-9


def test_two_flows_sharing_link_split_evenly():
    flow_paths = np.array([[True], [True]])
    link_caps = np.array([10.0])
    rates = solve_max_min_fair(flow_paths, link_caps)
    assert abs(rates[0] - 5.0) < 1e-9
    assert abs(rates[1] - 5.0) < 1e-9


def test_mixed_bottleneck_allocates_leftover():
    # 3 flows, 2 links of capacity [4, 10]:
    #   flow A uses link 0 alone (so it can use 4)
    #   flow B uses link 0 + link 1 (bottlenecked by link 0 at 4)
    #   flow C uses link 1 alone (uses leftover on link 1 after B claims 4 = 6)
    # Wait: max-min fair sharing on link 0 between A and B → 2 each.
    # Then link 1: B has 2 already, C can use 10 - 2 = 8.
    # Final: A=2, B=2, C=8.
    flow_paths = np.array([
        [True,  False],  # A
        [True,  True ],  # B
        [False, True ],  # C
    ])
    link_caps = np.array([4.0, 10.0])
    rates = solve_max_min_fair(flow_paths, link_caps)
    assert abs(rates[0] - 2.0) < 1e-9, rates
    assert abs(rates[1] - 2.0) < 1e-9, rates
    assert abs(rates[2] - 8.0) < 1e-9, rates
