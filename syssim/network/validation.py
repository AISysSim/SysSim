"""Analytical formula validators for collective communication algorithms.

This module provides closed-form performance formulas for collective algorithms
on idealized topologies (FullyConnectedTopology with no contention). These formulas
are used to validate the simulator's correctness.

All formulas assume LogGP model: T_msg = α + (m-1)*G
where α = L + 2*o (pipeline overhead) and G = 1/bandwidth.

References:
- "Optimization of Collective Communication Operations in MPICH" (Thakur et al., 2005)
- "A Better Model for Collective Communication" (Bruck et al., 1997)
"""

import math

from .loggp import LogGPParams


def validate_allreduce(
    num_ranks: int,
    total_size: float,
    loggp: LogGPParams,
    simulated_time: float,
    tolerance: float = 1e-5,
) -> tuple[bool, float, float]:
    """Validate ring allreduce against analytical formula.

    Ring allreduce: 2(P-1) steps, each sends M/P bytes
    Formula: T = 2(P-1) * (α + (M/P - 1) * G)

    Args:
        num_ranks: Number of participating ranks
        total_size: Total message size in bytes
        loggp: LogGP performance parameters
        simulated_time: Simulated makespan in seconds
        tolerance: Relative error tolerance (default 1e-6)

    Returns:
        (is_valid, analytical_time, relative_error)
    """
    P = num_ranks
    chunk_size = total_size / P

    # Ring allreduce: 2(P-1) steps
    num_steps = 2 * (P - 1)

    # Time per step: α + (chunk_size - 1) * G
    step_time = loggp.alpha + (chunk_size - 1) * loggp.G

    analytical_time = num_steps * step_time

    relative_error = abs(simulated_time - analytical_time) / analytical_time
    is_valid = relative_error <= tolerance

    return is_valid, analytical_time, relative_error


def validate_broadcast(
    num_ranks: int,
    total_size: float,
    loggp: LogGPParams,
    simulated_time: float,
    tolerance: float = 1e-5,
) -> tuple[bool, float, float]:
    """Validate binomial tree broadcast against analytical formula.

    Binomial broadcast: ⌈log₂ P⌉ steps, each sends M bytes
    Formula: T = ⌈log₂ P⌉ * (α + (M - 1) * G)

    Args:
        num_ranks: Number of participating ranks
        total_size: Message size in bytes
        loggp: LogGP performance parameters
        simulated_time: Simulated makespan in seconds
        tolerance: Relative error tolerance

    Returns:
        (is_valid, analytical_time, relative_error)
    """
    P = num_ranks
    num_steps = math.ceil(math.log2(P))

    # Time per step: α + (total_size - 1) * G
    step_time = loggp.alpha + (total_size - 1) * loggp.G

    analytical_time = num_steps * step_time

    relative_error = abs(simulated_time - analytical_time) / analytical_time
    is_valid = relative_error <= tolerance

    return is_valid, analytical_time, relative_error


def validate_reduce(
    num_ranks: int,
    total_size: float,
    loggp: LogGPParams,
    simulated_time: float,
    tolerance: float = 1e-5,
) -> tuple[bool, float, float]:
    """Validate binomial tree reduce against analytical formula.

    Binomial reduce: ⌈log₂ P⌉ steps, each sends M bytes
    Formula: T = ⌈log₂ P⌉ * (α + (M - 1) * G)

    Args:
        num_ranks: Number of participating ranks
        total_size: Message size in bytes
        loggp: LogGP performance parameters
        simulated_time: Simulated makespan in seconds
        tolerance: Relative error tolerance

    Returns:
        (is_valid, analytical_time, relative_error)
    """
    # Same formula as broadcast (binomial tree)
    return validate_broadcast(num_ranks, total_size, loggp, simulated_time, tolerance)


def validate_reduce_scatter(
    num_ranks: int,
    total_size: float,
    loggp: LogGPParams,
    simulated_time: float,
    tolerance: float = 1e-5,
) -> tuple[bool, float, float]:
    """Validate ring reduce-scatter against analytical formula.

    Ring reduce-scatter: (P-1) steps, each sends M/P bytes
    Formula: T = (P-1) * (α + (M/P - 1) * G)

    Args:
        num_ranks: Number of participating ranks
        total_size: Total message size in bytes
        loggp: LogGP performance parameters
        simulated_time: Simulated makespan in seconds
        tolerance: Relative error tolerance

    Returns:
        (is_valid, analytical_time, relative_error)
    """
    P = num_ranks
    chunk_size = total_size / P

    # Ring reduce-scatter: (P-1) steps
    num_steps = P - 1

    # Time per step: α + (chunk_size - 1) * G
    step_time = loggp.alpha + (chunk_size - 1) * loggp.G

    analytical_time = num_steps * step_time

    relative_error = abs(simulated_time - analytical_time) / analytical_time
    is_valid = relative_error <= tolerance

    return is_valid, analytical_time, relative_error


def validate_allgather(
    num_ranks: int,
    total_size: float,
    loggp: LogGPParams,
    simulated_time: float,
    tolerance: float = 1e-5,
) -> tuple[bool, float, float]:
    """Validate ring allgather against analytical formula.

    Ring allgather: (P-1) steps, each sends M/P bytes
    Formula: T = (P-1) * (α + (M/P - 1) * G)

    Args:
        num_ranks: Number of participating ranks
        total_size: Total message size in bytes
        loggp: LogGP performance parameters
        simulated_time: Simulated makespan in seconds
        tolerance: Relative error tolerance

    Returns:
        (is_valid, analytical_time, relative_error)
    """
    # Same formula as reduce_scatter (ring algorithm)
    return validate_reduce_scatter(num_ranks, total_size, loggp, simulated_time, tolerance)


def validate_alltoall(
    num_ranks: int,
    total_size: float,
    loggp: LogGPParams,
    simulated_time: float,
    tolerance: float = 1e-5,
) -> tuple[bool, float, float]:
    """Validate direct alltoall against analytical formula.

    Direct alltoall: (P-1) steps, each sends M/P bytes
    Formula: T = (P-1) * (α + (M/P - 1) * G)

    Args:
        num_ranks: Number of participating ranks
        total_size: Total message size in bytes
        loggp: LogGP performance parameters
        simulated_time: Simulated makespan in seconds
        tolerance: Relative error tolerance

    Returns:
        (is_valid, analytical_time, relative_error)
    """
    # Same formula as reduce_scatter (P-1 steps with M/P chunks)
    return validate_reduce_scatter(num_ranks, total_size, loggp, simulated_time, tolerance)


def validate_scatter(
    num_ranks: int,
    total_size: float,
    loggp: LogGPParams,
    simulated_time: float,
    tolerance: float = 1e-5,
) -> tuple[bool, float, float]:
    """Validate flat scatter against analytical formula.

    Flat scatter: (P-1) serialized sends of M/P bytes each
    Formula: T = (P-1) * (α + (M/P - 1) * G)

    Args:
        num_ranks: Number of participating ranks
        total_size: Total message size in bytes
        loggp: LogGP performance parameters
        simulated_time: Simulated makespan in seconds
        tolerance: Relative error tolerance

    Returns:
        (is_valid, analytical_time, relative_error)
    """
    # Same formula as reduce_scatter (serialized sends)
    return validate_reduce_scatter(num_ranks, total_size, loggp, simulated_time, tolerance)


def validate_gather(
    num_ranks: int,
    total_size: float,
    loggp: LogGPParams,
    bandwidth: float,
    simulated_time: float,
    tolerance: float = 1e-5,
) -> tuple[bool, float, float]:
    """Validate flat gather against analytical formula.

    Flat gather: (P-1) parallel sends to root, limited by root's receive bandwidth
    On FullyConnectedTopology: No contention, so time = α + (M/P - 1) * G

    However, on real topologies (like Switch), root receives (P-1) messages in parallel,
    causing bandwidth contention. For validation, we use FullyConnectedTopology which
    has no contention.

    Args:
        num_ranks: Number of participating ranks
        total_size: Total message size in bytes
        loggp: LogGP performance parameters
        bandwidth: Link bandwidth (used only for documentation, not in formula)
        simulated_time: Simulated makespan in seconds
        tolerance: Relative error tolerance

    Returns:
        (is_valid, analytical_time, relative_error)
    """
    P = num_ranks
    chunk_size = total_size / P

    # On FullyConnectedTopology: No contention, all sends complete in parallel
    # Time = α + (chunk_size - 1) * G (single message time)
    analytical_time = loggp.alpha + (chunk_size - 1) * loggp.G

    relative_error = abs(simulated_time - analytical_time) / analytical_time
    is_valid = relative_error <= tolerance

    return is_valid, analytical_time, relative_error
