"""Protocol change detection using Hoefler's lookahead algorithm.

This module implements the protocol detection algorithm from Hoefler et al.'s
"LogGP in Theory and Practice" (2009) paper. It detects changes in communication
protocols (e.g., eager → rendezvous) by analyzing PRTT measurements across
different message sizes.

The key insight: when the protocol changes, the relationship between message size
and gap time (Gall) deviates from the linear model. The lookahead algorithm
detects these deviations using least-squares fitting.

References:
    - "LogGP in Theory and Practice" (Hoefler et al., 2009), Section 4.3
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PRTTMeasurement:
    """PRTT (Parametrized Round Trip Time) measurement for a single message size.

    Attributes:
        size: Message size in bytes
        prtt_1_0: PRTT(1, 0, size) - single round-trip time
        prtt_n_0: PRTT(n, 0, size) - n iterations, no delay
        prtt_n_dG: PRTT(n, dG, size) - n iterations with dG delay
    """

    size: int
    prtt_1_0: float  # seconds
    prtt_n_0: float  # seconds
    prtt_n_dG: float  # seconds


@dataclass
class ProtocolRange:
    """A range of message sizes using the same protocol.

    Attributes:
        start_idx: Index of first measurement in range
        end_idx: Index of last measurement in range (inclusive)
        sizes: List of message sizes in this range
        g: Base gap parameter (seconds)
        G: Gap per byte parameter (seconds/byte)
        fit_error: Mean squared error of least-squares fit
    """

    start_idx: int
    end_idx: int
    sizes: List[int]
    g: float  # seconds
    G: float  # seconds/byte
    fit_error: float


def compute_gall(measurements: List[PRTTMeasurement], n: int = 10) -> List[float]:
    """Compute Gall(s) = [PRTT(n,0,s) - PRTT(1,0,s)] / (n-1).

    This represents the aggregate gap: Gall(s) = g + (s-1)*G

    Args:
        measurements: List of PRTT measurements
        n: Number of iterations used in PRTT(n,0,s)

    Returns:
        List of Gall values (seconds), one per measurement

    Example:
        >>> measurements = [
        ...     PRTTMeasurement(1024, 1e-5, 1.5e-4, 2e-4),
        ...     PRTTMeasurement(2048, 1.2e-5, 1.8e-4, 2.4e-4),
        ... ]
        >>> gall = compute_gall(measurements, n=10)
        >>> len(gall)
        2
    """
    gall = []
    for m in measurements:
        g_val = (m.prtt_n_0 - m.prtt_1_0) / (n - 1)
        gall.append(g_val)
    return gall


def least_squares_fit(sizes: List[int], gall: List[float]) -> Tuple[float, float, float]:
    """Fit Gall(s) = g + (s-1)*G using least squares.

    Args:
        sizes: Message sizes (bytes)
        gall: Measured Gall values (seconds)

    Returns:
        Tuple of (g, G, mse):
            - g: Base gap (seconds)
            - G: Gap per byte (seconds/byte)
            - mse: Mean squared error of fit

    Example:
        >>> sizes = [1024, 2048, 4096, 8192]
        >>> gall = [2e-6, 2.04e-6, 2.12e-6, 2.28e-6]  # g≈2e-6, G≈4e-11
        >>> g, G, mse = least_squares_fit(sizes, gall)
        >>> abs(g - 2e-6) < 1e-7
        True
        >>> abs(G - 4e-11) < 1e-12
        True
    """
    if len(sizes) != len(gall):
        raise ValueError(f"sizes and gall must have same length, got {len(sizes)} vs {len(gall)}")

    if len(sizes) < 2:
        raise ValueError(f"Need at least 2 points for fitting, got {len(sizes)}")

    # Convert to numpy arrays
    sizes_arr = np.array(sizes, dtype=np.float64)
    gall_arr = np.array(gall, dtype=np.float64)

    # Build design matrix: [1, s-1]
    # Gall(s) = g + (s-1)*G
    #         = g*1 + G*(s-1)
    X = np.column_stack([np.ones(len(sizes)), sizes_arr - 1])
    y = gall_arr

    # Solve: [g, G] = argmin ||X*[g, G]^T - y||^2
    params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    g = params[0]
    G = params[1]

    # Compute MSE
    predictions = X @ params
    mse = np.mean((y - predictions) ** 2)

    return g, G, mse


def detect_protocol_changes(
    measurements: List[PRTTMeasurement], n: int = 10, lookahead: int = 3, pfact: float = 2.0
) -> List[ProtocolRange]:
    """Detect protocol changes using Hoefler's lookahead algorithm.

    The algorithm works as follows:
    1. Start with first measurement as beginning of current protocol
    2. Fit g, G to all measurements from last change to current point
    3. Check next `lookahead` points: if ALL have >pfact worse fit error,
       declare a protocol change at current point
    4. Repeat from new protocol start

    Args:
        measurements: List of PRTT measurements (sorted by size)
        n: Number of iterations used in PRTT measurements (default 10)
        lookahead: Number of points to check ahead (default 3)
        pfact: Sensitivity factor for detecting changes (default 2.0)

    Returns:
        List of ProtocolRange objects, one per detected protocol

    Example:
        >>> # Synthetic eager (small) + rendezvous (large) protocols
        >>> measurements = [
        ...     PRTTMeasurement(1024, 1e-5, 1.5e-4, 2e-4),   # eager
        ...     PRTTMeasurement(2048, 1.1e-5, 1.6e-4, 2.1e-4),  # eager
        ...     PRTTMeasurement(8192, 1.5e-5, 2.5e-4, 3.2e-4),  # rendezvous (protocol change!)
        ...     PRTTMeasurement(16384, 2.0e-5, 3.5e-4, 4.5e-4),  # rendezvous
        ... ]
        >>> protocols = detect_protocol_changes(measurements, n=10, lookahead=2)
        >>> len(protocols) >= 1  # At least one protocol
        True
    """
    if len(measurements) < 2:
        raise ValueError(f"Need at least 2 measurements, got {len(measurements)}")

    # Compute Gall for all measurements
    gall = compute_gall(measurements, n)

    # Track protocol ranges
    ranges: List[ProtocolRange] = []
    last_change = 0  # Index of last protocol change

    i = 0
    while i < len(measurements):
        # Fit to current range [last_change : i+1]
        sizes_current = [m.size for m in measurements[last_change : i + 1]]
        gall_current = gall[last_change : i + 1]

        if len(sizes_current) < 2:
            # Need at least 2 points to fit
            i += 1
            continue

        g_current, G_current, mse_current = least_squares_fit(sizes_current, gall_current)

        # Check if next `lookahead` points all have worse fit
        protocol_changed = False
        if i + lookahead < len(measurements):
            # Check all points in lookahead window
            all_worse = True
            for offset in range(1, lookahead + 1):
                # Fit including next point
                sizes_next = [m.size for m in measurements[last_change : i + 1 + offset]]
                gall_next = gall[last_change : i + 1 + offset]
                _, _, mse_next = least_squares_fit(sizes_next, gall_next)

                # If ANY point has acceptable fit, no protocol change
                if mse_next <= pfact * mse_current:
                    all_worse = False
                    break

            if all_worse:
                protocol_changed = True

        if protocol_changed:
            # Save current protocol range
            ranges.append(
                ProtocolRange(
                    start_idx=last_change,
                    end_idx=i,
                    sizes=[m.size for m in measurements[last_change : i + 1]],
                    g=g_current,
                    G=G_current,
                    fit_error=mse_current,
                )
            )

            # Start new protocol
            last_change = i + 1

        i += 1

    # Save final protocol range
    if last_change < len(measurements):
        sizes_final = [m.size for m in measurements[last_change:]]
        gall_final = gall[last_change:]

        if len(sizes_final) >= 2:
            g_final, G_final, mse_final = least_squares_fit(sizes_final, gall_final)
            ranges.append(
                ProtocolRange(
                    start_idx=last_change,
                    end_idx=len(measurements) - 1,
                    sizes=sizes_final,
                    g=g_final,
                    G=G_final,
                    fit_error=mse_final,
                )
            )
        elif len(sizes_final) == 1:
            # Single point: use last fitted params if available
            if ranges:
                # Use parameters from previous protocol
                ranges.append(
                    ProtocolRange(
                        start_idx=last_change,
                        end_idx=len(measurements) - 1,
                        sizes=sizes_final,
                        g=ranges[-1].g,
                        G=ranges[-1].G,
                        fit_error=0.0,  # Single point, no error
                    )
                )
            else:
                # No previous protocol, can't fit
                raise ValueError("Cannot fit single measurement without protocol history")

    # If no ranges created, create single range with all data
    if not ranges:
        sizes_all = [m.size for m in measurements]
        gall_all = gall
        g_all, G_all, mse_all = least_squares_fit(sizes_all, gall_all)
        ranges.append(
            ProtocolRange(
                start_idx=0, end_idx=len(measurements) - 1, sizes=sizes_all, g=g_all, G=G_all, fit_error=mse_all
            )
        )

    return ranges
