"""Platform-agnostic operator profiling script for non-CPU/GPU targets.

This script collects operator execution times on a target accelerator platform
and saves clean CSV data.
The CSV can then be transferred to a CUDA-capable machine for model training.

Usage:
  python -m syssim.compute.compute_cost_profiler_custom \
      --operator gemm --output data/profiling/

  python -m syssim.compute.compute_cost_profiler_custom \
      --operator attn --output data/profiling/

WARNING:
  This script only performs profiling (data collection). Training efficiency
  models (XGBoost / MLP) requires CUDA and should be done on a separate
  CUDA-capable machine using:

    python -m syssim.compute.compute_cost_profiler \
        --operator <op> \
        --data-path <csv_from_this_script> \
        --output data/trained_models/<op>_<platform>_xgb.pth \
        --backend xgboost
"""

from __future__ import annotations

import argparse
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================================
# Hardware Info (manually specified for target platform)
# ==============================================================================

# TODO(platform): Fill in target platform's hardware specs and name.

PLATFORM_HW_NAME = "unknown"
PEAK_TFLOPS_MM = 0.0
PEAK_TFLOPS_MATH = 0.0
PEAK_MEMORY_BANDWIDTH_GBPS = 0.0


# ==============================================================================
# Sampling Utilities
# ==============================================================================

def _generate_proportional_samples(
    start: int, end: int, total_samples: int = 64, seed: int = 42,
) -> list[int]:
    """Generate samples with power-of-two anchors and proportional fill.

    Args:
        start: Minimum value (inclusive, must be positive).
        end: Maximum value (inclusive).
        total_samples: Target number of samples.
        seed: Random seed for reproducibility.

    Returns:
        Sorted list of unique sampled values.
    """
    rng = random.Random(seed)
    if start <= 0 or end <= 0:
        raise ValueError(f"Range must be positive: [{start}, {end}]")

    first_exp = math.ceil(math.log2(start))
    last_exp = math.floor(math.log2(end))
    powers_of_two = [2 ** i for i in range(first_exp, last_exp + 1) if start <= 2 ** i <= end]

    if not powers_of_two:
        return sorted(rng.sample(range(start, end + 1), min(total_samples, end - start + 1)))

    F = total_samples - len(powers_of_two)
    if F <= 0:
        return sorted(powers_of_two)

    W = powers_of_two[-1] - powers_of_two[0]
    if W == 0:
        return sorted(powers_of_two)

    all_samples = list(powers_of_two)
    for i in range(len(powers_of_two) - 1):
        lower, upper = powers_of_two[i], powers_of_two[i + 1]
        k_i = round(F * (upper - lower) / W)
        if k_i > 0:
            available = list(range(lower + 1, upper))
            if available:
                all_samples.extend(rng.sample(available, min(k_i, len(available))))

    return sorted(set(all_samples))


def _generate_power_of_two_range(start: int, end: int) -> list[int]:
    """Generate all power-of-two values in [start, end]."""
    first_exp = math.ceil(math.log2(max(1, start)))
    last_exp = math.floor(math.log2(end))
    return [2 ** e for e in range(first_exp, last_exp + 1) if start <= 2 ** e <= end]


# ==============================================================================
# Parameter Grids
# ==============================================================================

def _build_grids() -> dict[str, dict]:
    """Build parameter grids for all operator types."""
    seed = 42
    return {
        "gemm": {
            "M": _generate_proportional_samples(2, 131072, 64, seed),
            "N": _generate_proportional_samples(256, 65536, 64, seed + 1),
            "K": _generate_proportional_samples(256, 16384, 64, seed + 2),
        },
        "attn": {
            "bs": _generate_power_of_two_range(1, 16),
            "seq": _generate_proportional_samples(1, 131072, 64, seed),
            "nh": _generate_power_of_two_range(2, 128),
            "nkv": _generate_power_of_two_range(1, 8),
            "hd": [64, 128],
        },
        "rmsnorm": {
            "seq": _generate_proportional_samples(2, 131072, 128, seed),
            "dim": _generate_proportional_samples(128, 16384, 128, seed + 1),
        },
        "silu": {
            "seq": _generate_proportional_samples(2, 131072, 128, seed),
            "dim": _generate_proportional_samples(768, 106496, 128, seed + 1),
        },
    }


COMPUTE_GRIDS = _build_grids()


# ==============================================================================
# Single-config Profiling Functions
#
# Contract for each function:
#   1. Allocate tensors on target device
#   2. Warmup (run op several times)
#   3. Synchronize (ensure device execution completes)
#   4. Timed loop: op → sync → record wall time
#   5. Return median time in milliseconds, or -1.0 for OOM/timeout
# ==============================================================================

def _profile_gemm(m: int, n: int, k: int, num_runs: int = 100) -> float:
    """Profile GEMM: C[m,n] = A[m,k] @ B[k,n], FP16.

    Returns:
        Median execution time in milliseconds, or -1.0 for OOM/timeout.
    """
    # TODO(platform): Implement GEMM profiling.
    #
    # Reference (CUDA):
    #   a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    #   b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    #   for _ in range(5): torch.mm(a, b)         # warmup
    #   torch.cuda.synchronize()
    #   times = []
    #   for _ in range(num_runs):
    #       start = time.perf_counter()
    #       torch.mm(a, b)
    #       torch.cuda.synchronize()
    #       times.append((time.perf_counter() - start) * 1000)
    #   return float(np.median(times))
    raise NotImplementedError("GEMM profiling not implemented for this platform")


def _profile_attention(
    batch: int, num_heads: int, seq_len: int, head_dim: int,
    num_kv_heads: int | None = None, num_runs: int = 100,
) -> float:
    """Profile scaled dot-product attention (MHA/GQA).

    Shapes: Q[batch, num_heads, seq_len, head_dim],
            K/V[batch, num_kv_heads, seq_len, head_dim].

    Returns:
        Median execution time in milliseconds, or -1.0 for OOM/timeout.
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads

    # TODO(platform): Implement attention profiling.
    #
    # Reference (CUDA):
    #   q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    #   k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    #   v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    #   if num_kv_heads != num_heads:  # GQA expand
    #       k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    #       v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
    #   for _ in range(10): F.scaled_dot_product_attention(q, k, v)  # warmup
    #   torch.cuda.synchronize()
    #   times = []
    #   for _ in range(num_runs):
    #       start = time.perf_counter()
    #       F.scaled_dot_product_attention(q, k, v)
    #       torch.cuda.synchronize()
    #       times.append((time.perf_counter() - start) * 1000)
    #   return float(np.median(times))
    #
    # If no native SDPA, implement manually: softmax(Q @ K^T / sqrt(d)) @ V
    raise NotImplementedError("Attention profiling not implemented for this platform")


def _profile_rmsnorm(seq: int, dim: int, num_runs: int = 100) -> float:
    """Profile RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight.

    Shapes: x[seq, dim] FP32, weight[dim].

    Returns:
        Median execution time in milliseconds, or -1.0 for OOM/timeout.
    """
    # TODO(platform): Implement RMSNorm profiling.
    #
    # Reference (CUDA):
    #   x = torch.randn(seq, dim, device="cuda", dtype=torch.float32)
    #   w = torch.ones(dim, device="cuda", dtype=torch.float32)
    #   rmsnorm = lambda x: x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6) * w
    #   for _ in range(10): rmsnorm(x)  # warmup
    #   torch.cuda.synchronize()
    #   times = []
    #   for _ in range(num_runs):
    #       start = time.perf_counter()
    #       rmsnorm(x)
    #       torch.cuda.synchronize()
    #       times.append((time.perf_counter() - start) * 1000)
    #   return float(np.median(times))
    raise NotImplementedError("RMSNorm profiling not implemented for this platform")


def _profile_silu(seq: int, dim: int, num_runs: int = 100) -> float:
    """Profile SiLU: y = x * sigmoid(x).

    Shapes: x[seq, dim] FP16.

    Returns:
        Median execution time in milliseconds, or -1.0 for OOM/timeout.
    """
    # TODO(platform): Implement SiLU profiling.
    #
    # Reference (CUDA):
    #   x = torch.randn(seq, dim, device="cuda", dtype=torch.float16)
    #   silu = torch.nn.SiLU()
    #   for _ in range(10): silu(x)  # warmup
    #   torch.cuda.synchronize()
    #   times = []
    #   for _ in range(num_runs):
    #       start = time.perf_counter()
    #       silu(x)
    #       torch.cuda.synchronize()
    #       times.append((time.perf_counter() - start) * 1000)
    #   return float(np.median(times))
    #
    # SiLU = x * sigmoid(x), implement manually if no native op.
    raise NotImplementedError("SiLU profiling not implemented for this platform")


# ==============================================================================
# Grid Profiling
# ==============================================================================

def _profile_gemm_grid(grid: dict, num_runs: int) -> list[dict]:
    results = []
    total = len(grid["M"]) * len(grid["N"]) * len(grid["K"])
    count = 0
    for m in grid["M"]:
        for n in grid["N"]:
            for k in grid["K"]:
                count += 1
                if count % 100 == 0 or count == 1:
                    print(f"  [{count}/{total}] M={m}, N={n}, K={k}")
                try:
                    t = _profile_gemm(m, n, k, num_runs)
                except (RuntimeError, MemoryError):
                    t = -1.0
                results.append({"M": m, "N": n, "K": k, "t_measured_ms": t})
    return results


def _profile_attn_grid(grid: dict, num_runs: int) -> list[dict]:
    results = []
    total = (len(grid["bs"]) * len(grid["seq"]) * len(grid["nh"])
             * len(grid["nkv"]) * len(grid["hd"]))
    count = 0
    for bs in grid["bs"]:
        for seq in grid["seq"]:
            for nh in grid["nh"]:
                for nkv in grid["nkv"]:
                    for hd in grid["hd"]:
                        count += 1
                        if count % 100 == 0 or count == 1:
                            print(f"  [{count}/{total}] bs={bs}, seq={seq}, "
                                  f"nh={nh}, nkv={nkv}, hd={hd}")
                        try:
                            t = _profile_attention(
                                batch=bs, num_heads=nh, seq_len=seq,
                                head_dim=hd, num_kv_heads=nkv, num_runs=num_runs)
                        except (RuntimeError, MemoryError):
                            t = -1.0
                        results.append({"bs": bs, "seq": seq, "nh": nh,
                                        "nkv": nkv, "hd": hd, "t_measured_ms": t})
    return results


def _profile_rmsnorm_grid(grid: dict, num_runs: int) -> list[dict]:
    results = []
    total = len(grid["seq"]) * len(grid["dim"])
    count = 0
    for seq in grid["seq"]:
        for dim in grid["dim"]:
            count += 1
            if count % 100 == 0 or count == 1:
                print(f"  [{count}/{total}] seq={seq}, dim={dim}")
            try:
                t = _profile_rmsnorm(seq, dim, num_runs)
            except (RuntimeError, MemoryError):
                t = -1.0
            results.append({"seq": seq, "dim": dim, "t_measured_ms": t})
    return results


def _profile_silu_grid(grid: dict, num_runs: int) -> list[dict]:
    results = []
    total = len(grid["seq"]) * len(grid["dim"])
    count = 0
    for seq in grid["seq"]:
        for dim in grid["dim"]:
            count += 1
            if count % 100 == 0 or count == 1:
                print(f"  [{count}/{total}] seq={seq}, dim={dim}")
            try:
                t = _profile_silu(seq, dim, num_runs)
            except (RuntimeError, MemoryError):
                t = -1.0
            results.append({"seq": seq, "dim": dim, "t_measured_ms": t})
    return results


_GRID_DISPATCH = {
    "gemm": _profile_gemm_grid,
    "attn": _profile_attn_grid,
    "rmsnorm": _profile_rmsnorm_grid,
    "silu": _profile_silu_grid,
}


# ==============================================================================
# Entry Point
# ==============================================================================

def profile_operator(operator: str, output_dir: str, num_runs: int = 100) -> Path:
    """Profile operator across parameter grid and save CSV.

    Args:
        operator: One of "gemm", "attn", "rmsnorm", "silu".
        output_dir: Directory to save the CSV file.
        num_runs: Number of timed iterations per configuration.

    Returns:
        Path to the saved CSV file.
    """
    if PLATFORM_HW_NAME == "unknown":
        raise RuntimeError(
            "Hardware not configured. Set PLATFORM_HW_NAME, PEAK_TFLOPS_MM, "
            "PEAK_TFLOPS_MATH, PEAK_MEMORY_BANDWIDTH_GBPS at the top of this script."
        )
    if operator not in _GRID_DISPATCH:
        raise ValueError(f"Unknown operator: {operator}. Choose from {list(_GRID_DISPATCH)}")

    grid = COMPUTE_GRIDS[operator]

    if operator == "gemm":
        total = len(grid["M"]) * len(grid["N"]) * len(grid["K"])
    elif operator == "attn":
        total = (len(grid["bs"]) * len(grid["seq"]) * len(grid["nh"])
                 * len(grid["nkv"]) * len(grid["hd"]))
    else:
        total = len(grid["seq"]) * len(grid["dim"])

    print("=" * 70)
    print(f"  Profiling: {operator.upper()}")
    print(f"  Platform:  {PLATFORM_HW_NAME}")
    print(f"  Configs:   {total:,}")
    print(f"  Runs/cfg:  {num_runs}")
    print("=" * 70)

    results = _GRID_DISPATCH[operator](grid, num_runs)

    df = pd.DataFrame(results)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{operator}_{PLATFORM_HW_NAME}_data.csv"
    df.to_csv(csv_path, index=False)

    n_valid = int((df["t_measured_ms"] > 0).sum())
    n_failed = len(df) - n_valid

    print()
    print("=" * 70)
    print(f"  Saved to: {csv_path}")
    print(f"  Total: {len(df)}  Valid: {n_valid}  Failed: {n_failed}")
    print("=" * 70)

    warnings.warn(
        "\n"
        "  Training efficiency models (XGBoost / MLP) requires CUDA.\n"
        "  Copy the CSV to a CUDA-capable machine and run:\n"
        "\n"
        f"    python -m syssim.compute.compute_cost_profiler \\\n"
        f"        --operator {operator} \\\n"
        f"        --data-path {csv_path} \\\n"
        f"        --output data/trained_models/{operator}_{PLATFORM_HW_NAME}_xgb.pth \\\n"
        f"        --backend xgboost\n",
        stacklevel=2,
    )

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Profile operators on non-CPU/GPU platforms"
    )
    parser.add_argument(
        "--operator", required=True,
        choices=["gemm", "attn", "rmsnorm", "silu"],
        help="Operator type to profile",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for CSV data",
    )
    parser.add_argument(
        "--num-runs", type=int, default=100,
        help="Number of profiling runs per configuration (default: 100)",
    )
    args = parser.parse_args()
    profile_operator(args.operator, args.output, args.num_runs)


if __name__ == "__main__":
    main()
