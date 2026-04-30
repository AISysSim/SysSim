"""CLI tool for profiling compute operators and training efficiency models.

This module provides profiling utilities for:
- Benchmarking GEMM and attention operators across parameter grids
- Computing roofline predictions for each configuration
- Training MLP or XGBoost models to predict efficiency (realized / roofline)
- Saving trained models (.pth) and profiled data (.csv)

Usage:
    # MLP backend (default)
    python -m syssim.compute.compute_cost_profiler \
        --operator gemm \
        --output models/gemm_mlp.pth \
        --backend mlp \
        --epochs 300

    # XGBoost backend
    python -m syssim.compute.compute_cost_profiler \
        --operator gemm \
        --output models/gemm_mlp.pth \
        --backend xgboost
"""

from __future__ import annotations

import time
import math
import random
from pathlib import Path

import torch
import torch.nn as nn

from ..operator_graph import OperatorType
from ..config import HardwareInfo, get_hardware_info
from .compute_cost_predictor import roofline_estimate, aten

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    import xgboost as xgb
    import pickle
    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False


# ==============================================================================
# Proportional Bucket-Based Sampling (PROFILE_PLAN.md)
# ==============================================================================

def _generate_proportional_samples(
    start: int,
    end: int,
    total_samples: int = 64,
    seed: int = 42,
) -> list[int]:
    """Generate samples using proportional bucket-based strategy.

    Algorithm from PROFILE_PLAN.md:
    1. Include every power-of-two from 2^a to 2^b (P mandatory points)
    2. Calculate total width: W = 2^b - 2^a
    3. For each interval [2^i, 2^(i+1)), assign k_i = round(F × 2^i / W) fill points
    4. Draw k_i uniform random integers from (2^i, 2^(i+1))

    Args:
        start: Minimum value (inclusive)
        end: Maximum value (inclusive)
        total_samples: Target number of samples (default 64)
        seed: Random seed for reproducibility

    Returns:
        Sorted list of sampled values

    Example:
        >>> _generate_proportional_samples(2, 131072, total_samples=64)
        [2, 4, 8, ..., 131072]  # Powers of 2 + proportional fills
    """
    rng = random.Random(seed)

    if start <= 0 or end <= 0:
        raise ValueError(f"Range must be positive: [{start}, {end}]")

    # Step 1: Find all powers of 2 in range [start, end]
    first_exp = math.ceil(math.log2(start))
    last_exp = math.floor(math.log2(end))

    # Generate power-of-2 anchor points
    powers_of_two = [2 ** i for i in range(first_exp, last_exp + 1) if start <= 2 ** i <= end]

    if len(powers_of_two) == 0:
        # No powers of 2 in range - fall back to uniform random sampling
        samples = sorted(rng.sample(range(start, end + 1), min(total_samples, end - start + 1)))
        return samples

    P = len(powers_of_two)  # Number of mandatory anchor points

    # Step 2: Calculate fill budget
    F = total_samples - P  # Remaining samples to distribute

    if F <= 0:
        # Already have enough power-of-2 samples
        return sorted(powers_of_two)

    # Step 3: Calculate total interval width
    first_power = powers_of_two[0]
    last_power = powers_of_two[-1]
    W = last_power - first_power

    if W == 0:
        # Only one power of 2 in range
        return sorted(powers_of_two)

    # Step 4: Allocate fill samples proportionally to each interval
    all_samples = list(powers_of_two)  # Start with mandatory anchor points

    for i in range(len(powers_of_two) - 1):
        lower = powers_of_two[i]
        upper = powers_of_two[i + 1]

        # Interval width (proportional to 2^i)
        interval_width = upper - lower

        # Proportional allocation: k_i = round(F × interval_width / W)
        k_i = round(F * interval_width / W)

        if k_i > 0:
            # Draw k_i uniform random samples from (lower, upper) exclusive
            available = list(range(lower + 1, upper))
            if len(available) > 0:
                num_to_sample = min(k_i, len(available))
                fill_samples = rng.sample(available, num_to_sample)
                all_samples.extend(fill_samples)

    # Return sorted unique samples
    return sorted(set(all_samples))


def _generate_power_of_two_range(start: int, end: int) -> list[int]:
    """Generate all power-of-two values in range [start, end].

    Args:
        start: Minimum value (inclusive)
        end: Maximum value (inclusive)

    Returns:
        Sorted list of powers of 2 in range

    Example:
        >>> _generate_power_of_two_range(1, 16)
        [1, 2, 4, 8, 16]
    """
    first_exp = math.ceil(math.log2(max(1, start)))
    last_exp = math.floor(math.log2(end))

    powers = [2 ** exp for exp in range(first_exp, last_exp + 1)]
    return [p for p in powers if start <= p <= end]


# ==============================================================================
# Dataset Construction Functions (PROFILE_PLAN.md)
# ==============================================================================

def construct_dataset_gemm(total_samples: int = 64, seed: int = 42) -> dict:
    """Construct GEMM profiling dataset with proportional sampling.

    Ranges from PROFILE_PLAN.md:
    - M: [2, 131072]
    - N: [256, 65536]
    - K: [256, 16384]
    - 64 samples per dimension

    Args:
        total_samples: Target samples per dimension (default 64)
        seed: Random seed for reproducibility

    Returns:
        Dict with keys "M", "N", "K" containing sampled values
    """
    return {
        "M": _generate_proportional_samples(2, 131072, total_samples, seed),
        "N": _generate_proportional_samples(256, 65536, total_samples, seed + 1),
        "K": _generate_proportional_samples(256, 16384, total_samples, seed + 2),
    }


def construct_dataset_attn(total_samples: int = 64, seed: int = 42) -> dict:
    """Construct ATTN profiling dataset with strategic sparse sampling.

    Strategy from PROFILE_PLAN.md:
    - bs, nh, nkv: Powers-of-2 only (less critical dimensions)
    - seq: Proportional sampling (most critical - O(n²) complexity)
    - hd: Discrete values {64, 128}

    Ranges:
    - bs (batch): [1, 16] → Powers of 2 only
    - seq (seq_len): [1, 131072] → Proportional sampling (64 samples)
    - nh (num_heads): [2, 128] → Powers of 2 only
    - nkv (num_kv_heads): [1, 8] → Powers of 2 only
    - hd (head_dim): {64, 128} → Discrete

    Grid size: 5 × 64 × 7 × 4 × 2 = 17,920 configs (~9 hours)

    Args:
        total_samples: Target samples for seq only (default 64)
        seed: Random seed for reproducibility

    Returns:
        Dict with keys "bs", "seq", "nh", "nkv", "hd"
    """
    return {
        "bs": _generate_power_of_two_range(1, 16),  # [1, 2, 4, 8, 16] = 5 samples
        "seq": _generate_proportional_samples(1, 131072, total_samples, seed),  # 64 samples
        "nh": _generate_power_of_two_range(2, 128),  # [2, 4, 8, 16, 32, 64, 128] = 7 samples
        "nkv": _generate_power_of_two_range(1, 8),  # [1, 2, 4, 8] = 4 samples
        "hd": [64, 128],  # 2 discrete values
    }


def construct_dataset_rmsnorm(total_samples: int = 128, seed: int = 42) -> dict:
    """Construct RMSNorm profiling dataset with proportional sampling.

    Ranges from PROFILE_PLAN.md:
    - seq: [2, 131072]
    - dim: [128, 16384]
    - 128 samples per dimension

    Args:
        total_samples: Target samples per dimension (default 128)
        seed: Random seed for reproducibility

    Returns:
        Dict with keys "seq", "dim"
    """
    return {
        "seq": _generate_proportional_samples(2, 131072, total_samples, seed),
        "dim": _generate_proportional_samples(128, 16384, total_samples, seed + 1),
    }


def construct_dataset_silu(total_samples: int = 128, seed: int = 42) -> dict:
    """Construct SiLU profiling dataset with proportional sampling.

    Ranges from PROFILE_PLAN.md:
    - seq: [2, 131072]
    - dim: [768, 106496]
    - 128 samples per dimension

    Args:
        total_samples: Target samples per dimension (default 128)
        seed: Random seed for reproducibility

    Returns:
        Dict with keys "seq", "dim"
    """
    return {
        "seq": _generate_proportional_samples(2, 131072, total_samples, seed),
        "dim": _generate_proportional_samples(768, 106496, total_samples, seed + 1),
    }


# ==============================================================================
# Profiling Grids (Dynamically Constructed)
# ==============================================================================

# Configuration
GEMM_SAMPLES_PER_DIM = 64
ATTN_SAMPLES_SEQ = 64  # Only for seq dimension
MATH_SAMPLES_PER_DIM = 128
RANDOM_SEED = 42

# Dynamic grid construction
COMPUTE_GRIDS = {
    "gemm": construct_dataset_gemm(GEMM_SAMPLES_PER_DIM, RANDOM_SEED),
    "attn": construct_dataset_attn(ATTN_SAMPLES_SEQ, RANDOM_SEED),
    "rmsnorm": construct_dataset_rmsnorm(MATH_SAMPLES_PER_DIM, RANDOM_SEED),
    "silu": construct_dataset_silu(MATH_SAMPLES_PER_DIM, RANDOM_SEED),
}


def _profile_gemm(m: int, n: int, k: int, num_runs: int = 100, dtype: str = "fp16") -> float:
    """Profile a single GEMM at the requested precision; return median time (ms).

    Supported dtypes:
      "fp16" - PyTorch torch.mm with float16 inputs.
      "fp8"  - FlashInfer mm_fp8 with float8_e4m3fn inputs and per-tensor scale.
      "fp4"  - FlashInfer mm_fp4 with NVFP4 quantized inputs and block scales.

    Returns -1.0 on OOM/unsupported-shape failures (e.g. FP8/FP4 alignment violations);
    callers filter these rows before training. Returns positive median ms otherwise.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for profiling")

    device = torch.device("cuda")

    try:
        if dtype == "fp16":
            a = torch.randn(m, k, device=device, dtype=torch.float16)
            b = torch.randn(k, n, device=device, dtype=torch.float16)
            kernel = lambda: torch.mm(a, b)

        elif dtype == "fp8":
            # torch._scaled_mm is the general-purpose FP8 GEMM (Hopper/Blackwell).
            # FlashInfer mm_fp8 requires a TRT-LLM block layout (3D), unsuitable for
            # general profiling sweeps.
            if k % 16 != 0 or n % 16 != 0 or m == 0:
                return -1.0
            a_fp16 = torch.randn(m, k, device=device, dtype=torch.float16)
            b_fp16 = torch.randn(n, k, device=device, dtype=torch.float16)
            a_amax = a_fp16.abs().amax().clamp(min=1e-4)
            b_amax = b_fp16.abs().amax().clamp(min=1e-4)
            a = (a_fp16 * (448.0 / a_amax)).to(torch.float8_e4m3fn)
            b = (b_fp16 * (448.0 / b_amax)).to(torch.float8_e4m3fn)
            scale_a = (a_amax / 448.0).to(torch.float32).reshape(1)
            scale_b = (b_amax / 448.0).to(torch.float32).reshape(1)
            # torch._scaled_mm computes (scale_a * a) @ (scale_b * b_T).
            # b is laid out (N, K), b.t() gives (K, N) view as required.
            kernel = lambda: torch._scaled_mm(
                a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
            )

        elif dtype == "fp4":
            from flashinfer.gemm import mm_fp4
            from flashinfer import nvfp4_quantize
            # NVFP4 GEMM: K multiple of 32 (block size 16 + 128x4 scale layout)
            # and M/N alignment to 16 for GEMM tile.
            if k % 32 != 0 or n % 16 != 0 or m % 16 != 0 or m == 0:
                return -1.0
            a_fp16 = torch.randn(m, k, device=device, dtype=torch.float16)
            b_fp16 = torch.randn(n, k, device=device, dtype=torch.float16)
            # Global scale: (448 * 6) / amax (matches FlashInfer mm_fp4 example).
            a_global_sf = (448.0 * 6.0) / a_fp16.float().abs().nan_to_num().max()
            b_global_sf = (448.0 * 6.0) / b_fp16.float().abs().nan_to_num().max()
            a_q, a_sf = nvfp4_quantize(a_fp16, a_global_sf.reshape(1))
            b_q, b_sf = nvfp4_quantize(b_fp16, b_global_sf.reshape(1))
            alpha = (1.0 / (a_global_sf * b_global_sf)).reshape(1).to(torch.float32)
            # mm_fp4 expects b in (k, n) column-major layout; nvfp4_quantize returns
            # (N, K_packed), so .T gives the required column-major view.
            kernel = lambda: mm_fp4(
                a_q, b_q.T, a_sf, b_sf.T, alpha=alpha, out_dtype=torch.bfloat16
            )

        else:
            raise ValueError(f"Unknown dtype '{dtype}'")

        # Warmup
        for _ in range(5):
            kernel()
        torch.cuda.synchronize()

        # Profile
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            kernel()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return float(np.median(times))

    except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError, AssertionError):
        # Common in low-precision profiling: shape constraints, unsupported sizes.
        # ValueError covers FlashInfer K/N alignment errors; AssertionError covers
        # fp8/fp4 unsupported-backend paths.
        torch.cuda.empty_cache()
        return -1.0


def _profile_attention(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_kv_heads: int = None,
    num_runs: int = 100,
    dtype: str = "fp16",
) -> float:
    """Profile attention at the requested precision; return median time (ms).

    Supports MHA and GQA.

    Supported dtypes:
      "fp16" - torch.nn.functional.scaled_dot_product_attention with float16.
      "fp8"  - flashinfer.single_prefill_with_kv_cache with FP8 q/k/v + scale tensors.
               Single-batch only (FlashInfer single-prefill API).
      "fp4"  - Not supported by FlashInfer 0.6 attention path; returns -1.0.

    Returns -1.0 on unsupported configurations (e.g. fp8 with batch>1, fp4 always)
    so the row is filtered before training.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for profiling")
    if num_kv_heads is None:
        num_kv_heads = num_heads

    device = torch.device("cuda")
    try:
        if dtype == "fp16":
            q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            if num_kv_heads != num_heads:
                assert num_heads % num_kv_heads == 0, (
                    f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                )
                k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
                v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            kernel = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)

        elif dtype == "fp8":
            from flashinfer import single_prefill_with_kv_cache
            # single_prefill_with_kv_cache is single-batch (no batch dim) with NHD layout.
            if batch != 1:
                return -1.0
            q_fp16 = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
            k_fp16 = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
            v_fp16 = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
            scale_q = (q_fp16.abs().amax() / 448.0).clamp(min=1e-4).reshape(1)
            scale_k = (k_fp16.abs().amax() / 448.0).clamp(min=1e-4).reshape(1)
            scale_v = (v_fp16.abs().amax() / 448.0).clamp(min=1e-4).reshape(1)
            q = (q_fp16 / scale_q).to(torch.float8_e4m3fn)
            k = (k_fp16 / scale_k).to(torch.float8_e4m3fn)
            v = (v_fp16 / scale_v).to(torch.float8_e4m3fn)
            kernel = lambda: single_prefill_with_kv_cache(
                q, k, v,
                scale_q=scale_q, scale_k=scale_k, scale_v=scale_v,
                o_dtype=torch.bfloat16, kv_layout="NHD",
            )

        elif dtype == "fp4":
            # FlashInfer 0.6 has no FP4 attention path; record as N/A.
            return -1.0

        else:
            raise ValueError(f"Unknown dtype '{dtype}'")

        # Warmup
        for _ in range(10):
            kernel()
        torch.cuda.synchronize()

        # Profile
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            kernel()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return float(np.median(times))

    except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError, AssertionError):
        # AssertionError covers FlashInfer's "fa2 backend doesn't support fp8"
        # path (single_prefill on sm_120 falls back to fa2, which rejects FP8).
        torch.cuda.empty_cache()
        return -1.0


def _profile_math(batch: int, hidden: int, num_runs: int = 100) -> float:
    """Profile LayerNorm (representative MATH operator) and return median time in ms."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for profiling")

    device = torch.device("cuda")
    x = torch.randn(batch, hidden, device=device, dtype=torch.float16)
    layer_norm = nn.LayerNorm(hidden).to(device).half()

    # Warmup
    for _ in range(10):
        layer_norm(x)
    torch.cuda.synchronize()

    # Profile
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        layer_norm(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return float(np.median(times))


def _profile_rmsnorm(seq: int, dim: int, num_runs: int = 100) -> float:
    """Profile RMSNorm operator and return median time in ms.

    RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
    Commonly used in Llama, Mistral, Mixtral architectures.

    Args:
        seq: Sequence length
        dim: Hidden dimension
        num_runs: Number of profiling runs

    Returns:
        Median time in milliseconds
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for profiling")

    device = torch.device("cuda")
    x = torch.randn(seq, dim, device=device, dtype=torch.float32)

    # RMSNorm implementation (no bias)
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            # Compute RMS
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            # Normalize and scale
            return x / rms * self.weight

    rmsnorm = RMSNorm(dim).to(device).float()

    # Warmup
    for _ in range(10):
        rmsnorm(x)
    torch.cuda.synchronize()

    # Profile
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        rmsnorm(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return float(np.median(times))


def _profile_silu(seq: int, dim: int, num_runs: int = 100) -> float:
    """Profile SiLU (Swish) activation operator and return median time in ms.

    SiLU: y = x * sigmoid(x)
    Commonly used in Llama, GPT-J, Mistral architectures.

    Args:
        seq: Sequence length
        dim: Hidden dimension
        num_runs: Number of profiling runs

    Returns:
        Median time in milliseconds
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for profiling")

    device = torch.device("cuda")
    x = torch.randn(seq, dim, device=device, dtype=torch.float16)
    silu = nn.SiLU().to(device)

    # Warmup
    for _ in range(10):
        silu(x)
    torch.cuda.synchronize()

    # Profile
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        silu(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return float(np.median(times))


def _profile_gemm_grid(hw_info: HardwareInfo, grid: dict, num_runs: int, dtype: str = "fp16") -> list[dict]:
    """Profile GEMM operator across parameter grid at the requested precision.

    Saves ONLY clean measurement data (no roofline, no efficiency).
    Roofline features are computed on-the-fly during training.
    """
    results = []
    total = len(grid["M"]) * len(grid["N"]) * len(grid["K"])
    count = 0

    for m in grid["M"]:
        for n in grid["N"]:
            for k in grid["K"]:
                count += 1
                print(f"Profiling [{dtype}] {count}/{total}: M={m}, N={n}, K={k}")

                t_measured = _profile_gemm(m, n, k, num_runs, dtype=dtype)

                # Save only clean data: input features + measured time
                results.append({
                    "M": m,
                    "N": n,
                    "K": k,
                    "t_measured_ms": t_measured,
                })

    return results


def _profile_attn_grid(hw_info: HardwareInfo, grid: dict, num_runs: int, dtype: str = "fp16") -> list[dict]:
    """Profile attention operator across parameter grid with GQA support.

    Grid structure (from PROFILE_PLAN.md):
    - bs: batch size
    - seq: sequence length
    - nh: number of query heads
    - nkv: number of key/value heads (GQA)
    - hd: head dimension

    Saves ONLY clean measurement data (no roofline, no efficiency).
    Roofline features are computed on-the-fly during training.
    """
    results = []
    total = (len(grid["bs"]) * len(grid["seq"]) * len(grid["nh"]) *
             len(grid["nkv"]) * len(grid["hd"]))
    count = 0

    for bs in grid["bs"]:
        for seq in grid["seq"]:
            for nh in grid["nh"]:
                for nkv in grid["nkv"]:
                    for hd in grid["hd"]:
                        count += 1
                        print(f"Profiling [{dtype}] {count}/{total}: bs={bs}, seq={seq}, "
                              f"nh={nh}, nkv={nkv}, hd={hd}")

                        t_measured = _profile_attention(
                            batch=bs,
                            num_heads=nh,
                            seq_len=seq,
                            head_dim=hd,
                            num_kv_heads=nkv,
                            num_runs=num_runs,
                            dtype=dtype,
                        )

                        # Save only clean data: input features + measured time
                        results.append({
                            "bs": bs,
                            "seq": seq,
                            "nh": nh,
                            "nkv": nkv,
                            "hd": hd,
                            "t_measured_ms": t_measured,
                        })

    return results


def _profile_math_grid(hw_info: HardwareInfo, grid: dict, num_runs: int) -> list[dict]:
    """Profile MATH operator (LayerNorm) across parameter grid.

    Saves ONLY clean measurement data (no roofline, no efficiency).
    Roofline features are computed on-the-fly during training.
    """
    results = []
    total = len(grid["batch"]) * len(grid["hidden"])
    count = 0

    for batch in grid["batch"]:
        for hidden in grid["hidden"]:
            count += 1
            print(f"Profiling {count}/{total}: batch={batch}, hidden={hidden}")

            t_measured = _profile_math(batch, hidden, num_runs)

            # Save only clean data: input features + measured time
            results.append({
                "batch": batch,
                "hidden": hidden,
                "t_measured_ms": t_measured,
            })

    return results


def _profile_rmsnorm_grid(hw_info: HardwareInfo, grid: dict, num_runs: int) -> list[dict]:
    """Profile RMSNorm operator across parameter grid.

    Grid structure (from PROFILE_PLAN.md):
    - seq: sequence length [2, 131072], 128 samples
    - dim: hidden dimension [128, 16384], 128 samples

    Saves ONLY clean measurement data (no roofline, no efficiency).
    Roofline features are computed on-the-fly during training.
    """
    results = []
    total = len(grid["seq"]) * len(grid["dim"])
    count = 0

    for seq in grid["seq"]:
        for dim in grid["dim"]:
            count += 1
            print(f"Profiling {count}/{total}: seq={seq}, dim={dim}")

            t_measured = _profile_rmsnorm(seq, dim, num_runs)

            # Save only clean data: input features + measured time
            results.append({
                "seq": seq,
                "dim": dim,
                "t_measured_ms": t_measured,
            })

    return results


def _profile_silu_grid(hw_info: HardwareInfo, grid: dict, num_runs: int) -> list[dict]:
    """Profile SiLU operator across parameter grid.

    Grid structure (from PROFILE_PLAN.md):
    - seq: sequence length [2, 131072], 128 samples
    - dim: hidden dimension [768, 106496], 128 samples

    Saves ONLY clean measurement data (no roofline, no efficiency).
    Roofline features are computed on-the-fly during training.
    """
    results = []
    total = len(grid["seq"]) * len(grid["dim"])
    count = 0

    for seq in grid["seq"]:
        for dim in grid["dim"]:
            count += 1
            print(f"Profiling {count}/{total}: seq={seq}, dim={dim}")

            t_measured = _profile_silu(seq, dim, num_runs)

            # Save only clean data: input features + measured time
            results.append({
                "seq": seq,
                "dim": dim,
                "t_measured_ms": t_measured,
            })

    return results


def _bytes_per_element_for_dtype(dtype: str) -> float:
    """Return bytes per element for a profiling dtype string.

    FP16 = 2 bytes, FP8 = 1 byte, FP4 = 0.5 bytes (packed two-per-byte).
    """
    return {"fp16": 2.0, "fp8": 1.0, "fp4": 0.5}[dtype]


def _peak_tflops_for_dtype(hw_info: HardwareInfo, dtype: str) -> float:
    """Return matrix-unit peak TFLOP/s for the given dtype.

    Falls back to peak_tflops_mm when per-dtype peak is unset.
    """
    if dtype == "fp8":
        return (hw_info.peak_tflops_mm_fp8
                if hw_info.peak_tflops_mm_fp8 is not None else hw_info.peak_tflops_mm)
    if dtype == "fp4":
        return (hw_info.peak_tflops_mm_fp4
                if hw_info.peak_tflops_mm_fp4 is not None else hw_info.peak_tflops_mm)
    return hw_info.peak_tflops_mm


def _add_roofline_and_efficiency(
    df: "pd.DataFrame",
    hw_info: HardwareInfo,
    operator: str,
    dtype: str = "fp16",
) -> "pd.DataFrame":
    """Add roofline time and efficiency columns to profiling data.

    Enables data reuse: if roofline model changes, retrain without re-profiling.

    For FP16, uses roofline_estimate() from compute_cost_predictor (single source
    of truth). For FP8/FP4, computes the roofline directly here using the
    dtype-appropriate peak FLOP/s and bytes-per-element so efficiency lands in
    (0, 1] when measured times are at the FP8/FP4 throughput tier.

    Uses fake tensors (FakeTensorMode) for ZERO memory overhead - only tracks
    metadata (shape, dtype, device). roofline_estimate() only needs tensor
    metadata, not actual data.

    Args:
        df: Clean profiling data (input features + t_measured_ms only)
        hw_info: Hardware specifications
        operator: Operator type ("gemm", "attn", "rmsnorm", "silu", or "math")
        dtype: Profiling tensor precision ("fp16"/"fp8"/"fp4"). Affects compute
               peak (peak_tflops_mm_fp8/fp4) and per-element byte count.

    Returns:
        DataFrame with added columns: t_roofline_ms, efficiency
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    roofline_data = []
    total = len(df)

    print(f"Computing roofline for {total} configurations (dtype={dtype})...")

    # Create fake tensor mode - zero memory overhead!
    fake_mode = FakeTensorMode()

    # Pre-compute per-dtype constants
    dtype_bytes = _bytes_per_element_for_dtype(dtype)
    peak_flop_s_dtype = _peak_tflops_for_dtype(hw_info, dtype) * 1e12
    peak_bw_bs = hw_info.peak_memory_bandwidth_gbps * 1e9

    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{total} ({100 * (idx + 1) / total:.1f}%)")

        if operator == "gemm":
            m, n, k = int(row['M']), int(row['N']), int(row['K'])
            if dtype == "fp16":
                # Use fake tensors with the existing roofline_estimate path.
                with fake_mode:
                    a = torch.empty(m, k, dtype=torch.float16, device='cuda')
                    b = torch.empty(k, n, dtype=torch.float16, device='cuda')
                    out = torch.empty(m, n, dtype=torch.float16, device='cuda')
                result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)
                t_roofline_ms = result.t_roofline_ms
            else:
                # FP8/FP4: compute roofline directly with dtype-aware peak/bytes.
                flop_count = 2.0 * m * n * k
                bytes_transferred = dtype_bytes * (m * k + k * n + m * n)
                t_compute_ms = (flop_count / peak_flop_s_dtype) * 1000.0
                t_memory_ms = (bytes_transferred / peak_bw_bs) * 1000.0
                t_roofline_ms = max(t_compute_ms, t_memory_ms)

        elif operator == "attn":
            # Updated column names: bs, seq, nh, nkv, hd
            bs, seq, nh, nkv, hd = (
                int(row['bs']), int(row['seq']),
                int(row['nh']), int(row['nkv']),
                int(row['hd'])
            )
            if dtype == "fp16":
                with fake_mode:
                    q = torch.empty(bs, nh, seq, hd, dtype=torch.float16, device='cuda')
                    k = torch.empty(bs, nkv, seq, hd, dtype=torch.float16, device='cuda')
                    v = torch.empty(bs, nkv, seq, hd, dtype=torch.float16, device='cuda')
                result = roofline_estimate(
                    aten._scaled_dot_product_flash_attention,
                    (q, k, v), {}, q, hw_info, OperatorType.ATTN
                )
                t_roofline_ms = result.t_roofline_ms
            else:
                # SDPA FLOPs ≈ 4 * B * H * S^2 * D (2 GEMMs of B,H,S,D × B,H,D,S).
                flop_count = 4.0 * bs * nh * seq * seq * hd
                # Memory: read Q (B,H,S,D), K/V (B,Hkv,S,D) ×2, write O (B,H,S,D).
                bytes_transferred = dtype_bytes * (
                    bs * nh * seq * hd
                    + 2 * bs * nkv * seq * hd
                    + bs * nh * seq * hd
                )
                t_compute_ms = (flop_count / peak_flop_s_dtype) * 1000.0
                t_memory_ms = (bytes_transferred / peak_bw_bs) * 1000.0
                t_roofline_ms = max(t_compute_ms, t_memory_ms)

        elif operator == "rmsnorm":
            seq, dim = int(row['seq']), int(row['dim'])
            # RMSNorm: x^2, mean, sqrt, div, mul ≈ 6 ops per element
            # Memory-bound: read x, write y, read weight
            flop_count = 6 * seq * dim
            bytes_transferred = seq * dim * 2 * 3  # 3 tensors (input, output, weight), FP16

            # Roofline: max(compute_bound, memory_bound)
            peak_flop_s = hw_info.peak_tflops_math * 1e12  # TFLOP/s to FLOP/s
            t_compute_ms = (flop_count / peak_flop_s) * 1000  # s to ms

            peak_bw_bs = hw_info.peak_memory_bandwidth_gbps * 1e9  # GB/s to B/s
            t_memory_ms = (bytes_transferred / peak_bw_bs) * 1000  # s to ms

            t_roofline_ms = max(t_compute_ms, t_memory_ms)

        elif operator == "silu":
            seq, dim = int(row['seq']), int(row['dim'])
            # SiLU: sigmoid(x) + mul(x, sigmoid(x)) ≈ 8 ops per element
            # Memory-bound: read x, write y
            flop_count = 8 * seq * dim
            bytes_transferred = seq * dim * 2 * 2  # 2 tensors (input, output), FP16

            # Roofline: max(compute_bound, memory_bound)
            peak_flop_s = hw_info.peak_tflops_math * 1e12  # TFLOP/s to FLOP/s
            t_compute_ms = (flop_count / peak_flop_s) * 1000  # s to ms

            peak_bw_bs = hw_info.peak_memory_bandwidth_gbps * 1e9  # GB/s to B/s
            t_memory_ms = (bytes_transferred / peak_bw_bs) * 1000  # s to ms

            t_roofline_ms = max(t_compute_ms, t_memory_ms)

        elif operator == "math":
            batch, hidden = int(row['batch']), int(row['hidden'])
            # Roofline estimate (approximate as memory-bound operation)
            total_bytes = batch * hidden * 2 * 3  # read input, write output, params
            bw_gbps = hw_info.get_peak_memory_bandwidth_gbps()
            t_roofline_ms = (total_bytes / bw_gbps) / 1e6  # ns to ms

        else:
            raise ValueError(f"Unknown operator: {operator}")

        t_measured_ms = row['t_measured_ms']
        efficiency = t_roofline_ms / t_measured_ms if t_measured_ms > 0 else 0

        roofline_data.append({
            't_roofline_ms': t_roofline_ms,
            'efficiency': efficiency,
        })

    # Add roofline columns to original dataframe
    df_with_roofline = df.copy()
    roofline_df = pd.DataFrame(roofline_data)
    df_with_roofline['t_roofline_ms'] = roofline_df['t_roofline_ms'].values
    df_with_roofline['efficiency'] = roofline_df['efficiency'].values

    print(f"Roofline computation complete ({total} configs)")
    print(f"Efficiency: mean={df_with_roofline['efficiency'].mean():.3f}, "
          f"std={df_with_roofline['efficiency'].std():.3f}")

    return df_with_roofline


def _augment_gemm_data(df: "pd.DataFrame") -> "pd.DataFrame":
    """Augment GEMM data using transpose symmetry.

    For GEMM: C = A @ B where A is M×K, B is K×N, C is M×N
    Transpose: C.T = B.T @ A.T where B.T is N×K, A.T is K×M, C.T is N×M

    Both operations have the same arithmetic intensity and should have similar efficiency.
    This doubles the effective dataset size.

    Args:
        df: DataFrame with columns ["M", "N", "K", ...other metrics]

    Returns:
        Augmented DataFrame with original + transposed configs
    """
    # Create transposed versions: swap M ↔ N
    df_transposed = df.copy()
    df_transposed[["M", "N"]] = df[["N", "M"]].values

    # Concatenate original and transposed
    df_augmented = pd.concat([df, df_transposed], ignore_index=True)

    # Remove exact duplicates (happens when M == N)
    df_augmented = df_augmented.drop_duplicates(subset=["M", "N", "K"], keep="first")

    return df_augmented


# Operator shape configurations: maps CSV columns to tensor shapes
# To add new operator: just add entry here - no code changes needed!
OPERATOR_SHAPE_CONFIGS = {
    "gemm": {
        "inputs": [
            ["M", "K"],      # Input A: [M, K]
            ["K", "N"],      # Input B: [K, N]
        ],
        "output": ["M", "N"]  # Output: [M, N]
    },
    "attn": {
        "inputs": [
            ["bs", "nh", "seq", "hd"],   # q: [bs, nh, seq, hd]
            ["bs", "nkv", "seq", "hd"],  # k: [bs, nkv, seq, hd] (GQA support)
            ["bs", "nkv", "seq", "hd"],  # v: [bs, nkv, seq, hd] (GQA support)
        ],
        "output": ["bs", "nh", "seq", "hd"]  # out: [bs, nh, seq, hd]
    },
    "rmsnorm": {
        "inputs": [
            ["seq", "dim"],  # x: [seq, dim]
        ],
        "output": ["seq", "dim"]  # out: [seq, dim]
    },
    "silu": {
        "inputs": [
            ["seq", "dim"],  # x: [seq, dim]
        ],
        "output": ["seq", "dim"]  # out: [seq, dim]
    },
    "math": {
        "inputs": [
            ["batch", "hidden"],  # x: [B, H]
        ],
        "output": ["batch", "hidden"]  # out: [B, H]
    }
}


def _extract_shape_features(df: "pd.DataFrame", operator: str, base_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Extract shape-based features by concatenating input/output tensor shapes.

    GENERIC APPROACH (configuration-driven):
    - Uses OPERATOR_SHAPE_CONFIGS to map CSV columns → tensor shapes
    - Concatenates all input dimensions + output dimensions
    - Log-scales all dimensions (critical for MLP, neutral for XGBoost)
    - No operator-specific code - add new operators by extending config only!

    Examples:
      GEMM: mm(A: [M,K], B: [K,N]) -> [M,N]
            Features: [log(M), log(K), log(K), log(N), log(M), log(N)] (6 total)
      ATTN: sdpa(q/k/v: [B,H,S,D]) -> [B,H,S,D]
            Features: [log(B), log(H), log(S), log(D)] × 4 (16 total)
      MATH: layernorm(x: [B,H]) -> [B,H]
            Features: [log(B), log(H), log(B), log(H)] (4 total)

    Args:
        df: DataFrame with profiled data
        operator: Operator type (must be in OPERATOR_SHAPE_CONFIGS)
        base_cols: Base feature column names from CSV (e.g., ["M", "N", "K"])

    Returns:
        (X, feature_order): Feature matrix and ordered feature names

    Raises:
        ValueError: If operator not in OPERATOR_SHAPE_CONFIGS
    """
    if operator not in OPERATOR_SHAPE_CONFIGS:
        raise ValueError(
            f"Unknown operator '{operator}'. "
            f"Available: {list(OPERATOR_SHAPE_CONFIGS.keys())}"
        )

    config = OPERATOR_SHAPE_CONFIGS[operator]
    base_vals = df[base_cols].values

    # Create mapping: column name → values
    col_to_vals = {col: base_vals[:, i] for i, col in enumerate(base_cols)}

    features = []
    feature_names = []

    # Extract input tensor shapes (log-scaled for MLP compatibility)
    for input_idx, input_shape_cols in enumerate(config["inputs"]):
        for dim_idx, col_name in enumerate(input_shape_cols):
            dim_vals = col_to_vals[col_name]
            features.append(np.log(dim_vals + 1).reshape(-1, 1))
            feature_names.append(f"log_input{input_idx}_dim{dim_idx}")

    # Extract output tensor shape (log-scaled for MLP compatibility)
    for dim_idx, col_name in enumerate(config["output"]):
        dim_vals = col_to_vals[col_name]
        features.append(np.log(dim_vals + 1).reshape(-1, 1))
        feature_names.append(f"log_output_dim{dim_idx}")

    X = np.concatenate(features, axis=1)
    return X, feature_names


def _extract_roofline_features(
    df: "pd.DataFrame", hw_info: HardwareInfo, operator: str, dtype: str = "fp16"
) -> "pd.DataFrame":
    """Extract roofline envelope features for each configuration.

    Features include:
    - Constraint times (T_compute, T_memory)
    - Constraint ratios (T_memory / T_compute)
    - Dominant constraint (one-hot)

    For FP16, uses roofline_estimate() (FP16 fake tensors). For FP8/FP4, computes
    the constraints directly with dtype-aware peak FLOPs and bytes-per-element so
    features are consistent with the dtype-specific efficiency target.

    Uses fake tensors for zero memory overhead.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    features = []

    # Create fake tensor mode - zero memory overhead!
    fake_mode = FakeTensorMode()
    dtype_bytes = _bytes_per_element_for_dtype(dtype) if dtype != "fp16" else None
    peak_flop_s_dtype = _peak_tflops_for_dtype(hw_info, dtype) * 1e12
    peak_bw_bs = hw_info.peak_memory_bandwidth_gbps * 1e9

    def _direct_constraints(flop_count: float, bytes_transferred: float) -> dict:
        t_compute_ms = (flop_count / peak_flop_s_dtype) * 1000.0
        t_memory_ms = (bytes_transferred / peak_bw_bs) * 1000.0
        if t_compute_ms > 0:
            constraint_ratio = t_memory_ms / t_compute_ms
        else:
            constraint_ratio = float("inf") if t_memory_ms > 0 else 0.0
        is_compute_bound = 1.0 if t_compute_ms >= t_memory_ms else 0.0
        is_memory_bound = 1.0 if t_memory_ms > t_compute_ms else 0.0
        return dict(
            t_compute_ms=t_compute_ms,
            t_memory_ms=t_memory_ms,
            constraint_ratio=constraint_ratio,
            is_compute_bound=is_compute_bound,
            is_memory_bound=is_memory_bound,
        )

    for idx, row in df.iterrows():
        if operator == "gemm":
            m, n, k = int(row['M']), int(row['N']), int(row['K'])
            if dtype == "fp16":
                with fake_mode:
                    a = torch.empty(m, k, dtype=torch.float16, device='cuda')
                    b = torch.empty(k, n, dtype=torch.float16, device='cuda')
                    out = torch.empty(m, n, dtype=torch.float16, device='cuda')
                result = roofline_estimate(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)
            else:
                features.append(_direct_constraints(
                    flop_count=2.0 * m * n * k,
                    bytes_transferred=dtype_bytes * (m * k + k * n + m * n),
                ))
                continue

        elif operator == "attn":
            bs, seq, nh, nkv, hd = (
                int(row['bs']), int(row['seq']),
                int(row['nh']), int(row['nkv']),
                int(row['hd'])
            )
            if dtype == "fp16":
                with fake_mode:
                    q = torch.empty(bs, nh, seq, hd, dtype=torch.float16, device='cuda')
                    k = torch.empty(bs, nkv, seq, hd, dtype=torch.float16, device='cuda')
                    v = torch.empty(bs, nkv, seq, hd, dtype=torch.float16, device='cuda')
                result = roofline_estimate(
                    aten._scaled_dot_product_flash_attention,
                    (q, k, v), {}, q, hw_info, OperatorType.ATTN
                )
            else:
                features.append(_direct_constraints(
                    flop_count=4.0 * bs * nh * seq * seq * hd,
                    bytes_transferred=dtype_bytes * (
                        bs * nh * seq * hd
                        + 2 * bs * nkv * seq * hd
                        + bs * nh * seq * hd
                    ),
                ))
                continue

        else:  # math / rmsnorm / silu — memory-bound approximation
            # For math ops, we don't have a perfect roofline estimate, use memory-bound approximation
            features.append({
                't_compute_ms': 0.0,
                't_memory_ms': row['t_roofline_ms'],
                'constraint_ratio': 0.0,
                'is_compute_bound': 0.0,
                'is_memory_bound': 1.0,
            })
            continue

        # Extract constraint times
        t_compute_ms = 0.0
        t_memory_ms = 0.0
        for constraint in result.constraints:
            if constraint.work_type == "math":
                t_compute_ms = constraint.time_ms
            elif constraint.work_type == "memory":
                t_memory_ms = constraint.time_ms

        # Compute constraint ratio (avoid division by zero)
        if t_compute_ms > 0:
            constraint_ratio = t_memory_ms / t_compute_ms
        else:
            constraint_ratio = float('inf') if t_memory_ms > 0 else 0.0

        # Dominant constraint (one-hot)
        is_compute_bound = 1.0 if result.dominant_constraint[0] == "math" else 0.0
        is_memory_bound = 1.0 if result.dominant_constraint[0] == "memory" else 0.0

        features.append({
            't_compute_ms': t_compute_ms,
            't_memory_ms': t_memory_ms,
            'constraint_ratio': constraint_ratio,
            'is_compute_bound': is_compute_bound,
            'is_memory_bound': is_memory_bound,
        })

    # Convert to DataFrame
    roofline_df = pd.DataFrame(features)
    return roofline_df


def _build_training_features(
    df: "pd.DataFrame",
    hw_info: HardwareInfo,
    operator: str,
    base_cols: list[str],
    dtype: str = "fp16",
) -> tuple[np.ndarray, list[str]]:
    """Build complete feature matrix for training (shape + roofline features).

    Combines two feature types:
    1. Shape features: Log-scaled input/output tensor dimensions
    2. Roofline features: Analytical constraint envelope (compute/memory bounds)

    Used by both MLP and XGBoost training.
    """
    # Get shape features
    X_base, base_feature_names = _extract_shape_features(df, operator, base_cols)

    # Get roofline features (dtype-aware: uses FP8/FP4 peaks when relevant)
    roofline_df = _extract_roofline_features(df, hw_info, operator, dtype=dtype)

    # Log-scale roofline times (add 1e-10 to avoid log(0))
    roofline_features = np.column_stack([
        np.log(roofline_df['t_compute_ms'].values + 1e-10),
        np.log(roofline_df['t_memory_ms'].values + 1e-10),
        np.log(np.clip(roofline_df['constraint_ratio'].values, 1e-10, 1e10)),
        roofline_df['is_compute_bound'].values,
        roofline_df['is_memory_bound'].values,
    ])

    roofline_feature_names = [
        'log_t_compute',
        'log_t_memory',
        'log_constraint_ratio',
        'is_compute_bound',
        'is_memory_bound',
    ]

    # Concatenate all features
    X = np.concatenate([X_base, roofline_features], axis=1)
    feature_names = base_feature_names + roofline_feature_names

    return X, feature_names


def _build_mlp_model(input_dim: int) -> nn.Module:
    """Build improved MLP with batch normalization and deeper architecture."""
    return nn.Sequential(
        # Layer 1
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.1),

        # Layer 2
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.1),

        # Layer 3
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.1),

        # Output
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error loss function."""

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MAPE loss.

        MAPE = mean(|y_true - y_pred| / (|y_true| + epsilon)) * 100

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            MAPE loss value
        """
        return torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon))) * 100


def _train_and_validate_mlp(
    X: np.ndarray,
    y: np.ndarray,
    feature_order: list[str],
    epochs: int = 300,
    n_folds: int = 5,
) -> tuple[nn.Module, dict]:
    """Train MLP with k-fold cross-validation and return best model + metrics."""

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    best_model = None
    best_val_loss = float('inf')

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Training with {n_folds}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_t = torch.from_numpy(X_train).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
        X_val_t = torch.from_numpy(X_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().unsqueeze(1).to(device)

        # Build model and move to GPU
        input_dim = X.shape[1]
        model = _build_mlp_model(input_dim).to(device)

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, verbose=False
        )
        criterion = MAPELoss()

        # Early stopping
        patience = 30
        patience_counter = 0
        best_fold_val_loss = float('inf')
        best_fold_state = None

        for epoch in range(epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            loss_train = criterion(model(X_train_t), y_train_t)
            loss_train.backward()
            optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                loss_val = criterion(model(X_val_t), y_val_t)

            scheduler.step(loss_val)

            # Early stopping
            if loss_val < best_fold_val_loss:
                best_fold_val_loss = loss_val.item()
                best_fold_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Load best state for this fold
        model.load_state_dict(best_fold_state)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val_t).cpu().numpy().flatten()

        y_val_np = y_val

        # Compute metrics
        mask = (y_val_np > 0) & (y_pred > 0)
        if mask.sum() > 0:
            eff_mape = np.mean(np.abs((y_val_np[mask] - y_pred[mask]) / y_val_np[mask])) * 100
        else:
            eff_mape = float('inf')

        fold_metrics.append({
            'fold': fold + 1,
            'val_loss': best_fold_val_loss,
            'eff_mape': eff_mape,
        })

        print(f"  Validation loss: {best_fold_val_loss:.6f}")
        print(f"  Efficiency MAPE: {eff_mape:.2f}%")

        # Keep best model across all folds
        if best_fold_val_loss < best_val_loss:
            best_val_loss = best_fold_val_loss
            best_model = model

    # Compute average metrics
    avg_metrics = {
        'mean_val_loss': np.mean([m['val_loss'] for m in fold_metrics]),
        'std_val_loss': np.std([m['val_loss'] for m in fold_metrics]),
        'mean_eff_mape': np.mean([m['eff_mape'] for m in fold_metrics]),
        'std_eff_mape': np.std([m['eff_mape'] for m in fold_metrics]),
        'fold_metrics': fold_metrics,
    }

    print(f"\n--- Cross-Validation Summary ---")
    print(f"Mean val loss: {avg_metrics['mean_val_loss']:.6f} ± {avg_metrics['std_val_loss']:.6f}")
    print(f"Mean eff MAPE: {avg_metrics['mean_eff_mape']:.2f}% ± {avg_metrics['std_eff_mape']:.2f}%")

    return best_model, avg_metrics


def _train_and_validate_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    feature_order: list[str],
    n_folds: int = 5,
    xgb_params: dict = None,
) -> tuple:
    """Train XGBoost with k-fold cross-validation.

    Args:
        X: Feature matrix (N x D)
        y: Target efficiency values (N,)
        feature_order: Feature names
        n_folds: Number of CV folds
        xgb_params: XGBoost hyperparameters (optional)

    Returns:
        (best_model, cv_metrics)
    """

    # Default hyperparameters (tuned for efficiency regression)
    default_params = {
        "tree_method": "hist",          # Fast histogram-based algorithm
        "max_depth": 6,                 # Deeper trees for complex patterns
        "learning_rate": 0.05,          # Conservative learning rate
        "n_estimators": 500,            # Many trees, rely on early stopping
        "subsample": 0.8,               # Row sampling
        "colsample_bytree": 0.8,        # Column sampling
        "min_child_weight": 3,          # Prevent overfitting on small leaves
        "reg_alpha": 0.1,               # L1 regularization
        "reg_lambda": 1.0,              # L2 regularization
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_jobs": -1,                   # Use all CPU cores
        "early_stopping_rounds": 50,    # Early stopping in constructor
    }

    if xgb_params:
        default_params.update(xgb_params)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    best_model = None
    best_val_rmse = float('inf')

    print(f"\nTraining XGBoost with {n_folds}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train XGBoost
        model = xgb.XGBRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False  # Suppress per-iteration output
        )

        # Predict on validation set
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0.01, 1.0)  # Clip to valid efficiency range

        # Compute metrics
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        mask = (y_val > 0) & (y_pred > 0)
        if mask.sum() > 0:
            eff_mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
        else:
            eff_mape = float('inf')

        fold_metrics.append({
            'fold': fold + 1,
            'val_rmse': rmse,
            'eff_mape': eff_mape,
            'n_trees': model.best_iteration + 1 if hasattr(model, 'best_iteration') else default_params['n_estimators'],
        })

        print(f"  Val RMSE: {rmse:.6f}")
        print(f"  Eff MAPE: {eff_mape:.2f}%")
        print(f"  Trees used: {fold_metrics[-1]['n_trees']}")

        # Keep best model
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_model = model

    # Compute average metrics
    avg_metrics = {
        'mean_val_rmse': np.mean([m['val_rmse'] for m in fold_metrics]),
        'std_val_rmse': np.std([m['val_rmse'] for m in fold_metrics]),
        'mean_eff_mape': np.mean([m['eff_mape'] for m in fold_metrics]),
        'std_eff_mape': np.std([m['eff_mape'] for m in fold_metrics]),
        'mean_n_trees': np.mean([m['n_trees'] for m in fold_metrics]),
        'fold_metrics': fold_metrics,
    }

    print(f"\n--- Cross-Validation Summary ---")
    print(f"Mean val RMSE: {avg_metrics['mean_val_rmse']:.6f} ± {avg_metrics['std_val_rmse']:.6f}")
    print(f"Mean eff MAPE: {avg_metrics['mean_eff_mape']:.2f}% ± {avg_metrics['std_eff_mape']:.2f}%")
    print(f"Mean trees: {avg_metrics['mean_n_trees']:.0f}")

    return best_model, avg_metrics


def profile_operator(
    operator: str,
    output_dir: str,
    num_runs: int = 100,
    checkpoint_interval: int = 1000,
    dtype: str = "fp16",
) -> Path:
    """Profile operator and save clean data with checkpointing support.

    Args:
        operator: Operator type ("gemm", "attn", "rmsnorm", "silu", or "math")
        output_dir: Directory to save CSV data
        num_runs: Number of profiling runs per configuration
        checkpoint_interval: Save checkpoint every N configurations (default: 1000)
        dtype: Tensor precision ("fp16", "fp8", or "fp4"). FP8/FP4 only valid for
               GEMM and ATTN operators; rmsnorm/silu/math always use FP16.

    Returns:
        Path to saved CSV file

    Outputs:
        - {output_dir}/{operator}_{hw_name}_{dtype}_data.csv: Clean profiling data
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError("Install numpy and pandas for profiling")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for profiling")
    if dtype not in ("fp16", "fp8", "fp4"):
        raise ValueError(f"Unknown dtype '{dtype}'. Choose fp16/fp8/fp4")
    if dtype != "fp16" and operator in ("rmsnorm", "silu", "math"):
        raise ValueError(
            f"Operator '{operator}' is profiled only at fp16; got dtype={dtype}"
        )

    print(f"=" * 80)
    print(f"PROFILING MODE: {operator.upper()} (dtype={dtype})")
    print(f"=" * 80)

    # Auto-detect hardware
    hw_info, hw_name = get_hardware_info()
    print(f"\nDetected hardware: {hw_name}")
    print(f"  Peak TFLOP/s (MM):   {hw_info.peak_tflops_mm:.1f}")
    print(f"  Peak TFLOP/s (Math): {hw_info.peak_tflops_math:.1f}")
    print(f"  Peak Bandwidth:      {hw_info.peak_memory_bandwidth_gbps:.1f} GB/s")

    # Get profiling grid
    grid = COMPUTE_GRIDS[operator]
    print(f"\nProfiling {operator} operator...")

    # Calculate total configurations
    if operator == "gemm":
        total_configs = len(grid["M"]) * len(grid["N"]) * len(grid["K"])
    elif operator == "attn":
        total_configs = len(grid["bs"]) * len(grid["seq"]) * len(grid["nh"]) * len(grid["nkv"]) * len(grid["hd"])
    elif operator in ["rmsnorm", "silu"]:
        total_configs = len(grid["seq"]) * len(grid["dim"])
    elif operator == "math":
        total_configs = len(grid["batch"]) * len(grid["hidden"])
    else:
        total_configs = 0

    print(f"  Total configurations: {total_configs:,}")
    print(f"  Checkpoint interval: every {checkpoint_interval} configs")

    # Profile operator grid (clean data only)
    if operator == "gemm":
        results = _profile_gemm_grid(hw_info, grid, num_runs, dtype=dtype)
    elif operator == "attn":
        results = _profile_attn_grid(hw_info, grid, num_runs, dtype=dtype)
    elif operator == "rmsnorm":
        results = _profile_rmsnorm_grid(hw_info, grid, num_runs)
    elif operator == "silu":
        results = _profile_silu_grid(hw_info, grid, num_runs)
    elif operator == "math":
        results = _profile_math_grid(hw_info, grid, num_runs)
    else:
        raise ValueError(f"Unknown operator: {operator}")

    # Save clean data (model-agnostic). Dtype is in the filename so each
    # (op, dtype) combination gets its own CSV.
    df = pd.DataFrame(results)
    csv_path = Path(output_dir) / f"{operator}_{hw_name}_{dtype}_data.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"\n" + "=" * 80)
    print(f"✅ PROFILING COMPLETE")
    print(f"=" * 80)
    print(f"Profiled configurations: {len(results)}")
    print(f"Saved clean data to: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTo train models, run:")
    print(f"  python -m syssim.compute.compute_cost_profiler \\")
    print(f"    --operator {operator} \\")
    print(f"    --data-path {csv_path} \\")
    print(f"    --output {output_dir}/{operator}_{hw_name}_mlp.pth \\")
    print(f"    --backend mlp")
    print(f"\n  python -m syssim.compute.compute_cost_profiler \\")
    print(f"    --operator {operator} \\")
    print(f"    --data-path {csv_path} \\")
    print(f"    --output {output_dir}/{operator}_{hw_name}_xgb.pth \\")
    print(f"    --backend xgboost")
    print(f"=" * 80)

    return csv_path


def train_efficiency_model(
    operator: str,
    csv_path: Path,
    output_path: str,
    backend: str = "xgboost",
    epochs: int = 300,
    dtype: str = "fp16",
) -> tuple[dict, float, float]:
    """Train efficiency model from existing clean data. No profiling.

    Args:
        operator: Operator type ("gemm", "attn", "rmsnorm", "silu", or "math")
        csv_path: Path to existing CSV data file
        output_path: Path to save trained model (.pth)
        backend: "mlp" or "xgboost"
        epochs: Number of training epochs (MLP only)
        dtype: Tensor precision the data was collected at ("fp16"/"fp8"/"fp4").
               Used to pick the correct peak FLOP/s when computing roofline.

    Returns:
        (cv_metrics, eff_mape, time_mape)

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError("Install numpy, pandas, sklearn, and xgboost for training")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Run profiling first:\n"
            f"  python -m syssim.compute.compute_cost_profiler \\\n"
            f"    --operator {operator} \\\n"
            f"    --output {output_path}"
        )

    print(f"=" * 80)
    print(f"TRAINING MODE: {operator.upper()} (Backend: {backend.upper()})")
    print(f"=" * 80)

    # Auto-detect hardware (must match profiling hardware!)
    hw_info, hw_name = get_hardware_info()
    print(f"\nDetected hardware: {hw_name}")
    print(f"  Peak TFLOP/s (MM):   {hw_info.peak_tflops_mm:.1f}")
    print(f"  Peak TFLOP/s (Math): {hw_info.peak_tflops_math:.1f}")
    print(f"  Peak Bandwidth:      {hw_info.peak_memory_bandwidth_gbps:.1f} GB/s")

    # Load clean data
    print(f"\nLoading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} configurations")
    print(f"Columns: {list(df.columns)}")

    # Filter out invalid measurements (sentinel -1 for OOM/timeout failures)
    n_before = len(df)
    df = df[df['t_measured_ms'] > 0].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Filtered {n_dropped} invalid rows (t_measured_ms <= 0), {len(df)} remaining")

    # Auto-detect and compute roofline features on-the-fly
    if 't_roofline_ms' not in df.columns or 'efficiency' not in df.columns:
        print(f"\nComputing roofline features on-the-fly (dtype={dtype})...")
        df = _add_roofline_and_efficiency(df, hw_info, operator, dtype=dtype)
        print(f"Added columns: t_roofline_ms, efficiency")
        print(f"Efficiency: mean={df['efficiency'].mean():.3f}, "
              f"std={df['efficiency'].std():.3f}")
    else:
        print("\nUsing existing roofline features from CSV")

    # Get base feature columns
    if operator == "gemm":
        base_feature_cols = ["M", "N", "K"]
    elif operator == "attn":
        base_feature_cols = ["bs", "seq", "nh", "nkv", "hd"]
    elif operator in ["rmsnorm", "silu"]:
        base_feature_cols = ["seq", "dim"]
    elif operator == "math":
        base_feature_cols = ["batch", "hidden"]
    else:
        raise ValueError(f"Unknown operator: {operator}")

    # Data augmentation (GEMM only)
    if operator == "gemm":
        print("\nApplying data augmentation (transpose symmetry)...")
        df_train = _augment_gemm_data(df)
        print(f"  Augmented: {len(df)} → {len(df_train)} samples")
    else:
        df_train = df.copy()

    # Extract enhanced features with roofline envelope (dtype-aware)
    print(f"\nExtracting enhanced features (base + roofline envelope)...")
    X, feature_order = _build_training_features(
        df_train, hw_info, operator, base_feature_cols, dtype=dtype
    )
    y = df_train["efficiency"].values

    print(f"  Features: {len(feature_order)} total")
    print(f"  Samples:  {len(X)}")
    print(f"  Feature names: {feature_order}")

    # Train model based on backend
    if backend == "mlp":
        print(f"\nTraining MLP with k-fold cross valiation...")
        model, cv_metrics = _train_and_validate_mlp(X, y, feature_order, epochs, n_folds=5)
        model_type = "mlp"
    elif backend == "xgboost":
        print(f"\nTraining XGBoost with k-fold cross validation...")
        model, cv_metrics = _train_and_validate_xgboost(X, y, feature_order, n_folds=5)
        model_type = "xgboost"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Final evaluation on original (non-augmented) data
    print(f"\n--- Final Evaluation (Original Data) ---")
    X_eval, _ = _build_training_features(
        df, hw_info, operator, base_feature_cols, dtype=dtype
    )

    if backend == "mlp":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            X_eval_t = torch.from_numpy(X_eval).float().to(device)
            eta_pred = model(X_eval_t).cpu().numpy().flatten()
    else:  # xgboost
        eta_pred = model.predict(X_eval)
        eta_pred = np.clip(eta_pred, 0.01, 1.0)

    eta_true = df["efficiency"].values

    # Compute final metrics
    mask = (eta_true > 0) & (eta_pred > 0)
    eff_mape = np.mean(np.abs((eta_true[mask] - eta_pred[mask]) / eta_true[mask])) * 100

    # Compute time MAPE
    t_predicted = df["t_roofline_ms"].values / np.clip(eta_pred, 1e-10, 1.0)
    t_true = df["t_measured_ms"].values
    time_mape = np.mean(np.abs((t_true - t_predicted) / t_true)) * 100

    print(f"\nFinal metrics (on original {len(df)} configs):")
    print(f"  Efficiency MAPE: {eff_mape:.2f}%")
    print(f"  Time MAPE:       {time_mape:.2f}%")
    print(f"  Mean predicted:  {eta_pred.mean():.3f}")
    print(f"  Mean true:       {eta_true.mean():.3f}")

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if backend == "mlp":
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "hidden_dims": [256, 128, 64],
            "feature_order": feature_order,
            "operator": operator,
            "model_type": "mlp",
            "cv_metrics": cv_metrics,
            "final_metrics": {
                "eff_mape": eff_mape,
                "time_mape": time_mape,
            },
        }, output_path)
    else:  # xgboost
        model_bytes = pickle.dumps(model)
        feature_importance = dict(sorted(
            zip(feature_order, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        ))

        print(f"\nTop 5 Most Important Features:")
        for feat, imp in list(feature_importance.items())[:5]:
            print(f"  {feat}: {imp:.4f}")

        torch.save({
            "model_state_dict": model_bytes,  # Pickled XGBoost model
            "input_dim": X.shape[1],
            "feature_order": feature_order,
            "operator": operator,
            "model_type": "xgboost",
            "cv_metrics": cv_metrics,
            "final_metrics": {
                "eff_mape": eff_mape,
                "time_mape": time_mape,
            },
            "feature_importance": feature_importance,
        }, output_path)

    print(f"\n✅ TRAINING COMPLETE")
    print(f"Saved {backend.upper()} model to: {output_path}")
    print(f"=" * 80)

    return cv_metrics, eff_mape, time_mape


# CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Profile operators and train efficiency models (MLP or XGBoost)"
    )
    parser.add_argument(
        "--operator",
        required=True,
        choices=["gemm", "attn", "rmsnorm", "silu", "math"],
        help="Operator type to profile and train"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for trained model (.pth file). CSV data will be saved as {name}_data.csv"
    )
    parser.add_argument(
        "--backend",
        default="mlp",
        choices=["mlp", "xgboost"],
        help="Training backend: MLP (neural network) or XGBoost (gradient boosted trees)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (MLP only, default: 300)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of profiling runs per configuration (default: 100)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to existing CSV data file. If None, will profile and save data first."
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N configurations during profiling (default: 1000)"
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "fp8", "fp4"],
        help="Tensor precision: fp16 via PyTorch, fp8 via torch._scaled_mm, "
             "fp4 via FlashInfer mm_fp4 (NVFP4)."
    )

    args = parser.parse_args()

    if args.data_path is None:
        # PROFILING MODE: No data provided, must profile first
        print("\n" + "=" * 80)
        print("MODE: PROFILING (no --data-path provided)")
        print("=" * 80)

        output_dir = str(Path(args.output).parent)

        csv_path = profile_operator(
            operator=args.operator,
            output_dir=output_dir,
            num_runs=args.num_runs,
            checkpoint_interval=args.checkpoint_interval,
            dtype=args.dtype,
        )

    else:
        # TRAINING MODE: Data path provided, load and train
        print("\n" + "=" * 80)
        print("MODE: TRAINING (using --data-path)")
        print("=" * 80)

        csv_path = Path(args.data_path)

        # Validate output path
        _, hw_name = get_hardware_info()
        output_path = Path(args.output)
        backend_suffix = "xgb" if args.backend == "xgboost" else "mlp"
        expected_filename = f"{args.operator}_{hw_name}_{args.dtype}_{backend_suffix}.pth"

        if output_path.name != expected_filename:
            print(f"\nWARNING: Output filename doesn't match expected pattern!")
            print(f"  Detected hardware: {hw_name}")
            print(f"  Provided: {output_path.name}")
            print(f"  Expected: {expected_filename}")
            print(f"  (Will continue anyway...)\n")

        train_efficiency_model(
            operator=args.operator,
            csv_path=csv_path,
            output_path=args.output,
            backend=args.backend,
            epochs=args.epochs,
            dtype=args.dtype,
        )
