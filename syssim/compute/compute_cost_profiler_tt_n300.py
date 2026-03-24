"""Tenstorrent Wormhole N300 operator profiling via ttnn.

Collects operator execution times on a Tenstorrent Wormhole N300 device
using the ttnn API and saves CSV data compatible with SysSim's efficiency
model training pipeline.

Self-contained: only requires torch, ttnn, numpy, pandas on the target device.

Usage (on the N300 device):
  python compute_cost_profiler_tt_n300.py --operator gemm --output ./profiling_data/
  python compute_cost_profiler_tt_n300.py --operator attn --output ./profiling_data/
  python compute_cost_profiler_tt_n300.py --operator rmsnorm --output ./profiling_data/
  python compute_cost_profiler_tt_n300.py --operator silu --output ./profiling_data/
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
# Unit Conversion Constants (mirrors compute_cost_predictor.py)
# ==============================================================================

TFLOPS_TO_FLOPS = 1e12
GBPS_TO_BPS = 1e9
_PYTORCH_MIN_ALLOCATE = 512
LARGE_GEMM_THRESHOLD = 512

# ==============================================================================
# Hardware Info — Tenstorrent Wormhole N300 (single chip, device_id=0)
#
# N300 = 2x Wormhole b0 chips; profiling targets one chip.
# Per-chip specs (64 usable Tensix cores @ 1 GHz):
#   BF16 matrix engine (FPU): 74 TFLOP/s (advertised, accounts for data xfer)
#   FP32 vector engine (SFPU): 32 FLOPs/cycle * 64 cores * 1 GHz ≈ 2 TFLOP/s
#   DRAM: 12 GB GDDR6, 288 GB/s
# Sources: docs.tenstorrent.com, tt-metal/tech_reports/matrix_engine
# ==============================================================================

PLATFORM_HW_NAME = "tenstorrent_wh_n300"
PEAK_TFLOPS_MM = 74.0
PEAK_TFLOPS_MM_CONSERVATIVE = 74.0
PEAK_TFLOPS_MATH = 2.0
PEAK_MEMORY_BANDWIDTH_GBPS = 288.0

# ==============================================================================
# ttnn Device Singleton
# ==============================================================================

_TT_DEVICE = None


def _get_device():
    global _TT_DEVICE
    if _TT_DEVICE is None:
        import ttnn
        _TT_DEVICE = ttnn.open_device(device_id=0)
    return _TT_DEVICE


def _close_device():
    global _TT_DEVICE
    if _TT_DEVICE is not None:
        import ttnn
        ttnn.close_device(_TT_DEVICE)
        _TT_DEVICE = None


# ==============================================================================
# Sampling Utilities
# ==============================================================================

def _generate_proportional_samples(
    start: int, end: int, total_samples: int = 64, seed: int = 42,
) -> list[int]:
    """Generate samples with power-of-two anchors and proportional fill."""
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
# Roofline Computation
# ==============================================================================

def _aligned_tensor_bytes(numel: int, dtype_bytes: int = 2) -> int:
    raw = numel * dtype_bytes
    return math.ceil(raw / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE


def _roofline_gemm(m: int, n: int, k: int, dtype_bytes: int = 2) -> float:
    flops = 2 * m * n * k
    is_large = (m >= LARGE_GEMM_THRESHOLD
                and n >= LARGE_GEMM_THRESHOLD
                and k >= LARGE_GEMM_THRESHOLD)
    peak = PEAK_TFLOPS_MM if is_large else (PEAK_TFLOPS_MM_CONSERVATIVE or PEAK_TFLOPS_MM)
    compute_ns = (flops / (peak * TFLOPS_TO_FLOPS)) * 1e9

    bytes_a = _aligned_tensor_bytes(m * k, dtype_bytes)
    bytes_b = _aligned_tensor_bytes(k * n, dtype_bytes)
    bytes_c = _aligned_tensor_bytes(m * n, dtype_bytes)
    transfer_ns = (bytes_a + bytes_b + bytes_c) / PEAK_MEMORY_BANDWIDTH_GBPS

    return max(compute_ns, transfer_ns) / 1e6


def _roofline_attn(
    bs: int, nh: int, seq: int, hd: int, nkv: int,
    dtype_bytes: int = 2,
) -> float:
    flops = 4 * bs * nh * seq * seq * hd
    is_large = (bs * nh * seq >= 4096 and seq >= LARGE_GEMM_THRESHOLD)
    peak = PEAK_TFLOPS_MM if is_large else (PEAK_TFLOPS_MM_CONSERVATIVE or PEAK_TFLOPS_MM)
    compute_ns = (flops / (peak * TFLOPS_TO_FLOPS)) * 1e9

    bytes_q   = _aligned_tensor_bytes(bs * nh  * seq * hd, dtype_bytes)
    bytes_k   = _aligned_tensor_bytes(bs * nkv * seq * hd, dtype_bytes)
    bytes_v   = _aligned_tensor_bytes(bs * nkv * seq * hd, dtype_bytes)
    bytes_out = _aligned_tensor_bytes(bs * nh  * seq * hd, dtype_bytes)
    transfer_ns = (bytes_q + bytes_k + bytes_v + bytes_out) / PEAK_MEMORY_BANDWIDTH_GBPS

    return max(compute_ns, transfer_ns) / 1e6


def _roofline_rmsnorm(seq: int, dim: int) -> float:
    flops = 6 * seq * dim
    bytes_transferred = seq * dim * 2 * 3

    t_compute_ms = (flops / (PEAK_TFLOPS_MATH * TFLOPS_TO_FLOPS)) * 1000
    t_memory_ms = (bytes_transferred / (PEAK_MEMORY_BANDWIDTH_GBPS * GBPS_TO_BPS)) * 1000

    return max(t_compute_ms, t_memory_ms)


def _roofline_silu(seq: int, dim: int) -> float:
    flops = 8 * seq * dim
    bytes_transferred = seq * dim * 2 * 2

    t_compute_ms = (flops / (PEAK_TFLOPS_MATH * TFLOPS_TO_FLOPS)) * 1000
    t_memory_ms = (bytes_transferred / (PEAK_MEMORY_BANDWIDTH_GBPS * GBPS_TO_BPS)) * 1000

    return max(t_compute_ms, t_memory_ms)


def _add_roofline_and_efficiency(
    df: pd.DataFrame, operator: str,
) -> pd.DataFrame:
    roofline_data: list[dict] = []

    for _, row in df.iterrows():
        if operator == "gemm":
            t_roofline_ms = _roofline_gemm(
                int(row["M"]), int(row["N"]), int(row["K"]),
            )
        elif operator == "attn":
            t_roofline_ms = _roofline_attn(
                int(row["bs"]), int(row["nh"]), int(row["seq"]),
                int(row["hd"]), int(row["nkv"]),
            )
        elif operator == "rmsnorm":
            t_roofline_ms = _roofline_rmsnorm(int(row["seq"]), int(row["dim"]))
        elif operator == "silu":
            t_roofline_ms = _roofline_silu(int(row["seq"]), int(row["dim"]))
        else:
            raise ValueError(f"Unknown operator: {operator}")

        t_measured_ms = row["t_measured_ms"]
        efficiency = t_roofline_ms / t_measured_ms if t_measured_ms > 0 else 0

        roofline_data.append({
            "t_roofline_ms": t_roofline_ms,
            "efficiency": efficiency,
        })

    df_with_roofline = df.copy()
    roofline_df = pd.DataFrame(roofline_data)
    df_with_roofline["t_roofline_ms"] = roofline_df["t_roofline_ms"].values
    df_with_roofline["efficiency"] = roofline_df["efficiency"].values

    print(f"Roofline computation complete ({len(df)} configs)")
    print(f"Efficiency: mean={df_with_roofline['efficiency'].mean():.3f}, "
          f"std={df_with_roofline['efficiency'].std():.3f}")

    return df_with_roofline


# ==============================================================================
# Parameter Grids
#
# Upper bounds reduced vs CUDA version to stay within N300's 12 GB DRAM.
# 65536x65536 alloc triggers kernel OOM kill, so GEMM M capped at 32768.
# ==============================================================================

def _build_grids() -> dict[str, dict]:
    seed = 42
    return {
        "gemm": {
            "M": _generate_proportional_samples(2, 32768, 64, seed),
            "N": _generate_proportional_samples(256, 32768, 64, seed + 1),
            "K": _generate_proportional_samples(256, 16384, 64, seed + 2),
        },
        "attn": {
            "bs": _generate_power_of_two_range(1, 16),
            "seq": _generate_proportional_samples(1, 32768, 64, seed),
            "nh": _generate_power_of_two_range(2, 128),
            "nkv": _generate_power_of_two_range(1, 8),
            "hd": [64, 128],
        },
        "rmsnorm": {
            "seq": _generate_proportional_samples(2, 32768, 128, seed),
            "dim": _generate_proportional_samples(128, 16384, 128, seed + 1),
        },
        "silu": {
            "seq": _generate_proportional_samples(2, 32768, 128, seed),
            "dim": _generate_proportional_samples(768, 32768, 128, seed + 1),
        },
    }


COMPUTE_GRIDS = _build_grids()


# ==============================================================================
# Single-config Profiling Functions (ttnn backend)
#
# Contract:
#   1. Allocate tensors on TT device via ttnn.from_torch
#   2. Warmup iterations (let JIT/dispatch settle)
#   3. ttnn.synchronize_device to flush warmup
#   4. Timed loop: op → synchronize → record wall time
#   5. Deallocate device tensors
#   6. Return median time in ms, or -1.0 on failure
# ==============================================================================

def _profile_gemm(m: int, n: int, k: int, num_runs: int = 100) -> float:
    """Profile GEMM: C[m,n] = A[m,k] @ B[k,n], BF16."""
    try:
        import torch
        import ttnn

        device = _get_device()

        a = ttnn.from_torch(
            torch.randn(m, k, dtype=torch.float32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        b = ttnn.from_torch(
            torch.randn(k, n, dtype=torch.float32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

        for _ in range(5):
            c = ttnn.matmul(a, b)
            ttnn.deallocate(c)
        ttnn.synchronize_device(device)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            c = ttnn.matmul(a, b)
            ttnn.synchronize_device(device)
            end = time.perf_counter()
            ttnn.deallocate(c)
            times.append((end - start) * 1000)

        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return float(np.median(times))

    except (RuntimeError, MemoryError, Exception) as e:
        if "Out of Memory" in str(e) or "OOM" in str(e):
            return -1.0
        raise


def _profile_attention(
    batch: int, num_heads: int, seq_len: int, head_dim: int,
    num_kv_heads: int | None = None, num_runs: int = 100,
) -> float:
    """Profile scaled dot-product attention (MHA/GQA) via ttnn.transformer.scaled_dot_product_attention.

    Falls back to manual Q@K^T softmax @ V if the native op fails for the given shape.
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads

    try:
        import torch
        import ttnn

        device = _get_device()

        q = ttnn.from_torch(
            torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        k = ttnn.from_torch(
            torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        v = ttnn.from_torch(
            torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

        use_native_sdpa = True
        try:
            out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)
        except Exception:
            use_native_sdpa = False

        def _run_attn():
            if use_native_sdpa:
                return ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
            scale = head_dim ** -0.5
            if num_kv_heads != num_heads:
                k_expanded = ttnn.repeat_interleave(k, num_heads // num_kv_heads, dim=1)
                v_expanded = ttnn.repeat_interleave(v, num_heads // num_kv_heads, dim=1)
            else:
                k_expanded = k
                v_expanded = v
            scores = ttnn.matmul(q, ttnn.transpose(k_expanded, -2, -1))
            scores = ttnn.multiply(scores, scale)
            attn_weights = ttnn.softmax(scores, dim=-1)
            return ttnn.matmul(attn_weights, v_expanded)

        for _ in range(5):
            out = _run_attn()
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            out = _run_attn()
            ttnn.synchronize_device(device)
            end = time.perf_counter()
            ttnn.deallocate(out)
            times.append((end - start) * 1000)

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return float(np.median(times))

    except (RuntimeError, MemoryError, Exception) as e:
        if "Out of Memory" in str(e) or "OOM" in str(e):
            return -1.0
        raise


def _profile_rmsnorm(seq: int, dim: int, num_runs: int = 100) -> float:
    """Profile RMSNorm via ttnn.rms_norm. Input BF16 (ttnn uses BF16 natively)."""
    try:
        import torch
        import ttnn

        device = _get_device()

        x = ttnn.from_torch(
            torch.randn(1, 1, seq, dim, dtype=torch.float32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        w = ttnn.from_torch(
            torch.ones(dim, dtype=torch.float32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

        for _ in range(5):
            out = ttnn.rms_norm(x, weight=w)
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            out = ttnn.rms_norm(x, weight=w)
            ttnn.synchronize_device(device)
            end = time.perf_counter()
            ttnn.deallocate(out)
            times.append((end - start) * 1000)

        ttnn.deallocate(x)
        ttnn.deallocate(w)
        return float(np.median(times))

    except (RuntimeError, MemoryError, Exception) as e:
        if "Out of Memory" in str(e) or "OOM" in str(e):
            return -1.0
        raise


def _profile_silu(seq: int, dim: int, num_runs: int = 100) -> float:
    """Profile SiLU via ttnn.silu. BF16."""
    try:
        import torch
        import ttnn

        device = _get_device()

        x = ttnn.from_torch(
            torch.randn(1, 1, seq, dim, dtype=torch.float32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

        for _ in range(5):
            out = ttnn.silu(x)
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            out = ttnn.silu(x)
            ttnn.synchronize_device(device)
            end = time.perf_counter()
            ttnn.deallocate(out)
            times.append((end - start) * 1000)

        ttnn.deallocate(x)
        return float(np.median(times))

    except (RuntimeError, MemoryError, Exception) as e:
        if "Out of Memory" in str(e) or "OOM" in str(e):
            return -1.0
        raise


# ==============================================================================
# Grid Profiling
# ==============================================================================

_CSV_COLUMNS = {
    "gemm": ["M", "N", "K", "t_measured_ms"],
    "attn": ["bs", "seq", "nh", "nkv", "hd", "t_measured_ms"],
    "rmsnorm": ["seq", "dim", "t_measured_ms"],
    "silu": ["seq", "dim", "t_measured_ms"],
}


def _load_completed(csv_path: Path, key_columns: list[str]) -> set[tuple]:
    """Load already-profiled configs from existing CSV for resume support."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return {tuple(row[c] for c in key_columns) for _, row in df.iterrows()}
    except Exception:
        return set()


def _append_row(csv_path: Path, row: dict, columns: list[str]) -> None:
    """Append a single result row to CSV, writing header if file is new."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a") as f:
        if write_header:
            f.write(",".join(columns) + "\n")
        f.write(",".join(str(row[c]) for c in columns) + "\n")


def _profile_gemm_grid(grid: dict, num_runs: int, csv_path: Path) -> int:
    columns = _CSV_COLUMNS["gemm"]
    key_cols = ["M", "N", "K"]
    done = _load_completed(csv_path, key_cols)
    total = len(grid["M"]) * len(grid["N"]) * len(grid["K"])
    count, skipped = 0, 0
    for m in grid["M"]:
        for n in grid["N"]:
            for k in grid["K"]:
                count += 1
                if (m, n, k) in done:
                    skipped += 1
                    continue
                if (count - skipped) % 100 == 0 or count == 1:
                    print(f"  [{count}/{total}] M={m}, N={n}, K={k}  (skipped {skipped})")
                try:
                    t = _profile_gemm(m, n, k, num_runs)
                except (RuntimeError, MemoryError):
                    t = -1.0
                row = {"M": m, "N": n, "K": k, "t_measured_ms": t}
                _append_row(csv_path, row, columns)
    return total


def _profile_attn_grid(grid: dict, num_runs: int, csv_path: Path) -> int:
    columns = _CSV_COLUMNS["attn"]
    key_cols = ["bs", "seq", "nh", "nkv", "hd"]
    done = _load_completed(csv_path, key_cols)
    total = (len(grid["bs"]) * len(grid["seq"]) * len(grid["nh"])
             * len(grid["nkv"]) * len(grid["hd"]))
    count, skipped = 0, 0
    for bs in grid["bs"]:
        for seq in grid["seq"]:
            for nh in grid["nh"]:
                for nkv in grid["nkv"]:
                    for hd in grid["hd"]:
                        count += 1
                        if (bs, seq, nh, nkv, hd) in done:
                            skipped += 1
                            continue
                        if (count - skipped) % 100 == 0 or count == 1:
                            print(f"  [{count}/{total}] bs={bs}, seq={seq}, "
                                  f"nh={nh}, nkv={nkv}, hd={hd}  (skipped {skipped})")
                        try:
                            t = _profile_attention(
                                batch=bs, num_heads=nh, seq_len=seq,
                                head_dim=hd, num_kv_heads=nkv, num_runs=num_runs)
                        except (RuntimeError, MemoryError):
                            t = -1.0
                        row = {"bs": bs, "seq": seq, "nh": nh,
                               "nkv": nkv, "hd": hd, "t_measured_ms": t}
                        _append_row(csv_path, row, columns)
    return total


def _profile_rmsnorm_grid(grid: dict, num_runs: int, csv_path: Path) -> int:
    columns = _CSV_COLUMNS["rmsnorm"]
    key_cols = ["seq", "dim"]
    done = _load_completed(csv_path, key_cols)
    total = len(grid["seq"]) * len(grid["dim"])
    count, skipped = 0, 0
    for seq in grid["seq"]:
        for dim in grid["dim"]:
            count += 1
            if (seq, dim) in done:
                skipped += 1
                continue
            if (count - skipped) % 100 == 0 or count == 1:
                print(f"  [{count}/{total}] seq={seq}, dim={dim}  (skipped {skipped})")
            try:
                t = _profile_rmsnorm(seq, dim, num_runs)
            except (RuntimeError, MemoryError):
                t = -1.0
            row = {"seq": seq, "dim": dim, "t_measured_ms": t}
            _append_row(csv_path, row, columns)
    return total


def _profile_silu_grid(grid: dict, num_runs: int, csv_path: Path) -> int:
    columns = _CSV_COLUMNS["silu"]
    key_cols = ["seq", "dim"]
    done = _load_completed(csv_path, key_cols)
    total = len(grid["seq"]) * len(grid["dim"])
    count, skipped = 0, 0
    for seq in grid["seq"]:
        for dim in grid["dim"]:
            count += 1
            if (seq, dim) in done:
                skipped += 1
                continue
            if (count - skipped) % 100 == 0 or count == 1:
                print(f"  [{count}/{total}] seq={seq}, dim={dim}  (skipped {skipped})")
            try:
                t = _profile_silu(seq, dim, num_runs)
            except (RuntimeError, MemoryError):
                t = -1.0
            row = {"seq": seq, "dim": dim, "t_measured_ms": t}
            _append_row(csv_path, row, columns)
    return total


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
    if operator not in _GRID_DISPATCH:
        raise ValueError(f"Unknown operator: {operator}. Choose from {list(_GRID_DISPATCH)}")

    grid = COMPUTE_GRIDS[operator]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{operator}_{PLATFORM_HW_NAME}_data.csv"

    existing_count = 0
    if csv_path.exists():
        try:
            existing_count = len(pd.read_csv(csv_path))
        except Exception:
            pass
        if existing_count > 0:
            print(f"  Resuming: {existing_count} configs already in {csv_path}")

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
    print(f"  Output:    {csv_path}  (streaming)")
    print("=" * 70)

    _GRID_DISPATCH[operator](grid, num_runs, csv_path)

    _close_device()

    df = pd.read_csv(csv_path)

    print("\nComputing roofline estimates...")
    df = _add_roofline_and_efficiency(df, operator)
    df.to_csv(csv_path, index=False)

    n_valid = int((df["t_measured_ms"] > 0).sum())
    n_failed = len(df) - n_valid

    print()
    print("=" * 70)
    print(f"  Saved to: {csv_path}")
    print(f"  Total: {len(df)}  Valid: {n_valid}  Failed: {n_failed}")
    print(f"  Columns: {list(df.columns)}")
    print("=" * 70)

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Profile operators on Tenstorrent Wormhole N300"
    )
    parser.add_argument(
        "--operator", required=True,
        choices=["gemm", "attn", "rmsnorm", "silu"],
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for CSV data",
    )
    parser.add_argument(
        "--num-runs", type=int, default=100,
    )
    args = parser.parse_args()
    profile_operator(args.operator, args.output, args.num_runs)


if __name__ == "__main__":
    main()
