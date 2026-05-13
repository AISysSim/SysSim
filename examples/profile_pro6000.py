"""Driver script for Pro 6000 low-precision profiling.

Runs reduced-grid profiling for FP16/FP8/FP4 across operators.
Saves CSVs to data/profiling/{op}_pro6000_{dtype}_data.csv (same naming as
profile_operator produces) so the existing training pipeline picks them up.

Sample counts are reduced from production defaults (GEMM 64^3 ≈ 262K → 14^3 ≈ 2.7K)
so the full sweep finishes in ~hour rather than days.

Usage:
    python examples/profile_pro6000.py [--num-runs N] [--ops gemm,attn,rmsnorm,silu]
                                       [--dtypes fp16,fp8,fp4]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from syssim.compute.compute_cost_profiler import (
    _profile_gemm_grid,
    _profile_attn_grid,
    _profile_rmsnorm_grid,
    _profile_silu_grid,
    _generate_proportional_samples,
    _generate_power_of_two_range,
)
from syssim.config import get_hardware_info


def gemm_grid(samples: int = 14, seed: int = 42) -> dict:
    return {
        "M": _generate_proportional_samples(2, 16384, samples, seed),
        "N": _generate_proportional_samples(256, 16384, samples, seed + 1),
        "K": _generate_proportional_samples(256, 16384, samples, seed + 2),
    }


def attn_grid(samples_seq: int = 16, seed: int = 42) -> dict:
    return {
        "bs": [1],  # FP8 attention requires bs=1; keep consistent across dtypes
        "seq": _generate_proportional_samples(64, 16384, samples_seq, seed),
        "nh": [4, 8, 16, 32],
        "nkv": [1, 2, 4],
        "hd": [64, 128],
    }


def math_grid(samples: int = 32, seed: int = 42) -> dict:
    return {
        "seq": _generate_proportional_samples(64, 16384, samples, seed),
        "dim": _generate_proportional_samples(256, 8192, samples, seed + 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Pro 6000 low-precision profiling driver")
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument(
        "--ops",
        default="gemm,attn,rmsnorm,silu",
        help="Comma-separated operators to profile (gemm,attn,rmsnorm,silu)",
    )
    parser.add_argument(
        "--dtypes",
        default="fp16,fp8,fp4",
        help="Comma-separated dtypes (only applied to gemm/attn; rmsnorm/silu are fp16)",
    )
    parser.add_argument("--out-dir", default="data/profiling")
    parser.add_argument("--gemm-samples", type=int, default=14)
    parser.add_argument("--attn-seq-samples", type=int, default=16)
    parser.add_argument("--math-samples", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hw_info, hw_name = get_hardware_info()
    if hw_name != "pro6000":
        print(f"WARNING: detected hw={hw_name}, expected pro6000. Continuing anyway.")

    ops = args.ops.split(",")
    dtypes = args.dtypes.split(",")

    g_gemm = gemm_grid(args.gemm_samples)
    g_attn = attn_grid(args.attn_seq_samples)
    g_math = math_grid(args.math_samples)

    summary = []

    if "gemm" in ops:
        for dt in dtypes:
            t0 = time.perf_counter()
            print(f"\n{'='*70}\nGEMM {dt}\n{'='*70}")
            results = _profile_gemm_grid(hw_info, g_gemm, args.num_runs, dtype=dt)
            df = pd.DataFrame(results)
            csv = out_dir / f"gemm_{hw_name}_{dt}_data.csv"
            df.to_csv(csv, index=False)
            elapsed = time.perf_counter() - t0
            valid = (df["t_measured_ms"] > 0).sum()
            summary.append((f"gemm_{dt}", csv, len(df), valid, elapsed))
            print(f"\nWrote {csv}: {len(df)} rows ({valid} valid) in {elapsed/60:.1f} min")

    if "attn" in ops:
        for dt in dtypes:
            if dt == "fp4":
                # FlashInfer 0.6 has no FP4 attention path; skip to save time.
                continue
            t0 = time.perf_counter()
            print(f"\n{'='*70}\nATTN {dt}\n{'='*70}")
            results = _profile_attn_grid(hw_info, g_attn, args.num_runs, dtype=dt)
            df = pd.DataFrame(results)
            csv = out_dir / f"attn_{hw_name}_{dt}_data.csv"
            df.to_csv(csv, index=False)
            elapsed = time.perf_counter() - t0
            valid = (df["t_measured_ms"] > 0).sum()
            summary.append((f"attn_{dt}", csv, len(df), valid, elapsed))
            print(f"\nWrote {csv}: {len(df)} rows ({valid} valid) in {elapsed/60:.1f} min")

    if "rmsnorm" in ops:
        t0 = time.perf_counter()
        print(f"\n{'='*70}\nRMSNORM fp16\n{'='*70}")
        results = _profile_rmsnorm_grid(hw_info, g_math, args.num_runs)
        df = pd.DataFrame(results)
        csv = out_dir / f"rmsnorm_{hw_name}_fp16_data.csv"
        df.to_csv(csv, index=False)
        elapsed = time.perf_counter() - t0
        valid = (df["t_measured_ms"] > 0).sum()
        summary.append(("rmsnorm_fp16", csv, len(df), valid, elapsed))
        print(f"\nWrote {csv}: {len(df)} rows ({valid} valid) in {elapsed/60:.1f} min")

    if "silu" in ops:
        t0 = time.perf_counter()
        print(f"\n{'='*70}\nSILU fp16\n{'='*70}")
        results = _profile_silu_grid(hw_info, g_math, args.num_runs)
        df = pd.DataFrame(results)
        csv = out_dir / f"silu_{hw_name}_fp16_data.csv"
        df.to_csv(csv, index=False)
        elapsed = time.perf_counter() - t0
        valid = (df["t_measured_ms"] > 0).sum()
        summary.append(("silu_fp16", csv, len(df), valid, elapsed))
        print(f"\nWrote {csv}: {len(df)} rows ({valid} valid) in {elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    print(f"{'Run':<20} {'Total':<8} {'Valid':<8} {'Time (min)':<12}")
    for name, _csv, total, valid, elapsed in summary:
        print(f"{name:<20} {total:<8} {valid:<8} {elapsed/60:<12.1f}")


if __name__ == "__main__":
    main()
