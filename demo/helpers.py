"""Shared helpers for the ARIA tutorial Colab notebook.

Imported by demo/aria_tutorial.ipynb AND demo/smoke_test.py so the notebook
cells stay short and the logic is independently testable.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


def helpers_loaded() -> str:
    return "ok"


def _roofline_gemm_ms(m: int, n: int, k: int, peak_tflops: float,
                      peak_bw_GBps: float, dtype_bytes: int) -> float:
    """Roofline ms for an M×K @ K×N GEMM."""
    flops = 2.0 * m * n * k
    compute_s = flops / (peak_tflops * 1e12)
    bytes_moved = dtype_bytes * (m * k + k * n + m * n)
    memory_s = bytes_moved / (peak_bw_GBps * 1e9)
    return max(compute_s, memory_s) * 1000.0


def synthesize_gemm_csv(out_path: Path, peak_tflops: float, peak_bw_GBps: float,
                        dtype_bytes: int, seed: int = 42) -> Path:
    """Generate plausible GEMM profiling data.

    Measured time = roofline / (0.6 + 0.25 × U[0,1)) — gives the predictor
    a noisy-but-correlated target to learn.

    Shape grid: 5×5×5 = 125 rows, satisfying the ≥100 row requirement.
    """
    rng = random.Random(seed)
    shapes = [
        (m, n, k)
        for m in (512, 1024, 2048, 4096, 8192)
        for n in (512, 1024, 2048, 4096, 8192)
        for k in (512, 1024, 2048, 4096, 8192)
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "t_measured_ms"])
        for m, n, k in shapes:
            t_roof = _roofline_gemm_ms(m, n, k, peak_tflops, peak_bw_GBps, dtype_bytes)
            efficiency = 0.6 + 0.25 * rng.random()
            w.writerow([m, n, k, f"{t_roof / efficiency:.6f}"])
    return out_path


def synthesize_attn_csv(out_path: Path, peak_tflops: float, peak_bw_GBps: float,
                        dtype_bytes: int, seed: int = 42) -> Path:
    """Generate plausible attention (SDPA) profiling data.

    Shape grid: 2×4×2×2 = 32 cases × ... = 64 rows, satisfying the ≥50 row requirement.
    """
    rng = random.Random(seed)
    cases = [
        (bs, seq, nh, nkv, hd)
        for bs in (1, 2, 4, 8)
        for seq in (1024, 2048, 4096, 8192)
        for nh, nkv in ((32, 8), (32, 32))
        for hd in (64, 128)
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bs", "seq", "nh", "nkv", "hd", "t_measured_ms"])
        for bs, seq, nh, nkv, hd in cases:
            flops = 4.0 * bs * nh * seq * seq * hd
            compute_s = flops / (peak_tflops * 1e12)
            bytes_moved = dtype_bytes * 2 * bs * (nh + nkv) * seq * hd
            memory_s = bytes_moved / (peak_bw_GBps * 1e9)
            t_roof = max(compute_s, memory_s) * 1000.0
            efficiency = 0.45 + 0.25 * rng.random()
            w.writerow([bs, seq, nh, nkv, hd, f"{t_roof / efficiency:.6f}"])
    return out_path


def synthesize_rmsnorm_csv(out_path: Path, peak_bw_GBps: float, dtype_bytes: int,
                           seed: int = 42) -> Path:
    """Generate plausible RMSNorm profiling data (memory-bound).

    Shape grid: 4×3 = 12 cases, satisfying the ≥30 row requirement via 3 passes.
    Actually 4×4 = 16 dim values used; let's use 4 seq × 4 dim × 2 passes = 32 rows.
    """
    rng = random.Random(seed)
    cases = [
        (seq, dim)
        for seq in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
        for dim in (1024, 2048, 4096, 8192)
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "dim", "t_measured_ms"])
        for seq, dim in cases:
            bytes_moved = dtype_bytes * 2 * seq * dim
            t_roof = (bytes_moved / (peak_bw_GBps * 1e9)) * 1000.0
            efficiency = 0.55 + 0.25 * rng.random()
            w.writerow([seq, dim, f"{t_roof / efficiency:.6f}"])
    return out_path
