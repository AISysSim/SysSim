"""Train per-(operator, dtype) XGBoost efficiency models for Pro 6000.

Reads CSVs produced by examples/profile_pro6000.py and writes one .pth per
(operator, dtype) into data/trained_models/.

Usage:
    python examples/train_pro6000_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from syssim.compute.compute_cost_profiler import train_efficiency_model
from syssim.config import get_hardware_info


# (operator, dtype) combinations to train. rmsnorm/silu are fp16 only.
COMBOS = [
    ("gemm", "fp16"),
    ("gemm", "fp8"),
    ("gemm", "fp4"),
    ("attn", "fp16"),
    ("attn", "fp8"),
    ("rmsnorm", "fp16"),
    ("silu", "fp16"),
]


def main():
    _, hw_name = get_hardware_info()
    csv_dir = Path("data/profiling")
    out_dir = Path("data/trained_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for op, dt in COMBOS:
        csv = csv_dir / f"{op}_{hw_name}_{dt}_data.csv"
        if not csv.exists():
            print(f"[skip] {csv} not found")
            continue

        out_path = out_dir / f"{op}_{hw_name}_{dt}_xgb.pth"
        print(f"\n{'='*70}\nTraining {op} {dt}\n{'='*70}")
        try:
            cv_metrics, eff_mape, time_mape = train_efficiency_model(
                operator=op,
                csv_path=csv,
                output_path=str(out_path),
                backend="xgboost",
                dtype=dt,
            )
            summary.append((op, dt, eff_mape, time_mape, str(out_path)))
        except Exception as e:
            print(f"[FAIL] {op} {dt}: {e}")
            summary.append((op, dt, float("nan"), float("nan"), f"FAILED: {e}"))

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'op':<10} {'dtype':<6} {'eff_mape%':<12} {'time_mape%':<12} {'path'}")
    for op, dt, eff_mape, time_mape, path in summary:
        print(f"{op:<10} {dt:<6} {eff_mape:<12.2f} {time_mape:<12.2f} {path}")


if __name__ == "__main__":
    main()
