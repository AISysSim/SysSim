"""Blackhole end-to-end demo for SysSim.

Builds a Blackhole P150B HardwareInfo from TT_HW_DATABASE, loads the
trained efficiency models from ``data/trained_models/`` (or wherever
``RLSYSIM_MODEL_DIR`` points), and reports the predicted execution time
of a few representative GEMM, attention, RMSNorm, and SiLU shapes.

This script does NOT require CUDA — it exercises the cost-model layer
directly. The full ``trace_model_for_inference`` API still needs CUDA
for FakeTensor dispatch (see DESIGN.md).
"""

from __future__ import annotations

import os
import sys

import torch

from syssim.config import HardwareInfo
from syssim.compute.compute_cost_predictor import roofline_estimate, aten
from syssim.compute.compute_cost_profiler import TT_HW_DATABASE
from syssim.compute.efficiency_models import BackendManager
from syssim.operator_graph import OperatorType


PLATFORM = os.environ.get("SYSSIM_HW_NAME", "tt_bh_p150b")
MODEL_DIR = os.environ.get("RLSYSIM_MODEL_DIR", "data/trained_models")


def _hw() -> HardwareInfo:
    if PLATFORM not in TT_HW_DATABASE:
        sys.exit(f"unknown TT platform {PLATFORM!r}; choose from {list(TT_HW_DATABASE)}")
    mm, mm_c, math_t, bw = TT_HW_DATABASE[PLATFORM]
    return HardwareInfo(
        peak_tflops_mm=mm,
        peak_tflops_math=math_t,
        peak_memory_bandwidth_gbps=bw,
        peak_tflops_mm_conservative=mm_c,
    )


def _gemm_row(hw: HardwareInfo, mgr: BackendManager, m: int, n: int, k: int) -> None:
    a = torch.randn(m, k, dtype=torch.bfloat16)
    b = torch.randn(k, n, dtype=torch.bfloat16)
    out = torch.zeros(m, n, dtype=torch.bfloat16)
    r = roofline_estimate(aten.mm.default, (a, b), {}, out, hw, OperatorType.GEMM)
    model = mgr.get_model(OperatorType.GEMM)
    eff_str = "(no model)"
    if model is not None:
        from syssim.compute.compute_cost_predictor import efficiency_estimate
        eta = efficiency_estimate(
            aten.mm.default, (a, b), {}, out, hw, OperatorType.GEMM, r,
        )
        # The simulator's runtime estimate is roofline / efficiency.
        pred_ms = r.t_roofline_ms / eta if eta > 0 else r.t_roofline_ms
        eff_str = f"(eta={eta:.3f}, est={pred_ms:.4f} ms)"
    print(f"  gemm  {m:6d}x{n:6d}x{k:6d}  roofline={r.t_roofline_ms:.4f} ms  {eff_str}")


def _silu_row(hw: HardwareInfo, mgr: BackendManager, seq: int, dim: int) -> None:
    x = torch.randn(seq, dim, dtype=torch.bfloat16)
    y = torch.zeros_like(x)
    r = roofline_estimate(aten.silu.default, (x,), {}, y, hw, OperatorType.MATH)
    print(f"  silu  {seq:6d}x{dim:6d}                roofline={r.t_roofline_ms:.4f} ms")


def main() -> None:
    hw = _hw()
    print(f"Platform: {PLATFORM}")
    print(f"  peak_tflops_mm     = {hw.peak_tflops_mm}")
    print(f"  peak_tflops_math   = {hw.peak_tflops_math}")
    print(f"  peak_bandwidth_GBs = {hw.peak_memory_bandwidth_gbps}")
    print(f"Model dir: {MODEL_DIR}")
    mgr = BackendManager(MODEL_DIR, hw_name=PLATFORM)
    loaded = [op.value for op, m in mgr._models.items() if m is not None]
    print(f"  loaded models: {loaded if loaded else '(none — falling back to roofline)'}")
    print()

    print("=== GEMM (BF16) ===")
    for m, n, k in [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192),
                    (256, 4096, 4096), (4096, 4096, 256)]:
        _gemm_row(hw, mgr, m, n, k)

    print()
    print("=== SiLU (BF16) ===")
    for seq, dim in [(1024, 4096), (4096, 4096), (8192, 8192)]:
        _silu_row(hw, mgr, seq, dim)


if __name__ == "__main__":
    main()
