"""SysSim command-line interface."""

from __future__ import annotations

import argparse
import json
import sys


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="syssim")
    sub = p.add_subparsers(dest="command", required=True)

    def _common_args(sp):
        sp.add_argument("model", help="Path to model YAML")
        sp.add_argument("--hardware", required=True, help="Path to hardware YAML")
        sp.add_argument("--tp", type=int, default=1)
        sp.add_argument("--dp", type=int, default=1)
        sp.add_argument("--cp", type=int, default=1)
        sp.add_argument("--sp", action="store_true")
        sp.add_argument("--micro-batch", type=int, required=True)
        sp.add_argument("--global-batch", type=int, required=True)
        sp.add_argument("--dtype", choices=["fp16", "bf16", "fp8"], default="bf16")
        sp.add_argument("--recompute", choices=["selective", "full"], default=None)
        sp.add_argument("--format", choices=["table", "json", "yaml"], default="table")

    run = sub.add_parser("run"); _common_args(run)
    mem = sub.add_parser("memory"); _common_args(mem)
    summ = sub.add_parser("summary"); _common_args(summ)

    sweep = sub.add_parser("sweep")
    _common_args(sweep)
    sweep.add_argument("--over", action="append", default=[],
                       help="path=v1,v2,... (multiple allowed)")
    sweep.add_argument("--metric", default="mfu")

    prof = sub.add_parser("profile")
    prof.add_argument("--device", required=True)
    prof.add_argument("--out", required=True)
    prof.add_argument("--spec", default=None)
    prof.add_argument("--families", default="gemm")
    prof.add_argument("--reps", type=int, default=5)
    prof.add_argument("--dry-run", action="store_true")

    cal = sub.add_parser("calibrate")
    cal.add_argument("--device", required=True)
    cal.add_argument("--data", required=True)
    cal.add_argument("--families", default="gemm")
    cal.add_argument("--target", choices=["residual", "direct"], default="residual")

    return p


def _build_kwargs(args):
    import syssim
    return dict(
        model=args.model,
        hardware=args.hardware,
        parallelism=syssim.ParallelismConfig(tp=args.tp, dp=args.dp, sp=args.sp, cp=args.cp),
        training=syssim.TrainingConfig(
            micro_batch=args.micro_batch, global_batch=args.global_batch,
            dtype=args.dtype, recompute=args.recompute,
        ),
    )


def _print_report(report, fmt: str):
    if fmt == "json":
        print(report.to_json())
    elif fmt == "yaml":
        import yaml as _yaml
        print(_yaml.safe_dump(report.to_dict(), sort_keys=False))
    else:
        print(report)


def main(argv: list[str] | None = None) -> int:
    import syssim
    args = _parser().parse_args(argv)
    if args.command == "run":
        report = syssim.simulate(**_build_kwargs(args))
        _print_report(report, args.format)
        return 0
    if args.command == "memory":
        mem = syssim.estimate_memory(**_build_kwargs(args))
        print(f"peak_memory_gb = {mem.peak_memory_gb:.3f}")
        print(f"  param_bytes        = {mem.param_bytes}")
        print(f"  grad_bytes         = {mem.grad_bytes}")
        print(f"  optimizer_state_bytes = {mem.optimizer_state_bytes}")
        print(f"  activation_bytes   = {mem.activation_bytes}")
        return 0
    if args.command == "summary":
        kw = _build_kwargs(args)
        print(f"model:       {kw['model']}")
        print(f"hardware:    {kw['hardware']}")
        print(f"parallelism: tp={args.tp} dp={args.dp} cp={args.cp} sp={args.sp}  world_size={kw['parallelism'].world_size}")
        print(f"training:    micro_batch={args.micro_batch} global_batch={args.global_batch} dtype={args.dtype}")
        return 0
    if args.command == "sweep":
        over = {}
        for spec in args.over:
            path, _, vals = spec.partition("=")
            over[path] = [int(v) if v.isdigit() else v for v in vals.split(",")]
        kw = _build_kwargs(args)
        result = syssim.sweep(**kw, over=over)
        best = result.best(args.metric)
        print(f"best by {args.metric}: {best.inputs} → {best.metrics}")
        return 0
    if args.command == "profile":
        from .profiling.spec import load_profiling_spec, DEFAULT_SPEC_PATH
        from .profiling.catalog import gemm_worklist
        spec = load_profiling_spec(args.spec or DEFAULT_SPEC_PATH)
        wl = gemm_worklist(spec, token_points=[512, 4096])
        if args.dry_run:
            print(f"GEMM work-list: {len(wl)} items (dry-run, no kernels)")
            return 0
        from .profiling.measure import measure_gemm
        import os
        import pandas as pd
        os.makedirs(os.path.join(args.out, "prof"), exist_ok=True)
        rows, skipped = [], 0
        for it in wl:
            try:
                rows.append(measure_gemm(it["M"], it["K"], it["N"], it["dtype"],
                                         reps=args.reps))
            except Exception:
                skipped += 1   # e.g. fp8 _scaled_mm alignment constraints
        pd.DataFrame(rows).to_parquet(os.path.join(args.out, "prof", "gemm.parquet"))
        print(f"measured {len(rows)} GEMM kernels ({skipped} skipped) "
              f"-> {args.out}/prof/gemm.parquet")
        return 0
    if args.command == "calibrate":
        from .profiling.calibrate import calibrate_gemm
        from .config import get_hardware_info
        # sfu_peak from the device default (gh200 -> H100 default)
        try:
            hw, _ = get_hardware_info()
            sfu = hw.sfu_peak
        except Exception:
            sfu = 247.25
        metrics = calibrate_gemm(data_dir=args.data, device=args.device, sfu_peak=sfu,
                                 target=args.target)
        print(f"calibrated gemm: {metrics}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
