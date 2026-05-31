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

    # Profiling is IN-CONTEXT, LAYER-LEVEL only: build a real Megatron transformer layer for each
    # model, sweep input/sharded shapes, and record every dispatched op's real GPU time into one
    # profile.parquet. (There is no synthetic per-family microbenchmark path.)
    prof = sub.add_parser(
        "profile",
        help="profile real Megatron layers over the spec's shape space -> <out>/profile.parquet")
    prof.add_argument("--out", required=True, help="output data dir (writes <out>/profile.parquet)")
    prof.add_argument("--spec", default=None,
                      help="profiling spec YAML (architecture field lists + seq/token ranges); "
                           "default = syssim/profiling/default_spec.yaml")
    prof.add_argument("--num-workers", type=int, default=1,
                      help="GPU worker processes (1 = serial; N spawns a pool, each worker pins one "
                           "GPU). Mirrors the original profiler's --num-workers.")
    prof.add_argument("--dry-run", action="store_true",
                      help="print the job list (layer configs x tensor-parallel) without profiling")

    cal = sub.add_parser("calibrate", help="fit per-family residual trees from <data>/profile.parquet")
    cal.add_argument("--data", required=True, help="data dir containing profile.parquet")
    cal.add_argument("--hardware", required=True, help="hardware YAML")
    cal.add_argument("--families", default="gemm,elementwise,reduction")

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
        from .profiling.measure_layer import spec_configs, profile_worklist
        from .profiling.spec import load_profiling_spec, DEFAULT_SPEC_PATH
        import os
        import pandas as pd

        def powers_of_two(low, high):
            values, value = [], 1
            while value < low:
                value *= 2
            while value <= high:
                values.append(value)
                value *= 2
            return values

        spec = load_profiling_spec(args.spec or DEFAULT_SPEC_PATH)
        # seq lengths from seq_len_range; batch is derived (tokens = batch * seq) and the
        # (seq, batch) sweep is filtered to token_range inside the profiler; tensor-parallel shapes
        # come from parallelism.tensor_parallel. There is no "model": the unit is the (op, shape).
        seq_points = tuple(powers_of_two(int(spec.seq_len_range["min"]), int(spec.seq_len_range["max"])))
        token_low, token_high = int(spec.token_range["min"]), int(spec.token_range["max"])
        batch_points = tuple(powers_of_two(1, max(1, token_high // min(seq_points))))
        tensor_parallel_degrees = tuple(spec.parallelism.get("tensor_parallel", [1, 2, 4, 8]))

        # Each job builds one layer config at one tensor-parallel shape and times it.
        configs = spec_configs(spec)
        jobs = [(config, tensor_parallel)
                for config in configs
                for tensor_parallel in tensor_parallel_degrees
                if config.num_attention_heads % tensor_parallel == 0
                and config.num_query_groups % tensor_parallel == 0]
        if args.dry_run:
            print(f"dry run: {len(jobs)} layer-config jobs "
                  f"({len(configs)} configs x tensor-parallel {list(tensor_parallel_degrees)}, "
                  f"divisible only); seq points {list(seq_points)}")
            return 0
        print(f"profiling {len(jobs)} layer-config jobs with {args.num_workers} GPU worker(s)...",
              file=sys.stderr)

        profiled = profile_worklist(jobs, args.num_workers, seq_points, batch_points,
                                    (token_low, token_high))
        rows = [{"op": row["op"], "count": row["count"],
                 "per_instance_ns": row["per_instance_ns"],
                 "signature": json.dumps({"args": row["args"], "kwargs": row["kwargs"],
                                          "out": row["out"]})}
                for row in profiled]

        # The same (op, shape) recurs across configs; keep one row per unique op-shape (median time).
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = (frame.groupby(["op", "signature"], as_index=False)
                          .agg(count=("count", "first"),
                               per_instance_ns=("per_instance_ns", "median")))
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "profile.parquet")
        frame.to_parquet(out_path)
        print(f"profiled {len(frame)} unique (op, shape) rows from {len(jobs)} jobs -> {out_path}")
        return 0
    if args.command == "calibrate":
        from .profiling.calibrate_layer import calibrate_from_parquet
        import os
        families = [name.strip() for name in args.families.split(",")]
        metrics = calibrate_from_parquet(
            os.path.join(args.data, "profile.parquet"),
            data_dir=args.data, hardware=args.hardware, families=families)
        models = ", ".join(f"{name}_model.lgb" for name in metrics)
        print(f"calibrated {len(metrics)} families -> {args.data} ({models} + manifest.json)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
