"""Run one SysSim case and write {step_time_ms, peak_memory_gb, mfu, oom} JSON.

Run inside the localhost/mksit/syssim image on a GPU node — SysSim's tracer requires CUDA
(`torch.cuda.is_available()`). Architecture comes from the model YAML; hardware from the
isambard_gh200 hardware YAML.
"""
import argparse
import json

import syssim

_NONE = {"", "none", "None"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="model YAML path")
    p.add_argument("--hardware", required=True, help="hardware YAML path")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--mbs", type=int, default=1, help="micro batch size")
    p.add_argument("--gbs", type=int, default=1, help="global batch size")
    p.add_argument("--recompute", default=None, help="none|selective|full")
    p.add_argument("--distributed-optimizer", type=int, default=0,
                   help="1 = model ZeRO-1 optimizer-state sharding across the DP group")
    args = p.parse_args()

    recompute = None if (args.recompute is None or args.recompute in _NONE) else args.recompute
    report = syssim.simulate(
        model=args.model,
        hardware=args.hardware,
        parallelism=syssim.ParallelismConfig(tp=args.tp, dp=args.dp),
        training=syssim.TrainingConfig(
            micro_batch=args.mbs, global_batch=args.gbs, dtype="bf16", recompute=recompute,
            use_distributed_optimizer=bool(args.distributed_optimizer),
        ),
    )
    oom = report.bottlenecks.oom if report.bottlenecks is not None else None
    out = {
        "step_time_ms": report.step_time_ms,
        "peak_memory_gb": report.peak_memory_gb,
        "mfu": report.mfu,
        "oom": oom,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}: step={report.step_time_ms:.2f}ms "
          f"peak={report.peak_memory_gb:.2f}GB mfu={report.mfu:.3f} oom={oom}")


if __name__ == "__main__":
    main()
