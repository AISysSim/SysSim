"""End-to-end MoE expert-parallel (EP) example: gpt-oss-20b on a single GH200 node.

MoE uses expert parallelism ONLY (never TP/PP for experts). EP is carved from the
DP grid: world_size = tp*dp*cp*pp is unchanged; the experts shard over EP while
attention/embeddings stay DP-replicated. Configure with
``ParallelismConfig(dp=N, ep=M)``.

Two runs:
  1. dp=4, ep=4, no recompute -> activation-dominated, exceeds the 96 GB cap
     (a correctly-flagged OOM: the peak is real activation memory, not an
     expert-sharding bug -- experts ARE sharded num_experts/ep per rank).
  2. dp=4, ep=4, recompute_granularity="full" -> fits in 96 GB.

Run from the repo root with the syssim venv:
    PYTHONPATH=. python examples/moe_expert_parallel.py
"""

import os
import sys

# Run-as-script guard: `python examples/moe_expert_parallel.py` puts this directory
# first on sys.path, where the tracked `examples/megatron/` package would shadow the
# installed `megatron.core` that the tracer imports. Drop the script dir and put the
# repo root in front so the real Megatron-Core resolves.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _THIS_DIR:
    sys.path[0] = os.path.dirname(_THIS_DIR)

import syssim
from syssim.training.spec import (
    load_model_yaml, load_hardware_yaml, ParallelismConfig, TrainingConfig,
)


def _report(title, rep, gpu_cap_gb):
    oom = rep.peak_memory_gb > gpu_cap_gb
    print(f"\n=== {title} ===")
    print(f"step_time_ms   = {rep.step_time_ms:.2f}")
    print(f"fwd/bwd/opt ms = {rep.forward_ms:.1f} / {rep.backward_ms:.1f} / {rep.optimizer_ms:.1f}")
    print(f"MFU / HFU      = {rep.mfu:.4f} / {rep.hfu:.4f}")
    print(f"achieved_tflops= {rep.achieved_tflops:.1f}")
    print(f"peak_memory_gb = {rep.peak_memory_gb:.2f}  "
          f"({'OOM > ' if oom else 'fits <= '}{gpu_cap_gb:.0f} GB cap)")
    print(f"collective tot/exposed ms = "
          f"{rep.collective_total_ms:.2f} / {rep.collective_exposed_ms:.2f}")
    print(f"model_flops/step = {rep.model_flops_per_step:.3e}")


def main():
    m = load_model_yaml("examples/configs/models/gpt-oss-20b.yaml")
    hw = load_hardware_yaml("examples/configs/hardware/isambard_gh200_4gpu.yaml")
    par = ParallelismConfig(dp=4, ep=4)  # world=4, EP only: 32 experts -> 8/rank
    print(f"world_size={par.world_size} ep={par.expert_model_parallel_size} "
          f"dp={par.data_parallel_size} (32 experts -> "
          f"{32 // par.expert_model_parallel_size}/rank)")

    tr = TrainingConfig(micro_batch=1, global_batch=4, dtype="bf16",
                        use_distributed_optimizer=True)
    _report("gpt-oss-20b GH200x4 (dp4 ep4, no recompute)",
            syssim.simulate(model=m, hardware=hw, parallelism=par, training=tr),
            hw.gpu_memory_GB)

    tr_recompute = TrainingConfig(micro_batch=1, global_batch=4, dtype="bf16",
                                  use_distributed_optimizer=True, recompute="full")
    _report("gpt-oss-20b GH200x4 (dp4 ep4, recompute=full)",
            syssim.simulate(model=m, hardware=hw, parallelism=par, training=tr_recompute),
            hw.gpu_memory_GB)


if __name__ == "__main__":
    main()
