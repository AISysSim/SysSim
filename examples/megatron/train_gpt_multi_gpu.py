"""
Megatron-Core GPT-3 1.3B — Training Simulation (configurable TP)
=================================================================

Simulates one forward + backward step of a GPT-3-style 1.3B model sharded
with Tensor Parallelism using syssim. TP size is configurable via --tp-size.

Run command (from repo root):
  python examples/megatron/train_gpt_multi_gpu.py [--tp-size {1,2,4,8,16}]
"""

import argparse
import syssim


def main():
    parser = argparse.ArgumentParser(description="Simulate Megatron GPT-3 1.3B with syssim")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
        choices=[1, 2, 4, 8, 16],
        help="Tensor parallel size (default: 4).",
    )
    args = parser.parse_args()
    tp_size = args.tp_size

    # Use syssim's high-level simulate() API
    report = syssim.simulate(
        model="examples/configs/models/qwen3-1_7b.yaml",
        hardware="examples/configs/hardware/single_h100.yaml",
        parallelism=syssim.ParallelismConfig(tp=tp_size),
        training=syssim.TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
    )

    print("\n" + "=" * 60)
    print(f"  syssim — Megatron GPT-3 1.3B Simulation, TP={tp_size}")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
