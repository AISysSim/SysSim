# SysSim: parallelization search and large-model simulation

This runbook uses SysSim's existing CLI to compare parallelization configurations and simulate Llama-3-70B training.

## 1. Prepare the environment

Use a Linux CUDA machine with Python 3.10+, CUDA-enabled PyTorch 2.6 or later, and compatible Megatron-Core, Megatron-Bridge, and Transformer Engine installations. Run commands from the repository root:

```bash
python -m pip install -e .
python -m syssim --help
```

Set `SYSSIM_HARDWARE` to the absolute path of your target hardware YAML:

```bash
export SYSSIM_HARDWARE="$PWD/examples/configs/hardware/isambard_gh200_4node.yaml"  # 16 GPUs; or your own hardware YAML
```

The hardware file must specify compute peaks, memory bandwidth, `gpus_per_node`, `gpu_memory_GB`, and a `topology` block with intra-node and inter-node connectivity, bandwidth, and latency. Provide enough topology endpoints for the largest configuration below: 16 simulated GPUs. The bundled `isambard_gh200_*.yaml` files set `calibrated_model: data/gh200`, which needs `lightgbm` installed (without it every operator time silently falls back to zero). To use the default analytical estimator instead, omit `calibrated_model`.

Tracing requires a real CUDA device, but the execution host does not need to contain the entire simulated cluster. Model and hardware YAML files describe architecture and hardware; parallelization and training settings are CLI arguments.

## 2. Search parallelization configurations

Run the existing sweep command on the included Qwen3-1.7B model. This example varies tensor parallelism (TP) over 1, 2, and 4 while holding data parallelism (DP) and the training settings fixed:

```bash
python -m syssim sweep examples/configs/models/qwen3-1_7b.yaml \
  --hardware "$SYSSIM_HARDWARE" \
  --dp 1 --micro-batch 1 --global-batch 8 \
  --dtype bf16 --recompute full \
  --over parallelism.tp=1,2,4 --metric mfu
```

The command prints the evaluated configuration with the highest predicted model FLOPs utilization (MFU). This example uses different GPU counts for different TP values. Multiple `--over` arguments form a Cartesian product; they do not enforce a fixed total GPU count.

The current selector maximizes the supplied metric and does not filter out-of-memory configurations. Use `--metric mfu`; do not use `--metric step_time_ms` to find the shortest step time. Check the selected configuration with `run`, supplying its TP value:

```bash
export SYSSIM_SELECTED_TP=1  # Replace with the TP value printed by sweep.
python -m syssim run examples/configs/models/qwen3-1_7b.yaml \
  --hardware "$SYSSIM_HARDWARE" \
  --tp "$SYSSIM_SELECTED_TP" --dp 1 \
  --micro-batch 1 --global-batch 8 \
  --dtype bf16 --recompute full --format json
```

Check `bottlenecks.oom` before using the selected plan. If it is true, remove that TP value from the candidate list and repeat the sweep. Selection is limited to the supplied candidates.

## 3. Run a Llama-3-70B simulation

Use `examples/configs/models/llama3-70b.yaml`. It describes 80 layers, hidden size 8,192, 64 attention heads, eight key/value heads, feed-forward size 28,672, vocabulary size 128,256, and sequence length 8,192. Model weights and a training dataset are not required.

The commands below use TP8 and DP2, totaling 16 simulated GPUs, with BF16, microbatch size 1, global batch size 16, and full activation recomputation.

First, check the resolved configuration:

```bash
python -m syssim summary examples/configs/models/llama3-70b.yaml \
  --hardware "$SYSSIM_HARDWARE" \
  --tp 8 --dp 2 --micro-batch 1 --global-batch 16 \
  --dtype bf16 --recompute full
```

Estimate memory before running the timing simulation:

```bash
python -m syssim memory examples/configs/models/llama3-70b.yaml \
  --hardware "$SYSSIM_HARDWARE" \
  --tp 8 --dp 2 --micro-batch 1 --global-batch 16 \
  --dtype bf16 --recompute full
```

Compare the printed per-GPU peak memory with `gpu_memory_GB` in the hardware file. The configuration is a starting point, not a guarantee that it fits. The current CLI does not expose pipeline parallelism or the distributed optimizer; DP alone does not shard optimizer state in these commands. If memory exceeds capacity, adjust the supported parallelization settings or target hardware before proceeding, keeping the full model architecture intact.

Run the training-step simulation with the same settings:

```bash
python -m syssim run examples/configs/models/llama3-70b.yaml \
  --hardware "$SYSSIM_HARDWARE" \
  --tp 8 --dp 2 --micro-batch 1 --global-batch 16 \
  --dtype bf16 --recompute full --format json
```

For another configuration, keep the `summary`, `memory`, and `run` arguments consistent. TP must be compatible with the model's attention dimensions, and global batch size must be divisible by microbatch size times DP. Use `python -m syssim run --help` to inspect the available flags.



