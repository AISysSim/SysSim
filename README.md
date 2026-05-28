# SysSim — LLM Performance & Memory Simulator

[![CI](https://github.com/AISysSim/SysSim/actions/workflows/ci.yml/badge.svg)](https://github.com/AISysSim/SysSim/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**SysSim** estimates the step time and peak memory of LLM training — on hardware you don't have — without running real computation. It models tensor, sequence, data, and pipeline parallelism and reports step time, MFU, and per-GPU memory (including per-pipeline-stage peaks and OOM).

For in-depth technical architecture, see [docs/DESIGN.md](docs/DESIGN.md).

**Key use cases:**
- Estimate training step time and MFU on accelerators you can't access
- Compare parallelism strategies (TP / SP / DP / PP) before allocating a cluster
- Predict peak per-GPU memory and catch OOM ahead of time
- Find the runtime and memory bottleneck (top ops, heaviest pipeline stage)

---

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.6+ with CUDA support
- A CUDA-capable GPU
- [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) for the training simulator (`syssim.simulate`)

### Installation

```bash
git clone https://github.com/AISysSim/SysSim.git
cd SysSim
pip install -e .
```

### High-level Python APIs

Three entry points cover the common needs — run a simulation, check memory, or compare configs:

```python
import syssim

MODEL = "examples/configs/models/qwen3-1_7b.yaml"     # architecture YAML, ModelConfig, or HFModel
HW    = "examples/configs/hardware/dgx_h100.yaml"     # hardware + topology YAML, or HardwareConfig

# 1) simulate — full step-time / MFU / memory / bottleneck report
report = syssim.simulate(
    model=MODEL, hardware=HW,
    parallelism=syssim.ParallelismConfig(tp=2, dp=4),   # tp / dp / pp / cp / sp
    training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
)
print(report.step_time_ms, report.mfu, report.peak_memory_gb)

# 2) estimate_memory — per-GPU peak memory only (skips step-time estimation)
mem = syssim.estimate_memory(
    model=MODEL, hardware=HW,
    parallelism=syssim.ParallelismConfig(tp=2, dp=4),
    training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
)
print(mem.peak_memory_gb, mem.pp_stage_memory_gb)       # per-GPU peak, per-PP-stage

# 3) sweep — search a config axis, pick the best by a metric
result = syssim.sweep(
    model=MODEL, hardware=HW,
    parallelism=syssim.ParallelismConfig(dp=4),
    training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    over={"parallelism.tp": [1, 2, 4]},
)
best = result.best("mfu")
print(best.inputs, best.metrics)
```

### Simulate from the command line

```bash
# Full report (step time, MFU, memory, bottlenecks)
syssim run examples/configs/models/qwen3-1_7b.yaml \
    --hardware examples/configs/hardware/dgx_h100.yaml \
    --tp 2 --dp 4 --micro-batch 1 --global-batch 8

# Memory only — peak memory without step-time estimation
syssim memory examples/configs/models/qwen3-1_7b.yaml \
    --hardware examples/configs/hardware/dgx_h100.yaml \
    --micro-batch 1 --global-batch 8

# Sweep a config axis, pick the best by a metric
syssim sweep examples/configs/models/qwen3-1_7b.yaml \
    --hardware examples/configs/hardware/dgx_h100.yaml \
    --micro-batch 1 --global-batch 8 \
    --over parallelism.tp=1,2,4 --metric mfu
```

`run`, `memory`, and `summary` share the same flags: `--tp --dp --cp --sp --micro-batch --global-batch --dtype {fp16,bf16,fp8} --recompute {selective,full} --format {table,json,yaml}`. Pipeline parallelism (`pp`) is available through the Python API (`ParallelismConfig(pp=...)`).

---

## Configuration

SysSim uses two YAML files — **model architecture** and **hardware** — kept separate so a model can be simulated across machines and vice versa. Parallelism and training knobs are passed as Python kwargs / CLI flags, not YAML.

**Model YAML** — architecture only:

```yaml
# examples/configs/models/qwen3-1_7b.yaml
num_layers: 28
hidden_size: 2048
num_attention_heads: 16
num_query_groups: 8           # GQA
ffn_hidden_size: 6144
seq_length: 4096
max_position_embeddings: 40960
vocab_size: 151936
swiglu: true
rope: true
tie_word_embeddings: true
rms_norm_eps: 1.0e-6
```

**Hardware YAML** — accelerator peaks + interconnect, with a `topology:` block:

```yaml
# examples/configs/hardware/dgx_h100.yaml
peak_tflops_mm: 1979           # tensor-unit peak (TFLOP/s)
peak_tflops_math: 989          # vector/math peak (TFLOP/s)
peak_memory_bandwidth_GBps: 3350
peak_tflops_mm_fp8: 3958

gpus_per_node: 8
gpu_memory_GB: 80              # per-GPU HBM; enables OOM detection
inter_node_bandwidth_GBps: 200
inter_node_latency_us: 5

topology:
  type: simple                 # simple | arbitrary | two_layer_multipath | fat_tree
  num_nodes: 1
  intra_node_bandwidth_GBps: 900
  inter_node_bandwidth_GBps: 200
```

The number of nodes is derived from `world_size / gpus_per_node`. Bandwidth fields use neutral `intra_node_*` / `inter_node_*` names (not hardware-specific terms).

---

## What you get

`syssim.simulate(...)` returns a `SimulationReport`:

| Field | Meaning |
|---|---|
| `step_time_ms` | Estimated wall-clock step time |
| `forward_ms` / `backward_ms` / `optimizer_ms` | Time attributed per training phase |
| `collective_total_ms` / `collective_exposed_ms` | Total vs. non-overlapped collective time |
| `achieved_tflops`, `mfu`, `hfu` | Throughput and model/hardware FLOPs utilization |
| `peak_memory_gb` | Peak per-GPU memory (heaviest pipeline stage) |
| `pp_stage_memory_gb` | Per-pipeline-stage peak memory (one entry per stage) |
| `per_pp_rank_step_time_ms` | Per-stage finish time |
| `bottlenecks` | Top ops by time, dominant op type, longest collective, binding PP stage, peak module, and OOM (capacity / required / excess) when `gpu_memory_GB` is set |

`peak_memory_gb` is the heaviest pipeline stage; the report also breaks memory down into parameters, gradients, optimizer state, and activations.

---

## Parallelism support

| Strategy | Supported |
|---|---|
| Tensor parallel (TP) | ✅ |
| Sequence parallel (SP) | ✅ |
| Data parallel (DP) | ✅ |
| Context parallel (CP) | ✅ |
| Pipeline parallel (PP) | ✅ (1F1B schedule) |
| Expert parallel (EP) | 🚧 work in progress |

These are combinable (e.g. TP×DP×PP).

---

## Repository Structure

```
SysSim/
├── syssim/                  # Core package (tracer, operator-graph IR, config, CLI) + subpackages:
│   ├── compute/             # Per-operator cost models (pluggable estimators) + FLOP counting
│   ├── network/             # Network simulation: topologies, collectives, routing
│   ├── training/            # Distributed training simulator: configs, parallelism, memory, report
│   └── external/            # Optional isolated integrations (e.g. PLENA custom estimator)
├── examples/                # Runnable examples + model/hardware YAML configs
├── data/                    # Profiling CSVs and trained efficiency models
├── tests/                   # Test suite
├── third_party/             # Git submodules (e.g. PLENA_Simulator)
├── docs/                    # DESIGN.md (architecture)
└── pyproject.toml           # Package metadata
```
