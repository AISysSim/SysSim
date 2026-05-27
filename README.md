# SysSim — LLM Performance & Memory Simulator

[![CI](https://github.com/AISysSim/SysSim/actions/workflows/ci.yml/badge.svg)](https://github.com/AISysSim/SysSim/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**SysSim** estimates the step time and peak memory of LLM training — on hardware you don't have — without running real computation. It traces a model with fake CUDA tensors to build an operator graph, estimates per-operator time from roofline + ML-efficiency models, simulates collectives over a flow-level network model, and replays the graph through a stream-queue discrete-event simulator to get the critical path. For distributed runs it models tensor, sequence, data, and pipeline parallelism, and reports per-GPU memory (including pipeline-stage peaks and OOM).

For in-depth technical architecture, see [DESIGN.md](DESIGN.md).

**Key use cases:**
- Estimate training step time and MFU on accelerators you can't access
- Compare parallelism strategies (TP / SP / DP / PP) before allocating a cluster
- Predict peak per-GPU memory and catch OOM ahead of time
- Find the runtime and memory bottleneck (top ops, binding pipeline stage)

---

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.6+ with CUDA support
- A CUDA-capable GPU (required for tracing — FakeTensorMode dispatches to GPU kernel variants)
- [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) for the training simulator (`syssim.simulate`)

### Installation

```bash
git clone https://github.com/AISysSim/SysSim.git
cd SysSim

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install SysSim
pip install -e .                   # core
pip install -e ".[profiler]"       # + pandas, scikit-learn, xgboost
pip install -e ".[huggingface]"    # + transformers
pip install -e ".[dev]"            # + pytest, ruff, pytest-cov
pip install -e ".[all]"            # everything
```

### Simulate a training step (Python)

```python
import syssim

report = syssim.simulate(
    model="examples/configs/models/qwen3-1_7b.yaml",     # architecture YAML, ModelConfig, or HFModel
    hardware="examples/configs/hardware/dgx_h100.yaml",  # hardware + topology YAML, or HardwareConfig
    parallelism=syssim.ParallelismConfig(tp=2, dp=4),    # tp / dp / pp / cp / sp
    training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
)

print(report)                       # full breakdown (see "What you get")
print(report.step_time_ms)          # estimated step time (ms)
print(report.mfu)                   # model FLOPs utilization
print(report.peak_memory_gb)        # peak per-GPU memory (binding PP stage)
```

### Simulate from the command line

```bash
# Full report (step time, MFU, memory, bottlenecks)
syssim run examples/configs/models/qwen3-1_7b.yaml \
    --hardware examples/configs/hardware/dgx_h100.yaml \
    --tp 2 --dp 4 --micro-batch 1 --global-batch 8

# Fast memory-only estimate (no tracing, ~1 ms)
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
| `step_time_ms` | Wall-clock step time (critical path through the stream-queue DES) |
| `forward_ms` / `backward_ms` / `optimizer_ms` | Time attributed per training phase |
| `collective_total_ms` / `collective_exposed_ms` | Total vs. non-overlapped collective time |
| `achieved_tflops`, `mfu`, `hfu` | Throughput and model/hardware FLOPs utilization |
| `peak_memory_gb` | Peak per-GPU memory at the binding (max) PP stage |
| `pp_stage_memory_gb` | Per-pipeline-stage peak memory (one entry per stage) |
| `per_pp_rank_step_time_ms` | Per-stage finish time |
| `bottlenecks` | Top ops by time, dominant op type, longest collective, binding PP stage, peak module, and OOM (capacity / required / excess) when `gpu_memory_GB` is set |

Memory is captured by a dedicated single-microbatch [MemTracker](https://pytorch.org/docs/stable/distributed.tensor.html) pass and decomposed into persistent (parameters, gradients, optimizer state) vs. per-microbatch activations; the activation term is scaled per stage by the **1F1B** in-flight microbatch count, so earlier pipeline stages correctly show higher peaks.

---

## Parallelism support

| Strategy | Status |
|---|---|
| Tensor parallel (TP) | ✅ collectives traced and timed over the network model |
| Sequence parallel (SP) | ✅ |
| Data parallel (DP) | ✅ gradient all-reduce injected and timed |
| Context parallel (CP) | ✅ |
| Pipeline parallel (PP) | ✅ **1F1B schedule** (Megatron default); each stage traced as a separate process (MPMD), composed with timed P2P transfers |

Pipeline memory and runtime follow the **1F1B** schedule only. GPipe / interleaved (VPP) schedules are out of scope.

---

## Inspect the operator graph

`syssim.trace(...)` returns a `Trace` whose `.graph` is the operator DAG, useful for debugging or visualization:

```python
import syssim

t = syssim.trace(
    model="examples/configs/models/qwen3-1_7b.yaml",
    parallelism=syssim.ParallelismConfig(tp=2, dp=1),
    training=syssim.TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16"),
    hardware="examples/configs/hardware/dgx_h100.yaml",
    gpus_per_node=8,
)

print(t.graph.summary())                 # op counts + total time by type
with open("graph.dot", "w") as f:
    f.write(t.graph.to_dot())             # Graphviz

report = t.simulate_on(syssim.HardwareConfig(...))   # re-time the cached graph on other hardware
```

---

## Repository Structure

```
SysSim/
├── syssim/                  # Main package
│   ├── tracer.py            # TorchDispatchMode tracing → operator graph
│   ├── operator_graph.py    # DAG IR (operator types, summary, Graphviz/JSON export)
│   ├── config.py            # HardwareInfo, SimulatorConfig, hardware auto-detect
│   ├── cli.py               # `syssim run | memory | summary | sweep`
│   ├── compute/             # Per-operator time: roofline + ML-efficiency models, FLOP counter
│   ├── network/             # Flow-level network sim: topologies, collectives, max-min fair solver
│   └── training/            # Distributed training simulator: configs, parallelism, memory, report
├── examples/                # Runnable examples + model/hardware YAML configs
├── data/                    # Profiling CSVs and trained efficiency models
├── tests/                   # Test suite
├── DESIGN.md                # Technical design document
└── pyproject.toml           # Package metadata
```
