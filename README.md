<p align="center">
  <img src="assets/logo.svg" alt="SysSim" width="320">
</p>

<h1 align="center">An LLM Performance & Memory Simulator</h1>

<p align="center">
  <a href="https://github.com/AISysSim/SysSim/actions/workflows/ci.yml"><img src="https://github.com/AISysSim/SysSim/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://aisyssim.github.io/"><img src="https://img.shields.io/badge/docs-aisyssim.github.io-5b4ee5.svg" alt="Documentation"></a>
</p>

**SysSim** estimates the step time and peak memory of LLM training — on hardware you don't have — without running real computation. It models tensor, sequence, data, and pipeline parallelism and reports step time, MFU, and per-GPU memory (including per-pipeline-stage peaks and OOM).

📖 **Full documentation:** [aisyssim.github.io](https://aisyssim.github.io/) — for in-depth technical architecture, see [docs/DESIGN.md](docs/DESIGN.md).

**Key use cases:**
- Estimate training step time and MFU on accelerators you can't access
- Compare parallelism strategies (TP / SP / DP / PP) before allocating a cluster
- Predict peak per-GPU memory and catch OOM ahead of time
- Find the runtime and memory bottleneck (top ops, heaviest pipeline stage)

---

## Quick Start

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

MODEL = "examples/configs/models/qwen3-1_7b.yaml"             # architecture YAML, ModelConfig, or HFModel
HW    = "examples/configs/hardware/isambard_gh200_4gpu.yaml"  # hardware + topology YAML, or HardwareConfig

# 1) simulate — full step-time / MFU / memory / bottleneck report
report = syssim.simulate(
    model=MODEL, hardware=HW,
    parallelism=syssim.ParallelismConfig(tp=2, dp=2),   # tp / dp / pp / cp / sp
    training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
)
print(report.step_time_ms, report.mfu, report.peak_memory_gb)

# 2) estimate_memory — per-GPU peak memory only (skips step-time estimation)
mem = syssim.estimate_memory(
    model=MODEL, hardware=HW,
    parallelism=syssim.ParallelismConfig(tp=2, dp=2),
    training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
)
print(mem.peak_memory_gb, mem.pp_stage_memory_gb)       # per-GPU peak, per-PP-stage

# 3) sweep — search a config axis, pick the best by a metric
result = syssim.sweep(
    model=MODEL, hardware=HW,
    parallelism=syssim.ParallelismConfig(),
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
    --hardware examples/configs/hardware/isambard_gh200_4gpu.yaml \
    --tp 2 --dp 2 --micro-batch 1 --global-batch 8

# Memory only — peak memory without step-time estimation
syssim memory examples/configs/models/qwen3-1_7b.yaml \
    --hardware examples/configs/hardware/isambard_gh200_4gpu.yaml \
    --micro-batch 1 --global-batch 8

# Sweep a config axis, pick the best by a metric
syssim sweep examples/configs/models/qwen3-1_7b.yaml \
    --hardware examples/configs/hardware/isambard_gh200_4gpu.yaml \
    --micro-batch 1 --global-batch 8 \
    --over parallelism.tp=1,2,4 --metric mfu
```

`run`, `memory`, and `summary` share the same flags: `--tp --dp --cp --sp --micro-batch --global-batch --dtype {fp16,bf16,fp8} --recompute {selective,full} --format {table,json,yaml}`. Pipeline parallelism (`pp`) is available through the Python API (`ParallelismConfig(pp=...)`).

### Calibrated per-operator predictor (opt-in)

The default estimator is the **roofline** bound. For higher accuracy, attach a
**calibrated** estimator (`TreeEstimator`) — the roofline times a learned residual, one
regularized LightGBM tree per operator family (the in-context layer profile routes to GEMM,
elementwise, and reduction), with the bare roofline as the fallback for any uncalibrated op:

```python
from syssim.compute.tree_estimator import TreeEstimator
hw = HardwareConfig(..., estimator=TreeEstimator.load("data/gh200", hw_info))
```

Build the model for a GPU by profiling **real Megatron transformer layers** over the
architecture/shape space in `syssim/profiling/default_spec.yaml`, then fitting the trees on CPU
(`data/gh200/README.md` has the full reproduce recipe; profiling needs the GPU(s), calibration is
CPU-only):

```bash
# Profile real layers. --num-workers N spawns N workers, one pinned per GPU.
syssim profile   --out data/gh200 --num-workers 4
# Fit per-family residual trees from <data>/profile.parquet.
syssim calibrate --data data/gh200 --hardware examples/configs/hardware/isambard_gh200_4gpu.yaml
```

There is no model-file input to `profile` — the shape space is the committed spec. Preview the job
list (layer configs × tensor-parallel shapes) without touching the GPU via `syssim profile --dry-run`.

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

**Hardware YAML** — accelerator peaks + a per-dimension `topology:` block:

```yaml
# examples/configs/hardware/isambard_gh200_4gpu.yaml
peak_tflops_mm: 1979           # tensor-unit peak (TFLOP/s)
peak_tflops_math: 989          # vector/math peak (TFLOP/s)
peak_memory_bandwidth_GBps: 3350
peak_tflops_mm_fp8: 3958

gpus_per_node: 4
gpu_memory_GB: 96              # per-GPU HBM; enables OOM detection

# Each dimension lays down links that the flow simulator times directly (no analytical
# collective formula). `bandwidth` is the per-GPU UNI-directional aggregate; links are full-duplex,
# and the per-peer link bandwidth is derived from it by the dimension's node degree.
topology:
  dims:      [ fully_connected ]   # per dimension: fully_connected | switch | ring
  size:      [ 4 ]                 # endpoints in this dimension (4 NVLink-meshed GH200)
  bandwidth: [ 450 ]               # per-GPU uni-directional GB/s (900 NVLink bidir / 2)
  latency:   [ 12000 ]             # link latency (ns)
```

A multi-level fabric (e.g. intra-node NVLink + inter-node Slingshot) adds a second entry to each
list. The number of nodes is derived from `world_size / gpus_per_node`.

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
├── data/                    # Profiling data + per-device calibrated models
├── tests/                   # Test suite
├── third_party/             # Git submodules (e.g. PLENA_Simulator)
├── docs/                    # DESIGN.md (architecture)
└── pyproject.toml           # Package metadata
```
