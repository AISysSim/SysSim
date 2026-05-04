# SysSim — LLM Performance Simulator

[![CI](https://github.com/AISysSim/SysSim/actions/workflows/ci.yml/badge.svg)](https://github.com/AISysSim/SysSim/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**SysSim** is a PyTorch operator-level performance simulator that estimates neural network execution time without running actual computation. It traces model execution to build a computational graph (DAG), estimates per-operator runtime using roofline models and ML-based efficiency prediction, and computes the critical path through multi-stream execution.

**Key use cases:**
- Simulate model performance on hardware you don't have access to
- Predict training time before allocating resources
- Evaluate multi-stream execution and distributed training strategies
- Optimize model architecture for target accelerators

For in-depth technical architecture, see [DESIGN.md](DESIGN.md).

---

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.6+ with CUDA support
- A CUDA-capable GPU (required for tracing — FakeTensorMode needs CUDA)

### Installation

**From source (recommended for development):**

```bash
git clone https://github.com/AISysSim/SysSim.git
cd SysSim

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install SysSim with core dependencies (numpy, pandas, torch)
pip install -e .

# Or with optional extras
pip install -e ".[profiler]"       # + scikit-learn, xgboost (efficiency-model training)
pip install -e ".[huggingface]"    # + transformers
pip install -e ".[megatron]"       # + megatron-core
pip install -e ".[dev]"            # + pytest, ruff, pytest-cov
pip install -e ".[all]"            # everything
```

You must install the package itself (`pip install -e .`) — without it, the example scripts fail with `ModuleNotFoundError: No module named 'syssim'` when launched from outside the repo root.

### Basic Usage

```python
from syssim import trace_model_for_training, HardwareInfo, SimulatorConfig
import torch.nn as nn
import torch

# Define hardware specs
hw = HardwareInfo(
    peak_tflops_mm=1979.0,              # Tensor unit peak (TFLOP/s)
    peak_tflops_math=33.5,              # FP32 math peak (TFLOP/s)
    peak_memory_bandwidth_gbps=4900.0,  # Memory bandwidth (GB/s)
    peak_tflops_mm_conservative=535.0   # Conservative peak for small ops
)
config = SimulatorConfig(hw_info=hw)

# Model and inputs can be on CPU or meta device — tracer converts to fake CUDA internally
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
graph = trace_model_for_training(model, torch.randn(32, 128), config)

# Analyze
print(graph.summary())
critical_path_time = graph.compute_critical_path()
print(f"Critical path: {critical_path_time:.6e} ms")

# Export to Graphviz
with open("graph.dot", "w") as f:
    f.write(graph.to_dot())
```

### Auto-detect Hardware

If running on a supported GPU, skip manual hardware specs:

```python
from syssim import get_hardware_info, SimulatorConfig

hw_info, hw_name = get_hardware_info()  # auto-detects GPU
print(f"Detected: {hw_name}")
config = SimulatorConfig(hw_info=hw_info)
```

Supported hardware: GH200, H100, A100, V100, A40, RTX 4090, B100, B200, RTX PRO 6000 Blackwell, MI250, MI300. For unrecognized GPUs, either add an entry to `hw_database` in `syssim/config.py` or instantiate `HardwareInfo` directly.

---

## How It Works

SysSim uses a hybrid estimation approach:

1. **Trace** — intercepts PyTorch operations via `TorchDispatchMode` using fake CUDA tensors (no real computation)
2. **Classify** — categorizes each operation into one of 7 types: GEMM, ATTN, MATH, COLLECTIVE, MEMORY, BARRIER, STREAM_SYNC
3. **Estimate** — computes per-operator runtime:
   - *Compute ops*: `T_actual = T_roofline / efficiency`, where roofline gives the analytical ceiling and an ML model predicts real-world efficiency
   - *Collective ops*: event-driven network simulation with LogGP model and topology-aware bandwidth allocation
4. **Analyze** — builds a DAG with data and stream dependencies, computes critical path across multiple CUDA streams

---

## Examples

All examples require a CUDA-capable GPU.

### Basic Tracing — Diverse Operators

Traces a model with GEMM, ATTENTION, MATH ops in training, prefill, and decode modes:

```bash
python examples/trace_and_print.py
```

### Hugging Face — Qwen3-8B Single GPU

Traces a full Qwen3-8B training step (forward + backward) on a single GPU. Uses the published architecture with random weights built on the `meta` device (no download or weight allocation):

```bash
python examples/huggingface/train_qwen3_8b_single.py
```

Requires `transformers` (install via `pip install -e ".[huggingface]"` or pull in by `pip install -r requirements.txt`).

### Megatron-Core — GPT-3 1.3B Tensor Parallel (TP=4)

Traces a GPT-3 1.3B training step sharded across 4 tensor-parallel ranks. The model is built on the meta device (no real memory allocation). Runs on a single GPU — the script self-spawns 4 processes via `mp.spawn` using the `gloo` backend (SysSim uses FakeTensors so no multi-GPU hardware is required):

```bash
python examples/megatron/train_gpt_multi_gpu.py            # default TP=4
python examples/megatron/train_gpt_multi_gpu.py --tp-size 1
```

Requires `megatron-core` (install via `pip install -e ".[megatron]"`). On Slurm clusters wrap with `srun -N 1 --gpus 1 …`.

### Profiling

Train an efficiency model for a single operator type (`gemm`, `attn`, `rmsnorm`, `silu`, or `math`). The profiler runs the operator across many shapes, records measured time, and fits an MLP or XGBoost model that predicts arithmetic-intensity efficiency. Output `.pth` and `_data.csv` paths are user-chosen — make sure the parent directory exists first.

The profiler needs `scikit-learn` and `xgboost` on top of the core deps. Pull them in via the `profiler` extra:

```bash
pip install -e ".[profiler]"
```

Then run:

```bash
mkdir -p data/trained_models
python -m syssim.compute.compute_cost_profiler \
    --operator gemm \
    --output data/trained_models/gemm_gh200_mlp.pth \
    --backend mlp \
    --epochs 300
```

See `python -m syssim.compute.compute_cost_profiler --help` for the full set of flags (`--num-runs`, `--data-path` to reuse an existing CSV, `--checkpoint-interval`, etc.).

---

## Repository Structure

```
SysSim/
├── syssim/                              # Main package
│   ├── api.py                           # Public API
│   ├── config.py                        # HardwareInfo, SimulatorConfig, NetworkParams
│   ├── operator_graph.py                # DAG IR (operator types, critical path)
│   ├── tracer.py                        # TorchDispatchMode-based tracing
│   ├── compute/                         # Compute cost estimation
│   │   ├── compute_cost_predictor.py    # Roofline model
│   │   ├── compute_cost_profiler.py     # Profiler + ML model training
│   │   ├── efficiency_models.py         # ML efficiency model backends
│   │   ├── flop_counter.py              # FLOP counting registry
│   │   └── validate_profiler.py         # Profiler validation utilities
│   ├── network/                         # Network simulation
│   │   ├── simulator.py                 # Event-driven simulator
│   │   ├── collectives.py               # Collective op models
│   │   ├── topology.py                  # Topology models
│   │   ├── loggp.py                     # LogGP analytical model
│   │   ├── dag_builder.py               # Communication DAG builder
│   │   ├── device_mesh.py               # N-dimensional device mesh
│   │   ├── model_loader.py              # Network parameter loaders
│   │   ├── profiler.py                  # Network profiling helpers
│   │   ├── protocol_detector.py         # Protocol/transport detection
│   │   └── validation.py                # Network validation utilities
│   └── integrations/
│       └── huggingface.py               # HF Transformers training wrappers
├── examples/                            # Usage examples (basic, HF, Megatron, configs)
├── tests/                               # Test suite
├── data/
│   └── profiling/                       # Profiling CSVs (GEMM, ATTN, RMSNorm)
├── DESIGN.md                            # Technical design document
├── CONTRIBUTING.md                      # Contribution guide
├── CHANGELOG.md                         # Version history
├── pyproject.toml                       # Package metadata
└── requirements.txt
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

Check the [v0.1 Roadmap](https://github.com/AISysSim/SysSim/issues/1) for available tasks.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
