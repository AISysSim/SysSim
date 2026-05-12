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

# Install SysSim with core dependencies
pip install -e .

# Or with optional extras
pip install -e ".[profiler]"       # + pandas, scikit-learn, xgboost
pip install -e ".[huggingface]"    # + transformers
pip install -e ".[dev]"            # + pytest, ruff, pytest-cov
pip install -e ".[all]"            # everything
```

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

Supported hardware: GH200, H100, A100, V100, A40, RTX 4090, **RTX PRO 6000 (Blackwell)**, MI250, MI300. For unrecognized GPUs, either add an entry to `hw_database` in `syssim/config.py` or instantiate `HardwareInfo` directly.

On Blackwell-class hardware (RTX PRO 6000), `HardwareInfo` carries per-dtype tensor-unit peaks (`peak_tflops_mm`, `peak_tflops_mm_fp8`, `peak_tflops_mm_fp4`) and the roofline picks the right one per traced operator dtype.

---

## How It Works

SysSim uses a hybrid estimation approach:

1. **Trace** — intercepts PyTorch operations via `TorchDispatchMode` using fake CUDA tensors (no real computation)
2. **Classify** — categorizes each operation into one of 7 types: GEMM, ATTN, MATH, COLLECTIVE, MEMORY, BARRIER, STREAM_SYNC
3. **Estimate** — computes per-operator runtime:
   - *Compute ops*: `T_actual = T_roofline / efficiency`, where roofline gives the analytical ceiling and an ML model predicts real-world efficiency. Roofline FLOP/s and bytes-per-element are picked per-dtype, so FP8 and FP4 operators land on the correct peak.
   - *Collective ops*: event-driven network simulation with LogGP model and topology-aware bandwidth allocation
4. **Analyze** — builds a DAG with data and stream dependencies, computes critical path across multiple CUDA streams

Per-dtype efficiency models are routed by `(OperatorType, dtype)` in `BackendManager`. If a `(op, dtype)` model is missing — common for FP8 attention on consumer Blackwell today — the runtime falls back to the FP16 model rather than failing. See [docs/docs/2026-04-30-pro6000-low-precision-profiling.md](docs/docs/2026-04-30-pro6000-low-precision-profiling.md) for the end-to-end FP8/FP4 story.

---

## Examples

All examples require a CUDA-capable GPU.

### Basic Tracing — Diverse Operators

Traces a model with GEMM, ATTENTION, MATH ops in training, prefill, and decode modes:

```bash
python examples/trace_and_print.py
```

### Hugging Face — Qwen3-8B Single GPU

Traces a full Qwen3-8B training step (forward + backward) on a single GH200. Uses the published architecture with random weights (no download required):

```bash
python examples/huggingface/train_qwen3_8b_single.py
```

### Megatron-Core — GPT-3 1.3B Tensor Parallel (TP=4)

Traces a GPT-3 1.3B training step sharded across 4 tensor-parallel ranks. The model is built on the meta device (no real memory allocation). Runs on a single GPU — the script self-spawns 4 processes via `mp.spawn` using the `gloo` backend (SysSim uses FakeTensors so no multi-GPU hardware is required):

```bash
srun -N 1 --gpus 1 python examples/megatron/train_gpt_multi_gpu.py
```

### Profiling

```bash
# Profile all operators (~3-4 minutes on GH200)
./run_profiling.sh gh200

# Enhanced profiling with roofline features
python -m syssim.predictors.compute_cost_profiler \
    --operator gemm \
    --output data/trained_models/gemm_gh200_mlp.pth \
    --backend mlp \
    --epochs 300
```

### Low-Precision Profiling — FP8 / FP4 on RTX PRO 6000

Profile GEMM/attention/RMSNorm/SiLU across FP16, FP8, and FP4 on Blackwell, then train per-`(op, dtype)` XGBoost efficiency models. FP8 GEMM uses `torch._scaled_mm`, FP4 GEMM uses FlashInfer's NVFP4 path, and FP8 attention uses FlashInfer prefill (FP4 attention is not yet supported by FlashInfer 0.6 and falls back to the FP16 model at simulation time).

```bash
# Profile (~1 minute with reduced grid)
python examples/profile_pro6000.py --num-runs 15 \
    --gemm-samples 12 --attn-seq-samples 14 --math-samples 24

# Train per-(op, dtype) XGBoost models (~2 minutes)
python examples/train_pro6000_models.py

# End-to-end: trace a model with the per-dtype efficiency models loaded
RLSYSIM_MODEL_DIR=$(pwd)/data/trained_models \
    python examples/trace_and_print.py
```

Requires `flashinfer-python>=0.6` and `xgboost` (both in `requirements.txt`). See [docs/docs/2026-04-30-pro6000-low-precision-profiling.md](docs/docs/2026-04-30-pro6000-low-precision-profiling.md) for the reproduction recipe, MAPE numbers, and known FP8/FP4 limitations.

---

## Repository Structure

```
SysSim/
├── syssim/                         # Main package
│   ├── api.py                      # Public API (3 entry points)
│   ├── config.py                   # HardwareInfo, SimulatorConfig, NetworkParams
│   ├── operator_graph.py           # DAG IR (7 operator types, critical path)
│   ├── tracer.py                   # TorchDispatchMode-based tracing
│   ├── compute/                    # Compute cost estimation
│   │   ├── compute_cost_predictor.py   # Roofline model
│   │   ├── compute_cost_profiler.py    # Profiler + ML model training
│   │   ├── efficiency_models.py        # ML efficiency model backends
│   │   └── flop_counter.py             # FLOP counting registry
│   ├── network/                    # Network simulation
│   │   ├── simulator.py            # Event-driven simulator
│   │   ├── collectives.py          # 8 collective operations
│   │   ├── topology.py             # 5 topology models
│   │   ├── loggp.py                # LogGP analytical model
│   │   └── device_mesh.py          # N-dimensional device mesh
│   └── integrations/
│       └── huggingface.py          # HF Transformers training wrappers
├── examples/                       # Usage examples
│   ├── trace_and_print.py          # Basic tracing demo
│   ├── huggingface/                # HF Transformers (Qwen3-8B)
│   ├── megatron/                   # Megatron-Core TP example
│   ├── configs/                    # Mesh-based hierarchical profiling configs
│   ├── profile_pro6000.py          # FP16/FP8/FP4 profiling driver (Pro 6000)
│   └── train_pro6000_models.py     # Per-(op, dtype) XGBoost training driver
├── tests/                          # Test suite (~15k lines)
├── data/
│   ├── profiling/                  # Profiling CSVs (per-(op, hw, dtype))
│   └── trained_models/             # Trained efficiency models
├── docs/
│   ├── docs/                       # Engineering reports (e.g., Pro 6000 FP8/FP4 report)
│   └── tasks/                      # Task specs and planning docs
├── DESIGN.md                       # Technical design document
├── CONTRIBUTING.md                 # Contribution guide
├── CHANGELOG.md                    # Version history
└── pyproject.toml                  # Package metadata
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

Check the [v0.1 Roadmap](https://github.com/AISysSim/SysSim/issues/1) for available tasks.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
