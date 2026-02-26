# SysSim - LLM Performance Simulator

**SysSim** that traces neural network execution to build a computational graph and estimate runtime using roofline models and ML-based efficiency prediction.

For in-depth technical architecture, see [DESIGN.md](DESIGN.md) 

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from syssim import trace_model_for_inference, HardwareInfo, SimulatorConfig
import torch.nn as nn
import torch

# Define hardware specs
hw = HardwareInfo(
    peak_tflops_mm=1979.0,              # True tensor unit peak (GH200)
    peak_tflops_math=33.5,              # FP32 math peak
    peak_memory_bandwidth_gbps=4900.0,  # GB/s
    peak_tflops_mm_conservative=535.0   # Conservative peak (small ops)
)
config = SimulatorConfig(hw_info=hw)

# Model and inputs can be on CPU or meta device — tracer converts to fake CUDA internally
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
graph = trace_model_for_inference(model, torch.randn(32, 128), config)

# Analyze
print(graph.summary())
critical_path_time = graph.compute_critical_path()
print(f"Critical path: {critical_path_time:.6e} ms")

# Export
with open("graph.dot", "w") as f:
    f.write(graph.to_dot())
```

---

## 📁 Repository Structure

```
syssim/
├── syssim/                         # Main package
│   ├── api.py                      # Public API
│   ├── config.py                   # HardwareInfo, SimulatorConfig
│   ├── operator_graph.py           # IR (7 operator types, DAG, critical path)
│   ├── tracer.py                   # TorchDispatchMode tracing
│   ├── compute/                    # Compute cost models
│   │   ├── compute_cost_predictor.py   # Roofline model
│   │   ├── compute_cost_profiler.py    # Unified profiler (MLP + XGBoost)
│   │   ├── efficiency_models.py        # ML efficiency models
│   │   └── flop_counter.py             # FLOP counting registry
│   ├── network/                    # Network simulation
│   │   ├── collectives.py          # Collective op models
│   │   ├── loggp.py                # LogGP network model
│   │   ├── simulator.py            # Network simulator
│   │   └── topology.py             # Device topology
│   └── integrations/
│       └── huggingface.py          # HF Transformers training helpers
├── examples/
│   ├── trace_and_print.py          # Basic tracing (GEMM/ATTN/MATH)
│   ├── configs/                    # Hardware mesh configs
│   ├── huggingface/
│   │   └── train_qwen3_8b_single.py    # Qwen3-8B, single GPU
│   └── megatron/
│       └── train_gpt_multi_gpu.py      # GPT-3 1.3B, TP=4
├── tests/                          # Test suite
├── data/
│   ├── profiling/                  # Profiling CSVs (GEMM, ATTN, MATH)
│   ├── trained_models/             # Trained efficiency models (.pth)
│   └── network_models/             # LogGP network parameter files
├── logs/                           # Project documentation and reports
└── requirements.txt
```

---

## 🧪 Examples

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

Traces a GPT-3 1.3B training step sharded across 4 tensor-parallel ranks. The model is built on the meta device (no real memory allocation). Runs on a single GPU — the script self-spawns 4 processes via `mp.spawn` using the `gloo` backend (syssim uses FakeTensors so no multi-GPU hardware is required):

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
