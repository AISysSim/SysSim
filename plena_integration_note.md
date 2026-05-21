# PLENA Integration into SysSim

## Overview

This document describes the integration of PLENA's cycle-level analytic model into SysSim's performance estimation pipeline. The integration enables accurate simulation of LLM workloads on the PLENA custom accelerator alongside existing GPU targets.

## Architecture

```
User API (trace_model_for_plena / estimate_plena_inference)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   Op-Level Path           Layer-Level Path
   (Approach A)            (Approach B)
        │                       │
        ▼                       ▼
  PLENAEstimator         PLENAModelConfig
  (wraps PerfModel)      + LLaMAModel
        │                       │
        ▼                       ▼
  Per-op cycle count     End-to-end inference
  → OperatorGraph        → PLENAInferenceResult
```

### Two Integration Approaches

- **Approach A (Op-Level)**: Maps traced PyTorch ops to PLENA instructions for general models. Uses `trace_model_for_plena()` to trace any PyTorch model and estimate per-operator cycles.

- **Approach B (Layer-Level)**: Uses PLENA's native `LLaMAModel` for optimized LLM estimation. Provides accurate prefill/decode timing, TPS, and TTFT metrics.

## Files Created

### 1. `syssim/config_plena.py`

Configuration helpers for PLENA integration.

```python
from syssim import PLENAConfig

# Auto-locate config files in PLENA_Simulator submodule
config = PLENAConfig.from_plena_submodule()

# Or specify paths explicitly
config = PLENAConfig(
    settings_path="path/to/plena_settings.toml",
    isa_lib_path="path/to/customISA_lib.json",
    frequency_hz=1e9,  # 1 GHz default
)
```

**Key Components:**
- `PLENAConfig`: Dataclass holding paths to `plena_settings.toml` and `customISA_lib.json`
- `PLENAConfig.from_plena_submodule()`: Auto-locate config files in PLENA_Simulator/
- `is_plena_hardware()`: Type check helper to distinguish PLENA from GPU hardware
- `get_plena_perf_model()`: Load PLENA PerfModel from configuration

### 2. `syssim/compute/plena_backend.py`

Core PLENA estimation backend with cycle-level modeling.

```python
from syssim.compute.plena_backend import PLENAEstimator

estimator = PLENAEstimator(config)
time_ms = estimator.estimate_runtime(func_packet, args, kwargs, out, op_type)
```

**Key Components:**
- `PLENAHardwareInfo`: Wraps PerfModel + frequency for cycle→time conversion
- `estimate_plena_cycles_for_gemm()`: Maps M,N,K to PLENA M_MM instruction cycles
- `estimate_plena_cycles_for_attention()`: Calls `perf.flash_attention()`
- `estimate_plena_cycles_for_math()`: Maps vector ops to V_BASIC cycles
- `PLENAEstimator`: Drop-in replacement for GPU roofline estimation

### 3. `syssim/integrations/plena.py`

High-level model estimation API for LLM workloads.

```python
from syssim import PLENAModelConfig, estimate_plena_inference

config = PLENAModelConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_kv_heads=8,
    num_hidden_layers=32,
    intermediate_size=14336,
    vocab_size=128256,
)

result = estimate_plena_inference(
    config, "plena_settings.toml", "customISA_lib.json",
    batch_size=1, input_seq_len=2048, output_seq_len=128,
)
print(f"TPS: {result.tps:.2f}, TTFT: {result.ttft_ms:.2f} ms")
```

**Key Components:**
- `PLENAModelConfig`: Model params (hidden_size, num_heads, etc.)
- `PLENAModelConfig.from_hf_config()`: Extract from HuggingFace model config
- `PLENAInferenceResult`: Timing results (prefill_ms, decode_ms, tps, ttft_ms)
- `estimate_plena_inference()`: Layer-level estimation using LLaMAModel
- `trace_hf_model_for_plena()`: One-liner for HF models

### 4. `tests/test_plena_backend.py`

Unit tests for PLENA backend (24 tests).

### 5. `tests/test_plena_integration.py`

Integration tests for model-level estimation (21 tests).

## Files Modified

### 1. `syssim/compute/compute_cost_predictor.py`

Added PLENA dispatch to `estimate_runtime()`:

```python
def estimate_runtime(..., plena_estimator: Optional[PLENAEstimator] = None) -> float:
    # NEW: PLENA backend path
    if plena_estimator is not None:
        return plena_estimator.estimate_runtime(
            func_packet, args, kwargs, out, op_type, execution_mode, cache_seq_len
        )

    # Existing GPU roofline path (unchanged)
    roofline_result = roofline_estimate(...)
    ...
```

### 2. `syssim/tracer.py`

Threaded `plena_estimator` through tracing:
- Added `plena_estimator` param to `_OperatorGraphTracerMode.__init__()`
- Passed to `estimate_runtime()` call
- Added param to `OperatorGraphTracer.__init__()`

### 3. `syssim/api.py`

Added new API function:

```python
def trace_model_for_plena(
    model: nn.Module,
    example_inputs: Any,
    plena_config: PLENAConfig,
    mode: str = "prefill",
) -> OperatorGraph:
    """Trace model with PLENA cycle-based estimation."""
```

### 4. `syssim/__init__.py`

Exported new APIs:

```python
from .config_plena import PLENAConfig, is_plena_hardware
from .api import trace_model_for_plena
from .integrations.plena import (
    PLENAModelConfig, PLENAInferenceResult,
    estimate_plena_inference, trace_hf_model_for_plena,
)
```

## Op-to-PLENA Mapping Strategy

| SysSim Op Type | PyTorch Ops | PLENA Method/Instruction |
|----------------|-------------|--------------------------|
| GEMM | mm, addmm, bmm, matmul, linear | `M_MM` instruction cycles |
| ATTN | _scaled_dot_product_* | `perf.flash_attention()` |
| MATH (norm) | layernorm, rmsnorm | `V_BASIC` × 6 |
| MATH (act) | relu, gelu, silu | `V_BASIC` × multiplier (1-6) |
| MATH (other) | add, mul, softmax | `V_BASIC` × element count |

## Usage Examples

### Op-Level Tracing

```python
from syssim import trace_model_for_plena, PLENAConfig
import torch.nn as nn

# Create model
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
)

# Load PLENA config
config = PLENAConfig.from_plena_submodule()

# Trace with PLENA estimation
graph = trace_model_for_plena(model, inputs, config, mode="prefill")
print(f"Critical path: {graph.compute_critical_path():.2f} ms")
print(graph.summary())
```

### Layer-Level Estimation (Optimized for LLMs)

```python
from syssim import PLENAModelConfig, estimate_plena_inference

# Define model architecture
config = PLENAModelConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_kv_heads=8,
    num_hidden_layers=32,
    intermediate_size=14336,
    vocab_size=128256,
)

# Run estimation
result = estimate_plena_inference(
    config,
    "PLENA_Simulator/plena_settings.toml",
    "PLENA_Simulator/analytic_models/performance/customISA_lib.json",
    batch_size=4,
    input_seq_len=2048,
    output_seq_len=128,
)

print(f"Prefill: {result.prefill_ms:.2f} ms")
print(f"Decode: {result.decode_ms:.2f} ms")
print(f"TTFT: {result.ttft_ms:.2f} ms")
print(f"TPS: {result.tps:.2f}")
```

### HuggingFace Model Integration

```python
from syssim import trace_hf_model_for_plena
from transformers import AutoModelForCausalLM

# Load HF model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# One-liner estimation
result = trace_hf_model_for_plena(
    model,
    batch_size=1,
    input_seq_len=2048,
    output_seq_len=128,
)
print(f"TTFT: {result.ttft_ms:.2f} ms, TPS: {result.tps:.2f}")
```

## Dependencies

- PLENA_Simulator submodule must be initialized:
  ```bash
  git submodule update --init
  ```
- Required files in PLENA_Simulator:
  - `plena_settings.toml`
  - `analytic_models/performance/customISA_lib.json`

## Test Results

All tests pass:
- `tests/test_plena_backend.py`: 24 passed
- `tests/test_plena_integration.py`: 21 passed

Run tests with:
```bash
python -m pytest tests/test_plena_backend.py tests/test_plena_integration.py -v
```

## Comparison with Standalone PLENA

The SysSim integration produces identical results to the standalone `llama_model.py`:

```python
# SysSim result matches standalone PLENA llama_model.py
assert syssim_result.ttft_ms == pytest.approx(standalone_ttft * 1000, rel=0.01)
assert syssim_result.tps == pytest.approx(standalone_tps, rel=0.01)
```

## Future Enhancements

1. **GQA Detection**: Automatically detect grouped-query attention configuration from traced models
2. **MoE Support**: Add op-level mapping for Mixture-of-Experts layers
3. **Multi-Device**: Support device parallelism in layer-level estimation
4. **Sliding Window Attention**: Add support for sliding window attention patterns
