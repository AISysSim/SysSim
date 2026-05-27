# PLENA Integration into SysSim

## Overview

This document describes the integration of PLENA's cycle-level analytic model into SysSim's performance estimation pipeline. The integration enables accurate simulation of LLM workloads on the PLENA custom accelerator alongside existing GPU targets.

## Architecture

```
User API (trace_model_for_plena)
              │
              ▼
        PLENAEstimator
        (wraps PerfModel)
              │
              ▼
        Per-op cycle count
        → OperatorGraph
```

### Op-Level Integration

The integration maps traced PyTorch ops to PLENA instructions for general models. Uses `trace_model_for_plena()` to trace any PyTorch model and estimate per-operator cycles.

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

### 3. `tests/test_plena_backend.py`

Unit tests for PLENA backend (24 tests).

### 4. `tests/test_plena_integration.py`

Integration tests for op-level tracing (8 tests).

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
```

## Op-to-PLENA Mapping Strategy

| SysSim Op Type | PyTorch Ops | PLENA Method/Instruction |
|----------------|-------------|--------------------------|
| GEMM | mm, addmm, bmm, matmul, linear | `M_MM` instruction cycles |
| ATTN | _scaled_dot_product_* | `perf.flash_attention()` |
| MATH (norm) | layernorm, rmsnorm | `V_BASIC` × 6 |
| MATH (act) | relu, gelu, silu | `V_BASIC` × multiplier (1-6) |
| MATH (other) | add, mul, softmax | `V_BASIC` × element count |

## Usage Example

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
- `tests/test_plena_integration.py`: 8 passed

Run tests with:
```bash
python -m pytest tests/test_plena_backend.py tests/test_plena_integration.py -v
```

## Future Enhancements

1. **GQA Detection**: Automatically detect grouped-query attention configuration from traced models
2. **MoE Support**: Add op-level mapping for Mixture-of-Experts layers
3. **Multi-Device**: Support device parallelism in op-level estimation
4. **Sliding Window Attention**: Add support for sliding window attention patterns
