# SysSim ARIA Tutorial Colab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single Google Colab notebook (`demo/aria_tutorial.ipynb`) that demonstrates SysSim's five headline capabilities for the ARIA tutorial on 2026-05-22, with supporting YAML configs and a pytest-based smoke test, all under `demo/`.

**Architecture:** Helpers (synth CSV, ConstantEstimator, `simulated_hardware()`) live in `demo/helpers.py` and are imported by both the notebook cells and `demo/smoke_test.py`. This keeps the notebook short and the logic testable. Pure-Python helpers are TDD'd locally on Mac; CUDA-dependent integration tests are run in Colab.

**Tech Stack:** Python 3.10+, SysSim (`syssim` package on `master`), pytest, pandas, matplotlib, Jupyter (.ipynb v4 schema). Colab T4 runtime (L4/A100 fallback if flashinfer wheel fails on sm_75).

**Spec:** [demo/design.md](./design.md)

**Branch:** `lexu/demo-notebook` (already checked out)

---

## Critical Implementation Detail: hardware auto-detection

SysSim's `get_hardware_info()` (in `syssim/config.py:200`) inspects `torch.cuda.get_device_name(0)` against a hardcoded database. Two consequences for this demo:

1. **T4 is not in the hardware database** — `get_hardware_info()` raises `RuntimeError`. `BackendManager._load_models` catches it silently (warns, no models loaded), so `simulate(...)` with explicit YAML still works on T4 (returns pure roofline). But `train_efficiency_model(...)` does NOT catch the error and will fail.
2. **Trained model file lookup is hw-name-driven** — `BackendManager` searches for `{op}_{hw_name}_{dtype}_xgb.pth`, where `hw_name` comes from `get_hardware_info()`. So a model trained on synthesized "MI300X" data must be saved with `hw_name="mi300x"` in the filename, and `get_hardware_info()` must return `"mi300x"` when the loader runs — otherwise the loader can't find the file.

**Fix:** a `simulated_hardware(name, peaks)` context manager in `demo/helpers.py` that monkey-patches `get_hardware_info()` in both `syssim.config` and `syssim.compute.compute_cost_profiler` (the latter has a module-local binding from `from ..config import get_hardware_info` at line 35). Training + predictor activation happens inside the context; outside, behavior reverts.

This is honest about what the demo shows: the *workflow* of training a predictor from JSON for a target hardware. In production you'd profile on the actual hardware and the auto-detection would naturally match.

---

## File Structure

| Path | Status | Purpose |
|---|---|---|
| `demo/design.md` | exists | Spec |
| `demo/plan.md` | this file | Implementation plan |
| `demo/configs/models/llama3-8b.yaml` | create | Llama-3-8B architecture |
| `demo/configs/hardware/mi300x.yaml` | create | AMD MI300X hardware |
| `demo/helpers.py` | create | Shared logic (synth CSV, ConstantEstimator, simulated_hardware); imported by notebook + tests |
| `demo/smoke_test.py` | create | Pytest tests; pure-Python tests run locally, CUDA-gated tests run in Colab |
| `demo/aria_tutorial.ipynb` | create | The notebook (5 sections + setup) |
| `demo/__init__.py` | create (empty) | Makes `demo` a package so smoke_test can `import demo.helpers` |

Nothing outside `demo/` is modified.

---

## Pre-flight verification

```bash
cd /Users/lexu/Projects/SysSim
git status -sb                          # expect: ## lexu/demo-notebook (clean except .DS_Store)
git log -1 --oneline                    # expect: ac4bd8e demo: design doc ...
ls demo/                                # expect: configs/  design.md  plan.md
```

---

## Task 1: Bootstrap helpers.py + smoke_test.py skeletons

**Files:**
- Create: `demo/__init__.py`
- Create: `demo/helpers.py`
- Create: `demo/smoke_test.py`

- [ ] **Step 1: Create empty `demo/__init__.py`**

```bash
touch /Users/lexu/Projects/SysSim/demo/__init__.py
```

- [ ] **Step 2: Create `demo/helpers.py` with a marker function**

```python
"""Shared helpers for the ARIA tutorial Colab notebook.

Imported by demo/aria_tutorial.ipynb AND demo/smoke_test.py so the notebook
cells stay short and the logic is independently testable.
"""

from __future__ import annotations


def helpers_loaded() -> str:
    return "ok"
```

- [ ] **Step 3: Create `demo/smoke_test.py` with one passing test**

```python
"""Smoke tests for the ARIA tutorial Colab.

Pure-Python tests run anywhere. CUDA-gated integration tests are marked
with @requires_cuda; they run in Colab.

Run locally:   pytest demo/smoke_test.py -v
Run in Colab:  !pytest demo/smoke_test.py -v
"""

from __future__ import annotations

import pytest

import demo.helpers as helpers


def test_helpers_importable():
    assert helpers.helpers_loaded() == "ok"
```

- [ ] **Step 4: Run pytest to confirm the skeleton works**

```bash
cd /Users/lexu/Projects/SysSim
python -m pytest demo/smoke_test.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add demo/__init__.py demo/helpers.py demo/smoke_test.py
git commit -m "demo: bootstrap helpers.py and smoke_test.py skeletons"
```

---

## Task 2: Llama-3-8B model YAML

**Files:**
- Create: `demo/configs/models/llama3-8b.yaml`
- Modify: `demo/smoke_test.py`

- [ ] **Step 1: Add failing test to `demo/smoke_test.py`**

Append:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_YAML = REPO_ROOT / "demo" / "configs" / "models" / "llama3-8b.yaml"


def test_llama3_8b_yaml_loads():
    from syssim.training.spec import load_model_yaml
    cfg = load_model_yaml(str(LLAMA_YAML))
    assert cfg.num_layers == 32
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.num_query_groups == 8       # GQA: 8 KV heads
    assert cfg.ffn_hidden_size == 14336
    assert cfg.vocab_size == 128256
    assert cfg.swiglu is True
    assert cfg.rope is True
    assert cfg.tie_word_embeddings is False
```

- [ ] **Step 2: Run test, confirm it fails**

```bash
python -m pytest demo/smoke_test.py::test_llama3_8b_yaml_loads -v
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Create the YAML**

Write `demo/configs/models/llama3-8b.yaml`:

```yaml
# Llama-3-8B architecture
# Source: Meta's published config (huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json)
num_layers: 32
hidden_size: 4096
num_attention_heads: 32
num_query_groups: 8           # GQA: 32 Q heads, 8 KV heads
ffn_hidden_size: 14336
seq_length: 8192
max_position_embeddings: 8192
vocab_size: 128256
swiglu: true
rope: true
rope_theta: 500000.0
tie_word_embeddings: false
rms_norm_eps: 1.0e-5
```

- [ ] **Step 4: Run test, confirm it passes**

```bash
python -m pytest demo/smoke_test.py::test_llama3_8b_yaml_loads -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add demo/configs/models/llama3-8b.yaml demo/smoke_test.py
git commit -m "demo: add Llama-3-8B model YAML + schema test"
```

---

## Task 3: MI300X hardware YAML

**Files:**
- Create: `demo/configs/hardware/mi300x.yaml`
- Modify: `demo/smoke_test.py`

- [ ] **Step 1: Add failing test**

Append to `demo/smoke_test.py`:

```python
MI300X_YAML = REPO_ROOT / "demo" / "configs" / "hardware" / "mi300x.yaml"


def test_mi300x_yaml_loads():
    from syssim.training.spec import load_hardware_yaml
    hw = load_hardware_yaml(str(MI300X_YAML))
    assert hw.peak_tflops_mm == 1307          # FP16 matrix peak (MI300X spec)
    assert hw.peak_tflops_mm_fp8 == 2615      # FP8 matrix peak (~2x FP16)
    assert hw.peak_memory_bandwidth_GBps == 5300   # HBM3 bandwidth
    assert hw.gpu_memory_GB == 192
    assert hw.gpus_per_node == 8
```

- [ ] **Step 2: Run test, confirm it fails**

```bash
python -m pytest demo/smoke_test.py::test_mi300x_yaml_loads -v
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Create the YAML**

Write `demo/configs/hardware/mi300x.yaml`:

```yaml
# AMD Instinct MI300X (CDNA3) — 8x GPU OAM platform
# Source: AMD public spec sheet
peak_tflops_mm: 1307            # FP16 matrix peak (TFLOP/s)
peak_tflops_math: 163.4         # FP16 vector peak (TFLOP/s)
peak_memory_bandwidth_GBps: 5300  # HBM3 peak bandwidth
peak_tflops_mm_fp8: 2615        # FP8 matrix peak (~2x FP16)

gpus_per_node: 8
gpu_memory_GB: 192              # HBM3 capacity per GPU
inter_node_bandwidth_GBps: 400  # 8x 400 Gbps NICs typical (Infiniband NDR)
inter_node_latency_us: 5

topology:
  type: simple
  num_nodes: 1
  intra_node_bandwidth_GBps: 896  # Infinity Fabric: 7 links x 128 GB/s
  inter_node_bandwidth_GBps: 400
```

- [ ] **Step 4: Run test, confirm it passes**

```bash
python -m pytest demo/smoke_test.py::test_mi300x_yaml_loads -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add demo/configs/hardware/mi300x.yaml demo/smoke_test.py
git commit -m "demo: add AMD MI300X hardware YAML + schema test"
```

---

## Task 4: Synth profiling CSV helpers

**Files:**
- Modify: `demo/helpers.py`
- Modify: `demo/smoke_test.py`

Generate plausible profiling data for §3 (AMD) and §4 (FP8). Schema matches what the trainer expects (verified from `data/profiling/gemm_pro6000_fp16_data.csv` columns):

- GEMM: `M,N,K,t_measured_ms`
- Attention: `bs,seq,nh,nkv,hd,t_measured_ms`
- RMSNorm: `seq,dim,t_measured_ms`

- [ ] **Step 1: Add failing tests**

Append to `demo/smoke_test.py`:

```python
import csv
import tempfile


def _read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def test_synthesize_gemm_csv_schema():
    with tempfile.TemporaryDirectory() as tmp:
        out = helpers.synthesize_gemm_csv(
            out_path=Path(tmp) / "gemm.csv",
            peak_tflops=1307, peak_bw_GBps=5300, dtype_bytes=2, seed=42,
        )
        fields, rows = _read_csv(out)
        assert fields == ["M", "N", "K", "t_measured_ms"]
        assert len(rows) >= 100
        for r in rows[:5]:
            assert float(r["t_measured_ms"]) > 0
            assert int(r["M"]) > 0 and int(r["N"]) > 0 and int(r["K"]) > 0


def test_synthesize_attn_csv_schema():
    with tempfile.TemporaryDirectory() as tmp:
        out = helpers.synthesize_attn_csv(
            out_path=Path(tmp) / "attn.csv",
            peak_tflops=1307, peak_bw_GBps=5300, dtype_bytes=2, seed=42,
        )
        fields, rows = _read_csv(out)
        assert fields == ["bs", "seq", "nh", "nkv", "hd", "t_measured_ms"]
        assert len(rows) >= 50


def test_synthesize_rmsnorm_csv_schema():
    with tempfile.TemporaryDirectory() as tmp:
        out = helpers.synthesize_rmsnorm_csv(
            out_path=Path(tmp) / "rmsnorm.csv",
            peak_bw_GBps=5300, dtype_bytes=2, seed=42,
        )
        fields, rows = _read_csv(out)
        assert fields == ["seq", "dim", "t_measured_ms"]
        assert len(rows) >= 30


def test_synthesize_gemm_csv_seed_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        a = helpers.synthesize_gemm_csv(Path(tmp) / "a.csv", 1307, 5300, 2, seed=42)
        b = helpers.synthesize_gemm_csv(Path(tmp) / "b.csv", 1307, 5300, 2, seed=42)
        assert a.read_text() == b.read_text()
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
python -m pytest demo/smoke_test.py -k synthesize -v
```

Expected: 4 failures (`AttributeError: module 'demo.helpers' has no attribute 'synthesize_gemm_csv'`).

- [ ] **Step 3: Implement helpers in `demo/helpers.py`**

Append:

```python
import csv
import random
from pathlib import Path


def _roofline_gemm_ms(m: int, n: int, k: int, peak_tflops: float,
                     peak_bw_GBps: float, dtype_bytes: int) -> float:
    """Roofline ms for an M×K @ K×N GEMM."""
    flops = 2.0 * m * n * k
    compute_s = flops / (peak_tflops * 1e12)
    bytes_moved = dtype_bytes * (m * k + k * n + m * n)
    memory_s = bytes_moved / (peak_bw_GBps * 1e9)
    return max(compute_s, memory_s) * 1000.0


def synthesize_gemm_csv(out_path: Path, peak_tflops: float, peak_bw_GBps: float,
                       dtype_bytes: int, seed: int = 42) -> Path:
    """Generate plausible GEMM profiling data.

    Measured time = roofline / (0.6 + 0.25 × U[0,1)) — gives the predictor
    a noisy-but-correlated target to learn.
    """
    rng = random.Random(seed)
    shapes = [
        (m, n, k)
        for m in (512, 1024, 2048, 4096)
        for n in (512, 1024, 2048, 4096, 8192)
        for k in (512, 1024, 2048, 4096)
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "t_measured_ms"])
        for m, n, k in shapes:
            t_roof = _roofline_gemm_ms(m, n, k, peak_tflops, peak_bw_GBps, dtype_bytes)
            efficiency = 0.6 + 0.25 * rng.random()
            w.writerow([m, n, k, f"{t_roof / efficiency:.6f}"])
    return out_path


def synthesize_attn_csv(out_path: Path, peak_tflops: float, peak_bw_GBps: float,
                       dtype_bytes: int, seed: int = 42) -> Path:
    """Generate plausible attention (SDPA) profiling data."""
    rng = random.Random(seed)
    cases = [
        (bs, seq, nh, nkv, hd)
        for bs in (1, 2)
        for seq in (1024, 2048, 4096, 8192)
        for nh, nkv in ((32, 8), (32, 32))
        for hd in (64, 128)
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bs", "seq", "nh", "nkv", "hd", "t_measured_ms"])
        for bs, seq, nh, nkv, hd in cases:
            flops = 4.0 * bs * nh * seq * seq * hd
            compute_s = flops / (peak_tflops * 1e12)
            bytes_moved = dtype_bytes * 2 * bs * (nh + nkv) * seq * hd
            memory_s = bytes_moved / (peak_bw_GBps * 1e9)
            t_roof = max(compute_s, memory_s) * 1000.0
            efficiency = 0.45 + 0.25 * rng.random()
            w.writerow([bs, seq, nh, nkv, hd, f"{t_roof / efficiency:.6f}"])
    return out_path


def synthesize_rmsnorm_csv(out_path: Path, peak_bw_GBps: float, dtype_bytes: int,
                          seed: int = 42) -> Path:
    """Generate plausible RMSNorm profiling data (memory-bound)."""
    rng = random.Random(seed)
    cases = [(seq, dim) for seq in (1024, 2048, 4096, 8192)
                       for dim in (2048, 4096, 8192)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "dim", "t_measured_ms"])
        for seq, dim in cases:
            bytes_moved = dtype_bytes * 2 * seq * dim
            t_roof = (bytes_moved / (peak_bw_GBps * 1e9)) * 1000.0
            efficiency = 0.55 + 0.25 * rng.random()
            w.writerow([seq, dim, f"{t_roof / efficiency:.6f}"])
    return out_path
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
python -m pytest demo/smoke_test.py -k synthesize -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add demo/helpers.py demo/smoke_test.py
git commit -m "demo: synth profiling CSV helpers + schema tests"
```

---

## Task 5: ConstantEstimator class

**Files:**
- Modify: `demo/helpers.py`
- Modify: `demo/smoke_test.py`

- [ ] **Step 1: Add failing tests**

Append to `demo/smoke_test.py`:

```python
def test_constant_estimator_protocol():
    from syssim.compute.estimator import Estimator
    est = helpers.ConstantEstimator(constant_ms=1.0)
    assert isinstance(est, Estimator)
    assert est.estimate_op(None, (), {}, None, None) == 1.0
    assert est.estimate_op("anything", (1, 2), {"a": 3}, None, None) == 1.0


def test_constant_estimator_custom_value():
    est = helpers.ConstantEstimator(constant_ms=2.5)
    assert est.estimate_op(None, (), {}, None, None) == 2.5
```

- [ ] **Step 2: Run, confirm fail**

```bash
python -m pytest demo/smoke_test.py -k constant_estimator -v
```

Expected: `AttributeError: module 'demo.helpers' has no attribute 'ConstantEstimator'`.

- [ ] **Step 3: Implement in `demo/helpers.py`**

Append:

```python
class ConstantEstimator:
    """Toy custom Estimator that returns a constant ms per operator.

    Implements syssim.compute.estimator.Estimator (Protocol). Used in §5
    of the notebook to demonstrate the estimator swap mechanism. The hook
    is HardwareConfig.estimator (see syssim/training/spec.py:203 and
    syssim/training/runner.py:374).

    Example:
        from syssim.training.spec import load_hardware_yaml
        hw = load_hardware_yaml("examples/configs/hardware/dgx_h100.yaml")
        hw.estimator = ConstantEstimator(constant_ms=1.0)
        report = syssim.simulate(model=..., hardware=hw, ...)
    """

    def __init__(self, constant_ms: float = 1.0):
        self.constant_ms = constant_ms

    def estimate_op(
        self, func_packet, args, kwargs, out, op_type,
        execution_mode=None, cache_seq_len: int = 0,
    ) -> float:
        return self.constant_ms
```

- [ ] **Step 4: Run, confirm pass**

```bash
python -m pytest demo/smoke_test.py -k constant_estimator -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add demo/helpers.py demo/smoke_test.py
git commit -m "demo: ConstantEstimator for §5 cost-model demo"
```

---

## Task 6: `simulated_hardware()` context manager

**Files:**
- Modify: `demo/helpers.py`
- Modify: `demo/smoke_test.py`

See "Critical Implementation Detail" at the top of this plan. We need to override `get_hardware_info()` for the duration of training + predictor activation, so that:
(a) `train_efficiency_model(...)` succeeds on Colab GPUs that aren't in SysSim's hw_database
(b) `BackendManager` looks for trained model files using OUR hw_name (e.g., "mi300x"), so it finds the files we just saved
(c) Roofline normalization inside `_add_roofline_and_efficiency` uses target hardware peaks, not the Colab GPU's

- [ ] **Step 1: Add failing tests to `demo/smoke_test.py`**

Append:

```python
MI300X_PEAKS = {
    "peak_tflops_mm": 1307.0,
    "peak_tflops_math": 163.4,
    "peak_memory_bandwidth_gbps": 5300.0,
    "peak_tflops_mm_fp8": 2615.0,
    "peak_tflops_mm_fp4": None,
}


def test_simulated_hardware_overrides_get_hardware_info():
    """Inside the context, syssim.config.get_hardware_info returns our HW."""
    import syssim.config as sc
    import syssim.compute.compute_cost_profiler as ccp

    orig_sc = sc.get_hardware_info
    orig_ccp = ccp.get_hardware_info

    with helpers.simulated_hardware("mi300x_test", MI300X_PEAKS) as (hw, name):
        assert name == "mi300x_test"
        assert hw.peak_tflops_mm == 1307.0
        # Patched in BOTH places — local binding in ccp matters
        hw_sc, name_sc = sc.get_hardware_info()
        hw_ccp, name_ccp = ccp.get_hardware_info()
        assert name_sc == "mi300x_test" and name_ccp == "mi300x_test"
        assert hw_sc.peak_tflops_mm == 1307.0
        assert hw_ccp.peak_tflops_mm == 1307.0

    # Restored after exit
    assert sc.get_hardware_info is orig_sc
    assert ccp.get_hardware_info is orig_ccp


def test_simulated_hardware_restores_on_exception():
    import syssim.config as sc
    orig = sc.get_hardware_info
    with pytest.raises(ValueError, match="intentional"):
        with helpers.simulated_hardware("x", MI300X_PEAKS):
            raise ValueError("intentional")
    assert sc.get_hardware_info is orig
```

- [ ] **Step 2: Run, confirm fail**

```bash
python -m pytest demo/smoke_test.py -k simulated_hardware -v
```

Expected: `AttributeError: module 'demo.helpers' has no attribute 'simulated_hardware'`.

- [ ] **Step 3: Implement in `demo/helpers.py`**

Append:

```python
from contextlib import contextmanager


@contextmanager
def simulated_hardware(name: str, peaks: dict):
    """Override SysSim's auto-detected hardware for the duration of the block.

    SysSim's get_hardware_info() inspects torch.cuda.get_device_name(0) against
    a hardcoded database (syssim/config.py:200). The Colab T4 isn't in that
    database, so training a predictor fails. Even on supported runtimes, a
    predictor trained on synthesized MI300X data ends up saved with the wrong
    hw_name in its filename (BackendManager looks up files by runtime hw_name).

    This patches get_hardware_info() in both syssim.config AND
    syssim.compute.compute_cost_profiler (which has a module-local binding
    from `from ..config import get_hardware_info` at line 35).

    Usage:
        with simulated_hardware("mi300x", {...}) as (hw, name):
            train_efficiency_model("gemm", csv, f"models/gemm_{name}_fp16_xgb.pth", ...)
            set_efficiency_model_dir("models/")
            report = syssim.simulate(...)   # predictor active inside context

    `peaks` keys (all required except the optional ones):
        peak_tflops_mm, peak_tflops_math, peak_memory_bandwidth_gbps,
        peak_tflops_mm_fp8 (optional), peak_tflops_mm_fp4 (optional).
    """
    import syssim.config as sc
    import syssim.compute.compute_cost_profiler as ccp
    from syssim.config import HardwareInfo

    hw_info = HardwareInfo(
        peak_tflops_mm=peaks["peak_tflops_mm"],
        peak_tflops_math=peaks["peak_tflops_math"],
        peak_memory_bandwidth_gbps=peaks["peak_memory_bandwidth_gbps"],
        peak_tflops_mm_fp8=peaks.get("peak_tflops_mm_fp8"),
        peak_tflops_mm_fp4=peaks.get("peak_tflops_mm_fp4"),
    )

    def _fake_get_hardware_info():
        return hw_info, name

    orig_sc = sc.get_hardware_info
    orig_ccp = ccp.get_hardware_info
    sc.get_hardware_info = _fake_get_hardware_info
    ccp.get_hardware_info = _fake_get_hardware_info
    try:
        yield hw_info, name
    finally:
        sc.get_hardware_info = orig_sc
        ccp.get_hardware_info = orig_ccp
```

- [ ] **Step 4: Run, confirm pass**

```bash
python -m pytest demo/smoke_test.py -k simulated_hardware -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add demo/helpers.py demo/smoke_test.py
git commit -m "demo: simulated_hardware() context manager for cross-HW training"
```

---

## Task 7: CUDA-gated integration tests

**Files:**
- Modify: `demo/smoke_test.py`

These tests exercise `syssim.simulate(...)` and `train_efficiency_model(...)`. They skip on Mac (no CUDA); they MUST pass in Colab before any notebook task is considered done. They are acceptance criteria for Tasks 8-12.

- [ ] **Step 1: Add CUDA-gated tests**

Append to `demo/smoke_test.py`:

```python
import os

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA (run in Colab)")

QWEN3_8B_YAML = REPO_ROOT / "examples" / "configs" / "models" / "qwen3-8b.yaml"
DGX_H100_YAML = REPO_ROOT / "examples" / "configs" / "hardware" / "dgx_h100.yaml"

H100_FP8_PEAKS = {
    "peak_tflops_mm": 1979.0,
    "peak_tflops_math": 989.0,
    "peak_memory_bandwidth_gbps": 3350.0,
    "peak_tflops_mm_fp8": 3958.0,
    "peak_tflops_mm_fp4": None,
}


@requires_cuda
def test_simulate_qwen3_8b_on_h100():
    """§1 smoke: simulate Qwen3-8B on H100."""
    import syssim
    r = syssim.simulate(
        model=str(QWEN3_8B_YAML),
        hardware=str(DGX_H100_YAML),
        parallelism=syssim.ParallelismConfig(tp=2, dp=4),
        training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    )
    assert r.step_time_ms > 0
    assert 0 < r.mfu < 1
    assert r.peak_memory_gb > 0


@requires_cuda
def test_simulate_llama3_8b_on_h100():
    """§1 smoke: simulate Llama-3-8B on H100 (validates demo/configs/models/llama3-8b.yaml end-to-end)."""
    import syssim
    r = syssim.simulate(
        model=str(LLAMA_YAML),
        hardware=str(DGX_H100_YAML),
        parallelism=syssim.ParallelismConfig(tp=2, dp=4),
        training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    )
    assert r.step_time_ms > 0


@requires_cuda
def test_simulate_mi300x_roofline():
    """§3a smoke: simulate on MI300X with default (no efficiency model) — pure roofline."""
    import syssim
    r = syssim.simulate(
        model=str(QWEN3_8B_YAML),
        hardware=str(MI300X_YAML),
        parallelism=syssim.ParallelismConfig(tp=8),
        training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    )
    assert r.step_time_ms > 0


@requires_cuda
def test_train_mi300x_predictor_and_swap():
    """§3c smoke: synth MI300X data → train predictors → set dir → simulate."""
    import syssim
    from syssim.compute.compute_cost_profiler import train_efficiency_model
    from syssim.api import set_efficiency_model_dir

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prof_dir = tmp / "profiling"
        prof_dir.mkdir()
        model_dir = tmp / "models"
        model_dir.mkdir()

        gemm_csv = helpers.synthesize_gemm_csv(prof_dir / "gemm.csv", 1307, 5300, 2)
        attn_csv = helpers.synthesize_attn_csv(prof_dir / "attn.csv", 1307, 5300, 2)
        rms_csv = helpers.synthesize_rmsnorm_csv(prof_dir / "rms.csv", 5300, 2)

        with helpers.simulated_hardware("mi300x", MI300X_PEAKS) as (_, hw_name):
            # Train ONE predictor per operator
            train_efficiency_model(
                "gemm", gemm_csv,
                str(model_dir / f"gemm_{hw_name}_fp16_xgb.pth"),
                backend="xgboost", dtype="fp16",
            )
            train_efficiency_model(
                "attn", attn_csv,
                str(model_dir / f"attn_{hw_name}_fp16_xgb.pth"),
                backend="xgboost", dtype="fp16",
            )
            train_efficiency_model(
                "rmsnorm", rms_csv,
                str(model_dir / f"rmsnorm_{hw_name}_fp16_xgb.pth"),
                backend="xgboost", dtype="fp16",
            )
            set_efficiency_model_dir(str(model_dir))

            r = syssim.simulate(
                model=str(QWEN3_8B_YAML),
                hardware=str(MI300X_YAML),
                parallelism=syssim.ParallelismConfig(tp=8),
                training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
            )
            assert r.step_time_ms > 0

        # Reset predictor outside context
        set_efficiency_model_dir("")


@requires_cuda
def test_train_h100_fp8_predictor_and_swap():
    """§4c smoke: same as above but for H100 FP8."""
    import syssim
    from syssim.compute.compute_cost_profiler import train_efficiency_model
    from syssim.api import set_efficiency_model_dir

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prof_dir = tmp / "profiling"
        prof_dir.mkdir()
        model_dir = tmp / "models"
        model_dir.mkdir()

        gemm_csv = helpers.synthesize_gemm_csv(prof_dir / "gemm.csv", 3958, 3350, 1)
        attn_csv = helpers.synthesize_attn_csv(prof_dir / "attn.csv", 3958, 3350, 1)
        rms_csv = helpers.synthesize_rmsnorm_csv(prof_dir / "rms.csv", 3350, 1)

        with helpers.simulated_hardware("h100", H100_FP8_PEAKS) as (_, hw_name):
            train_efficiency_model(
                "gemm", gemm_csv,
                str(model_dir / f"gemm_{hw_name}_fp8_xgb.pth"),
                backend="xgboost", dtype="fp8",
            )
            train_efficiency_model(
                "attn", attn_csv,
                str(model_dir / f"attn_{hw_name}_fp8_xgb.pth"),
                backend="xgboost", dtype="fp8",
            )
            train_efficiency_model(
                "rmsnorm", rms_csv,
                str(model_dir / f"rmsnorm_{hw_name}_fp8_xgb.pth"),
                backend="xgboost", dtype="fp8",
            )
            set_efficiency_model_dir(str(model_dir))
            os.environ["SYSSIM_FORCE_DTYPE"] = "fp8"
            try:
                r = syssim.simulate(
                    model=str(QWEN3_8B_YAML),
                    hardware=str(DGX_H100_YAML),
                    parallelism=syssim.ParallelismConfig(tp=2, dp=4),
                    training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="fp8"),
                )
                assert r.step_time_ms > 0
            finally:
                del os.environ["SYSSIM_FORCE_DTYPE"]
        set_efficiency_model_dir("")


@requires_cuda
def test_constant_estimator_in_simulate():
    """§5 smoke: plug ConstantEstimator into HardwareConfig, simulate."""
    import syssim
    from syssim.training.spec import load_hardware_yaml

    hw = load_hardware_yaml(str(DGX_H100_YAML))
    hw.estimator = helpers.ConstantEstimator(constant_ms=1.0)
    r = syssim.simulate(
        model=str(QWEN3_8B_YAML),
        hardware=hw,
        parallelism=syssim.ParallelismConfig(tp=2, dp=4),
        training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"),
    )
    assert r.step_time_ms > 0
```

- [ ] **Step 2: Run locally, confirm CUDA tests SKIP and pure-Python tests still pass**

```bash
python -m pytest demo/smoke_test.py -v
```

Expected: all pure-Python tests pass; all `@requires_cuda` tests show `SKIPPED [requires CUDA (run in Colab)]`.

- [ ] **Step 3: Commit**

```bash
git add demo/smoke_test.py
git commit -m "demo: CUDA-gated integration tests for simulate / predictor / estimator"
```

- [ ] **Step 4: Run the smoke test in Colab (T4 runtime)**

In a fresh Colab cell (don't need the notebook yet — bootstrap manually):

```python
!git config --global url."https://github.com/".insteadOf "git@github.com:"
!git clone -b lexu/demo-notebook --recurse-submodules https://github.com/AISysSim/SysSim.git
%cd SysSim
!pip install -q -e .
!pip install -q pytest
!python -m pytest demo/smoke_test.py -v
```

Expected: all tests pass — no skips, no failures.

**If anything fails**, this is where to learn ground truth before writing notebook cells. Common issues:
- `flashinfer-python` wheel install fails on T4 → switch to L4 runtime and re-run.
- `train_efficiency_model` raises on FP8 path → check `_add_roofline_and_efficiency`'s FP8 branch (line 825+); may need to pass extra arg or accept that FP8 training writes `_mlp.pth` instead of `_xgb.pth`. Adjust filename and re-test.
- Predictor file naming mismatch → enable DEBUG logging temporarily: `logging.getLogger("syssim.compute.efficiency_models").setLevel(logging.INFO)` to see which paths it's checking.

**Do not proceed to Task 8 until ALL CUDA tests pass in Colab.**

---

## Task 8: Notebook scaffold + setup cell + §1 Models

**Files:**
- Create: `demo/aria_tutorial.ipynb`

The notebook is a JSON file (Jupyter v4 schema). Author via `nbformat` from a one-shot Python script, then commit the `.ipynb`. Below is the cell content; the implementer materializes them as ipynb JSON.

- [ ] **Step 1: Generate the notebook with header + setup + §1 cells**

Use a throwaway Python script:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: header + badge
cells.append(nbf.v4.new_markdown_cell("""# SysSim — ARIA Tutorial (2026-05-22)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AISysSim/SysSim/blob/lexu/demo-notebook/demo/aria_tutorial.ipynb)

SysSim estimates the step time and peak memory of LLM training on hardware
you don't have, without running real computation. This notebook walks through
five demos:

1. **Models** — Qwen3-8B vs. Llama-3-8B (dense)
2. **Configs** — Batch / Seqlen / TP / PP sweeps
3. **GPU vendor** — AMD MI300X (roofline → trained predictor)
4. **Precision** — FP8 (roofline → trained predictor)
5. **Cost model** — modifying `estimate_runtime()`

**Before you run anything:** _Runtime → Change runtime type → T4 GPU_.
(If the install cell fails on T4, fall back to L4 or A100.)
"""))

# Cell 2: CUDA assertion
cells.append(nbf.v4.new_code_cell("""import torch
assert torch.cuda.is_available(), (
    "SysSim requires a GPU runtime. "
    "Runtime → Change runtime type → T4 GPU (or L4/A100), then re-run."
)
print(f"GPU: {torch.cuda.get_device_name(0)}")"""))

# Cell 3: install header
cells.append(nbf.v4.new_markdown_cell("## Install SysSim (~3-5 min on a fresh runtime)"))

# Cell 4: install
cells.append(nbf.v4.new_code_cell("""import os, subprocess

if not os.path.exists("SysSim"):
    subprocess.run(
        'git config --global url."https://github.com/".insteadOf "git@github.com:"',
        shell=True, check=True,
    )
    subprocess.run(
        "git clone -b lexu/demo-notebook --recurse-submodules "
        "https://github.com/AISysSim/SysSim.git",
        shell=True, check=True,
    )

%cd SysSim
!pip install -q -e .
import syssim
print(f"SysSim version: {syssim.__version__}")"""))

# Cell 5: import helpers
cells.append(nbf.v4.new_code_cell("""import sys
sys.path.insert(0, ".")
from demo import helpers
print("Helpers loaded:", helpers.helpers_loaded())"""))

# Cell 6: §1 header
cells.append(nbf.v4.new_markdown_cell("""## §1. Models — Qwen3-8B vs. Llama-3-8B

Same hardware (H100 DGX), same parallelism (TP=2, DP=4). The simulator
is architecture-aware — GQA group count, MLP ratio, RoPE settings all
flow through."""))

# Cell 7: load + show architectures
cells.append(nbf.v4.new_code_cell("""from syssim.training.spec import load_model_yaml

QWEN3 = "examples/configs/models/qwen3-8b.yaml"
LLAMA = "demo/configs/models/llama3-8b.yaml"

for path in (QWEN3, LLAMA):
    cfg = load_model_yaml(path)
    print(f"{path}:")
    print(f"  layers={cfg.num_layers}  hidden={cfg.hidden_size}  "
          f"heads={cfg.num_attention_heads} (GQA groups={cfg.num_query_groups})  "
          f"ffn={cfg.ffn_hidden_size}  vocab={cfg.vocab_size}")"""))

# Cell 8: simulate both, side-by-side
cells.append(nbf.v4.new_code_cell("""import syssim
import pandas as pd

HW = "examples/configs/hardware/dgx_h100.yaml"
PAR = syssim.ParallelismConfig(tp=2, dp=4)
TR = syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16")

rows = []
for name, path in [("Qwen3-8B", QWEN3), ("Llama-3-8B", LLAMA)]:
    r = syssim.simulate(model=path, hardware=HW, parallelism=PAR, training=TR)
    rows.append({
        "model": name,
        "step_time_ms": round(r.step_time_ms, 2),
        "forward_ms": round(r.forward_ms, 2),
        "backward_ms": round(r.backward_ms, 2),
        "mfu": round(r.mfu, 3),
        "peak_memory_gb": round(r.peak_memory_gb, 2),
    })
pd.DataFrame(rows)"""))

nb.cells = cells
nbf.write(nb, "demo/aria_tutorial.ipynb")
print("Wrote demo/aria_tutorial.ipynb")
```

Run it from repo root: `python /tmp/gen_notebook.py` (save the script anywhere outside repo).

- [ ] **Step 2: Validate JSON**

```bash
python -c "import nbformat; nb = nbformat.read('demo/aria_tutorial.ipynb', as_version=4); print(f'{len(nb.cells)} cells, schema v{nb.nbformat}.{nb.nbformat_minor}')"
```

Expected: `8 cells, schema v4.5`.

- [ ] **Step 3: Open in Colab and execute Cells 1-8 end-to-end**

URL: `https://colab.research.google.com/github/AISysSim/SysSim/blob/lexu/demo-notebook/demo/aria_tutorial.ipynb`

Expected: setup + install succeeds; §1 table shows two rows with sensible step_time / mfu / peak_memory.

- [ ] **Step 4: Commit**

```bash
git add demo/aria_tutorial.ipynb
git commit -m "demo: notebook scaffold + setup + §1 Models"
```

---

## Task 9: §2 Configs — Batch / Seqlen / TP / PP sweeps

**Files:**
- Modify: `demo/aria_tutorial.ipynb`

- [ ] **Step 1: Append §2 cells via nbformat**

Append script:

```python
import nbformat as nbf

nb = nbf.read("demo/aria_tutorial.ipynb", as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("""## §2. Configs — Batch / Seqlen / TP / PP

Hold model = Qwen3-8B and HW = H100 DGX fixed. Sweep one knob at a time."""))

nb.cells.append(nbf.v4.new_code_cell("""import matplotlib.pyplot as plt

def run_sweep(axis_label, axis_key, values):
    rows = []
    for v in values:
        par = syssim.ParallelismConfig(tp=2, dp=4)
        tr = syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16")
        if axis_key == "micro_batch":
            tr = syssim.TrainingConfig(micro_batch=v, global_batch=max(8, v*4), dtype="bf16")
        elif axis_key == "tp":
            par = syssim.ParallelismConfig(tp=v, dp=8 // v)
        elif axis_key == "pp":
            par = syssim.ParallelismConfig(pp=v, dp=8 // v)
        # NOTE: seq_length sweep below requires generating temp model YAMLs
        # because seq_length lives in the model YAML, not TrainingConfig.
        r = syssim.simulate(model=QWEN3, hardware=HW, parallelism=par, training=tr)
        rows.append({axis_label: v, "step_time_ms": round(r.step_time_ms, 2),
                     "peak_memory_gb": round(r.peak_memory_gb, 2), "mfu": round(r.mfu, 3)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([str(v) for v in df[axis_label]], df["step_time_ms"])
    ax.set_xlabel(axis_label); ax.set_ylabel("step_time_ms")
    ax.set_title(f"Sweep: {axis_label}")
    plt.show()
    return df

for label, key, vals in [
    ("micro_batch", "micro_batch", [1, 2, 4]),
    ("TP",          "tp",          [1, 2, 4]),
    ("PP",          "pp",          [1, 2, 4]),
]:
    print(f"\\n=== Sweep: {label} ===")
    display(run_sweep(label, key, vals))"""))

nb.cells.append(nbf.v4.new_code_cell("""# Seqlen sweep: generate temp model YAMLs with different seq_length values
import tempfile, yaml
from pathlib import Path

base_cfg = yaml.safe_load(open(QWEN3))
seq_rows = []
with tempfile.TemporaryDirectory() as tmp:
    for seq in (2048, 4096, 8192):
        cfg = dict(base_cfg); cfg["seq_length"] = seq
        path = Path(tmp) / f"qwen3-8b_seq{seq}.yaml"
        path.write_text(yaml.dump(cfg))
        r = syssim.simulate(model=str(path), hardware=HW,
                            parallelism=syssim.ParallelismConfig(tp=2, dp=4),
                            training=syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="bf16"))
        seq_rows.append({"seq_length": seq, "step_time_ms": round(r.step_time_ms, 2),
                         "mfu": round(r.mfu, 3)})

seq_df = pd.DataFrame(seq_rows)
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar([str(v) for v in seq_df["seq_length"]], seq_df["step_time_ms"])
ax.set_xlabel("seq_length"); ax.set_ylabel("step_time_ms"); ax.set_title("Sweep: seq_length")
plt.show()
seq_df"""))

nbf.write(nb, "demo/aria_tutorial.ipynb")
print(f"Notebook now has {len(nb.cells)} cells")
```

- [ ] **Step 2: Validate + run in Colab**

```bash
python -c "import nbformat; nb = nbformat.read('demo/aria_tutorial.ipynb', as_version=4); print(f'{len(nb.cells)} cells')"
```

Expected: 11 cells (8 from Task 8 + 3 added).

Run §2 cells in Colab. Expected: four bar charts (TP, PP, micro_batch, seq_length), each with a DataFrame.

- [ ] **Step 3: Commit**

```bash
git add demo/aria_tutorial.ipynb
git commit -m "demo: §2 Configs sweeps (batch, seqlen, TP, PP) with charts"
```

---

## Task 10: §3 AMD GPU vendor

**Files:**
- Modify: `demo/aria_tutorial.ipynb`

§3 uses `simulated_hardware()` to override SysSim's hw detection during training (see Critical Implementation Detail at the top of this plan).

- [ ] **Step 1: Append §3 cells**

```python
import nbformat as nbf

nb = nbf.read("demo/aria_tutorial.ipynb", as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("""## §3. GPU vendor — AMD MI300X (Mike)

Three sub-demos:
1. **Roofline-only** on `mi300x.yaml`
2. **Synthesize MI300X profiling CSVs** in-cell
3. **Train predictor → re-run**, compare Roofline vs. Trained

Because we're running on a Colab GPU that isn't MI300X, we use a
`simulated_hardware()` context manager (defined in `demo/helpers.py`)
to override SysSim's hardware auto-detection while training the predictor.
In production you'd profile on the actual MI300X and skip the override."""))

nb.cells.append(nbf.v4.new_code_cell("""# §3a: Roofline-only
MI300X = "demo/configs/hardware/mi300x.yaml"
PAR_MI = syssim.ParallelismConfig(tp=8)

print("=== MI300X — Roofline-only (no efficiency model) ===")
r_roof_mi = syssim.simulate(model=QWEN3, hardware=MI300X, parallelism=PAR_MI, training=TR)
print(f"step_time_ms={r_roof_mi.step_time_ms:.2f}  mfu={r_roof_mi.mfu:.3f}  "
      f"peak_mem_gb={r_roof_mi.peak_memory_gb:.2f}")
roofline_ms_mi = r_roof_mi.step_time_ms"""))

nb.cells.append(nbf.v4.new_code_cell("""# §3b: Synthesize MI300X profiling data
import tempfile
from pathlib import Path

MI300X_PEAKS = {
    "peak_tflops_mm": 1307.0, "peak_tflops_math": 163.4,
    "peak_memory_bandwidth_gbps": 5300.0, "peak_tflops_mm_fp8": 2615.0,
    "peak_tflops_mm_fp4": None,
}

PROF_DIR_MI = Path(tempfile.mkdtemp(prefix="mi300x_profiling_"))
helpers.synthesize_gemm_csv(PROF_DIR_MI / "gemm_mi300x_fp16_data.csv",
                            peak_tflops=1307, peak_bw_GBps=5300, dtype_bytes=2)
helpers.synthesize_attn_csv(PROF_DIR_MI / "attn_mi300x_fp16_data.csv",
                            peak_tflops=1307, peak_bw_GBps=5300, dtype_bytes=2)
helpers.synthesize_rmsnorm_csv(PROF_DIR_MI / "rmsnorm_mi300x_fp16_data.csv",
                               peak_bw_GBps=5300, dtype_bytes=2)
print(f"Wrote profiling CSVs to {PROF_DIR_MI}:")
for p in sorted(PROF_DIR_MI.glob("*.csv")):
    print(f"  {p.name}  ({p.stat().st_size} bytes)")"""))

nb.cells.append(nbf.v4.new_code_cell("""# §3c: Train predictor → swap → re-simulate
from syssim.compute.compute_cost_profiler import train_efficiency_model
from syssim.api import set_efficiency_model_dir

MODEL_DIR_MI = Path(tempfile.mkdtemp(prefix="mi300x_models_"))

with helpers.simulated_hardware("mi300x", MI300X_PEAKS) as (_, hw_name):
    train_efficiency_model(
        "gemm", PROF_DIR_MI / "gemm_mi300x_fp16_data.csv",
        str(MODEL_DIR_MI / f"gemm_{hw_name}_fp16_xgb.pth"),
        backend="xgboost", dtype="fp16",
    )
    train_efficiency_model(
        "attn", PROF_DIR_MI / "attn_mi300x_fp16_data.csv",
        str(MODEL_DIR_MI / f"attn_{hw_name}_fp16_xgb.pth"),
        backend="xgboost", dtype="fp16",
    )
    train_efficiency_model(
        "rmsnorm", PROF_DIR_MI / "rmsnorm_mi300x_fp16_data.csv",
        str(MODEL_DIR_MI / f"rmsnorm_{hw_name}_fp16_xgb.pth"),
        backend="xgboost", dtype="fp16",
    )
    set_efficiency_model_dir(str(MODEL_DIR_MI))

    r_trained_mi = syssim.simulate(model=QWEN3, hardware=MI300X,
                                   parallelism=PAR_MI, training=TR)
    trained_ms_mi = r_trained_mi.step_time_ms

# Reset for clean §4
set_efficiency_model_dir("")

pd.DataFrame([
    {"estimator": "Roofline-only", "step_time_ms": round(roofline_ms_mi, 2), "delta_pct": "—"},
    {"estimator": "Trained predictor (synth)",
     "step_time_ms": round(trained_ms_mi, 2),
     "delta_pct": f"{100 * (trained_ms_mi - roofline_ms_mi) / roofline_ms_mi:+.1f}%"},
])"""))

nbf.write(nb, "demo/aria_tutorial.ipynb")
print(f"Notebook now has {len(nb.cells)} cells")
```

- [ ] **Step 2: Run §3 in Colab, verify both numbers print and Δ% is non-trivial**

Expected: 2-row DataFrame, trained number differs from roofline by ~10-50%. If train_efficiency_model fails for one operator (e.g. rmsnorm trains as MLP not XGBoost), accept the resulting filename — update the per-op filename suffix from `_xgb.pth` to whatever the trainer actually wrote.

- [ ] **Step 3: Commit**

```bash
git add demo/aria_tutorial.ipynb
git commit -m "demo: §3 AMD vendor (roofline → synth → trained predictor)"
```

---

## Task 11: §4 Precision FP8

**Files:**
- Modify: `demo/aria_tutorial.ipynb`

- [ ] **Step 1: Append §4 cells**

```python
import nbformat as nbf

nb = nbf.read("demo/aria_tutorial.ipynb", as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("""## §4. Precision FP8 (Dayou)

Same workflow as §3, but the dimension is precision rather than vendor.
H100's `peak_tflops_mm_fp8` (3958) gives ~2× throughput vs bf16."""))

nb.cells.append(nbf.v4.new_code_cell("""# §4a: FP8 roofline
TR_FP8 = syssim.TrainingConfig(micro_batch=1, global_batch=8, dtype="fp8")

print("=== H100 FP8 — Roofline-only ===")
r_roof_fp8 = syssim.simulate(model=QWEN3, hardware=HW, parallelism=PAR, training=TR_FP8)
print(f"step_time_ms={r_roof_fp8.step_time_ms:.2f}  mfu={r_roof_fp8.mfu:.3f}  "
      f"peak_mem_gb={r_roof_fp8.peak_memory_gb:.2f}")
roofline_ms_fp8 = r_roof_fp8.step_time_ms"""))

nb.cells.append(nbf.v4.new_code_cell("""# §4b: Synthesize H100 FP8 profiling data (1 byte per FP8 element)
H100_FP8_PEAKS = {
    "peak_tflops_mm": 1979.0, "peak_tflops_math": 989.0,
    "peak_memory_bandwidth_gbps": 3350.0, "peak_tflops_mm_fp8": 3958.0,
    "peak_tflops_mm_fp4": None,
}

PROF_DIR_FP8 = Path(tempfile.mkdtemp(prefix="h100_fp8_profiling_"))
helpers.synthesize_gemm_csv(PROF_DIR_FP8 / "gemm_h100_fp8_data.csv",
                            peak_tflops=3958, peak_bw_GBps=3350, dtype_bytes=1)
helpers.synthesize_attn_csv(PROF_DIR_FP8 / "attn_h100_fp8_data.csv",
                            peak_tflops=3958, peak_bw_GBps=3350, dtype_bytes=1)
helpers.synthesize_rmsnorm_csv(PROF_DIR_FP8 / "rmsnorm_h100_fp8_data.csv",
                               peak_bw_GBps=3350, dtype_bytes=1)
print(f"Wrote FP8 profiling CSVs to {PROF_DIR_FP8}")"""))

nb.cells.append(nbf.v4.new_code_cell("""# §4c: Train FP8 predictor → swap → re-simulate
import os
MODEL_DIR_FP8 = Path(tempfile.mkdtemp(prefix="h100_fp8_models_"))

with helpers.simulated_hardware("h100", H100_FP8_PEAKS) as (_, hw_name):
    train_efficiency_model(
        "gemm", PROF_DIR_FP8 / "gemm_h100_fp8_data.csv",
        str(MODEL_DIR_FP8 / f"gemm_{hw_name}_fp8_xgb.pth"),
        backend="xgboost", dtype="fp8",
    )
    train_efficiency_model(
        "attn", PROF_DIR_FP8 / "attn_h100_fp8_data.csv",
        str(MODEL_DIR_FP8 / f"attn_{hw_name}_fp8_xgb.pth"),
        backend="xgboost", dtype="fp8",
    )
    train_efficiency_model(
        "rmsnorm", PROF_DIR_FP8 / "rmsnorm_h100_fp8_data.csv",
        str(MODEL_DIR_FP8 / f"rmsnorm_{hw_name}_fp8_xgb.pth"),
        backend="xgboost", dtype="fp8",
    )
    set_efficiency_model_dir(str(MODEL_DIR_FP8))
    os.environ["SYSSIM_FORCE_DTYPE"] = "fp8"
    try:
        r_trained_fp8 = syssim.simulate(model=QWEN3, hardware=HW,
                                        parallelism=PAR, training=TR_FP8)
        trained_ms_fp8 = r_trained_fp8.step_time_ms
    finally:
        del os.environ["SYSSIM_FORCE_DTYPE"]

set_efficiency_model_dir("")  # reset for §5

pd.DataFrame([
    {"estimator": "FP8 Roofline-only", "step_time_ms": round(roofline_ms_fp8, 2), "delta_pct": "—"},
    {"estimator": "FP8 Trained predictor (synth)",
     "step_time_ms": round(trained_ms_fp8, 2),
     "delta_pct": f"{100 * (trained_ms_fp8 - roofline_ms_fp8) / roofline_ms_fp8:+.1f}%"},
])"""))

nbf.write(nb, "demo/aria_tutorial.ipynb")
print(f"Notebook now has {len(nb.cells)} cells")
```

- [ ] **Step 2: Run §4 in Colab, verify FP8 numbers**

Expected: FP8 roofline step_time noticeably lower than §1 bf16 (~1.5-2× faster).

- [ ] **Step 3: Commit**

```bash
git add demo/aria_tutorial.ipynb
git commit -m "demo: §4 Precision FP8 (roofline → synth → trained predictor)"
```

---

## Task 12: §5 Cost model — modifying estimate_runtime()

**Files:**
- Modify: `demo/aria_tutorial.ipynb`

- [ ] **Step 1: Append §5 cells**

```python
import nbformat as nbf

nb = nbf.read("demo/aria_tutorial.ipynb", as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("""## §5. Cost model — modifying `estimate_runtime()` (Dayou)

SysSim's per-op estimator is a Protocol (`syssim.compute.estimator.Estimator`)
with one method: `estimate_op(...)`. The default is `RooflineEstimator`;
custom backends (e.g. PLENA at `syssim/external/plena/backend.py:282`)
implement the same protocol and slot in via `HardwareConfig.estimator`.

Below: a toy `ConstantEstimator` that returns 1ms per op."""))

nb.cells.append(nbf.v4.new_code_cell("""# The pluggable estimator protocol
import inspect
from syssim.compute.estimator import Estimator, RooflineEstimator
print(inspect.getsource(Estimator))"""))

nb.cells.append(nbf.v4.new_code_cell("""# Our toy estimator (defined once in demo/helpers.py)
print(inspect.getsource(helpers.ConstantEstimator))

# Load the H100 config and attach the custom estimator
from syssim.training.spec import load_hardware_yaml
hw_cfg = load_hardware_yaml(HW)
hw_cfg.estimator = helpers.ConstantEstimator(constant_ms=1.0)

r_const = syssim.simulate(model=QWEN3, hardware=hw_cfg, parallelism=PAR, training=TR)
print(f"With ConstantEstimator(1ms): step_time_ms = {r_const.step_time_ms:.2f}")
print(f"(Compare to §1 Qwen3-8B bf16 roofline.)")"""))

nb.cells.append(nbf.v4.new_markdown_cell("""For a real custom estimator, see [`syssim/external/plena/backend.py`](https://github.com/AISysSim/SysSim/blob/master/syssim/external/plena/backend.py)
— PLENA maps PyTorch ops to cycle-level performance on a custom accelerator
using the same `Estimator` protocol."""))

nbf.write(nb, "demo/aria_tutorial.ipynb")
print(f"Notebook now has {len(nb.cells)} cells")
```

- [ ] **Step 2: Run §5 in Colab**

Expected: prints the Protocol source, prints ConstantEstimator source, prints a step_time_ms ≈ (num_ops × 1ms).

- [ ] **Step 3: Commit**

```bash
git add demo/aria_tutorial.ipynb
git commit -m "demo: §5 Cost model — custom Estimator via ConstantEstimator"
```

---

## Task 13: End-to-end Colab run + polish + push

**Files:**
- Modify: `demo/aria_tutorial.ipynb` (small fixups only, if anything breaks)

- [ ] **Step 1: Restart Colab runtime and run all cells top-to-bottom**

In Colab: _Runtime → Disconnect and delete runtime_ → reconnect with T4 GPU → _Runtime → Run all_.

Watch for:
- Install cell completes without error (or fails on T4 → switch to L4, document in Cell 3 prose)
- Every section's output renders
- No leftover state between sections (§3 and §4 both reset via `set_efficiency_model_dir("")`)
- Total runtime in the 5-10 min range

- [ ] **Step 2: Run smoke test in the same Colab session as a final regression check**

```python
!python -m pytest demo/smoke_test.py -v
```

Expected: all tests pass (no skips).

- [ ] **Step 3: Fix anything that broke, one commit per fix**

For each issue:
- Adjust the offending cell in `demo/aria_tutorial.ipynb` (re-generate via nbformat or hand-edit JSON)
- Re-run that cell + all downstream
- `git commit -m "demo: fix <thing>"` — one commit per fix, not a batched "polish"

- [ ] **Step 4: Verify smoke test runs locally (skipping CUDA tests)**

```bash
cd /Users/lexu/Projects/SysSim
python -m pytest demo/smoke_test.py -v
```

Expected: pure-Python tests pass, CUDA tests SKIP.

- [ ] **Step 5: Push branch**

```bash
git status -sb
git log --oneline lexu/demo-notebook ^master   # expect ~12-14 commits
git push -u origin lexu/demo-notebook
```

Tell the user: "Branch pushed. Colab URL: https://colab.research.google.com/github/AISysSim/SysSim/blob/lexu/demo-notebook/demo/aria_tutorial.ipynb — Mike/Dayou ready to polish their sections."

---

## Acceptance Criteria

The implementation is complete when:

1. ✅ `demo/` contains: `design.md`, `plan.md`, `__init__.py`, `aria_tutorial.ipynb`, `helpers.py`, `smoke_test.py`, `configs/models/llama3-8b.yaml`, `configs/hardware/mi300x.yaml`.
2. ✅ `pytest demo/smoke_test.py -v` passes locally (pure-Python tests pass, CUDA tests skip).
3. ✅ `pytest demo/smoke_test.py -v` passes in Colab T4/L4/A100 (all pass, no skips).
4. ✅ The notebook runs top-to-bottom in a fresh Colab T4 runtime (or L4 if flashinfer issue) in 5-10 min.
5. ✅ Each of §1-§5 produces visible output (table, chart, or printed numbers).
6. ✅ Branch `lexu/demo-notebook` pushed to origin.
7. ✅ Open-in-Colab badge URL works.

## Known Gotchas

- **`flashinfer-python` on T4 (Risk R1):** if install fails, switch to L4/A100 and update Cell 1 prose. Don't try to pin flashinfer unless L4 also fails.
- **`train_efficiency_model` per-operator signature:** the actual signature is `train_efficiency_model(operator, csv_path, output_path, backend="xgboost", epochs=300, dtype="fp16")` — one call per operator type, single CSV per call, output is a single `.pth` file. Verified from `compute_cost_profiler.py:1613`.
- **Backend XGBoost vs MLP output filename:** trainer saves to whatever `output_path` you pass, but BackendManager looks for `{op}_{hw_name}_{dtype}_xgb.pth` (XGBoost) or `{op}_{hw_name}_{dtype}_mlp.pth` (MLP). Match the suffix to the chosen backend.
- **`simulated_hardware()` context scope:** training + activation MUST be inside the context, so that `train_efficiency_model` and `BackendManager._load_models` both see the patched hw. `simulate(...)` can be inside or outside — once `BackendManager` is constructed with the trained models, it's cached.
- **`set_efficiency_model_dir("")` reset:** creates a new BackendManager pointing at an empty path; `_load_models` warns and exits without loading; `efficiency_estimate` returns 1.0 (pure roofline). Confirmed from reading `BackendManager.__init__` and `_load_models`.
- **FP8 dtype routing (§4):** the trainer writes `{op}_{hw}_fp8_xgb.pth`; the lookup via `efficiency_estimate` derives dtype from the output tensor (`fp8_e4m3fn` → "fp8"). If auto-detection misses on a fresh simulate, `os.environ["SYSSIM_FORCE_DTYPE"] = "fp8"` forces routing.
- **Seq_length sweep in §2:** `seq_length` is in the model YAML, not `TrainingConfig`. The plan generates temp YAMLs per seq_length (Task 9). Verify `TrainingConfig` doesn't actually accept seq_length first — if it does, simplify.
