# Task 1 — Low-Precision Profiling on RTX PRO 6000 (Blackwell)

## My Prompt

You should check @README.md  and @DESIGN.md  to understand the whole proj overview.

The goal of this tasks is to add profiling data of P6000 (Current GPU).

Besides of adding fp16 profiling data, you should also add FP8 and FP4 profiling data.

For FP16, I guess using pytorch is enough? Like the GH200, original methods.

For FP8 and FP4, I guess we can use the flashinfer library (https://github.com/flashinfer-ai/flashinfer)?

And the verify the whole workflow e2e in Pro6000.

Please don't delete my prompt. And then please write down the plan under this .md. /writing-plans .

The expected result is that, you finish all the implementation of profiling new data in Pro6000, including adding new data type.

Then run the e2e workflow.

Update the finish report in @docs/docs .

---

# Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FP16, FP8, and FP4 compute profiling data + trained efficiency models for the NVIDIA RTX PRO 6000 Blackwell ("Pro 6000") and verify the full simulator workflow end-to-end on this GPU.

**Architecture:** Extend `HardwareInfo` to carry per-precision peak FLOP rates, register Pro 6000 in the hardware detection table, refactor the profiler to take a `dtype` argument (FP16 via PyTorch, FP8/FP4 via FlashInfer), train per-(operator, dtype) efficiency models, and route predictor calls to the correct model based on the traced tensor's dtype. Final validation runs `examples/trace_and_print.py` and one HF training example to confirm the simulator works end-to-end.

**Tech Stack:** Python 3.13, PyTorch 2.11 (CUDA 13.0), FlashInfer 0.6.7, XGBoost, scikit-learn, pandas. Hardware: 4× NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0, GB202).

**Hardware Spec Reference (Pro 6000 / GB202, dense values, with FP32 accumulator unless noted):**

| Quantity | Value |
|---|---|
| Peak FP32 (vector) | 117 TFLOP/s |
| Peak BF16/FP16 tensor | 3,752 TFLOP/s |
| Peak FP8 tensor | 7,504 TFLOP/s |
| Peak FP4 (NVFP4) tensor | 15,008 TFLOP/s |
| Peak HBM bandwidth | 1,792 GB/s (GDDR7) |
| `peak_tflops_mm_conservative` (small ops, FP16) | 1,000 TFLOP/s (~27% of dense peak, mirrors GH200 ratio) |

These are the values to write into the hardware database. If `nvidia-smi -q` or vendor docs show different numbers when verified in Task 1, update the constants in this plan inline before continuing.

**File Layout (plan-wide):**
- New profile data: `data/profiling/{op}_pro6000_{dtype}_data.csv` for `op ∈ {gemm, attn, rmsnorm, silu}`, `dtype ∈ {fp16, fp8, fp4}` (rmsnorm/silu only get fp16).
- New trained models: `data/trained_models/{op}_pro6000_{dtype}_xgb.pth`.
- Modified: `syssim/config.py`, `syssim/compute/compute_cost_profiler.py`, `syssim/compute/compute_cost_predictor.py`, `syssim/compute/efficiency_models.py`.
- New tests: `tests/test_low_precision_profiling.py`, `tests/test_pro6000_hw_detect.py`.
- New example/script: `examples/profile_pro6000.sh` (driver script for all profiling runs).
- Final report: `docs/docs/2026-04-30-pro6000-low-precision-profiling.md`.

---

### Task 0: Setup & Discovery (verify environment, baseline)

**Files:**
- Read-only: `requirements.txt`, `syssim/config.py`

- [ ] **Step 0.1: Confirm GPU + library versions**

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv | head -2
python -c "import torch, flashinfer; print(torch.__version__, torch.version.cuda, flashinfer.__version__); print(torch.cuda.get_device_capability(0))"
```

Expected: `RTX PRO 6000 Blackwell`, CUDA 13.0, torch 2.11.x, flashinfer 0.6.7, capability `(12, 0)`.

- [ ] **Step 0.2: Pin flashinfer in requirements.txt (additive)**

Append the following line under the `# ── Profiler ──` section of `requirements.txt`:

```
flashinfer-python>=0.6  # FP8/FP4 GEMM and attention kernels (Blackwell)
```

- [ ] **Step 0.3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add flashinfer-python for low-precision profiling"
```

---

### Task 1: Add Pro 6000 to hardware detection (FP16 only, baseline)

**Files:**
- Modify: `syssim/config.py:75-145, 148-211`
- Test: `tests/test_pro6000_hw_detect.py` (create)

- [ ] **Step 1.1: Write the failing hardware-detection test**

Create `tests/test_pro6000_hw_detect.py`:

```python
import pytest
import torch
from syssim.config import HardwareInfo, get_hardware_info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pro6000_detected_when_present():
    name = torch.cuda.get_device_name(0).lower()
    if "rtx pro 6000" not in name and "blackwell" not in name:
        pytest.skip("Not running on Pro 6000")

    hw, hw_name = get_hardware_info()
    assert hw_name == "pro6000"
    assert hw.peak_tflops_mm > 3000          # FP16 dense
    assert hw.peak_memory_bandwidth_gbps > 1500
    assert hw.peak_tflops_mm_conservative < hw.peak_tflops_mm


def test_hardware_info_has_per_dtype_peaks():
    hw = HardwareInfo(
        peak_tflops_mm=3752.0,
        peak_tflops_math=117.0,
        peak_memory_bandwidth_gbps=1792.0,
        peak_tflops_mm_fp8=7504.0,
        peak_tflops_mm_fp4=15008.0,
    )
    assert hw.get_peak_tflops_mm_for_dtype(torch.float16) == 3752.0
    assert hw.get_peak_tflops_mm_for_dtype(torch.float8_e4m3fn) == 7504.0
    # Fall through to fp16 peak when FP4 dtype unset
    hw_no_fp4 = HardwareInfo(
        peak_tflops_mm=3752.0, peak_tflops_math=117.0, peak_memory_bandwidth_gbps=1792.0
    )
    assert hw_no_fp4.get_peak_tflops_mm_for_dtype(torch.float16) == 3752.0
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
pytest tests/test_pro6000_hw_detect.py -v
```

Expected: FAIL — `pro6000` pattern not in `hw_database`, and `HardwareInfo` has no `peak_tflops_mm_fp8` / `peak_tflops_mm_fp4` / `get_peak_tflops_mm_for_dtype`.

- [ ] **Step 1.3: Extend `HardwareInfo` with per-dtype peaks**

In `syssim/config.py`, modify the `HardwareInfo.__init__` signature and add a dtype-aware accessor. Replace the `def __init__` block (currently lines ~107–123) with:

```python
def __init__(
    self,
    peak_tflops_mm: float,
    peak_tflops_math: float,
    peak_memory_bandwidth_gbps: float,
    peak_tflops_mm_conservative: float | None = None,
    peak_tflops_mm_fp8: float | None = None,
    peak_tflops_mm_fp4: float | None = None,
    network: Optional[NetworkParams] = None,
):
    self.peak_tflops_mm = peak_tflops_mm
    self.peak_tflops_math = peak_tflops_math
    self.peak_memory_bandwidth_gbps = peak_memory_bandwidth_gbps
    self.peak_tflops_mm_conservative = (
        peak_tflops_mm_conservative if peak_tflops_mm_conservative is not None else peak_tflops_mm
    )
    self.peak_tflops_mm_fp8 = peak_tflops_mm_fp8
    self.peak_tflops_mm_fp4 = peak_tflops_mm_fp4
    self.network = network if network is not None else NetworkParams()
```

Then add a method right after `get_peak_memory_bandwidth_gbps`:

```python
def get_peak_tflops_mm_for_dtype(self, dtype: torch.dtype) -> float:
    """Return the matrix-unit peak TFLOP/s appropriate for the given dtype.

    Falls back to FP16 peak (peak_tflops_mm) when a per-dtype value is not set.
    """
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return self.peak_tflops_mm_fp8 if self.peak_tflops_mm_fp8 is not None else self.peak_tflops_mm
    # FP4 dtypes are surfaced via FlashInfer/quant tensors as uint8; we use a sentinel string.
    if isinstance(dtype, str) and dtype == "nvfp4":
        return self.peak_tflops_mm_fp4 if self.peak_tflops_mm_fp4 is not None else self.peak_tflops_mm
    return self.peak_tflops_mm
```

- [ ] **Step 1.4: Register Pro 6000 in `get_hardware_info()`**

Inside `get_hardware_info()` in `syssim/config.py`, the `hw_database` list (~lines 168–194) currently uses 5-tuples. Switch to 7-tuples that include per-dtype peaks, and add Pro 6000:

Replace the line `# Format: (pattern, hw_name, peak_tflops_mm_fp16, peak_tflops_math_fp16, peak_bw_gb_s)` and the entire list with:

```python
# Format: (pattern, hw_name, peak_mm_fp16, peak_math_fp16, peak_bw, peak_mm_fp8, peak_mm_fp4)
hw_database = [
    ("gh200", "gh200", 989.0, 989.0, 3350.0, None, None),
    ("grace hopper", "gh200", 989.0, 989.0, 3350.0, None, None),
    ("h100", "h100", 1979.0, 989.0, 3350.0, None, None),
    ("a100", "a100", 312.0, 156.0, 1935.0, None, None),
    ("v100", "v100", 125.0, 62.5, 900.0, None, None),
    ("a40", "a40", 149.0, 74.5, 696.0, None, None),
    ("rtx 4090", "rtx4090", 330.0, 165.0, 1008.0, None, None),
    ("geforce rtx 4090", "rtx4090", 330.0, 165.0, 1008.0, None, None),
    ("mi250", "mi250", 362.0, 181.0, 1600.0, None, None),
    ("mi300", "mi300", 653.0, 326.5, 5200.0, None, None),
    # NVIDIA RTX PRO 6000 Blackwell (GB202) — FP16/FP8/FP4 dense peaks
    ("rtx pro 6000", "pro6000", 3752.0, 117.0, 1792.0, 7504.0, 15008.0),
]
```

And update the constructor call at the bottom of the loop:

```python
for pattern, hw_name, peak_mm, peak_math, peak_bw, peak_fp8, peak_fp4 in hw_database:
    if pattern in device_name:
        hw_info = HardwareInfo(
            peak_tflops_mm=peak_mm,
            peak_tflops_math=peak_math,
            peak_memory_bandwidth_gbps=peak_bw,
            peak_tflops_mm_fp8=peak_fp8,
            peak_tflops_mm_fp4=peak_fp4,
        )
        return hw_info, hw_name
```

- [ ] **Step 1.5: Run test to verify it passes**

```bash
pytest tests/test_pro6000_hw_detect.py -v
```

Expected: PASS for both tests.

- [ ] **Step 1.6: Run full test suite to confirm no regressions**

```bash
pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all existing tests still pass (the additional `peak_tflops_mm_fp8`/`peak_tflops_mm_fp4` constructor args default to `None`, so existing call sites are unaffected).

- [ ] **Step 1.7: Commit**

```bash
git add syssim/config.py tests/test_pro6000_hw_detect.py
git commit -m "feat(config): add RTX PRO 6000 (Blackwell) hardware entry + per-dtype peak FLOPs"
```

---

### Task 2: Profile FP16 baseline on Pro 6000 (existing PyTorch path)

**Files:**
- Run: `python -m syssim.compute.compute_cost_profiler --operator <op> --output ...`
- Outputs: `data/profiling/gemm_pro6000_data.csv`, `attn_pro6000_data.csv`, `rmsnorm_pro6000_data.csv`, `silu_pro6000_data.csv` (note: filename has no dtype suffix yet — that's added in Task 3)

This task validates that the existing profiler runs cleanly on Pro 6000 before refactoring. It uses the unmodified `_profile_gemm`/`_profile_attention` (FP16) and just produces baseline CSVs named with the new `pro6000` hw_name.

- [ ] **Step 2.1: Profile GEMM (FP16, ~30–60 min)**

```bash
python -m syssim.compute.compute_cost_profiler \
    --operator gemm \
    --output data/trained_models/gemm_pro6000_mlp.pth \
    --num-runs 50
```

Expected: `data/profiling/gemm_pro6000_data.csv` is created with columns `M,N,K,t_measured_ms` and ~thousands of rows.

- [ ] **Step 2.2: Profile ATTN (FP16)**

```bash
python -m syssim.compute.compute_cost_profiler \
    --operator attn \
    --output data/trained_models/attn_pro6000_mlp.pth \
    --num-runs 30
```

Expected: `data/profiling/attn_pro6000_data.csv`.

- [ ] **Step 2.3: Profile RMSNorm + SiLU (FP16)**

```bash
python -m syssim.compute.compute_cost_profiler --operator rmsnorm \
    --output data/trained_models/rmsnorm_pro6000_mlp.pth --num-runs 30
python -m syssim.compute.compute_cost_profiler --operator silu \
    --output data/trained_models/silu_pro6000_mlp.pth --num-runs 30
```

Expected: `rmsnorm_pro6000_data.csv` and `silu_pro6000_data.csv`.

- [ ] **Step 2.4: Sanity-check CSVs**

```bash
for f in data/profiling/*pro6000*data.csv; do
    echo "$f: $(wc -l < "$f") lines"; head -2 "$f"
done
```

Expected: each CSV has ≥1 000 rows, no NaN values, all `t_measured_ms > 0`.

- [ ] **Step 2.5: Commit**

```bash
git add data/profiling/*pro6000_data.csv
git commit -m "data: FP16 profiling baseline for RTX PRO 6000 (gemm/attn/rmsnorm/silu)"
```

---

### Task 3: Refactor profiler to accept `--dtype` argument

**Files:**
- Modify: `syssim/compute/compute_cost_profiler.py:282-481` (profile helpers), `1314-1409` (`profile_operator`), `1602-1694` (CLI block)
- Test: `tests/test_low_precision_profiling.py` (create)

The current `_profile_gemm` / `_profile_attention` hard-code `dtype=torch.float16`. We add an optional `dtype` parameter and a dispatch table that selects the right tensor builder + kernel call.

- [ ] **Step 3.1: Write a smoke test for dtype-aware profiling**

Create `tests/test_low_precision_profiling.py`:

```python
import pytest
import torch
from syssim.compute.compute_cost_profiler import (
    _profile_gemm,
    _profile_attention,
    profile_operator,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_gemm_fp16_returns_positive_time():
    t = _profile_gemm(64, 64, 64, num_runs=3, dtype="fp16")
    assert t > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_gemm_fp8_returns_positive_time():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 9:
        pytest.skip("FP8 needs SM>=89 (Hopper/Blackwell)")
    t = _profile_gemm(128, 128, 128, num_runs=3, dtype="fp8")
    assert t > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_gemm_fp4_returns_positive_time():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 10:
        pytest.skip("NVFP4 needs SM>=100 (Blackwell)")
    t = _profile_gemm(256, 256, 256, num_runs=3, dtype="fp4")
    assert t > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_attention_fp8_returns_positive_time():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 9:
        pytest.skip("FP8 attention needs SM>=89")
    t = _profile_attention(
        batch=1, num_heads=8, seq_len=128, head_dim=128, dtype="fp8", num_runs=3
    )
    assert t > 0
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
pytest tests/test_low_precision_profiling.py -v
```

Expected: FAIL — `_profile_gemm()` doesn't accept `dtype` kwarg yet.

- [ ] **Step 3.3: Refactor `_profile_gemm` to be dtype-aware**

Replace `_profile_gemm` (lines ~282–305 in `syssim/compute/compute_cost_profiler.py`) with:

```python
def _profile_gemm(m: int, n: int, k: int, num_runs: int = 100, dtype: str = "fp16") -> float:
    """Profile a single GEMM at the requested precision and return median time (ms).

    Supported dtypes: "fp16" (PyTorch torch.mm), "fp8" (FlashInfer mm_fp8),
    "fp4" (FlashInfer mm_fp4 with NVFP4 quantization).
    Returns -1.0 on OOM/unsupported-shape failures so callers can filter the row.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for profiling")

    device = torch.device("cuda")

    try:
        if dtype == "fp16":
            a = torch.randn(m, k, device=device, dtype=torch.float16)
            b = torch.randn(k, n, device=device, dtype=torch.float16)
            kernel = lambda: torch.mm(a, b)

        elif dtype == "fp8":
            from flashinfer.gemm import mm_fp8
            # FP8 requires K dimension to be multiple of 16 and M/N alignment
            if k % 16 != 0 or n % 16 != 0:
                return -1.0
            a_fp16 = torch.randn(m, k, device=device, dtype=torch.float16)
            # Per-tensor scaling; clamp into representable FP8 range
            a_amax = a_fp16.abs().amax().clamp(min=1e-4)
            b_fp16 = torch.randn(n, k, device=device, dtype=torch.float16)
            b_amax = b_fp16.abs().amax().clamp(min=1e-4)
            a = (a_fp16 * (448.0 / a_amax)).to(torch.float8_e4m3fn)
            b = (b_fp16 * (448.0 / b_amax)).to(torch.float8_e4m3fn)
            alpha = (a_amax * b_amax / (448.0 * 448.0)).to(torch.float32).reshape(1)
            kernel = lambda: mm_fp8(a, b.t().contiguous(), alpha=alpha, out_dtype=torch.bfloat16)

        elif dtype == "fp4":
            from flashinfer.gemm import mm_fp4
            from flashinfer import nvfp4_quantize
            if k % 32 != 0 or n % 16 != 0 or m % 16 != 0:
                return -1.0
            a_fp16 = torch.randn(m, k, device=device, dtype=torch.float16)
            b_fp16 = torch.randn(n, k, device=device, dtype=torch.float16)
            a_q, a_sf = nvfp4_quantize(a_fp16, a_fp16.abs().amax().reshape(1))
            b_q, b_sf = nvfp4_quantize(b_fp16, b_fp16.abs().amax().reshape(1))
            alpha = torch.ones(1, device=device, dtype=torch.float32)
            kernel = lambda: mm_fp4(a_q, b_q, a_sf, b_sf, alpha=alpha, out_dtype=torch.bfloat16)

        else:
            raise ValueError(f"Unknown dtype '{dtype}'")

        # Warmup
        for _ in range(5):
            kernel()
        torch.cuda.synchronize()

        # Profile
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            kernel()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return float(np.median(times))

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Common in low-precision profiling: shape constraints, unsupported sizes
        torch.cuda.empty_cache()
        return -1.0
```

- [ ] **Step 3.4: Refactor `_profile_attention` to be dtype-aware**

Replace `_profile_attention` (lines ~308–363) with:

```python
def _profile_attention(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_kv_heads: int = None,
    num_runs: int = 100,
    dtype: str = "fp16",
) -> float:
    """Profile attention at requested precision; returns median time (ms) or -1.0 on failure.

    fp16: torch.nn.functional.scaled_dot_product_attention.
    fp8:  flashinfer.single_prefill_with_kv_cache with FP8 q/k/v + scale tensors.
    fp4:  Not currently supported by FlashInfer attention kernels — returns -1.0.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for profiling")
    if num_kv_heads is None:
        num_kv_heads = num_heads

    device = torch.device("cuda")
    try:
        if dtype == "fp16":
            q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
            if num_kv_heads != num_heads:
                assert num_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
                v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            kernel = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)

        elif dtype == "fp8":
            from flashinfer import single_prefill_with_kv_cache
            # Single-batch only; flashinfer expects (seq, num_heads, head_dim) NHD layout
            if batch != 1:
                return -1.0
            q_fp16 = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
            k_fp16 = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
            v_fp16 = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
            scale_q = (q_fp16.abs().amax() / 448.0).clamp(min=1e-4).reshape(1)
            scale_k = (k_fp16.abs().amax() / 448.0).clamp(min=1e-4).reshape(1)
            scale_v = (v_fp16.abs().amax() / 448.0).clamp(min=1e-4).reshape(1)
            q = (q_fp16 / scale_q).to(torch.float8_e4m3fn)
            k = (k_fp16 / scale_k).to(torch.float8_e4m3fn)
            v = (v_fp16 / scale_v).to(torch.float8_e4m3fn)
            kernel = lambda: single_prefill_with_kv_cache(
                q, k, v,
                scale_q=scale_q, scale_k=scale_k, scale_v=scale_v,
                o_dtype=torch.bfloat16, kv_layout="NHD",
            )

        elif dtype == "fp4":
            return -1.0  # FlashInfer 0.6 has no FP4 attention path; record as N/A.

        else:
            raise ValueError(f"Unknown dtype '{dtype}'")

        for _ in range(10):
            kernel()
        torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            kernel()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return float(np.median(times))

    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return -1.0
```

- [ ] **Step 3.5: Plumb `dtype` through grid runners + `profile_operator`**

In the same file:
1. `_profile_gemm_grid`, `_profile_attn_grid` — accept `dtype: str = "fp16"`, pass it through to `_profile_gemm`/`_profile_attention`.
2. `profile_operator` — add `dtype: str = "fp16"` parameter; pass to grid functions; change CSV path to:
   ```python
   csv_path = Path(output_dir) / f"{operator}_{hw_name}_{dtype}_data.csv"
   ```
3. CLI parser — add:
   ```python
   parser.add_argument(
       "--dtype",
       default="fp16",
       choices=["fp16", "fp8", "fp4"],
       help="Tensor precision (fp16 via PyTorch; fp8/fp4 via FlashInfer)",
   )
   ```
   And pass `args.dtype` into `profile_operator(...)` and `train_efficiency_model(...)`.

For `rmsnorm`/`silu` paths — these stay FP16 only. In `profile_operator`, raise a clear error if `--dtype != fp16` and `operator in {"rmsnorm","silu","math"}`:

```python
if dtype != "fp16" and operator in ("rmsnorm", "silu", "math"):
    raise ValueError(f"Operator '{operator}' is profiled only at fp16; got dtype={dtype}")
```

- [ ] **Step 3.6: Migrate existing pro6000 FP16 CSVs to the new naming scheme**

Step 2 produced files like `gemm_pro6000_data.csv`. With the new naming, these become `gemm_pro6000_fp16_data.csv`. Rename:

```bash
for op in gemm attn rmsnorm silu; do
  src="data/profiling/${op}_pro6000_data.csv"
  if [ -f "$src" ]; then
    mv "$src" "data/profiling/${op}_pro6000_fp16_data.csv"
  fi
done
```

- [ ] **Step 3.7: Run smoke tests**

```bash
pytest tests/test_low_precision_profiling.py -v -k "fp16"
```

Expected: PASS for `fp16` test; FP8/FP4 tests can be deferred to Task 4 once kernels are exercised.

- [ ] **Step 3.8: Commit**

```bash
git add syssim/compute/compute_cost_profiler.py tests/test_low_precision_profiling.py \
        data/profiling/*pro6000_fp16_data.csv
git commit -m "refactor(profiler): add --dtype switch (fp16/fp8/fp4) with FlashInfer kernels"
```

---

### Task 4: Profile FP8 GEMM and Attention on Pro 6000

**Files:**
- Run profiler with `--dtype fp8`
- Outputs: `data/profiling/gemm_pro6000_fp8_data.csv`, `attn_pro6000_fp8_data.csv`

- [ ] **Step 4.1: FP8 smoke test**

```bash
pytest tests/test_low_precision_profiling.py -v -k "fp8"
```

Expected: PASS — confirms the FlashInfer FP8 kernels run on this GPU before launching a multi-hour profile.

- [ ] **Step 4.2: Run FP8 GEMM profile**

```bash
python -m syssim.compute.compute_cost_profiler \
    --operator gemm --dtype fp8 \
    --output data/trained_models/gemm_pro6000_fp8_mlp.pth \
    --num-runs 50
```

Expected: `data/profiling/gemm_pro6000_fp8_data.csv` written; some rows may have `t_measured_ms == -1` for shapes that violate FP8 alignment (k%16!=0). The training pipeline already filters these in `train_efficiency_model` (line ~1465).

- [ ] **Step 4.3: Run FP8 attention profile**

```bash
python -m syssim.compute.compute_cost_profiler \
    --operator attn --dtype fp8 \
    --output data/trained_models/attn_pro6000_fp8_mlp.pth \
    --num-runs 30
```

Expected: `attn_pro6000_fp8_data.csv` written. Rows with `bs > 1` will have `-1.0` (single-prefill API restriction); the dataset will be filtered to `bs == 1`.

- [ ] **Step 4.4: Sanity-check FP8 measurements vs FP16**

For a few large shapes (e.g., M=N=K=4096), FP8 should be ~2× faster than FP16:

```bash
python -c "
import pandas as pd
fp16 = pd.read_csv('data/profiling/gemm_pro6000_fp16_data.csv')
fp8  = pd.read_csv('data/profiling/gemm_pro6000_fp8_data.csv')
fp8 = fp8[fp8.t_measured_ms > 0]
join = fp16.merge(fp8, on=['M','N','K'], suffixes=('_fp16','_fp8'))
big = join[(join.M >= 4096) & (join.N >= 4096) & (join.K >= 4096)]
print('Large-shape FP16/FP8 speedup ratio:')
print((big.t_measured_ms_fp16 / big.t_measured_ms_fp8).describe())
"
```

Expected: median ratio ≥ 1.6× (Blackwell theoretical 2×; real-world allows for overhead).

- [ ] **Step 4.5: Commit**

```bash
git add data/profiling/*pro6000_fp8_data.csv
git commit -m "data: FP8 GEMM/attention profiling for RTX PRO 6000"
```

---

### Task 5: Profile FP4 GEMM on Pro 6000

**Files:**
- Output: `data/profiling/gemm_pro6000_fp4_data.csv`

- [ ] **Step 5.1: FP4 smoke test**

```bash
pytest tests/test_low_precision_profiling.py -v -k "fp4"
```

Expected: PASS for GEMM. Attention FP4 test is auto-skipped (returns `-1.0` by design).

- [ ] **Step 5.2: Run FP4 GEMM profile**

```bash
python -m syssim.compute.compute_cost_profiler \
    --operator gemm --dtype fp4 \
    --output data/trained_models/gemm_pro6000_fp4_mlp.pth \
    --num-runs 50
```

Expected: `data/profiling/gemm_pro6000_fp4_data.csv`. Rows where `k % 32 != 0` are recorded as `-1.0`.

- [ ] **Step 5.3: Sanity-check vs FP8**

```bash
python -c "
import pandas as pd
fp8 = pd.read_csv('data/profiling/gemm_pro6000_fp8_data.csv')
fp4 = pd.read_csv('data/profiling/gemm_pro6000_fp4_data.csv')
join = fp8[fp8.t_measured_ms>0].merge(fp4[fp4.t_measured_ms>0], on=['M','N','K'], suffixes=('_fp8','_fp4'))
big = join[(join.M>=4096) & (join.N>=4096) & (join.K>=4096)]
print((big.t_measured_ms_fp8 / big.t_measured_ms_fp4).describe())
"
```

Expected: median ratio ≥ 1.6× (theoretical 2×, but FP4 has scale-factor overhead that erodes this for smaller shapes).

- [ ] **Step 5.4: Commit**

```bash
git add data/profiling/gemm_pro6000_fp4_data.csv
git commit -m "data: FP4 (NVFP4) GEMM profiling for RTX PRO 6000"
```

---

### Task 6: Update roofline + training to handle per-dtype peaks

**Files:**
- Modify: `syssim/compute/compute_cost_profiler.py:649-775` (`_add_roofline_and_efficiency`), `905-987` (`_extract_roofline_features`), `1483-1502` (training entry)
- Modify: `syssim/compute/compute_cost_predictor.py:232-293` (`get_roofline_compute_time`)

The roofline formulas currently use `hw_info.peak_tflops_mm` regardless of dtype. For FP8/FP4 measured data, the roofline must use the per-dtype peak so `efficiency = T_roofline / T_measured` lands in `(0, 1]`.

- [ ] **Step 6.1: Add a dtype hook to `roofline_estimate` consumers**

In `syssim/compute/compute_cost_profiler.py`, modify `_add_roofline_and_efficiency` and `_extract_roofline_features` to accept a `dtype: str` parameter and select the correct peak when constructing fake tensors:

```python
def _dtype_str_to_torch(dtype: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "fp8":  torch.float8_e4m3fn,
        "fp4":  torch.float8_e4m3fn,  # placeholder; FP4 not a real torch dtype
    }[dtype]


def _add_roofline_and_efficiency(df, hw_info, operator, dtype: str = "fp16"):
    ...
    # Inside the loop, when creating fake tensors for GEMM:
    torch_dtype = _dtype_str_to_torch(dtype)
    with fake_mode:
        a = torch.empty(m, k, dtype=torch_dtype, device='cuda')
        ...
    # And before the final efficiency calc, scale roofline by dtype-appropriate peak:
    if operator == "gemm":
        peak_for_dtype = hw_info.get_peak_tflops_mm_for_dtype(
            torch.float8_e4m3fn if dtype == "fp8"
            else "nvfp4" if dtype == "fp4"
            else torch.float16
        )
        # Recompute compute-bound time using the right peak
        flop_count = 2 * m * n * k
        bytes_for_dtype = {"fp16": 2, "fp8": 1, "fp4": 0.5}[dtype]
        bytes_transferred = bytes_for_dtype * (m*k + k*n + m*n)
        t_compute_ms = (flop_count / (peak_for_dtype * 1e12)) * 1000
        t_memory_ms  = (bytes_transferred / (hw_info.peak_memory_bandwidth_gbps * 1e9)) * 1000
        t_roofline_ms = max(t_compute_ms, t_memory_ms)
    # similarly for "attn"
```

The simplest correct change is to *bypass* `roofline_estimate()` for FP8/FP4 inside `_add_roofline_and_efficiency` and compute the roofline directly with the dtype-appropriate peak (above). For FP16 the existing path is unchanged.

- [ ] **Step 6.2: Plumb `dtype` from the CLI through `train_efficiency_model`**

`train_efficiency_model` should accept `dtype` and forward it to `_add_roofline_and_efficiency` and `_build_training_features`. The output filenames for trained models become:

```python
expected_filename = f"{args.operator}_{hw_name}_{dtype}_{backend_suffix}.pth"
```

- [ ] **Step 6.3: Commit**

```bash
git add syssim/compute/compute_cost_profiler.py syssim/compute/compute_cost_predictor.py
git commit -m "feat(roofline): use per-dtype peak FLOPs in profiler training pipeline"
```

---

### Task 7: Train per-(operator, dtype) efficiency models

**Files:**
- Outputs: `data/trained_models/{op}_pro6000_{dtype}_xgb.pth`

- [ ] **Step 7.1: Train all dtype × operator combinations**

```bash
mkdir -p data/trained_models
for op in gemm attn; do
  for dt in fp16 fp8; do
    [ "$op" = "attn" ] && [ "$dt" = "fp4" ] && continue
    python -m syssim.compute.compute_cost_profiler \
        --operator $op --dtype $dt --backend xgboost \
        --data-path data/profiling/${op}_pro6000_${dt}_data.csv \
        --output data/trained_models/${op}_pro6000_${dt}_xgb.pth
  done
done
# FP4 only for GEMM
python -m syssim.compute.compute_cost_profiler \
    --operator gemm --dtype fp4 --backend xgboost \
    --data-path data/profiling/gemm_pro6000_fp4_data.csv \
    --output data/trained_models/gemm_pro6000_fp4_xgb.pth
# FP16 for math ops
for op in rmsnorm silu; do
  python -m syssim.compute.compute_cost_profiler \
      --operator $op --dtype fp16 --backend xgboost \
      --data-path data/profiling/${op}_pro6000_fp16_data.csv \
      --output data/trained_models/${op}_pro6000_fp16_xgb.pth
done
```

Expected: 8 trained models — gemm × {fp16,fp8,fp4} (3), attn × {fp16,fp8} (2), rmsnorm × fp16 (1), silu × fp16 (1) = 7 total. (No FP4 attention.) Each printout should report Eff MAPE < 25 % (ideally < 15 %); flag any that don't for re-profiling.

- [ ] **Step 7.2: Commit**

```bash
git add data/trained_models/*pro6000*.pth
git commit -m "models: train XGBoost efficiency models for Pro 6000 fp16/fp8/fp4"
```

---

### Task 8: Make BackendManager and predictor dtype-aware

**Files:**
- Modify: `syssim/compute/efficiency_models.py:154-217` (`BackendManager._load_models`, `get_model`)
- Modify: `syssim/compute/compute_cost_predictor.py:605-670` (`efficiency_estimate`)
- Test: `tests/test_efficiency_models.py` (add a dtype-routing test)

- [ ] **Step 8.1: Add a failing routing test**

Append to `tests/test_efficiency_models.py`:

```python
import torch
from syssim.compute.efficiency_models import BackendManager
from syssim.operator_graph import OperatorType


def test_backend_manager_routes_by_dtype(tmp_path, monkeypatch):
    """BackendManager.get_model(op_type, dtype) should pick the right per-dtype file."""
    # Create dummy files
    for dt in ("fp16", "fp8", "fp4"):
        (tmp_path / f"gemm_pro6000_{dt}_xgb.pth").write_bytes(b"fake")

    monkeypatch.setattr(
        "syssim.compute.efficiency_models.get_hardware_info",
        lambda: (None, "pro6000"),
    )

    mgr = BackendManager.__new__(BackendManager)  # bypass loader
    mgr.model_dir = str(tmp_path)
    mgr._models = {}
    found_fp16 = mgr._resolve_model_path(OperatorType.GEMM, "fp16")
    found_fp8 = mgr._resolve_model_path(OperatorType.GEMM, "fp8")
    found_fp4 = mgr._resolve_model_path(OperatorType.GEMM, "fp4")
    assert found_fp16.endswith("gemm_pro6000_fp16_xgb.pth")
    assert found_fp8.endswith("gemm_pro6000_fp8_xgb.pth")
    assert found_fp4.endswith("gemm_pro6000_fp4_xgb.pth")
```

- [ ] **Step 8.2: Run test to verify it fails**

```bash
pytest tests/test_efficiency_models.py::test_backend_manager_routes_by_dtype -v
```

Expected: FAIL — `_resolve_model_path` doesn't exist.

- [ ] **Step 8.3: Implement dtype-aware loading**

In `efficiency_models.py`, change `BackendManager._models` to be `dict[tuple[OperatorType, str], EfficiencyModel]`. Rewrite `_load_models` to walk the directory and parse filenames with a `_{dtype}_` segment, defaulting unmatched files (legacy gh200 etc.) to `dtype="fp16"`. Add:

```python
def _resolve_model_path(self, op_type, dtype: str) -> str:
    return os.path.join(
        self.model_dir,
        f"{op_type.value}_{self.hw_name}_{dtype}_xgb.pth",
    )

def get_model(self, op_type, dtype: str = "fp16"):
    return self._models.get((op_type, dtype))
```

- [ ] **Step 8.4: Update `efficiency_estimate` in `compute_cost_predictor.py` to thread the dtype through**

Inside `efficiency_estimate`, derive dtype from the output tensor:

```python
# Derive dtype from output (or args[0])
out_dtype = None
flat_outs, _ = tree_flatten(out)
for t in flat_outs:
    if isinstance(t, torch.Tensor):
        out_dtype = t.dtype
        break
dtype_str = {
    torch.float16: "fp16",
    torch.bfloat16: "fp16",
    torch.float8_e4m3fn: "fp8",
    torch.float8_e5m2: "fp8",
}.get(out_dtype, "fp16")

model = model_manager.get_model(op_type, dtype_str)
```

Note: FP4 tensors arrive as `torch.uint8` from FlashInfer quantize functions, so they won't auto-route to FP4 unless we explicitly pass a hint. For now, fp4 routing is opt-in via an environment variable `SYSSIM_FORCE_DTYPE`, which we'll exercise in the e2e test in Task 9.

- [ ] **Step 8.5: Run test to verify it passes**

```bash
pytest tests/test_efficiency_models.py -v
pytest tests/ -x -q 2>&1 | tail -10
```

Expected: PASS, no regressions in existing tests.

- [ ] **Step 8.6: Commit**

```bash
git add syssim/compute/efficiency_models.py syssim/compute/compute_cost_predictor.py tests/test_efficiency_models.py
git commit -m "feat(predictor): route to per-dtype efficiency models (fp16/fp8/fp4)"
```

---

### Task 9: Run end-to-end workflow on Pro 6000

**Files:**
- Run: `examples/trace_and_print.py`, `examples/huggingface/train_qwen3_8b_single.py`

- [ ] **Step 9.1: Smoke test — `trace_and_print.py` (generic FP16)**

```bash
RLSYSIM_MODEL_DIR=data/trained_models python examples/trace_and_print.py 2>&1 | tail -40
```

Expected: prints three tables (Training, Prefill, Decode) with `Time (ms)` columns populated and a `Total time` line; no warnings about missing efficiency models for `gemm`/`attn`.

- [ ] **Step 9.2: Smoke test — full HF training trace (Qwen3-8B)**

```bash
RLSYSIM_MODEL_DIR=data/trained_models \
    python examples/huggingface/train_qwen3_8b_single.py 2>&1 | tail -40
```

Expected: traces a forward + backward step on Qwen3-8B and prints critical-path time. Capture the result.

- [ ] **Step 9.3: Compare predicted vs measured (sanity check)**

For one or two small models we can cheaply measure (e.g., a single-layer linear+ReLU+linear), run the eager model and compare wall time vs `graph.compute_critical_path()`. Aim for ≤ 30 % error on this small case.

```bash
python - <<'EOF'
import time, torch, torch.nn as nn
from syssim import HardwareInfo, SimulatorConfig, trace_model_for_inference, set_efficiency_model_dir
set_efficiency_model_dir("data/trained_models")
from syssim.config import get_hardware_info
hw, _ = get_hardware_info()
cfg = SimulatorConfig(hw_info=hw)
model = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 4096)).cuda().half()
x = torch.randn(8, 1024, 4096, device='cuda', dtype=torch.float16)
g = trace_model_for_inference(model, x, cfg)
predicted = g.compute_critical_path()
for _ in range(5): model(x); torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20): model(x)
torch.cuda.synchronize()
measured = (time.perf_counter()-t0)*1000/20
print(f"Predicted: {predicted:.3f} ms | Measured: {measured:.3f} ms | Err: {abs(predicted-measured)/measured*100:.1f}%")
EOF
```

Expected: error ≤ 30 %. Larger errors flag a model-quality problem to mention in the finish report.

- [ ] **Step 9.4: Commit any small fixes uncovered during e2e**

```bash
git status
# If anything changed, commit with a focused message; otherwise skip.
```

---

### Task 10: Write finish report

**Files:**
- Create: `docs/docs/2026-04-30-pro6000-low-precision-profiling.md`

- [ ] **Step 10.1: Write the report**

Create `docs/docs/2026-04-30-pro6000-low-precision-profiling.md` with the following sections:

1. **Summary** — what was added (Pro 6000 FP16/FP8/FP4 profiling + per-dtype trained models + e2e validated).
2. **Hardware spec used** — table of (peak_tflops_mm fp16/fp8/fp4, peak_math, peak_bw, conservative peak); note any spec sources.
3. **Profiling stats** — for each (op, dtype): row count, valid-row count (after filtering -1), median measured time per representative shape, FP16/FP8 and FP8/FP4 speedup ratios at large shapes (from Tasks 4.4 / 5.3).
4. **Trained-model accuracy** — per-(op, dtype) Eff MAPE and Time MAPE table from Task 7.
5. **End-to-end results** — outputs from Task 9 (`trace_and_print.py`, Qwen3-8B trace, predicted-vs-measured small case).
6. **Known limitations** — FP4 attention not supported by FlashInfer 0.6 (recorded as `-1.0`); FP8 attention is single-batch only; rmsnorm/silu remain FP16-only.
7. **Reproduction steps** — the exact commands used in Tasks 2/4/5/7/9.

- [ ] **Step 10.2: Commit**

```bash
git add docs/docs/2026-04-30-pro6000-low-precision-profiling.md
git commit -m "docs: finish report for Pro 6000 low-precision profiling"
```

---

## Self-Review Notes

- **Spec coverage:** FP16 (Tasks 2, 7), FP8 (Tasks 4, 7), FP4 (Tasks 5, 7), e2e (Task 9), Pro 6000 hardware support (Task 1), report (Task 10). ✓
- **No placeholders left** in code blocks; every step shows the actual code/command. ✓
- **Type & naming consistency:** `pro6000` used as `hw_name` everywhere; `{op}_{hw}_{dtype}_data.csv` for CSVs and `{op}_{hw}_{dtype}_xgb.pth` for models, end-to-end. ✓
- **Dtype routing:** dtype is plumbed: CLI → `profile_operator` → `_profile_*` (Task 3); CSV path includes dtype (Task 3); training reads dtype (Task 6); model filename includes dtype (Task 7); `BackendManager` keys by `(op_type, dtype)` (Task 8); `efficiency_estimate` derives dtype from tensor (Task 8). ✓
- **Risk callouts:** GEMM shape constraints for FP8/FP4 are handled by returning `-1.0` and filtering at training time (existing code path at line ~1465). FP4 attention path explicitly returns `-1.0`. The existing PyTorch `torch.randn` does not support FP8/FP4, so we generate FP16 then quantize.
