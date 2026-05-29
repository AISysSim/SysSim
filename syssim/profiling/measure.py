"""Real-kernel measurement (GPU). The only component that runs real kernels."""
from __future__ import annotations

import statistics
from typing import Any

import torch
import torch.nn.functional as F

from ..operator_graph import OperatorType

_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}

# Norm and softmax kernels are launch-jitter-dominated at small sizes — a one-shot
# CUDA-event sample is non-physical there (non-monotonic in size, pinned near the launch
# floor). Batch this many back-to-back launches so the sample reflects steady-state
# per-kernel cost. GEMM/attention are large enough that one launch suffices.
_SMALL_INNER = 100


def median_of_reps(samples: list[float], warmup: int = 2) -> float:
    """Median over samples after discarding the first `warmup` (pure; testable)."""
    kept = samples[warmup:] if len(samples) > warmup else samples
    return float(statistics.median(kept))


def _gemm_kernel(a, b, dtype: str):
    """Run the right GEMM kernel for the dtype. bf16/fp16 -> `@`; fp8 -> _scaled_mm."""
    if dtype.startswith("fp8"):
        scale = torch.ones(1, device=a.device)
        return torch._scaled_mm(a, b.t().contiguous().t(), scale_a=scale, scale_b=scale,
                                out_dtype=torch.bfloat16)
    return a @ b


def measure_gemm(M: int, K: int, N: int, dtype: str, reps: int = 5,
                 warmup: int = 2, hw_info: Any = None) -> dict[str, Any]:
    """Time one MxKxN GEMM on the current CUDA device (median over reps) AND record
    the roofline anchor components (tensor_ns, mem_ns) via the SAME roofline code
    inference uses — guaranteeing the calibrate-time anchor == the inference-time
    anchor (train/inference parity)."""
    dt = _DTYPE[dtype]
    a = torch.randn(M, K, device="cuda").to(dt)
    b = torch.randn(K, N, device="cuda").to(dt)
    samples = []
    for _ in range(reps + warmup):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = _gemm_kernel(a, b, dtype)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e6)   # ms -> ns

    from ..compute.predictor.roofline import roofline
    from ..operator_graph import OperatorType
    if hw_info is None:
        from ..config import get_hardware_info
        hw_info, _ = get_hardware_info()
    aten = torch.ops.aten
    out = torch.empty(M, N, dtype=dt, device="cuda")
    bound = roofline(aten.mm, (a, b), {}, out, hw_info, OperatorType.GEMM)
    return {"M": M, "K": K, "N": N, "dtype": dtype,
            "latency_ns": median_of_reps(samples, warmup),
            "tensor_ns": bound.tensor_ns, "mem_ns": bound.mem_ns}


def _time_ns(fn, reps: int = 5, warmup: int = 2, inner: int = 1) -> float:
    """Median CUDA-event time (ns) of `fn` over reps, discarding the first `warmup`.

    `inner` back-to-back launches are timed per sample and the elapsed time is divided
    by `inner` to recover the per-kernel cost. A one-shot event measurement of a tiny
    memory-bound kernel (norm) is dominated by launch/queue jitter — a floor that swamps
    the true few-us work and is non-monotonic in size; batching amortizes it away.
    """
    samples = []
    for _ in range(reps + warmup):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1e6 / inner)   # ms -> ns, per launch
    return median_of_reps(samples, warmup)


def _anchor(func, args, kwargs, out, hw_info, op_type) -> dict:
    """Roofline anchor components via the SAME roofline inference uses
    (train/inference parity)."""
    from ..compute.predictor.roofline import roofline
    if hw_info is None:
        from ..config import get_hardware_info
        hw_info, _ = get_hardware_info()
    b = roofline(func, args, kwargs, out, hw_info, op_type)
    return {"tensor_ns": b.tensor_ns, "mem_ns": b.mem_ns,
            "fma_ns": b.fma_ns, "sfu_ns": b.sfu_ns}


def measure_attention(B, H_q, H_kv, S, D, causal, dtype, reps=5, warmup=2, hw_info=None):
    dt = _DTYPE[dtype]
    q = torch.randn(B, H_q, S, D, device="cuda", dtype=dt)
    k = torch.randn(B, H_kv, S, D, device="cuda", dtype=dt)
    v = torch.randn(B, H_kv, S, D, device="cuda", dtype=dt)
    gqa = H_kv != H_q
    lat = _time_ns(lambda: F.scaled_dot_product_attention(
        q, k, v, is_causal=bool(causal), enable_gqa=gqa), reps, warmup)
    anchor = _anchor(torch.ops.aten._scaled_dot_product_flash_attention,
                     (q, k, v, 0.0, bool(causal)), {}, q, hw_info, OperatorType.ATTN)
    return {"B": B, "H_q": H_q, "H_kv": H_kv, "S": S, "D": D, "causal": bool(causal),
            "dtype": dtype, "latency_ns": lat, **anchor}


def measure_norm(tokens, hidden, op_subtype, dtype, reps=5, warmup=2, hw_info=None):
    dt = _DTYPE[dtype]
    x = torch.randn(tokens, hidden, device="cuda", dtype=dt)
    w = torch.randn(hidden, device="cuda", dtype=dt)
    aten = torch.ops.aten
    if op_subtype == "rmsnorm":
        lat = _time_ns(lambda: F.rms_norm(x, (hidden,), w), reps, warmup, inner=_SMALL_INNER)
        func, fargs = aten.native_layer_norm, (x, [hidden], w, None, 1e-6)
    else:
        b = torch.randn(hidden, device="cuda", dtype=dt)
        lat = _time_ns(lambda: F.layer_norm(x, (hidden,), w, b), reps, warmup, inner=_SMALL_INNER)
        func, fargs = aten.native_layer_norm, (x, [hidden], w, b, 1e-5)
    anchor = _anchor(func, fargs, {}, x, hw_info, OperatorType.MATH)
    return {"tokens": tokens, "hidden": hidden, "op_subtype": op_subtype,
            "dtype": dtype, "latency_ns": lat, **anchor}


def measure_elementwise(total_elements, op_subtype, dtype, reps=5, warmup=2, hw_info=None):
    dt = _DTYPE[dtype]
    x = torch.randn(total_elements, device="cuda", dtype=dt)
    aten = torch.ops.aten
    if op_subtype == "gelu":
        fn, func, fargs = (lambda: F.gelu(x)), aten.gelu, (x,)
    elif op_subtype == "silu":
        fn, func, fargs = (lambda: F.silu(x)), aten.silu, (x,)
    elif op_subtype == "add":
        y = torch.randn(total_elements, device="cuda", dtype=dt)
        fn, func, fargs = (lambda: x + y), aten.add, (x, y)
    else:   # mul
        y = torch.randn(total_elements, device="cuda", dtype=dt)
        fn, func, fargs = (lambda: x * y), aten.mul, (x, y)
    lat = _time_ns(fn, reps, warmup, inner=_SMALL_INNER)
    anchor = _anchor(func, fargs, {}, x, hw_info, OperatorType.MATH)
    return {"total_elements": total_elements, "op_subtype": op_subtype,
            "dtype": dtype, "latency_ns": lat, **anchor}


def measure_reduction(B, H, S, dtype, reps=5, warmup=2, hw_info=None):
    dt = _DTYPE[dtype]
    x = torch.randn(B, H, S, S, device="cuda", dtype=dt)
    lat = _time_ns(lambda: F.softmax(x, dim=-1), reps, warmup, inner=_SMALL_INNER)
    anchor = _anchor(torch.ops.aten._softmax, (x, -1, False), {}, x, hw_info, OperatorType.MATH)
    return {"B": B, "H": H, "S": S, "op_subtype": "_softmax",
            "dtype": dtype, "latency_ns": lat, **anchor}


def _dispatch_measure(item):
    """Route a family-tagged work item to its measurer (tags the row with family)."""
    fam = item.get("family")
    if fam == "gemm":
        r = measure_gemm(item["M"], item["K"], item["N"], item["dtype"])
    elif fam == "attention":
        r = measure_attention(item["B"], item["H_q"], item["H_kv"], item["S"], item["D"],
                              item["causal"], item["dtype"])
    elif fam == "normalization":
        r = measure_norm(item["tokens"], item["hidden"], item["op_subtype"], item["dtype"])
    elif fam == "elementwise":
        r = measure_elementwise(item["total_elements"], item["op_subtype"], item["dtype"])
    elif fam == "reduction":
        r = measure_reduction(item["B"], item["H"], item["S"], item["dtype"])
    else:
        raise ValueError(f"unknown family: {fam!r}")
    r["family"] = fam
    return r


def _worker_loop(rank, ngpus, in_q, out_q):
    """Multi-GPU worker: pin to a device, drain the queue, emit a row or None per item."""
    torch.cuda.set_device(rank % ngpus)
    while True:
        item = in_q.get()
        if item is None:
            break
        try:
            out_q.put(_dispatch_measure(item))
        except Exception:
            out_q.put(None)   # skip marker (still counts toward expected)


def measure_worklist(items, num_workers: int = 1, runner=None) -> list:
    """Measure each family-tagged work item; per-item failures are skipped (not fatal).

    num_workers=1 -> sequential, in-process (CPU-testable via an injected `runner`).
    num_workers>1 -> a torch.multiprocessing (spawn) pool: each worker pins to GPU
    `rank % device_count` and pulls items from a shared queue. The real runner is
    `_dispatch_measure`; `runner` injection is for the sequential (test) path.
    """
    run = runner or _dispatch_measure
    if num_workers <= 1:
        rows = []
        for it in items:
            try:
                rows.append(run(it))
            except Exception:
                pass
        return rows
    import torch.multiprocessing as mp
    ngpus = max(1, torch.cuda.device_count())
    ctx = mp.get_context("spawn")
    in_q = ctx.Queue()
    out_q = ctx.Queue()
    for it in items:
        in_q.put(it)
    for _ in range(num_workers):
        in_q.put(None)
    procs = [ctx.Process(target=_worker_loop, args=(r, ngpus, in_q, out_q))
             for r in range(num_workers)]
    for p in procs:
        p.start()
    rows = []
    for _ in range(len(items)):
        msg = out_q.get()
        if msg is not None:
            rows.append(msg)
    for p in procs:
        p.join()
    return rows
