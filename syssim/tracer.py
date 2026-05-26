"""OperatorGraphTracer: builds OperatorGraph from PyTorch model execution.

Uses TorchDispatchMode + FakeTensorMode to intercept operations without
running actual computation. Tracks tensor storage for aliasing, monkey-patches
CUDA events for cross-stream dependencies, and classifies operations into
OperatorType categories.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torch._subclasses import FakeTensorMode
from torch._subclasses.fake_tensor import DataDependentOutputException
from torch._subclasses.fake_tensor import FakeTensor as _FakeTensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils.module_tracker import ModuleTracker

from .compute.compute_cost_predictor import (
    _ATTN_OPS,
    _CREATE_OPS,
    _GEMM_OPS,
    _IGNORE_OPS,
    _VIEW_OPS,
)
from .config import ExecutionMode
from .operator_graph import (
    OperatorGraph,
    OperatorNode,
    OperatorType,
    TensorMeta,
)

log = logging.getLogger(__name__)

aten = torch.ops.aten

_TRACE_DEVICE = "cuda:0"

# ---------------------------------------------------------------------------
# Metadata ops: compared by string name to avoid import-time errors for ops
# that may not exist in all PyTorch versions.
# ---------------------------------------------------------------------------
_METADATA_PACKET_NAMES = frozenset(
    {
        "prim.device",
        "prim.dtype",
        "prim.layout",
        "aten.sym_size",
        "aten.sym_stride",
        "aten.sym_storage_offset",
        "aten.sym_numel",
        "aten.is_contiguous",
        "aten.is_strides_like_format",
        "aten.is_non_overlapping_and_dense",
        "aten.dim",
        "aten.size",
        "aten.stride",
        "aten.storage_offset",
        "aten.numel",
    }
)

# ---------------------------------------------------------------------------
# Fake-CUDA helpers
# ---------------------------------------------------------------------------


def _to_fake_device(
    tensor: torch.Tensor,
    fake_mode: FakeTensorMode,
    device: str,
) -> torch.Tensor:
    """Convert a real tensor to a fake tensor on *device* (e.g. ``cuda:0``).

    Creates an intermediate meta tensor, then wraps it as a FakeTensor that
    claims to live on *device*.  Must be called outside any active
    FakeTensorMode context to avoid double-wrapping.
    """
    meta = tensor.to(device="meta")
    return _FakeTensor(fake_mode, meta, torch.device(device))


def _convert_model_to_fake(
    model: nn.Module,
    fake_mode: FakeTensorMode,
    device: str,
) -> list[tuple[nn.Module, str, dict[str, torch.Tensor], bool]]:
    """Replace every parameter and buffer in *model* with fake CUDA tensors.

    Returns a *restore_log* that :func:`_restore_model` can use to undo the
    swap.  Each entry is ``(module, dict_name, original_dict, is_params)``.
    """
    restore_log: list[tuple[nn.Module, str, dict[str, torch.Tensor], bool]] = []
    for mod in model.modules():
        # --- parameters ---
        orig_params = dict(mod._parameters)
        new_params: dict[str, Optional[torch.Tensor]] = {}
        changed = False
        for k, v in orig_params.items():
            if v is not None:
                fake = _to_fake_device(v.data, fake_mode, device)
                if v.requires_grad:
                    fake.requires_grad_(True)
                new_params[k] = fake
                changed = True
            else:
                new_params[k] = None
        if changed:
            restore_log.append((mod, "_parameters", orig_params, True))
            mod._parameters = new_params  # type: ignore[assignment]

        # --- buffers ---
        orig_bufs = dict(mod._buffers)
        new_bufs: dict[str, Optional[torch.Tensor]] = {}
        changed = False
        for k, v in orig_bufs.items():
            if v is not None:
                new_bufs[k] = _to_fake_device(v.data, fake_mode, device)
                changed = True
            else:
                new_bufs[k] = None
        if changed:
            restore_log.append((mod, "_buffers", orig_bufs, False))
            mod._buffers = new_bufs  # type: ignore[assignment]
    return restore_log


def _restore_model(
    restore_log: list[tuple[nn.Module, str, dict[str, torch.Tensor], bool]],
) -> None:
    """Undo the parameter/buffer swap performed by :func:`_convert_model_to_fake`."""
    for mod, dict_name, orig_dict, _is_params in restore_log:
        setattr(mod, dict_name, orig_dict)


def _make_fake_inputs(
    inputs: Any,
    fake_mode: FakeTensorMode,
    device: str,
) -> "tuple | dict":
    """Normalize *inputs* to a tuple (or dict) and convert every tensor to fake CUDA."""

    def _convert(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return _to_fake_device(x, fake_mode, device)
        return x

    if isinstance(inputs, dict):
        return {k: tree_map(_convert, v) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    elif isinstance(inputs, (list, tuple)):
        inputs = tuple(inputs)
    else:
        inputs = (inputs,)
    return tuple(tree_map(_convert, inputs))


# ---------------------------------------------------------------------------
# Storage tracking
# ---------------------------------------------------------------------------


def _storage_key(tensor: torch.Tensor) -> int:
    """Return a key identifying the tensor's underlying storage.

    Uses id(untyped_storage()) which works for FakeTensors where data_ptr() == 0.
    Views share the same untyped_storage object, so aliases are correctly identified.
    """
    return id(tensor.untyped_storage())


class TensorStorageTracker:
    """Maps tensor storage to the producer operator name."""

    def __init__(self) -> None:
        self._storage_to_producer: dict[int, str] = {}

    def register_output(self, tensor: torch.Tensor, producer_name: str) -> None:
        if isinstance(tensor, torch.Tensor):
            key = _storage_key(tensor)
            self._storage_to_producer[key] = producer_name

    def register_outputs(self, pytree_val: Any, producer_name: str) -> None:
        flat, _ = tree_flatten(pytree_val)
        for val in flat:
            if isinstance(val, torch.Tensor):
                self.register_output(val, producer_name)

    def get_producer(self, tensor: torch.Tensor) -> Optional[str]:
        if isinstance(tensor, torch.Tensor):
            key = _storage_key(tensor)
            return self._storage_to_producer.get(key)
        return None

    def register_alias(self, alias: torch.Tensor, source: torch.Tensor) -> None:
        """Safety net: if view op creates a new storage object somehow, link it."""
        if isinstance(alias, torch.Tensor) and isinstance(source, torch.Tensor):
            src_key = _storage_key(source)
            producer = self._storage_to_producer.get(src_key)
            if producer is not None:
                alias_key = _storage_key(alias)
                self._storage_to_producer[alias_key] = producer


# ---------------------------------------------------------------------------
# CUDA Event tracking
# ---------------------------------------------------------------------------


class CUDAEventTracker:
    """Monkey-patches torch.cuda.Event to intercept record/wait for sync tracking."""

    def __init__(
        self,
        graph: OperatorGraph,
        last_op_on_stream: dict[int, str],
        op_counter: list[int],
    ) -> None:
        self._graph = graph
        self._last_op_on_stream = last_op_on_stream
        self._op_counter = op_counter
        self._event_to_stream: dict[int, int] = {}
        self._event_to_last_op: dict[int, str] = {}
        self._orig_record: Any = None
        self._orig_wait: Any = None
        self._orig_init: Any = None

    def install_hooks(self) -> None:
        tracker = self

        self._orig_record = torch.cuda.Event.record
        self._orig_wait = torch.cuda.Event.wait

        def patched_record(event_self, stream=None):
            stream_id = 0 if stream is None else getattr(stream, "stream_id", 0)
            event_id = id(event_self)
            tracker._event_to_stream[event_id] = stream_id
            last_op = tracker._last_op_on_stream.get(stream_id)
            if last_op is not None:
                tracker._event_to_last_op[event_id] = last_op
            # No node creation - just tracking

        def patched_wait(event_self, stream=None):
            stream_id = 0 if stream is None else getattr(stream, "stream_id", 0)
            event_id = id(event_self)
            src_stream = tracker._event_to_stream.get(event_id)
            src_last_op = tracker._event_to_last_op.get(event_id)

            idx = tracker._op_counter[0]
            tracker._op_counter[0] += 1
            name = f"stream_sync_{idx}"
            node = OperatorNode(
                name=name,
                op_type=OperatorType.STREAM_SYNC,
                stream_id=stream_id,
                config={"target_stream": src_stream},
            )
            last_op = tracker._last_op_on_stream.get(stream_id)
            if last_op is not None:
                node.stream_deps.append(last_op)
            if src_last_op is not None:
                node.data_deps.append(src_last_op)
            tracker._graph.add_operator(node)
            tracker._last_op_on_stream[stream_id] = name

        torch.cuda.Event.record = patched_record
        torch.cuda.Event.wait = patched_wait

    def remove_hooks(self) -> None:
        if self._orig_record is not None:
            torch.cuda.Event.record = self._orig_record
        if self._orig_wait is not None:
            torch.cuda.Event.wait = self._orig_wait


# ---------------------------------------------------------------------------
# Op classification
# ---------------------------------------------------------------------------


def _extract_gemm_config(args: tuple) -> dict[str, Any]:
    """Extract M, N, K from matrix multiply args."""
    config: dict[str, Any] = {}
    if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
        a, b = args[0], args[1]
        if a.dim() == 2 and b.dim() == 2:
            config["M"] = a.shape[0]
            config["K"] = a.shape[1]
            config["N"] = b.shape[1]
        elif a.dim() == 3 and b.dim() == 3:
            config["batch"] = a.shape[0]
            config["M"] = a.shape[1]
            config["K"] = a.shape[2]
            config["N"] = b.shape[2]
    # addmm: (bias, a, b)
    if len(args) >= 3 and isinstance(args[1], torch.Tensor) and isinstance(args[2], torch.Tensor):
        a, b = args[1], args[2]
        if a.dim() == 2 and b.dim() == 2:
            config["M"] = a.shape[0]
            config["K"] = a.shape[1]
            config["N"] = b.shape[1]
    return config


def _extract_attention_config(args: tuple) -> dict[str, Any]:
    """Extract attention config from SDPA-style args (query, key, value, ...)."""
    config: dict[str, Any] = {}
    if len(args) >= 3:
        q = args[0]
        if isinstance(q, torch.Tensor) and q.dim() == 4:
            config["batch"] = q.shape[0]
            config["num_heads"] = q.shape[1]
            config["seq_len"] = q.shape[2]
            config["head_dim"] = q.shape[3]
    return config


def _is_cross_device_copy(func_packet, args, kwargs) -> Optional[OperatorType]:
    """Detect cross-device copy (_to_copy or copy_ with different src/dst devices)."""
    if func_packet == aten._to_copy:
        src = args[0] if args and isinstance(args[0], torch.Tensor) else None
        dst_device = kwargs.get("device")
        if src is not None and dst_device is not None:
            if str(src.device) != str(dst_device):
                return OperatorType.MEMORY  # Single type for all copies
    return None


def _classify_op(
    func_packet: Any, func: Any, args: tuple, kwargs: dict
) -> tuple[Optional[OperatorType], dict[str, Any]]:
    """Classify a dispatched op into (OperatorType, config) or (None, {})."""
    func_name = str(func_packet) if func_packet is not None else ""

    # Collective ops
    if "c10d" in func_name:
        return OperatorType.COLLECTIVE, {}

    # Cross-device copy
    copy_type = _is_cross_device_copy(func_packet, args, kwargs)
    if copy_type is not None:
        config: dict[str, Any] = {}
        if args and isinstance(args[0], torch.Tensor):
            t = args[0]
            config["size_bytes"] = t.numel() * t.element_size()
            config["non_blocking"] = kwargs.get("non_blocking", False)
        return copy_type, config

    # GEMM (matrix multiply)
    if func_packet in _GEMM_OPS:
        return OperatorType.GEMM, _extract_gemm_config(args)

    # ATTENTION (scaled dot-product attention)
    if func_packet in _ATTN_OPS:
        return OperatorType.ATTN, _extract_attention_config(args)

    # Default: generic math/compute
    return OperatorType.MATH, {}


# ---------------------------------------------------------------------------
# TensorMeta helpers
# ---------------------------------------------------------------------------


def _make_tensor_meta(t: torch.Tensor) -> TensorMeta:
    return TensorMeta(
        shape=tuple(t.shape),
        dtype=str(t.dtype),
        device=str(t.device),
    )


def _collect_tensor_metas(pytree_val: Any) -> list[TensorMeta]:
    flat, _ = tree_flatten(pytree_val)
    return [_make_tensor_meta(v) for v in flat if isinstance(v, torch.Tensor)]


# ---------------------------------------------------------------------------
# _OperatorGraphTracerMode: the inner TorchDispatchMode
# ---------------------------------------------------------------------------


class _OperatorGraphTracerMode(TorchDispatchMode):
    """Intercepts PyTorch dispatched ops and builds an OperatorGraph."""

    def __init__(
        self,
        graph: OperatorGraph,
        storage_tracker: TensorStorageTracker,
        last_op_on_stream: dict[int, str],
        op_counter: list[int],
        hw_info: Any = None,
        execution_mode: Optional[ExecutionMode] = None,
        cache_seq_len: int = 0,
        plena_estimator: Any = None,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._storage = storage_tracker
        self._last_op_on_stream = last_op_on_stream
        self._op_counter = op_counter
        self._hw_info = hw_info
        self._execution_mode = execution_mode
        self._cache_seq_len = cache_seq_len
        self._plena_estimator = plena_estimator

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        func_packet = func._overloadpacket
        packet_name = str(func_packet)

        # 1. Metadata ops: pass through
        if packet_name in _METADATA_PACKET_NAMES:
            return func(*args, **kwargs)

        # 2. View ops: execute, track aliasing, no node
        if func_packet in _VIEW_OPS:
            out = func(*args, **kwargs)
            # Track aliasing: find the first tensor arg as source
            flat_args, _ = tree_flatten(args)
            source = None
            for a in flat_args:
                if isinstance(a, torch.Tensor):
                    source = a
                    break
            if source is not None:
                flat_out, _ = tree_flatten(out)
                for o in flat_out:
                    if isinstance(o, torch.Tensor):
                        self._storage.register_alias(o, source)
            return out

        # 3. Create ops: execute, zero-time node, register output
        if func_packet in _CREATE_OPS:
            out = func(*args, **kwargs)
            idx = self._op_counter[0]
            self._op_counter[0] += 1
            name = f"op_{idx}_{packet_name}"
            node = OperatorNode(
                name=name,
                op_type=OperatorType.MATH,
                estimated_time_ms=0.0,
                outputs=_collect_tensor_metas(out),
            )
            self._graph.add_operator(node)
            self._storage.register_outputs(out, name)
            self._last_op_on_stream[node.stream_id] = name
            return out

        # Execute the op — skip NCCL for collective ops (fake tensors crash real NCCL)
        if "c10d" in packet_name:
            # Megatron TP collectives are in-place on the first arg; return it as-is.
            out = args[0] if args else None
        else:
            try:
                out = func(*args, **kwargs)
            except DataDependentOutputException:
                # aten._local_scalar_dense (tensor.item()) — FakeTensorMode
                # cannot simulate data-dependent scalar ops. Return 0 as a
                # placeholder; this is safe because callers only use the value
                # for masking/slicing decisions, not for FLOP-relevant shapes.
                log.debug(f"DataDependentOutputException for {packet_name}, returning 0")
                return 0

        # 4. Classify the operation
        op_type, config = _classify_op(func_packet, func, args, kwargs)
        if op_type is None:
            # Unknown op -- still register output for dependency tracking
            return out

        # 5. Collect data dependencies from input tensors
        flat_args, _ = tree_flatten((args, kwargs))
        data_deps: list[str] = []
        seen_deps: set[str] = set()
        for a in flat_args:
            if isinstance(a, torch.Tensor):
                producer = self._storage.get_producer(a)
                if producer is not None and producer not in seen_deps:
                    data_deps.append(producer)
                    seen_deps.add(producer)

        # 6. Stream dependency (same-stream ordering)
        stream_id = 0  # Default stream
        stream_deps: list[str] = []
        prev_op = self._last_op_on_stream.get(stream_id)
        if prev_op is not None and prev_op not in seen_deps:
            stream_deps.append(prev_op)

        # 7. Estimate time using predictor (decoupled)
        estimated_time_ms = 0.0
        if (self._hw_info is not None or self._plena_estimator is not None) and func_packet not in _IGNORE_OPS:
            try:
                from .compute.compute_cost_predictor import estimate_runtime

                estimated_time_ms = estimate_runtime(
                    func_packet,
                    args,
                    kwargs,
                    out,
                    self._hw_info,
                    op_type,
                    execution_mode=self._execution_mode,
                    cache_seq_len=self._cache_seq_len,
                    plena_estimator=self._plena_estimator,
                )
            except Exception as e:
                log.debug(f"Runtime estimation failed for {packet_name}: {e}")
                estimated_time_ms = 0.0

        # 8. Create node
        idx = self._op_counter[0]
        self._op_counter[0] += 1
        name = f"op_{idx}_{packet_name}"

        node = OperatorNode(
            name=name,
            op_type=op_type,
            config=config,
            data_deps=data_deps,
            stream_deps=stream_deps,
            stream_id=stream_id,
            inputs=_collect_tensor_metas((args, kwargs)),
            outputs=_collect_tensor_metas(out),
            estimated_time_ms=estimated_time_ms,
        )

        self._graph.add_operator(node)
        self._storage.register_outputs(out, name)
        self._last_op_on_stream[stream_id] = name

        return out


# ---------------------------------------------------------------------------
# Distributed collective no-op context (for tracing with fake tensors)
# ---------------------------------------------------------------------------


class _MockDistHandle:
    """Mock handle returned by async distributed collectives during tracing.

    Megatron uses async collectives (e.g. all_reduce with async_op=True) and
    calls handle.wait() in the backward pass. Since we replace real collectives
    with no-ops, we must return an object that responds to .wait().
    """

    def wait(self):
        pass

    def is_completed(self):
        return True


@contextlib.contextmanager
def _dist_noop_context():
    """Monkey-patch torch.distributed collectives to be no-ops during tracing.

    torch.distributed.all_reduce / all_gather / reduce_scatter call directly
    into C++ ProcessGroup bindings which cannot accept FakeTensors. This
    context manager replaces them with no-ops for the duration of the trace.
    In-place ops (all_reduce, broadcast) leave tensors unchanged; out-of-place
    ops (all_gather_into_tensor, reduce_scatter_tensor) expect pre-allocated
    output tensors with the right shapes — Megatron always pre-allocates them.
    Returns a _MockDistHandle for async ops so that handle.wait() doesn't crash.
    """
    import torch.distributed as dist

    # Only patch if distributed is available (no-op otherwise)
    if not dist.is_available():
        yield
        return

    # Save originals
    orig = {
        name: getattr(dist, name)
        for name in (
            "all_reduce",
            "broadcast",
            "all_gather",
            "all_gather_into_tensor",
            "reduce_scatter",
            "reduce_scatter_tensor",
            "barrier",
        )
        if hasattr(dist, name)
    }

    _handle = _MockDistHandle()

    def _noop(*args, **kwargs):
        return _handle  # Return mock handle so callers can call .wait()

    try:
        for name in orig:
            setattr(dist, name, _noop)
        yield
    finally:
        for name, fn in orig.items():
            setattr(dist, name, fn)


# ---------------------------------------------------------------------------
# OperatorGraphTracer: public API
# ---------------------------------------------------------------------------


class OperatorGraphTracer:
    """Traces a PyTorch model and builds an OperatorGraph.

    Usage::

        tracer = OperatorGraphTracer(hw_info=hw)
        graph = tracer.trace(model, example_inputs)
        print(graph.summary())
    """

    def __init__(
        self,
        hw_info: Any = None,
        execution_mode: Optional[ExecutionMode] = None,
        cache_seq_len: int = 0,
        plena_estimator: Any = None,
    ) -> None:
        self._hw_info = hw_info
        self._execution_mode = execution_mode
        self._cache_seq_len = cache_seq_len
        self._plena_estimator = plena_estimator

    def trace(
        self,
        model: nn.Module,
        example_inputs: Any,
        forward_backward: bool = False,
        loss_fn: Any = None,
    ) -> OperatorGraph:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "rlsysim requires a CUDA-capable device for tracing. "
                "Fake CUDA tensors are used internally so that PyTorch "
                "dispatches to GPU kernel variants (flash attention, etc.)."
            )

        graph = OperatorGraph(name=type(model).__name__)
        storage_tracker = TensorStorageTracker()
        last_op_on_stream: dict[int, str] = {}
        op_counter = [0]  # Mutable counter shared with sub-components

        cuda_tracker = CUDAEventTracker(graph, last_op_on_stream, op_counter)
        mod_tracker = ModuleTracker()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        tracer_mode = _OperatorGraphTracerMode(
            graph=graph,
            storage_tracker=storage_tracker,
            last_op_on_stream=last_op_on_stream,
            op_counter=op_counter,
            hw_info=self._hw_info,
            execution_mode=self._execution_mode,
            cache_seq_len=self._cache_seq_len,
            plena_estimator=self._plena_estimator,
        )

        if loss_fn is None:

            def loss_fn(out):
                return out.sum() if isinstance(out, torch.Tensor) else out[0].sum()

        # Use current CUDA device so multi-rank jobs (TP/PP/DP) trace on the
        # correct per-rank device rather than always cuda:0.
        trace_device = f"cuda:{torch.cuda.current_device()}"

        cuda_tracker.install_hooks()
        try:
            # Phase 1: convert model + inputs to fake CUDA (NO dispatch mode active)
            restore_log = _convert_model_to_fake(model, fake_mode, trace_device)
            fake_inputs = _make_fake_inputs(example_inputs, fake_mode, trace_device)
            try:
                # Phase 2: trace with fake_mode + tracer_mode active.
                # _dist_noop_context patches torch.distributed collectives to
                # be no-ops so that Megatron's all_reduce / all_gather calls
                # don't try to pass FakeTensors to C++ NCCL bindings.
                with _dist_noop_context(), fake_mode, mod_tracker, tracer_mode:
                    out = model(**fake_inputs) if isinstance(fake_inputs, dict) else model(*fake_inputs)
                    if forward_backward:
                        loss = loss_fn(out)
                        loss.backward()
            finally:
                _restore_model(restore_log)
        finally:
            cuda_tracker.remove_hooks()

        return graph
