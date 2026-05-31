"""OperatorGraphTracer: builds OperatorGraph from PyTorch model execution.

Uses TorchDispatchMode + FakeTensorMode to intercept operations without
running actual computation. Tracks tensor storage for aliasing, monkey-patches
CUDA events for cross-stream dependencies, and classifies operations into
OperatorType categories.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, ClassVar, Optional

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils.module_tracker import ModuleTracker
from torch._subclasses import FakeTensorMode
from torch._subclasses.fake_tensor import FakeTensor as _FakeTensor
from torch._subclasses.fake_tensor import DataDependentOutputException

from .config import ExecutionMode
from .operator_graph import (
    OperatorType,
    TensorMeta,
    OperatorNode,
    OperatorGraph,
)
from .compute.predictor.roofline import (
    _VIEW_OPS,
    _CREATE_OPS,
    _GEMM_OPS,
    _ATTN_OPS,
    _IGNORE_OPS,
)

log = logging.getLogger(__name__)

aten = torch.ops.aten

_TRACE_DEVICE = "cuda:0"

# ---------------------------------------------------------------------------
# Metadata ops: compared by string name to avoid import-time errors for ops
# that may not exist in all PyTorch versions.
# ---------------------------------------------------------------------------
_METADATA_PACKET_NAMES = frozenset({
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
})

# ---------------------------------------------------------------------------
# Fake-CUDA helpers
# ---------------------------------------------------------------------------

def _to_fake_device(
    tensor: torch.Tensor,
    fake_mode: FakeTensorMode,
    device: str,
) -> torch.Tensor:
    """Create a fake tensor on *device* with *tensor*'s shape/dtype/stride.

    Built via an allocation op *inside* ``fake_mode`` so the resulting fake storage is tracked by
    the MemTracker on *device* -- exactly like activations created during the traced forward. The
    previous approach (``FakeTensor(fake_mode, tensor.to("meta"), device)``) left the underlying
    storage on the ``meta`` device, so model weights were charged to a phantom ``meta`` bucket and
    double-counted against the distributed optimizer's real fake-cuda param buffer.
    """
    with fake_mode:
        return torch.empty_strided(
            tuple(tensor.shape), tuple(tensor.stride()),
            dtype=tensor.dtype, device=torch.device(device),
        )


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


def _build_megatron_ddp_optimizer(model, provider, use_distributed_optimizer):
    """Wrap *model* in the real Megatron DistributedDataParallel + get_megatron_optimizer, exactly
    as run_megatron.py does, so the trace runs the SAME training stack as real execution: DDP gives
    the fp32 gradient buffer (grad_reduce_in_fp32); the optimizer gives the fp32 master weights and
    Adam m/v, ZeRO-sharded across the DP group when the distributed optimizer is on.

    Returns ``(forward_backward_model, optimizer)``; falls back to ``(model, None)`` when the stack
    can't be built (e.g. provider is None). Does NOT call ``optimizer.step()`` -- the caller decides
    when (the memory pass steps before the backward to allocate m/v; the timing pass steps after the
    backward so the Adam update's ops are captured and timed).
    """
    if provider is None:
        return model, None
    try:
        from megatron.core.distributed import (
            DistributedDataParallel, DistributedDataParallelConfig)
        from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
        from megatron.core import tensor_parallel, parallel_state
        # The fake tracing process group doesn't build the "partial data parallel" subgroups that
        # DDP/the distributed optimizer query (regular + GLOO variants); with a single optimizer
        # instance they are just the full data-parallel group.
        dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        if getattr(parallel_state, "_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP", None) is None:
            parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = dp_group
        if getattr(parallel_state, "_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO", None) is None:
            parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = dp_group
        for param in model.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        ddp_model = DistributedDataParallel(
            config=provider,
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=False,
                use_distributed_optimizer=use_distributed_optimizer),
            module=model)
        optimizer = get_megatron_optimizer(
            OptimizerConfig(optimizer="adam", lr=1e-4, bf16=True,
                            use_distributed_optimizer=use_distributed_optimizer),
            [ddp_model])
        return ddp_model, optimizer
    except Exception as exc:
        log.debug(f"DDP+megatron optimizer build failed: {exc}")
        return model, None



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
            predecessors: list[str] = []
            last_op = tracker._last_op_on_stream.get(stream_id)
            if last_op is not None:
                predecessors.append(last_op)
            if src_last_op is not None and src_last_op not in predecessors:
                predecessors.append(src_last_op)
            node = OperatorNode(
                name=name,
                op_type=OperatorType.STREAM_SYNC,
                stream_id=stream_id,
                config={"target_stream": src_stream},
                predecessors=predecessors,
            )
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
        last_op_on_stream: dict[int, str],
        op_counter: list[int],
        hw_info: Any = None,
        execution_mode: Optional[ExecutionMode] = None,
        cache_seq_len: int = 0,
        mod_tracker: Any = None,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._last_op_on_stream = last_op_on_stream
        self._op_counter = op_counter
        self._hw_info = hw_info
        self._execution_mode = execution_mode
        self._cache_seq_len = cache_seq_len
        # ModuleTracker whose `is_bw` flag tags each captured op's training phase.
        self._mod_tracker = mod_tracker

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        func_packet = func._overloadpacket
        packet_name = str(func_packet)

        # 1. Metadata ops: pass through
        if packet_name in _METADATA_PACKET_NAMES:
            return func(*args, **kwargs)

        # 2. View ops: execute, no node, no storage tracking
        if func_packet in _VIEW_OPS:
            return func(*args, **kwargs)

        # 3. Create ops: execute, zero-time node
        if func_packet in _CREATE_OPS:
            out = func(*args, **kwargs)
            idx = self._op_counter[0]
            self._op_counter[0] += 1
            name = f"op_{idx}_{packet_name}"
            # Read the live CUDA stream
            stream_id = torch.cuda.current_stream().stream_id
            prev_op = self._last_op_on_stream.get(stream_id)
            predecessors = [prev_op] if prev_op is not None else []
            node = OperatorNode(
                name=name,
                op_type=OperatorType.MATH,
                estimated_time_ms=0.0,
                outputs=_collect_tensor_metas(out),
                stream_id=stream_id,
                predecessors=predecessors,
            )
            self._graph.add_operator(node)
            self._last_op_on_stream[stream_id] = name
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
            return out

        # Tag the training phase from the module tracker's backward flag so the
        # report can split forward vs backward compute time.
        if self._mod_tracker is not None:
            if config is None:
                config = {}
            config.setdefault("phase", "backward" if self._mod_tracker.is_bw else "forward")

        # 5. Read the live CUDA stream — was hardcoded to 0
        stream_id = torch.cuda.current_stream().stream_id

        # Single predecessors list: stream FIFO only. Cross-stream causality comes
        # from explicit sync primitives (collective capture, CUDAEventTracker,
        # _MockDistHandle.wait), not from inferred producer→consumer relations.
        prev_op = self._last_op_on_stream.get(stream_id)
        predecessors = [prev_op] if prev_op is not None else []

        # 6. Estimate time using predictor (decoupled)
        estimated_time_ms = 0.0
        if self._hw_info is not None and func_packet not in _IGNORE_OPS:
            try:
                from .compute.estimator import estimate_runtime
                estimated_time_ms = estimate_runtime(
                    func_packet, args, kwargs, out, self._hw_info, op_type,
                    execution_mode=self._execution_mode,
                    cache_seq_len=self._cache_seq_len,
                )
            except Exception as e:
                log.debug(f"Runtime estimation failed for {packet_name}: {e}")
                estimated_time_ms = 0.0

        # 7. Create node
        idx = self._op_counter[0]
        self._op_counter[0] += 1
        name = f"op_{idx}_{packet_name}"

        node = OperatorNode(
            name=name,
            op_type=op_type,
            config=config,
            predecessors=predecessors,
            stream_id=stream_id,
            inputs=_collect_tensor_metas((args, kwargs)),
            outputs=_collect_tensor_metas(out),
            estimated_time_ms=estimated_time_ms,
        )

        self._graph.add_operator(node)
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
    # Set by _dist_noop_context for async ops; None for sync ops where the
    # caller-stream sync was already inserted at dispatch time.
    _producer_collective: Optional[str] = None
    # Set by _dist_noop_context — shared closure into the active capture state.
    _capture_graph: ClassVar[Optional["OperatorGraph"]] = None
    _capture_last_op: ClassVar[Optional[dict[int, str]]] = None

    def wait(self) -> None:
        if (self._producer_collective is not None
                and _MockDistHandle._capture_graph is not None
                and _MockDistHandle._capture_last_op is not None):
            import torch
            g = _MockDistHandle._capture_graph
            last_op = _MockDistHandle._capture_last_op
            caller = torch.cuda.current_stream().stream_id
            prev = last_op.get(caller)
            idx = len(g.operators)
            sync_name = f"async_wait_{idx}"
            preds = [self._producer_collective] + ([prev] if prev else [])
            g.add_operator(OperatorNode(
                name=sync_name,
                op_type=OperatorType.STREAM_SYNC,
                config={"reason": "async_collective_wait", "target_stream": 1},
                predecessors=preds,
                stream_id=caller,
            ))
            last_op[caller] = sync_name

    def is_completed(self) -> bool:
        return True


@contextlib.contextmanager
def _dist_noop_context(
    graph: Optional["OperatorGraph"] = None,
    last_op_on_stream: Optional[dict] = None,
):
    """Patch torch.distributed collectives to no-ops during tracing.

    Default (graph=None): preserves the original no-op-with-mock-handle behavior.

    Opt-in capture (graph != None and last_op_on_stream != None):
      - Each patched call records a COLLECTIVE OperatorNode on stream_id=1.
      - predecessors = [last_op_on_stream[caller_stream], last_op_on_stream[1]] (filtered Nones)
      - For SYNC collectives (async_op=False, default): also insert a STREAM_SYNC
        on the caller stream with predecessors=[COLLECTIVE]; this becomes the new
        last_op_on_stream[caller], so subsequent compute on the caller stream
        waits for the collective. Mirrors NCCL's post-sync semantics.
      - For ASYNC collectives (async_op=True): the returned _MockDistHandle
        carries the COLLECTIVE name; its .wait() emits a STREAM_SYNC then.
      - barrier() → BARRIER node on the caller stream.
    """
    import torch.distributed as dist
    if not dist.is_available():
        yield
        return

    orig = {
        name: getattr(dist, name)
        for name in (
            "all_reduce", "broadcast",
            "all_gather", "all_gather_into_tensor",
            "reduce_scatter", "reduce_scatter_tensor",
            "barrier",
            "isend", "irecv", "send", "recv",
            "batch_isend_irecv",
        )
        if hasattr(dist, name)
    }

    def _caller_stream_id() -> int:
        return torch.cuda.current_stream().stream_id

    def _record_collective(name, tensor, group):
        """Return the new COLLECTIVE node name (or None if not capturing)."""
        if graph is None or last_op_on_stream is None or tensor is None or not hasattr(tensor, "numel"):
            return None
        try:
            ranks = (dist.get_process_group_ranks(group) if group is not None
                     else list(range(dist.get_world_size())))
        except Exception:
            ranks = []
        caller = _caller_stream_id()
        preds = [last_op_on_stream.get(caller), last_op_on_stream.get(1)]
        preds = [p for p in preds if p is not None]
        idx = len(graph.operators)
        node_name = f"collective_{idx}_{name}"
        graph.add_operator(OperatorNode(
            name=node_name,
            op_type=OperatorType.COLLECTIVE,
            config={
                "collective": name,
                "bytes": tensor.numel() * tensor.element_size(),
                "group_ranks": ranks,
            },
            predecessors=preds,
            stream_id=1,
        ))
        last_op_on_stream[1] = node_name
        return node_name

    def _record_sync_post(collective_name: str) -> None:
        """For sync collectives, insert a STREAM_SYNC on caller stream after the collective."""
        if graph is None or last_op_on_stream is None or collective_name is None:
            return
        caller = _caller_stream_id()
        prev_caller = last_op_on_stream.get(caller)
        idx = len(graph.operators)
        sync_name = f"sync_post_{idx}"
        preds = [collective_name] + ([prev_caller] if prev_caller is not None else [])
        graph.add_operator(OperatorNode(
            name=sync_name,
            op_type=OperatorType.STREAM_SYNC,
            config={"reason": "sync_collective_caller_wait", "target_stream": 1},
            predecessors=preds,
            stream_id=caller,
        ))
        last_op_on_stream[caller] = sync_name

    def _wrap_inplace(name):
        def fn(tensor, *args, **kwargs):
            async_op = kwargs.get("async_op", False)
            coll_name = _record_collective(name, tensor, kwargs.get("group"))
            handle = _MockDistHandle()
            handle._producer_collective = coll_name           # for async wait
            if not async_op:
                _record_sync_post(coll_name)
            return handle
        return fn

    def _wrap_outofplace(name):
        def fn(output, input_, *args, **kwargs):
            async_op = kwargs.get("async_op", False)
            t = output if hasattr(output, "numel") else input_
            coll_name = _record_collective(name, t, kwargs.get("group"))
            handle = _MockDistHandle()
            handle._producer_collective = coll_name
            if not async_op:
                _record_sync_post(coll_name)
            return handle
        return fn

    def _wrap_barrier(_name):
        def fn(*args, **kwargs):
            if graph is not None and last_op_on_stream is not None:
                caller = _caller_stream_id()
                prev = last_op_on_stream.get(caller)
                idx = len(graph.operators)
                node_name = f"barrier_{idx}"
                preds = [prev] if prev is not None else []
                graph.add_operator(OperatorNode(
                    name=node_name,
                    op_type=OperatorType.BARRIER,
                    predecessors=preds,
                    stream_id=caller,
                ))
                last_op_on_stream[caller] = node_name
            return _MockDistHandle()
        return fn

    def _record_p2p(direction: str, tensor, peer: int, tag: int) -> Optional[str]:
        """Emit a COLLECTIVE node tagged kind=p2p; return its name."""
        if graph is None or last_op_on_stream is None or tensor is None:
            return None
        caller = _caller_stream_id()
        preds = [last_op_on_stream.get(caller), last_op_on_stream.get(1)]
        preds = [p for p in preds if p is not None]
        idx = len(graph.operators)
        node_name = f"p2p_{direction}_{idx}"
        graph.add_operator(OperatorNode(
            name=node_name,
            op_type=OperatorType.COLLECTIVE,
            config={
                "kind": "p2p",
                "direction": direction,
                "peer_rank": peer,
                "tag": tag,
                "bytes": tensor.numel() * tensor.element_size(),
            },
            predecessors=preds,
            stream_id=1,
        ))
        last_op_on_stream[1] = node_name
        return node_name

    def _wrap_p2p(direction: str, peer_kwarg: str):
        """direction is 'send' or 'recv'; peer_kwarg is 'dst' (send) or 'src' (recv)."""
        def fn(tensor, *args, **kwargs):
            peer = kwargs.get(peer_kwarg, args[0] if args else -1)
            tag = kwargs.get("tag", 0)
            coll_name = _record_p2p(direction, tensor, peer, tag)
            handle = _MockDistHandle()
            handle._producer_collective = coll_name
            return handle
        return fn

    def _wrap_batch_isend_irecv(_name):
        def fn(p2p_op_list, *args, **kwargs):
            handles = []
            for p2p_op in p2p_op_list:
                op_func = getattr(p2p_op, "op", None)
                tensor = getattr(p2p_op, "tensor", None)
                peer = getattr(p2p_op, "peer", -1)
                tag = getattr(p2p_op, "tag", 0)
                direction = "send" if op_func is dist.isend else "recv"
                coll_name = _record_p2p(direction, tensor, peer, tag)
                h = _MockDistHandle()
                h._producer_collective = coll_name
                handles.append(h)
            return handles
        return fn

    patched = {
        "all_reduce":              _wrap_inplace("all_reduce"),
        "broadcast":               _wrap_inplace("broadcast"),
        "all_gather":              _wrap_outofplace("all_gather"),
        "all_gather_into_tensor":  _wrap_outofplace("all_gather_into_tensor"),
        "reduce_scatter":          _wrap_outofplace("reduce_scatter"),
        "reduce_scatter_tensor":   _wrap_outofplace("reduce_scatter_tensor"),
        "barrier":                 _wrap_barrier("barrier"),
        "isend":                   _wrap_p2p("send", "dst"),
        "irecv":                   _wrap_p2p("recv", "src"),
        "send":                    _wrap_p2p("send", "dst"),
        "recv":                    _wrap_p2p("recv", "src"),
        "batch_isend_irecv":       _wrap_batch_isend_irecv("batch_isend_irecv"),
    }
    # Populate the ClassVars so _MockDistHandle.wait() can emit STREAM_SYNC nodes
    # even when called outside OperatorGraphTracer (e.g. in unit tests).
    if graph is not None and last_op_on_stream is not None:
        _MockDistHandle._capture_graph = graph
        _MockDistHandle._capture_last_op = last_op_on_stream
    # P2POp.__new__ calls _check_op which validates op against the *module-level*
    # `isend`/`irecv` names inside distributed_c10d — not against dist.isend/irecv.
    # Patching dist.isend/irecv doesn't update those internal references, so
    # P2POp construction fails during tracing. Bypass the check with a no-op.
    import torch.distributed.distributed_c10d as _d10d
    _orig_check_op = _d10d._check_op
    try:
        for name in orig:
            setattr(dist, name, patched[name])
        _d10d._check_op = lambda op: None  # allow mocked isend/irecv in P2POp
        yield
    finally:
        for name, fn in orig.items():
            setattr(dist, name, fn)
        _d10d._check_op = _orig_check_op
        # Only clear ClassVars if we set them (don't clobber an outer context)
        if graph is not None and last_op_on_stream is not None:
            _MockDistHandle._capture_graph = None
            _MockDistHandle._capture_last_op = None


# ---------------------------------------------------------------------------
# OperatorGraphTracer: public API
# ---------------------------------------------------------------------------

class OperatorGraphTracer:
    """Traces a PyTorch model and builds an OperatorGraph.

    Megatron-shaped signature: the caller supplies a `forward_backward_func`
    (typically the return value of
    `megatron.core.pipeline_parallel.get_forward_backward_func()`). The tracer
    sets up fake CUDA + dispatch contexts and invokes `forward_backward_func`
    with the kwargs passed through verbatim.
    """

    def __init__(
        self,
        hw_info: Any = None,
        execution_mode: Optional[ExecutionMode] = None,
        cache_seq_len: int = 0,
    ) -> None:
        self._hw_info = hw_info
        self._execution_mode = execution_mode
        self._cache_seq_len = cache_seq_len
        from .training.mem_profile import MemoryProfile
        self.memory_estimate = MemoryProfile()

    def trace(
        self,
        model: nn.Module,
        forward_backward_func,
        *,
        forward_step_func,
        data_iterator,
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        forward_only: bool = False,
    ) -> OperatorGraph:
        """Run `forward_backward_func(...)` under FakeTensorMode + TorchDispatchMode."""
        if torch.version.cuda is None:
            raise RuntimeError(
                f"syssim tracer needs a CUDA build of PyTorch, but a CPU-only build is "
                f"installed (torch {torch.__version__}). Fake CUDA tensors require a CUDA "
                f"wheel so PyTorch dispatches GPU kernel variants; reinstall a CUDA torch."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "syssim tracer requires a CUDA-capable device. "
                "Fake CUDA tensors are used internally so PyTorch dispatches "
                "to GPU kernel variants (flash attention, etc.)."
            )

        graph = OperatorGraph(name=type(model).__name__)
        last_op_on_stream: dict[int, str] = {}
        op_counter = [0]

        cuda_tracker = CUDAEventTracker(graph, last_op_on_stream, op_counter)
        mod_tracker = ModuleTracker()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        tracer_mode = _OperatorGraphTracerMode(
            graph=graph,
            last_op_on_stream=last_op_on_stream,
            op_counter=op_counter,
            hw_info=self._hw_info,
            execution_mode=self._execution_mode,
            cache_seq_len=self._cache_seq_len,
            mod_tracker=mod_tracker,
        )

        trace_device = f"cuda:{torch.cuda.current_device()}"

        # Same fake-safe CUDA RNG as estimate_memory: full recompute checkpoints the RNG state,
        # which hits real CUDA under FakeTensor; it affects neither memory nor timing.
        import torch.cuda.random as cuda_random
        import megatron.core.tensor_parallel.random as megatron_random
        orig_get_rng_state = cuda_random.get_rng_state
        orig_set_rng_state = cuda_random.set_rng_state
        orig_megatron_set_rng = megatron_random._set_cuda_rng_state
        fake_get_rng_state = lambda *args, **kwargs: torch.zeros(16, dtype=torch.uint8)
        fake_set_rng_state = lambda *args, **kwargs: None
        cuda_random.get_rng_state = torch.cuda.get_rng_state = fake_get_rng_state
        cuda_random.set_rng_state = torch.cuda.set_rng_state = fake_set_rng_state
        megatron_random._set_cuda_rng_state = fake_set_rng_state

        cuda_tracker.install_hooks()
        try:
            restore_log = _convert_model_to_fake(model, fake_mode, trace_device)
            _MockDistHandle._capture_graph = graph
            _MockDistHandle._capture_last_op = last_op_on_stream
            try:
                with _dist_noop_context(graph=graph,
                                        last_op_on_stream=last_op_on_stream), \
                     fake_mode, mod_tracker, tracer_mode:
                    # NOTE: the optimizer step is NOT traced here. Real Megatron fuses Adam into a
                    # single multi_tensor_apply kernel; under FakeTensor that fused kernel has no
                    # fake impl and decomposes into hundreds of per-param ops touching ~100x the
                    # memory of the fused kernel, so the captured time is meaningless. The optimizer
                    # is instead modeled analytically (memory-bound) in runner.inject_optimizer_step.
                    forward_backward_func(
                        forward_step_func=forward_step_func,
                        data_iterator=data_iterator,
                        model=model,
                        num_microbatches=num_microbatches,
                        seq_length=seq_length,
                        micro_batch_size=micro_batch_size,
                        forward_only=forward_only,
                    )
            finally:
                _restore_model(restore_log)
                _MockDistHandle._capture_graph = None
                _MockDistHandle._capture_last_op = None
        finally:
            cuda_tracker.remove_hooks()
            cuda_random.get_rng_state = torch.cuda.get_rng_state = orig_get_rng_state
            cuda_random.set_rng_state = torch.cuda.set_rng_state = orig_set_rng_state
            megatron_random._set_cuda_rng_state = orig_megatron_set_rng

        return graph

    def estimate_memory(
        self,
        model: nn.Module,
        forward_backward_func,
        *,
        forward_step_func,
        data_iterator,
        seq_length: int,
        micro_batch_size: int,
        use_distributed_optimizer: bool = False,
        provider=None,
    ):
        """Run ONE microbatch under MemTracker + a fake AdamW step to capture
        the per-microbatch memory footprint. Sets and returns
        ``self.memory_estimate`` (a MemoryProfile). Must run on a freshly-built
        model BEFORE any runtime pass mutates gradients.
        """
        from .training.mem_tracker import MemTracker, _MemRefType
        from .training.mem_profile import MemoryProfile

        # The memory pass builds no runtime op graph (no cuda_tracker/tracer_mode),
        # but forward_backward_func still issues collective calls (TP/SP, PP
        # send/recv). `graph` + `_dist_noop_context` are load-bearing: they make
        # those collectives no-op on fake tensors instead of hitting real
        # distributed. The graph itself is a throwaway and is never returned.
        graph = OperatorGraph(name=type(model).__name__ + "_mem")
        last_op_on_stream: dict[int, str] = {}
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        trace_device = f"cuda:{torch.cuda.current_device()}"

        mem_tracker = MemTracker()
        mem_tracker.track_external(model)

        restore_log = _convert_model_to_fake(model, fake_mode, trace_device)
        _MockDistHandle._capture_graph = graph
        _MockDistHandle._capture_last_op = last_op_on_stream
        # Full recompute checkpoints the CUDA RNG state for determinism; under FakeTensor
        # torch.cuda.get_rng_state hits real CUDA (illegal access). It affects neither memory nor
        # timing, so make it fake-safe for the trace — the recompute still re-runs the forward.
        import torch.cuda.random as cuda_random
        import megatron.core.tensor_parallel.random as megatron_random
        orig_get_rng_state = cuda_random.get_rng_state
        orig_set_rng_state = cuda_random.set_rng_state
        orig_megatron_set_rng = megatron_random._set_cuda_rng_state
        fake_get_rng_state = lambda *args, **kwargs: torch.zeros(16, dtype=torch.uint8)
        fake_set_rng_state = lambda *args, **kwargs: None
        cuda_random.get_rng_state = torch.cuda.get_rng_state = fake_get_rng_state
        cuda_random.set_rng_state = torch.cuda.set_rng_state = fake_set_rng_state
        megatron_random._set_cuda_rng_state = fake_set_rng_state  # raw CUDA call in fork/recompute
        try:
            with _dist_noop_context(graph=graph, last_op_on_stream=last_op_on_stream), \
                 fake_mode, mem_tracker:
                # Build the real Megatron mixed-precision training stack on the fake model so the
                # MemTracker MEASURES the true footprint (not a hand-built estimate). Step the
                # optimizer once BEFORE the backward so the persistent optimizer (master + Adam m/v)
                # and grad buffer are present at the backward peak (the binding peak without recompute).
                forward_backward_model, megatron_optimizer = _build_megatron_ddp_optimizer(
                    model, provider, use_distributed_optimizer)
                if megatron_optimizer is not None:
                    try:
                        megatron_optimizer.step()   # steady state: allocate Adam m/v pre-backward
                    except Exception:
                        pass
                forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=forward_backward_model,
                    num_microbatches=1,
                    seq_length=seq_length,
                    micro_batch_size=micro_batch_size,
                    forward_only=False,
                )
                del megatron_optimizer
            try:
                self.memory_estimate = MemoryProfile.from_mem_tracker(mem_tracker)
            except Exception as e:
                log.debug(f"MemTracker profile extraction failed: {e}")
                self.memory_estimate = MemoryProfile()
        finally:
            _restore_model(restore_log)
            _MockDistHandle._capture_graph = None
            _MockDistHandle._capture_last_op = None
            cuda_random.get_rng_state = torch.cuda.get_rng_state = orig_get_rng_state
            cuda_random.set_rng_state = torch.cuda.set_rng_state = orig_set_rng_state
            megatron_random._set_cuda_rng_state = orig_megatron_set_rng
        return self.memory_estimate
