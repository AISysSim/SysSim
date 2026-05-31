"""In-context calibration data: profile a REAL Megatron transformer layer and emit, per
(op, input-shape), the op's full signature + its real GPU time.

Unlike measure.py (which times synthetic, isolated kernels with guessed inputs), this builds the
actual Megatron GPTModel (local/explicit-attention spec) on real CUDA, runs a real forward+backward,
and records every dispatched op. Two passes over the same model:
  1. a TorchDispatchMode pass that captures each op's exact signature — operator, per-arg
     shapes+dtypes (incl. any fp32 upcast, bool masks), kwargs, and output shapes+dtypes;
  2. a torch.profiler pass that records each op's real self-GPU time, keyed by (op, input shapes).
The two are joined so the calibrator can reconstruct the EXACT op the simulator sees at inference
(same shapes, dtypes, out) and learn its residual from the TRUE in-context runtime. Sweeping
seq/batch (tp=1, no sharding, no collectives) covers the op-size feature space.

Run on a GPU node inside the container, single rank:
  python -m syssim.profiling.measure_layer <model.yaml> --seq 1024,2048,4096,8192 --batch 1,2 --out f
"""
from __future__ import annotations

import os


def init_single_rank(tp: int) -> int:
    import torch
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29577")
    # Use whichever GPU the caller already pinned (a multi-worker pool pins rank % device_count
    # before calling this); default current device is 0 for the single-worker path.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    # tp=1 here: a single real rank can't run TP collectives. Sharded shapes are reached by the
    # tree interpolating on op size; tp>1 in-context profiling (fake group) is a follow-up.
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                                             pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(42)
    return 0


def _build_real_model(provider, max_seq: int, vocab: int):
    """Real (cuda) GPTModel with the local/explicit-attention spec — same model run_megatron.py
    trains, but built directly on cuda (no meta redirect)."""
    from megatron.core import tensor_parallel
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.module import Float16Module
    model = GPTModel(
        config=provider, transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab, max_sequence_length=max_seq,
        pre_process=True, post_process=True, parallel_output=True,
    ).cuda().train()
    for p in model.parameters():
        tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(p)
    if getattr(provider, "bf16", False) or getattr(provider, "fp16", False):
        model = Float16Module(provider, model)
    return model


def _ser_val(v):
    """Serialize one arg/kwarg/output to a JSON-safe signature element."""
    import torch
    if isinstance(v, torch.Tensor):
        return {"t": list(v.shape), "dt": str(v.dtype)}
    if isinstance(v, bool) or isinstance(v, int) or isinstance(v, float) or v is None:
        return {"v": v}
    if isinstance(v, (list, tuple)):
        return {"seq": [_ser_val(x) for x in v]}
    return {"v": None}


def _tensor_shape_key(args):
    """Join key: tensor-arg shapes in flatten order (matches torch.profiler input_shapes once
    its empty-list scalar placeholders are dropped)."""
    import torch
    from torch.utils._pytree import tree_flatten
    flat, _ = tree_flatten(args)
    return tuple(tuple(int(d) for d in t.shape) for t in flat if isinstance(t, torch.Tensor))


def _make_sig_capture():
    """A TorchDispatchMode subclass instance that records each op's signature, keyed by
    (op-name, tensor-arg-shapes). Defined lazily so the module imports without torch."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class _SigCapture(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.sigs: dict = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            try:
                name = str(func.overloadpacket).rsplit(".", 1)[-1]
                key = (name, _tensor_shape_key(args))
                if key not in self.sigs:
                    self.sigs[key] = {
                        "op": name,
                        "args": [_ser_val(a) for a in args],
                        "kwargs": {k: _ser_val(v) for k, v in kwargs.items()},
                        "out": _ser_val(out),
                    }
            except Exception:
                pass
            return out

    return _SigCapture()


def _profile_one(model, vocab, heads, seq_points, batch_points, reps, warmup, token_range):
    """Run the two-pass profile (sig-capture + timing) on one built model over a seq/batch sweep;
    return calibration rows joined by (op, input-shape)."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    from ..training.runner import make_lm_forward_step, make_lm_data_iterator
    from megatron.core.pipeline_parallel import get_forward_backward_func

    fb = get_forward_backward_func()
    fwd_step = make_lm_forward_step(vocab)

    def step(data_iter, S, b, sync=True):
        for p in model.parameters():
            p.grad = None
        fb(forward_step_func=fwd_step, data_iterator=data_iter, model=[model],
           num_microbatches=1, seq_length=S, micro_batch_size=b, forward_only=False)
        if sync:
            torch.cuda.synchronize()

    # batch is not a free axis: tokens = batch*seq, so keep only (S, b) whose token count lands in
    # token_range (derives the batch sweep from token_range + seq_len_range, no batch field). Then
    # drop pairs whose attention scores [b, heads, S, S] (bf16) or lm-head logits [tokens, vocab]
    # (fp32, fwd+bwd) would not fit GPU memory; coverage above these caps is reached by the residual
    # tree interpolating on op size. These are the two tensors that dominate per-step peak memory.
    # The lm-head logits exist as bf16 AND fp32, forward AND backward (~12 bytes/elem peak), so the
    # 2*tokens*vocab*4 proxy must stay well under GPU memory once the (up to 16384-hidden) model
    # weights+grads are added; 22 GB caps tokens at 16384, which fits the largest config.
    token_low, token_high = token_range
    scores_budget, logits_budget = 16e9, 22e9
    sweep = [(S, b) for S in seq_points for b in batch_points
             if token_low <= S * b <= token_high
             and b * heads * S * S * 2 <= scores_budget
             and 2 * (S * b) * vocab * 4 <= logits_budget]
    if not sweep:
        return []
    max_seq, b0 = max(S for S, _ in sweep), min(b for _, b in sweep)

    cap = _make_sig_capture()       # pass 1: exact op signatures (shapes/dtypes/out) — timing-agnostic
    with cap:
        for S, b in sweep:
            step(make_lm_data_iterator(vocab, b, S), S, b)

    for _ in range(warmup):         # pin the boost clock right before timing (sustained, no inner sync)
        step(make_lm_data_iterator(vocab, b0, max_seq), max_seq, b0, sync=False)
    torch.cuda.synchronize()

    # pass 2: real per-(op, input-shape) GPU self-time (CPU activity needed for shape attribution).
    # Run each (S,b)'s reps back-to-back WITHOUT an inner sync so they pipeline at the GPU's sustained
    # boost clock (a per-step sync idles the GPU between steps, drops the clock, over-measures). Sync
    # ONCE per (S,b) group: frequent enough to keep torch.profiler's per-kernel self-time attribution
    # clean (a whole-sweep no-sync queue both degrades the attribution AND OOMs big configs), rare
    # enough to stay near the sustained clock. Profile single-worker (--num-workers 1): concurrent
    # workers contend for the node power/thermal envelope and inflate measured kernel time ~15%.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for S, b in sweep:
            di = make_lm_data_iterator(vocab, b, S)
            for _ in range(reps):
                step(di, S, b, sync=False)
            torch.cuda.synchronize()

    rows = []
    for e in prof.key_averages(group_by_input_shape=True):
        t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        n = max(int(getattr(e, "count", 1) or 1), 1)
        if t <= 0 or not e.key.startswith("aten::"):
            continue
        name = e.key.split("::", 1)[-1]
        shape_key = tuple(tuple(int(d) for d in s) for s in (e.input_shapes or []) if s)
        sig = cap.sigs.get((name, shape_key))
        if sig is None:
            continue
        rows.append({**sig, "count": n, "per_instance_ns": t * 1000.0 / n})
    return rows


# Representative architecture. Each profiling-spec field is swept over its own list while the
# other five are held here — covering each op-size axis independently (the cartesian product of the
# per-field lists would be ~80k configs; the residual tree interpolates op size, so 1-D coverage of
# every axis suffices). There is no "model": the unit is the (op, shape, dtype) the layer dispatches.
_BASE = dict(hidden=4096, heads=32, kv=8, head_dim=128, ffn=14336, vocab=128256)


def _make_layer_config(hidden, heads, query_groups, head_dim, ffn, vocab):
    import math
    from ..training.spec import ModelConfig
    if heads % query_groups:                # num_query_groups must divide num_attention_heads
        query_groups = math.gcd(heads, query_groups) or 1
    return ModelConfig(num_layers=1, hidden_size=hidden, num_attention_heads=heads,
                       num_query_groups=query_groups, kv_channels=head_dim, ffn_hidden_size=ffn,
                       seq_length=4096, max_position_embeddings=131072, vocab_size=vocab,
                       swiglu=True, rope=True)


def spec_configs(spec) -> list:
    """One layer config per (field, value) in the profiling spec: walk each architectural field over
    its list holding the other five at _BASE. De-duped on the architecture tuple."""
    base = _BASE
    configs = {}

    def add(config):
        configs.setdefault((config.hidden_size, config.num_attention_heads, config.num_query_groups,
                            config.kv_channels, config.ffn_hidden_size, config.vocab_size), config)

    for hidden in spec.hidden_sizes:
        add(_make_layer_config(hidden, base["heads"], base["kv"], base["head_dim"], base["ffn"], base["vocab"]))
    for ffn in spec.ffn_hidden_sizes:
        add(_make_layer_config(base["hidden"], base["heads"], base["kv"], base["head_dim"], ffn, base["vocab"]))
    for heads in spec.num_attention_heads:
        add(_make_layer_config(base["hidden"], heads, base["kv"], base["head_dim"], base["ffn"], base["vocab"]))
    for query_groups in spec.num_query_groups:
        add(_make_layer_config(base["hidden"], base["heads"], query_groups, base["head_dim"], base["ffn"], base["vocab"]))
    for head_dim in spec.head_dims:
        add(_make_layer_config(base["hidden"], base["heads"], base["kv"], head_dim, base["ffn"], base["vocab"]))
    for vocab in spec.vocab_sizes:
        add(_make_layer_config(base["hidden"], base["heads"], base["kv"], base["head_dim"], base["ffn"], vocab))
    return list(configs.values())


def profile_layer(model_config, tp: int, seq_points, batch_points, token_range,
                  layers: int = 1, reps: int = 5, warmup: int = 3) -> list[dict]:
    """Build one (config, tp-shard) per-rank 1-layer Megatron model on CUDA and profile it over the
    (seq, batch) sweep. The tp shard divides heads/kv/ffn/vocab so the per-rank shapes the simulator
    sees under TP are profiled DIRECTLY (head_dim preserved). Returns op-signature rows joined with
    real per-instance GPU ns; [] if this shard doesn't divide evenly or every seq point is too big.
    `init_single_rank` must have been called once before the first item."""
    import gc
    import torch
    from ..training.spec import ParallelismConfig, TrainingConfig
    from ..training.sources import resolve_megatron_provider
    from ..training.runner import _vocab_size_for_tp

    base_query_groups = model_config.num_query_groups or model_config.num_attention_heads
    if model_config.num_attention_heads % tp or base_query_groups % tp:
        return []
    gc.collect()                           # reclaim the previous job's tensors before building
    torch.cuda.empty_cache()
    head_dim = model_config.kv_channels or (model_config.hidden_size // model_config.num_attention_heads)
    seq_points = sorted(seq_points)
    max_position = max(model_config.max_position_embeddings or seq_points[-1], seq_points[-1])
    provider = resolve_megatron_provider(model_config, ParallelismConfig(tp=1, dp=1),
                                         TrainingConfig(micro_batch=1, global_batch=1,
                                                        dtype="bf16", recompute=None))
    provider.num_layers = layers
    provider.recompute_granularity = None
    provider.kv_channels = head_dim        # preserve head_dim when sharding heads
    provider.num_attention_heads = model_config.num_attention_heads // tp
    provider.num_query_groups = base_query_groups // tp
    provider.ffn_hidden_size = model_config.ffn_hidden_size // tp
    vocab = _vocab_size_for_tp(model_config.vocab_size, tp)
    model = _build_real_model(provider, max_position, vocab)
    # _profile_one drops (seq, batch) pairs whose scores/logits tensors would OOM (per-rank heads).
    rows = _profile_one(model, vocab, provider.num_attention_heads, seq_points, batch_points,
                        reps, warmup, token_range)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def _job_label(model_config, tp) -> str:
    return (f"hidden={model_config.hidden_size} heads={model_config.num_attention_heads} "
            f"kv={model_config.num_query_groups} head_dim={model_config.kv_channels} "
            f"ffn={model_config.ffn_hidden_size} vocab={model_config.vocab_size} tp={tp}")


def _make_progress_bar(total, desc):
    """A tqdm progress bar if tqdm is installed, else a minimal stdout counter with the same
    update()/write()/close() surface so the caller is identical either way."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="job")
    except ImportError:
        class _Counter:
            def __init__(self):
                self.done = 0
            def update(self, n=1):
                self.done += n
                print(f"{desc}: {self.done}/{total} jobs done", flush=True)
            def write(self, message):
                print(message, flush=True)
            def close(self):
                pass
        return _Counter()


def _profile_worker_loop(rank, num_gpus, job_queue, result_queue, seq_points, batch_points, token_range):
    """One pool worker: pin to GPU `rank % num_gpus`, initialise its own single-rank Megatron context
    on a unique port, then drain the job queue calling profile_layer on each (config, tp). Results go
    back as (label, gpu, rows, error) so the parent owns all printing (one progress bar, no races)."""
    import torch
    gpu = rank % num_gpus
    torch.cuda.set_device(gpu)
    os.environ["MASTER_PORT"] = str(29577 + rank)        # one rendezvous port per worker
    init_single_rank(1)
    while True:
        job = job_queue.get()
        if job is None:
            break
        model_config, tp = job
        label = _job_label(model_config, tp)
        try:
            rows = profile_layer(model_config, tp, seq_points, batch_points, token_range)
            result_queue.put((label, gpu, rows, None))
        except Exception as error:
            result_queue.put((label, gpu, None, f"{type(error).__name__}: {error}"))


def profile_worklist(jobs, num_workers, seq_points, batch_points, token_range) -> list[dict]:
    """Profile every (model_config, tp) job and return the concatenated op-signature rows.

    num_workers=1 -> sequential, in this process.
    num_workers>1 -> a torch.multiprocessing (spawn) pool: each worker pins to GPU
    `rank % torch.cuda.device_count()` and pulls jobs from a shared queue. Per-job failures are
    skipped, not fatal. Mirrors measure.measure_worklist; the layer addition is per-worker Megatron
    init on a distinct port."""
    if num_workers <= 1:
        init_single_rank(1)
        bar = _make_progress_bar(len(jobs), "profiling")
        rows = []
        for model_config, tp in jobs:
            label = _job_label(model_config, tp)
            try:
                item_rows = profile_layer(model_config, tp, seq_points, batch_points, token_range)
                bar.write(f"  {label} -> {len(item_rows)} rows")
                rows.extend(item_rows)
            except Exception as error:
                bar.write(f"  {label} -> FAILED ({type(error).__name__}: {error})")
            bar.update(1)
        bar.close()
        return rows

    import torch
    import torch.multiprocessing as mp
    num_gpus = max(1, torch.cuda.device_count())
    ctx = mp.get_context("spawn")
    job_queue, result_queue = ctx.Queue(), ctx.Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in range(num_workers):
        job_queue.put(None)                              # one stop sentinel per worker
    procs = [ctx.Process(target=_profile_worker_loop,
                         args=(rank, num_gpus, job_queue, result_queue,
                               seq_points, batch_points, token_range))
             for rank in range(num_workers)]
    for proc in procs:
        proc.start()

    import queue as _queue
    bar = _make_progress_bar(len(jobs), "profiling")
    rows = []
    for _ in range(len(jobs)):
        try:
            label, gpu, item_rows, error = result_queue.get(timeout=1800)  # one job never needs 30 min
        except _queue.Empty:
            bar.write("  a worker died; stopping collection")
            break
        if error is None:
            rows.extend(item_rows)
            bar.write(f"  [gpu{gpu}] {label} -> {len(item_rows)} rows")
        else:
            bar.write(f"  [gpu{gpu}] {label} -> FAILED ({error})")
        bar.update(1)
    bar.close()
    for proc in procs:
        proc.join(timeout=60)
    return rows
