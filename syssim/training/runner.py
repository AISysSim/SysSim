"""Orchestrate config-driven training simulation."""

from __future__ import annotations

import copy
import math


def dp_group_ranks(tp_size: int, dp_size: int) -> list[int]:
    """Global ranks of the data-parallel group containing rank 0.

    Megatron lays out ranks tensor-parallel-fastest, so the data-parallel replicas of a given
    tensor-parallel shard are strided by ``tp_size``: ``[0, tp, 2*tp, ...]``. (Using
    ``list(range(dp))`` instead selects the first ``dp`` *tensor-parallel* ranks, which share a
    node — that mis-routes the inter-node DP collective onto the intra-node fabric.)
    """
    return [r * tp_size for r in range(dp_size)]


def _simulate_on_hardware(trace, hardware):
    """Inject + DES + build report. Used by Trace.simulate_on AND by simulate()."""
    from .runtime import simulate_runtime, phase_breakdown_ms, collective_exposed_ms
    from .report import (
        SimulationReport, aggregate_by_op_type, compute_efficiency, compute_model_flops_budget,
        build_bottlenecks,
    )
    from .memory import grad_bytes_per_rank, param_bytes_per_rank, sharded_param_bytes_per_rank
    from .collectives import (
        inject_dp_gradient_allreduce, inject_optimizer_step,
        last_backward_op_per_pp_stage,
    )
    from .spec import derive_num_nodes, ModelConfig
    from .sources import HFModel

    _ = derive_num_nodes(trace.parallelism, hardware)  # validates divisibility + inter_node_bw

    if hardware.topology is None:
        raise ValueError(
            "HardwareConfig.topology is required: the training simulator drives all "
            "collective times through the flow-level network simulator."
        )
    from ..network import build_topology_from_config
    topology = build_topology_from_config(hardware)

    # Work on a copy so the cached trace.graph is reusable across hardware
    graph = copy.deepcopy(trace.graph)

    # Fill in estimated_time_ms for captured TP/SP collectives that came out
    # of the tracer with time 0. P2P collectives are timed in
    # pipeline.compose_multi_rank_graph; injected DP allreduce is timed below.
    from ..operator_graph import OperatorType
    for op in graph.operators.values():
        if op.op_type != OperatorType.COLLECTIVE:
            continue
        if op.estimated_time_ms > 0:
            continue
        if op.config.get("kind") == "p2p":
            continue
        coll_name = op.config.get("collective")
        if not coll_name:
            continue
        op.estimated_time_ms = _estimate_collective_ms(
            collective_name=coll_name,
            total_bytes=int(op.config.get("bytes", 0)),
            ranks=list(op.config.get("group_ranks", [])),
            topology=topology,
        )

    # Resolve a ModelConfig view of the model for memory + budget calculations
    if isinstance(trace.model, ModelConfig):
        model_for_mem = trace.model
    else:  # HFModel — resolve via Megatron-Bridge
        from .sources import resolve_megatron_provider
        provider = resolve_megatron_provider(trace.model, trace.parallelism, trace.training)
        model_for_mem = _provider_to_model_config(provider)

    pp = trace.parallelism.pipeline_model_parallel_size
    dp = trace.parallelism.data_parallel_size
    # DP replicas are strided by tp_size (Megatron's tp-fastest rank order), so the DP group
    # spans nodes when DP does — timing the DP collective over the correct (inter-node) links.
    dp_ranks = dp_group_ranks(trace.parallelism.tensor_model_parallel_size, dp)
    grad_bytes = grad_bytes_per_rank(model_for_mem, trace.parallelism, trace.training)
    pb = param_bytes_per_rank(model_for_mem, trace.parallelism, trace.training)
    spb = sharded_param_bytes_per_rank(model_for_mem, trace.parallelism, trace.training)

    # Fused mixed-precision Adam is bandwidth-bound (one multi_tensor_apply kernel). Per bf16 param
    # (param_bytes = 2B): read+write fp32 master+m+v (12B/param = 6x bf16 -> 12*pb), read fp32 grad
    # (grad_bytes), write bf16 param (pb). The distributed optimizer shards the update by 1/dp.
    adam_bytes = 12 * pb + grad_bytes + pb
    use_dist_opt = bool(getattr(trace.training, "use_distributed_optimizer", False)) and dp > 1
    if use_dist_opt:
        adam_bytes //= dp

    if pp == 1:
        if graph.operators:
            last_backward = next(reversed(graph.operators))
            inject_dp_gradient_allreduce(
                graph, total_grad_bytes=grad_bytes, dp_ranks=dp_ranks,
                last_backward_op_name=last_backward,
                estimated_time_ms=_estimate_dp_comm_ms(
                    grad_bytes=grad_bytes, sharded_param_bytes=spb, dp_ranks=dp_ranks,
                    topology=topology, distributed_optimizer=use_dist_opt),
            )
            inject_optimizer_step(
                graph, bytes_moved=adam_bytes,
                peak_memory_bandwidth_GBps=hardware.peak_memory_bandwidth_GBps,
                last_op_name=next(reversed(graph.operators)),
            )
    else:
        # PP > 1: inject DP allreduce + optimizer per PP stage
        last_per_stage = last_backward_op_per_pp_stage(graph)
        for pp_rank, last_name in last_per_stage.items():
            inject_dp_gradient_allreduce(
                graph, total_grad_bytes=grad_bytes, dp_ranks=dp_ranks,
                last_backward_op_name=last_name,
                estimated_time_ms=_estimate_dp_comm_ms(
                    grad_bytes=grad_bytes, sharded_param_bytes=spb, dp_ranks=dp_ranks,
                    topology=topology, distributed_optimizer=use_dist_opt),
            )
        for pp_rank in range(pp):
            anchor = last_per_stage.get(pp_rank)
            if anchor is not None:
                inject_optimizer_step(
                    graph, bytes_moved=adam_bytes,
                    peak_memory_bandwidth_GBps=hardware.peak_memory_bandwidth_GBps,
                    last_op_name=anchor,
                )

    res = simulate_runtime(graph)
    phases = phase_breakdown_ms(graph)
    by_type = aggregate_by_op_type(graph)
    budget = compute_model_flops_budget(model_for_mem, trace.parallelism, trace.training)
    eff = compute_efficiency(
        budget=budget, step_time_ms=res.step_time_ms,
        world_size=trace.parallelism.world_size, peak_tflops_mm=hardware.peak_tflops_mm,
    )

    # Memory: per-microbatch footprint (from the mb=1 MemTracker pass) scaled by
    # the 1F1B in-flight microbatch count per PP stage. Binding stage = max peak.
    from .pipeline import in_flight_microbatches
    profiles = trace.per_stage_profiles or []
    num_mb = max(
        1,
        trace.training.global_batch_size
        // (trace.training.micro_batch_size * trace.parallelism.data_parallel_size),
    )
    pp_stage_memory_gb: list[float] = []
    stage_by_type: list[dict] = []
    for stage_rank, prof in enumerate(profiles):
        infl = in_flight_microbatches(pp, stage_rank, num_mb)
        pp_stage_memory_gb.append(prof.peak_gb(infl))
        stage_by_type.append(prof.peak_by_type(infl))
    if pp_stage_memory_gb:
        binding_stage = max(range(len(pp_stage_memory_gb)),
                            key=lambda s: pp_stage_memory_gb[s])
        peak_gb = pp_stage_memory_gb[binding_stage]
        mem_by_type = stage_by_type[binding_stage]
        peak_module = profiles[binding_stage].peak_module
    else:
        binding_stage, peak_gb, mem_by_type, peak_module = 0, 0.0, {}, ""
        pp_stage_memory_gb = [0.0]
    report_param_bytes = mem_by_type.get("Parameter", 0) + mem_by_type.get("Buffer", 0)
    report_grad_bytes = mem_by_type.get("Gradient", 0)
    report_opt_bytes = mem_by_type.get("Optstate", 0)
    report_act_bytes = mem_by_type.get("Activation", 0) + mem_by_type.get("Temp", 0)

    # ZeRO-1 sharding of the optimizer state is already reflected in the tracked peak: the real
    # distributed optimizer runs under the fake DP group in the trace and shards master/m/v by 1/dp.
    per_stage_mem = pp_stage_memory_gb

    bottlenecks = build_bottlenecks(
        graph, peak_memory_gb=peak_gb, gpu_memory_GB=hardware.gpu_memory_GB,
        peak_module=peak_module, memory_by_type=mem_by_type, binding_stage=binding_stage,
    )

    per_rank_finish_ms: list[float] = []
    for pp_rank in range(pp):
        stage_ops = [op for op in graph.operators.values()
                     if op.config.get("pp_rank") == pp_rank]
        if stage_ops:
            per_rank_finish_ms.append(max(
                res.per_op_finish_ms.get(op.name, 0.0) for op in stage_ops
            ))
        else:
            per_rank_finish_ms.append(res.step_time_ms)
    if not per_rank_finish_ms:
        per_rank_finish_ms = [res.step_time_ms]

    return SimulationReport(
        step_time_ms=res.step_time_ms,
        forward_ms=phases["forward"], backward_ms=phases["backward"],
        optimizer_ms=phases["optimizer"],
        collective_total_ms=by_type.get("collective", 0.0),
        collective_exposed_ms=collective_exposed_ms(graph, res),
        by_op_type_ms=by_type,
        model_flops_per_step=int(budget.model_flops_per_step),
        achieved_tflops=eff["achieved_tflops"], mfu=eff["mfu"], hfu=eff["hfu"],
        param_bytes=report_param_bytes, grad_bytes=report_grad_bytes,
        optimizer_state_bytes=report_opt_bytes,
        activation_bytes=report_act_bytes, peak_memory_gb=peak_gb,
        pp_stage_memory_gb=per_stage_mem,
        per_pp_rank_step_time_ms=per_rank_finish_ms,
        model=trace.model, parallelism=trace.parallelism, training=trace.training,
        hardware=hardware, bottlenecks=bottlenecks,
    )


def _estimate_dp_allreduce_ms(total_bytes, dp_ranks, topology):
    """DP all-reduce time in ms via the flow-level network simulator.

    Decomposes the all-reduce into a ring over `dp_ranks` and runs
    `simulate(ops, topology)` against the given Topology.
    """
    P = len(dp_ranks)
    if P <= 1 or total_bytes <= 0:
        return 0.0
    from ..network import allreduce, simulate
    ops = allreduce(ranks=list(dp_ranks), total_bytes=int(total_bytes), tag="dp_ar")
    return simulate(ops, topology).makespan * 1000.0


def _estimate_dp_comm_ms(*, grad_bytes, sharded_param_bytes, dp_ranks, topology, distributed_optimizer):
    """Data-parallel communication time (ms) via the flow-level network simulator.

    Measured on GH200 (NCCL_DEBUG=COLL, qwen3-8b TP4·DP2): with the distributed optimizer (ZeRO-1)
    the exposed inter-node DP comm is a single **bf16 all-gather of the TP-sharded parameters** — no
    reduce-scatter appears on the wire, and the datatype is bf16 even though gradients accumulate in
    fp32 (`grad_reduce_in_fp32`). The replicated embeddings/norms are not DP-gathered. Plain DDP
    all-reduces the bf16 gradient over the DP group.
    """
    if len(dp_ranks) <= 1:
        return 0.0
    if distributed_optimizer:
        return _estimate_collective_ms(collective_name="all_gather",
                                       total_bytes=int(sharded_param_bytes),
                                       ranks=dp_ranks, topology=topology)
    return _estimate_collective_ms(collective_name="all_reduce", total_bytes=int(grad_bytes),
                                   ranks=dp_ranks, topology=topology)


def _estimate_collective_ms(*, collective_name, total_bytes, ranks, topology):
    """Estimated time (ms) for one captured collective via the flow-level simulator.

    Decomposes the collective into an Op list (ring for allreduce / allgather /
    reduce_scatter; binary tree for broadcast) and runs `simulate(ops, topology)`.
    Returns 0.0 for degenerate cases (P <= 1 or total_bytes <= 0) and for
    collective names that don't correspond to a modeled collective (e.g. barrier).
    """
    P = len(ranks)
    if P <= 1 or total_bytes <= 0:
        return 0.0

    # PyTorch "_into_tensor" / "_tensor" suffixes are tensor-variant aliases
    # of the same underlying collective.
    name = collective_name.replace("_into_tensor", "").replace("_tensor", "")

    from ..network import (
        allreduce as _ar, allgather as _ag,
        reduce_scatter as _rs, broadcast as _bc, simulate as _sim,
    )
    if name == "all_reduce":
        ops = _ar(ranks=list(ranks), total_bytes=int(total_bytes), tag="tp_ar")
    elif name == "all_gather":
        ops = _ag(ranks=list(ranks), total_bytes=int(total_bytes), tag="tp_ag")
    elif name == "reduce_scatter":
        ops = _rs(ranks=list(ranks), total_bytes=int(total_bytes), tag="tp_rs")
    elif name == "broadcast":
        ops = _bc(ranks=list(ranks), total_bytes=int(total_bytes),
                  root=ranks[0], tag="tp_bc")
    else:
        return 0.0
    return _sim(ops, topology).makespan * 1000.0


def _build_gpt_model_on_meta(provider, max_sequence_length: int, vocab_size: int):
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from .cuda_redirect import redirect_cuda_alloc_to_meta
    from ._norm_meta import install_norm_meta_kernels

    install_norm_meta_kernels()  # make apex + TE fused norm kernels fake-safe for tracing
    has_embedding = parallel_state.is_pipeline_first_stage()
    has_lm_head = parallel_state.is_pipeline_last_stage()
    layer_spec = get_gpt_layer_local_spec()
    with redirect_cuda_alloc_to_meta():
        model = GPTModel(
            config=provider, transformer_layer_spec=layer_spec,
            vocab_size=vocab_size, max_sequence_length=max_sequence_length,
            pre_process=has_embedding, post_process=has_lm_head, parallel_output=True,
        )
        model = model.train()
        # Match the real run (run_megatron.py): wrap in Float16Module so the traced
        # forward runs in bf16/fp16. Without this the trace stays fp32 — every mem-bound
        # op (softmax/norm/bmm/dropout) counts 2x the bytes and the bf16-calibrated
        # anchors no longer match the traced fp32 anchors.
        if getattr(provider, "bf16", False) or getattr(provider, "fp16", False):
            from megatron.core.transformer.module import Float16Module
            model = Float16Module(provider, model)
    return model


def _vocab_size_for_tp(vocab_size: int, tp_size: int) -> int:
    return math.ceil(vocab_size / tp_size) * tp_size


def _provider_to_model_config(provider):
    """Adapter: extract a ModelConfig-shaped view from a Megatron TransformerConfig."""
    from .spec import ModelConfig
    return ModelConfig(
        num_layers=provider.num_layers,
        hidden_size=provider.hidden_size,
        num_attention_heads=provider.num_attention_heads,
        num_query_groups=getattr(provider, "num_query_groups", provider.num_attention_heads),
        ffn_hidden_size=provider.ffn_hidden_size,
        seq_length=getattr(provider, "seq_length", 2048),
        max_position_embeddings=getattr(provider, "max_position_embeddings", 2048),
        vocab_size=getattr(provider, "padded_vocab_size", getattr(provider, "vocab_size", 0)),
        swiglu=getattr(provider, "swiglu", True),
        rope=getattr(provider, "rotary_percent", 1.0) > 0,
        tie_word_embeddings=getattr(provider, "share_embeddings_and_output_weights", False),
        rms_norm_eps=getattr(provider, "layernorm_epsilon", 1e-6),
    )


def make_lm_forward_step(vocab_size: int):
    """Causal-LM forward_step_func for Megatron's get_forward_backward_func.

    Returns (output_tensor, loss_func) per Megatron's API contract:
      - output_tensor: the logits tensor (or loss on last stage before loss_func call)
      - loss_func: callable(output_tensor) -> (loss_scalar, {"lm_loss": ...})
    """
    import torch.nn.functional as F

    def forward_step(data_iterator, model):
        batch = next(data_iterator)
        out = model(batch["input_ids"], batch["position_ids"], None)
        logits = out if isinstance(out, _torch_tensor_type()) else out[0]
        labels = batch["labels"]

        def loss_func(output_tensor):
            loss = F.cross_entropy(
                output_tensor.reshape(-1, output_tensor.size(-1)),
                labels.reshape(-1),
            )
            return loss, {"lm_loss": loss.detach()}

        return logits, loss_func
    return forward_step


def make_lm_data_iterator(vocab_size: int, micro_batch_size: int, seq_length: int):
    import torch
    def gen():
        while True:
            yield {
                "input_ids":    torch.randint(0, vocab_size, (micro_batch_size, seq_length), device="cuda"),
                "position_ids": torch.arange(seq_length, device="cuda").unsqueeze(0).expand(micro_batch_size, -1),
                "labels":       torch.randint(0, vocab_size, (micro_batch_size, seq_length), device="cuda"),
            }
    return gen()


def _torch_tensor_type():
    import torch
    return torch.Tensor


def _trace_rank(spawn_rank: int, model_arg, par_arg, tr_arg, hw_arg, out_path: str) -> None:
    """Run on one PP stage. Initializes a fake process group with the full
    Megatron world size, sets this process's global rank to that stage's
    (tp=0, dp=0, cp=0) slice, traces, writes graph to out_path.
    """
    import json
    import torch
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from .pipeline import pp_rank_to_global_rank
    from .dist_setup import init_fake_process_group, destroy_process_group
    from .sources import resolve_megatron_provider
    from .spec import HardwareConfig
    from ..tracer import OperatorGraphTracer

    world_size = par_arg.world_size
    global_rank = pp_rank_to_global_rank(spawn_rank, par_arg)
    init_fake_process_group(world_size=world_size, rank=global_rank)
    torch.cuda.set_device(0)

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=par_arg.tensor_model_parallel_size,
            pipeline_model_parallel_size=par_arg.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=par_arg.virtual_pipeline_model_parallel_size,
            context_parallel_size=par_arg.context_parallel_size,
            create_gloo_process_groups=False,
        )
    model_parallel_cuda_manual_seed(42)

    provider = resolve_megatron_provider(model_arg, par_arg, tr_arg)
    vocab = _vocab_size_for_tp(
        provider.padded_vocab_size if hasattr(provider, "padded_vocab_size")
        else (model_arg.vocab_size if hasattr(model_arg, "vocab_size") and model_arg.vocab_size
              else 151936),
        par_arg.tensor_model_parallel_size,
    )
    seq_length = (model_arg.seq_length if hasattr(model_arg, "seq_length") and model_arg.seq_length
                  else provider.seq_length if hasattr(provider, "seq_length")
                  else 2048)
    max_pos = (model_arg.max_position_embeddings if hasattr(model_arg, "max_position_embeddings")
               and model_arg.max_position_embeddings else seq_length)
    model = _build_gpt_model_on_meta(provider, max_sequence_length=max_pos, vocab_size=vocab)

    num_microbatches_per_rank = max(1, tr_arg.global_batch_size
                                       // (tr_arg.micro_batch_size * par_arg.data_parallel_size))

    forward_backward_func = get_forward_backward_func()
    hw_info = None
    if hw_arg is not None and isinstance(hw_arg, HardwareConfig):
        from ..config import HardwareInfo
        hw_info = HardwareInfo(
            peak_tflops_mm=hw_arg.peak_tflops_mm,
            peak_tflops_math=hw_arg.peak_tflops_math,
            peak_memory_bandwidth_gbps=hw_arg.peak_memory_bandwidth_GBps,
            peak_tflops_mm_fp8=hw_arg.peak_tflops_mm_fp8,
            peak_tflops_mm_fp4=hw_arg.peak_tflops_mm_fp4,
            sfu_peak=hw_arg.sfu_peak,
            estimator=hw_arg.estimator,
            calibrated_model=getattr(hw_arg, "calibrated_model", None),
        )
    tracer = OperatorGraphTracer(hw_info=hw_info)
    # Estimate per-microbatch memory first, on the freshly-built model, so the
    # footprint reflects a clean state before the runtime pass runs.
    tracer.estimate_memory(
        model=model,
        forward_backward_func=forward_backward_func,
        forward_step_func=make_lm_forward_step(vocab),
        data_iterator=make_lm_data_iterator(vocab, tr_arg.micro_batch_size, seq_length),
        seq_length=seq_length,
        micro_batch_size=tr_arg.micro_batch_size,
        use_distributed_optimizer=getattr(tr_arg, "use_distributed_optimizer", False),
        provider=provider,
    )
    # estimate_memory ran fwd/bwd on fake copies and restored the originals, so
    # these grads are already clean; null them defensively before the runtime trace.
    for p in model.parameters():
        p.grad = None
    graph = tracer.trace(
        model=model,
        forward_backward_func=forward_backward_func,
        forward_step_func=make_lm_forward_step(vocab),
        data_iterator=make_lm_data_iterator(vocab, tr_arg.micro_batch_size, seq_length),
        num_microbatches=num_microbatches_per_rank,
        seq_length=seq_length,
        micro_batch_size=tr_arg.micro_batch_size,
        forward_only=False,
    )

    with open(out_path, "w") as f:
        json.dump({
            "graph": json.loads(graph.to_json()),
            "pp_rank": spawn_rank,
            # key kept as "memory_profile" for a stable cross-process serialization name
            "memory_profile": tracer.memory_estimate.to_dict(),
        }, f)

    parallel_state.destroy_model_parallel()
    destroy_process_group()


def _spawn_wrapper(spawn_rank, model_arg, par, tr, hw, per_stage_paths):
    _trace_rank(spawn_rank, model_arg, par, tr, hw, per_stage_paths[spawn_rank])


def trace(
    *,
    model,
    parallelism=None,
    training=None,
    hardware=None,
    gpus_per_node: int,
    workdir=None,
):
    """Run the trace phase under mp.spawn and return a bare-graph Trace.

    NO injection happens here — the returned Trace.graph is just what the
    tracer captured (fwd + bwd + TP/SP collectives). Injection of DP all-reduce
    and optimizer step happens in Trace.simulate_on(hardware).
    """
    import json, os, tempfile
    from .spec import (
        load_model_yaml, ParallelismConfig, TrainingConfig, ModelConfig,
    )
    from .sources import HFModel
    from .trace import Trace
    from ..operator_graph import OperatorGraph, OperatorNode, OperatorType

    if isinstance(model, str):
        model = load_model_yaml(model)
    elif not isinstance(model, (ModelConfig, HFModel)):
        raise TypeError(f"model must be a path, ModelConfig, or HFModel; got {type(model).__name__}")
    parallelism = parallelism or ParallelismConfig()
    training    = training or TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")

    workdir = str(workdir) if workdir else tempfile.mkdtemp(prefix="syssim_trace_")
    os.makedirs(workdir, exist_ok=True)
    pp = parallelism.pipeline_model_parallel_size

    per_stage_paths = [os.path.join(workdir, f"pp_rank{r}.json") for r in range(pp)]

    if pp == 1:
        _trace_rank(0, model, parallelism, training, hardware, per_stage_paths[0])
    else:
        import torch.multiprocessing as mp
        mp.spawn(
            _spawn_wrapper,
            args=(model, parallelism, training, hardware, per_stage_paths),
            nprocs=pp,
            join=True,
        )

    from .mem_profile import MemoryProfile

    per_stage_graphs: dict[int, OperatorGraph] = {}
    per_stage_profiles: dict[int, MemoryProfile] = {}
    for pp_rank, path in enumerate(per_stage_paths):
        with open(path) as f:
            stage_data = json.load(f)
        graph_data = stage_data["graph"]
        per_stage_profiles[pp_rank] = MemoryProfile.from_dict(
            stage_data.get("memory_profile", {})
        )
        g = OperatorGraph(name=graph_data.get("name", f"stage{pp_rank}"))
        for d in graph_data.get("operators", []):
            g.add_operator(OperatorNode(
                name=d["name"], op_type=OperatorType(d["op_type"]),
                config=d.get("config", {}),
                predecessors=list(d.get("predecessors", [])),
                stream_id=d.get("stream_id", 0), device_id=d.get("device_id", 0),
                estimated_time_ms=d.get("estimated_time_ms", 0.0),
            ))
        per_stage_graphs[pp_rank] = g

    if pp == 1:
        composed = per_stage_graphs[0]
    else:
        if hardware is None:
            raise ValueError("hardware is required for PP > 1 (needed to time P2P transfers)")
        from .pipeline import compose_multi_rank_graph
        composed = compose_multi_rank_graph(
            per_stage_graphs=per_stage_graphs,
            parallelism=parallelism,
            hardware=hardware,
        )

    per_stage_profiles_list = [per_stage_profiles[r] for r in range(pp)]

    return Trace(graph=composed, model=model, parallelism=parallelism,
                 training=training, gpus_per_node=gpus_per_node,
                 per_stage_profiles=per_stage_profiles_list)


def simulate(*, model, hardware, parallelism=None, training=None, workdir=None):
    """Thin wrapper: trace + Trace.simulate_on(hardware) -> SimulationReport.

    Equivalent to:
        t = trace(model=..., parallelism=..., training=..., hardware=...,
                  gpus_per_node=hardware.gpus_per_node, workdir=...)
        return t.simulate_on(hardware)
    """
    from .spec import load_hardware_yaml, HardwareConfig, ParallelismConfig, TrainingConfig

    if isinstance(hardware, str):
        hardware = load_hardware_yaml(hardware)
    elif not isinstance(hardware, HardwareConfig):
        raise TypeError(f"hardware must be a path or HardwareConfig; got {type(hardware).__name__}")
    parallelism = parallelism or ParallelismConfig()
    training    = training or TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")
    t = trace(model=model, parallelism=parallelism, training=training,
              hardware=hardware, gpus_per_node=hardware.gpus_per_node, workdir=workdir)
    return t.simulate_on(hardware)


def estimate_memory(*, model, hardware, parallelism=None, training=None, workdir=None):
    """Per-GPU peak memory via the MemTracker per-microbatch footprint scaled by the
    1F1B in-flight count. Runs the trace's memory pass, SKIPS the runtime DES, and
    returns a `SimulationReport` with only the memory fields populated (runtime
    fields zero). Same memory path as `simulate()`."""
    from .spec import load_hardware_yaml, ParallelismConfig, TrainingConfig
    from .pipeline import in_flight_microbatches
    from .report import SimulationReport, build_bottlenecks

    if isinstance(hardware, str):
        hardware = load_hardware_yaml(hardware)
    parallelism = parallelism or ParallelismConfig()
    training = training or TrainingConfig(micro_batch=1, global_batch=1, dtype="bf16")

    t = trace(model=model, parallelism=parallelism, training=training,
              hardware=hardware, gpus_per_node=hardware.gpus_per_node, workdir=workdir)

    # Same memory computation as _simulate_on_hardware: per-stage MemTracker
    # footprint scaled by the 1F1B in-flight count; binding stage = max peak.
    pp = parallelism.pipeline_model_parallel_size
    profiles = t.per_stage_profiles or []
    num_mb = max(1, training.global_batch_size
                 // (training.micro_batch_size * parallelism.data_parallel_size))
    pp_stage_memory_gb: list[float] = []
    stage_by_type: list[dict] = []
    for stage_rank, prof in enumerate(profiles):
        infl = in_flight_microbatches(pp, stage_rank, num_mb)
        pp_stage_memory_gb.append(prof.peak_gb(infl))
        stage_by_type.append(prof.peak_by_type(infl))
    if pp_stage_memory_gb:
        binding_stage = max(range(len(pp_stage_memory_gb)),
                            key=lambda s: pp_stage_memory_gb[s])
        peak_gb = pp_stage_memory_gb[binding_stage]
        mem_by_type = stage_by_type[binding_stage]
        peak_module = profiles[binding_stage].peak_module
    else:
        binding_stage, peak_gb, mem_by_type, peak_module = 0, 0.0, {}, ""
        pp_stage_memory_gb = [0.0]

    bottlenecks = build_bottlenecks(
        t.graph, peak_memory_gb=peak_gb, gpu_memory_GB=hardware.gpu_memory_GB,
        peak_module=peak_module, memory_by_type=mem_by_type, binding_stage=binding_stage,
    )
    return SimulationReport(
        step_time_ms=0.0, forward_ms=0.0, backward_ms=0.0, optimizer_ms=0.0,
        collective_total_ms=0.0, collective_exposed_ms=0.0, by_op_type_ms={},
        model_flops_per_step=0, achieved_tflops=0.0, mfu=0.0, hfu=0.0,
        param_bytes=mem_by_type.get("Parameter", 0) + mem_by_type.get("Buffer", 0),
        grad_bytes=mem_by_type.get("Gradient", 0),
        optimizer_state_bytes=mem_by_type.get("Optstate", 0),
        activation_bytes=mem_by_type.get("Activation", 0) + mem_by_type.get("Temp", 0),
        peak_memory_gb=peak_gb, pp_stage_memory_gb=pp_stage_memory_gb,
        per_pp_rank_step_time_ms=[],
        model=model, parallelism=parallelism, training=training,
        hardware=hardware, bottlenecks=bottlenecks,
    )
