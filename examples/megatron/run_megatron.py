"""Real megatron-core training micro-benchmark — ground truth for SysSim validation.

Runs a short real fwd+bwd+optimizer training loop for ONE model x parallelism case and
writes {step_time_ms, peak_memory_gb, oom} JSON. This is the "real Megatron" side; the
"sim" side is examples/megatron/_sim_case.py and compare.py joins them.

Apples-to-apples: the trained model is the SAME megatron-core GPTModel that SysSim traces
(via syssim.training.resolve_megatron_provider + the local GPT layer spec), built on REAL
cuda with real params/grads/optimizer.

Launch under torchrun (WORLD_SIZE must equal TP*DP; DP = WORLD_SIZE // TP):
  torchrun --nproc_per_node=<TP*DP> --nnodes=1 examples/megatron/run_megatron.py \
    --model examples/megatron/qwen3-8b.yaml --tp 4 --dp 1 --mbs 1 --gbs 1 \
    --recompute none --train-iters 8 --warmup 3 --out <path.real.json>
"""
import argparse
import json
import os
import time

import torch

from syssim.training.spec import load_model_yaml, ParallelismConfig, TrainingConfig
from syssim.training.sources import resolve_megatron_provider
from syssim.training.runner import make_lm_data_iterator, _vocab_size_for_tp

_NONE = {"", "none", "None"}


def make_lm_forward_step():
    """forward_step_func for Megatron's schedule, for REAL training.

    Unlike syssim.training.runner.make_lm_forward_step (which is FakeTensorMode-only:
    it pulls raw `parallel_output=True` logits and runs a plain F.cross_entropy, valid
    only under tracing), this feeds `labels` straight into GPTModel so Megatron computes
    its own vocab-parallel cross-entropy on the TP-sharded logits — the correct, real path.
    Returns (loss_per_token, loss_func) per Megatron's contract.
    """
    def forward_step(data_iterator, model):
        batch = next(data_iterator)
        output = model(batch["input_ids"], batch["position_ids"], None,
                       labels=batch["labels"])  # [b, s] per-token loss

        def loss_func(loss_tensor):
            loss = loss_tensor.mean()
            return loss, {"lm_loss": loss.detach()}

        return output, loss_func
    return forward_step


def build_model(provider, vocab, max_pos):
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.module import Float16Module
    from megatron.core.distributed import (
        DistributedDataParallel as DDP, DistributedDataParallelConfig as DDPConfig,
    )

    model = GPTModel(
        config=provider, transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab, max_sequence_length=max_pos,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
        parallel_output=True,
    ).cuda()
    model = model.train()
    for p in model.parameters():
        tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(p)
    if provider.bf16 or provider.fp16:
        model = Float16Module(provider, model)
    return model, DDP, DDPConfig


def run(args):
    tp, dp = args.tp, args.dp
    world = int(os.environ["WORLD_SIZE"])
    if world != tp * dp:
        raise ValueError(f"WORLD_SIZE ({world}) must equal TP*DP ({tp*dp})")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
    from megatron.core.pipeline_parallel import get_forward_backward_func

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(42)

    recompute = None if args.recompute in _NONE else args.recompute
    model_cfg = load_model_yaml(args.model)
    par = ParallelismConfig(tp=tp, dp=dp)
    train = TrainingConfig(micro_batch=args.mbs, global_batch=args.gbs,
                           dtype="bf16", recompute=recompute)
    provider = resolve_megatron_provider(model_cfg, par, train)
    # Flow --recompute through the real provider, matching what SysSim simulates.
    provider.recompute_granularity = recompute
    if recompute == "full":
        provider.recompute_method = "uniform"
        provider.recompute_num_layers = 1
    elif recompute == "selective":
        provider.recompute_granularity = "selective"

    vocab = _vocab_size_for_tp(model_cfg.vocab_size, tp)
    seq = model_cfg.seq_length
    max_pos = model_cfg.max_position_embeddings or seq

    use_dist_opt = dp > 1 and args.distributed_optimizer
    model, DDP, DDPConfig = build_model(provider, vocab, max_pos)
    ddp_cfg = DDPConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=use_dist_opt,
    )
    model = DDP(config=provider, ddp_config=ddp_cfg, module=model)

    opt_cfg = OptimizerConfig(
        optimizer="adam", lr=1e-4, bf16=True,
        use_distributed_optimizer=use_dist_opt,
    )
    optimizer = get_megatron_optimizer(opt_cfg, [model])

    fb = get_forward_backward_func()
    num_mb = max(1, args.gbs // (args.mbs * dp))
    fwd_step = make_lm_forward_step()
    data_iter = make_lm_data_iterator(vocab, args.mbs, seq)

    per_iter_ms = []
    for i in range(args.train_iters):
        if i == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        t0 = time.perf_counter()
        fb(forward_step_func=fwd_step, data_iterator=data_iter, model=[model],
           num_microbatches=num_mb, seq_length=seq, micro_batch_size=args.mbs,
           forward_only=False)
        optimizer.step()
        torch.cuda.synchronize()
        per_iter_ms.append((time.perf_counter() - t0) * 1000.0)

    if os.environ.get("SYSSIM_PROFILE"):
        import sys
        from torch.profiler import profile, ProfilerActivity
        # All ranks must run the profiled fwd/bwd: fb() contains TP collectives that deadlock if
        # only rank 0 calls them. Only rank 0 prints the aggregated breakdown.
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                optimizer.zero_grad()
                fb(forward_step_func=fwd_step, data_iterator=data_iter, model=[model],
                   num_microbatches=num_mb, seq_length=seq, micro_batch_size=args.mbs,
                   forward_only=False)
                optimizer.step()
                torch.cuda.synchronize()
        if rank == 0:
            ka = prof.key_averages()
            cu = lambda e: (getattr(e, "self_device_time_total", 0) or
                            getattr(e, "self_cuda_time_total", 0))
            total = sum(cu(e) for e in ka) / 1000.0 / 3.0
            print("REAL GPU-busy ms/step (rank0) = %.1f" % total, file=sys.stderr)
            fam = {}
            for e in ka:
                nm = e.key.lower()
                if any(k in nm for k in ("nccl", "allreduce", "reduce_scatter", "all_gather")):
                    f = "collective"
                elif "multi_tensor" in nm or "fused_adam" in nm or "adam" in nm:
                    f = "optimizer"
                elif any(k in nm for k in ("nvjet", "gemm", "cutlass", "xmma", "wgmma", "s16816")):
                    f = "gemm"
                elif "softmax" in nm:
                    f = "softmax"
                elif "dropout" in nm:
                    f = "dropout"
                elif "norm" in nm:
                    f = "norm"
                elif "memcpy" in nm or "memset" in nm:
                    f = "memcpy"
                elif "elementwise" in nm:
                    f = "elementwise"
                else:
                    f = "other"
                fam[f] = fam.get(f, 0) + cu(e)
            for f in sorted(fam, key=lambda x: -fam[x]):
                print("  FAM %-12s %.1f" % (f, fam[f] / 1000.0 / 3.0), file=sys.stderr)
            for e in sorted(ka, key=lambda e: -cu(e))[:14]:
                print("  %-46s %.2f" % (e.key[:46], cu(e) / 1000.0 / 3.0), file=sys.stderr)

    steady = sorted(per_iter_ms[args.warmup:])
    step_time_ms = steady[len(steady) // 2] if steady else None

    peak = torch.tensor([torch.cuda.max_memory_allocated() / 1e9], device="cuda")
    torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
    peak_memory_gb = float(peak.item())

    return rank, step_time_ms, peak_memory_gb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="model YAML path")
    p.add_argument("--hardware", default=None, help="accepted+ignored for parity with sim runner")
    p.add_argument("--out", required=True, help="output JSON path (rank 0 writes)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--mbs", type=int, default=1, help="micro batch size")
    p.add_argument("--gbs", type=int, default=1, help="global batch size")
    p.add_argument("--recompute", default="none", help="none|selective|full")
    p.add_argument("--distributed-optimizer", action="store_true",
                   help="enable ZeRO-1 distributed optimizer (only takes effect when DP>1)")
    p.add_argument("--train-iters", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    oom = False
    step_time_ms = peak_memory_gb = None
    rank = int(os.environ.get("RANK", 0))
    try:
        rank, step_time_ms, peak_memory_gb = run(args)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e).lower()
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg:
            oom = True
            step_time_ms = None
            try:
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
            except Exception:
                peak_memory_gb = None
        else:
            raise

    # Skip the barrier on OOM: ranks are in an inconsistent state (some may have
    # OOM'd mid-collective), so a barrier would hang rather than synchronize.
    if not oom and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        out = {"step_time_ms": step_time_ms, "peak_memory_gb": peak_memory_gb, "oom": oom}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}: step={step_time_ms} peak={peak_memory_gb} oom={oom}")

    from megatron.core import parallel_state
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
