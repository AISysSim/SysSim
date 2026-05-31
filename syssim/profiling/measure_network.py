"""Network profiling: measure real NCCL collective bus-bandwidth + latency floor on the hardware
and derive the per-dimension `topology:` parameters for a hardware YAML.

This is the network analogue of the compute layer profiler: it promotes the ad-hoc NCCL bandwidth
probe into the standard `syssim profile --network` workflow, so users calibrate the network model
from measurement instead of hand-entering datasheet numbers.

`run_network_profile` is a torch.distributed program — launch it under torchrun (single node) or the
multi-node recipe in examples/megatron/run_megatron_multi.sh (srun --mpi=pmi2 + podman-hpc
--openmpi-pmi2 + LD_PRELOAD the container NCCL). Every rank joins the collectives; global rank 0
aggregates the measurements and (via `derive_topology`) turns them into the topology block.
"""
from __future__ import annotations

import json
import os

# Message-size sweep (bytes): the largest fixes the saturated bus-bandwidth.
_DEFAULT_SIZES = (1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20)  # 1 .. 256 MiB
# A tiny message isolates the pure latency floor (a 1 MB message still has meaningful transfer time,
# which would inflate the derived per-hop latency).
_LATENCY_PROBE_BYTES = 1024


def _bench_allreduce(grp, n_members, sizes, *, is_timer, iters, warmup):
    """Ring all-reduce over the process group `grp`. ALL members must call this (the collectives are
    group-collective); only the designated timer member records. Returns the measurement dict on the
    timer rank, else None."""
    import torch
    import torch.distributed as dist

    def run(nbytes):
        x = torch.ones(max(1, nbytes // 2), dtype=torch.bfloat16, device="cuda")
        for _ in range(warmup):
            dist.all_reduce(x, group=grp)
        torch.cuda.synchronize()
        dist.barrier(group=grp)
        if is_timer:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
        for _ in range(iters):
            dist.all_reduce(x, group=grp)
        if is_timer:
            e1.record()
        torch.cuda.synchronize()
        return (e0.elapsed_time(e1) / iters / 1e3) if is_timer else None  # seconds

    t_big = run(max(sizes))
    t_small = run(_LATENCY_PROBE_BYTES)   # tiny message -> pure latency floor
    if not is_timer:
        return None
    # busBW (algorithm-bandwidth x 2(n-1)/n): the standard NCCL bus-bandwidth metric.
    busbw = 2 * (n_members - 1) / n_members * max(sizes) / t_big / 1e9
    return {"busBW_GBps": busbw, "latency_floor_us": t_small * 1e6,
            "n": n_members, "size_bytes": int(max(sizes))}


def run_network_profile(*, gpus_per_node, nodes, sizes=None, iters=20, warmup=5):
    """Measure intra-node and (when nodes>1) inter-node all-reduce busBW + latency floor.

    torch.distributed program: reads WORLD_SIZE/RANK/LOCAL_RANK from the env. Returns
    (rank, measurements) where measurements is a dict on global rank 0 (None elsewhere):
        {"intra": {busBW_GBps, latency_floor_us, n, size_bytes}, "inter": {...same...}}
    The inter group is the strided cross-node group [0, gpn, 2*gpn, ...] (one GPU per node),
    matching how a tensor-parallel-then-data-parallel DP group spans nodes.
    """
    import torch
    import torch.distributed as dist

    sizes = tuple(sizes) if sizes else _DEFAULT_SIZES
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != gpus_per_node * nodes:
        raise ValueError(f"world_size ({world}) != gpus_per_node*nodes ({gpus_per_node*nodes})")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % gpus_per_node)))

    # Every rank must create every group (new_group is collective), even non-members.
    intra_ranks = list(range(gpus_per_node))                     # the GPUs of node 0
    intra_grp = dist.new_group(intra_ranks)
    inter_ranks = list(range(0, world, gpus_per_node)) if nodes > 1 else []
    inter_grp = dist.new_group(inter_ranks) if nodes > 1 else None

    meas = {}
    if rank in intra_ranks:
        r = _bench_allreduce(intra_grp, len(intra_ranks), sizes,
                             is_timer=(rank == intra_ranks[0]), iters=iters, warmup=warmup)
        if r is not None:
            meas["intra"] = r
    if nodes > 1 and rank in inter_ranks:
        r = _bench_allreduce(inter_grp, len(inter_ranks), sizes,
                             is_timer=(rank == inter_ranks[0]), iters=iters, warmup=warmup)
        if r is not None:
            meas["inter"] = r

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return rank, (meas if rank == 0 else None)


def derive_topology(measurements, *, gpus_per_node, nodes):
    """Pure: measured busBW/latency-floor -> a per-dimension `topology:` dict.

    dim 0 = intra-node NVLink mesh (`fully_connected`, gpus_per_node endpoints); dim 1 (if nodes>1)
    = inter-node fabric (`switch`, one uplink per node, `nodes` endpoints).

    bandwidth: the topology `bandwidth[d]` is the per-endpoint aggregate the flow sim divides by the
      dimension degree (fully_connected -> size-1 peer links; switch -> 1 uplink). A ring all-reduce
      over a full-duplex dimension saturates at ~the per-link bandwidth, so per-link ~= measured
      busBW and aggregate = busBW * degree.
    latency: a ring all-reduce chains 2*(size-1) hops, so per-hop latency = floor / (2*(size-1)),
      mirroring the calibrated intra-node 12000 ns (= 72 us floor / 6 hops).
    """
    dims, size, bandwidth, latency = [], [], [], []

    def _one(typ, n, m):
        degree = max(1, n - 1) if typ == "fully_connected" else 1
        hops = max(1, 2 * (n - 1))
        dims.append(typ)
        size.append(int(n))
        bandwidth.append(round(float(m["busBW_GBps"]) * degree))
        latency.append(round(float(m["latency_floor_us"]) * 1000.0 / hops))

    _one("fully_connected", gpus_per_node, measurements["intra"])
    if nodes > 1:
        _one("switch", nodes, measurements["inter"])
    return {"dims": dims, "size": size, "bandwidth": bandwidth, "latency": latency}


def emit_topology(topology, measurements, *, out_dir=None, hardware_path=None):
    """Write `<out_dir>/network.json` (raw measurements + derived block), print the topology block,
    and (when hardware_path is given) update that hardware YAML's `topology:` in place."""
    block_lines = ["topology:",
                   f"  dims:      [ {', '.join(topology['dims'])} ]",
                   f"  size:      [ {', '.join(str(s) for s in topology['size'])} ]",
                   f"  bandwidth: [ {', '.join(str(b) for b in topology['bandwidth'])} ]",
                   f"  latency:   [ {', '.join(str(l) for l in topology['latency'])} ]"]
    block = "\n".join(block_lines)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "network.json"), "w") as f:
            json.dump({"measurements": measurements, "topology": topology}, f, indent=2)

    if hardware_path:
        import yaml as _yaml
        with open(hardware_path) as f:
            hw = _yaml.safe_load(f) or {}
        hw["topology"] = topology
        with open(hardware_path, "w") as f:
            _yaml.safe_dump(hw, f, sort_keys=False)

    return block
