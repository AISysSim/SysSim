#!/usr/bin/env bash
# Multi-node real Megatron-core micro-benchmark: one model x parallelism -> <out>.real.json.
# Same per-case runner as run_megatron.sh (run_megatron.py), but launched ACROSS nodes: torchrun's
# rendezvous (node count, this node's rank, the master endpoint) is derived from the SLURM env, so
# --nnodes is NOT hardcoded — the identical script serves 2-node, 4-node, N-node jobs.
#
# Launch model: the batch script runs this ONCE PER NODE (srun --ntasks-per-node=1) inside the
# container; each instance starts a local torchrun of GPUS_PER_NODE procs and they rendezvous over
# c10d at MASTER_ADDR:MASTER_PORT. WORLD_SIZE = NNODES*GPUS_PER_NODE must equal TP*DP.
#
#   run_megatron_multi.sh <model.yaml> <tp> <dp> <mbs> <gbs> <recompute:none|selective|full> <distopt:0|1> <out.json>
#
# Required env (exported by the batch script and forwarded into the container with podman-hpc --env):
#   SLURM_NNODES        node count for this job
#   SLURM_NODEID        this node's rank in [0, NNODES)
#   MASTER_ADDR         hostname/IP of node 0 (computed on the host: scontrol show hostnames | head -1)
# Optional:
#   GPUS_PER_NODE       procs per node (default 4 — a full GH200 node)
#   MASTER_PORT         rendezvous port (default 29500)
#   SLURM_JOB_ID        used as the rdzv id (default falls back to a fixed tag)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="$1"; TP="$2"; DP="$3"; MBS="$4"; GBS="$5"; RC="$6"; DISTOPT="$7"; OUT="$8"
DOPT=""; [ "$DISTOPT" = "1" ] && DOPT="--distributed-optimizer"

NNODES="${SLURM_NNODES:?run_megatron_multi.sh must run under SLURM (SLURM_NNODES unset)}"
NODE_RANK="${SLURM_NODEID:?SLURM_NODEID unset: launch with srun --ntasks-per-node=1}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR unset: the batch script must export the node-0 hostname}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_ID="${SLURM_JOB_ID:-syssim-mn}"

echo "=== run_megatron_multi: node ${NODE_RANK}/${NNODES} -> ${MASTER_ADDR}:${MASTER_PORT}" \
     "(nproc_per_node=${GPUS_PER_NODE}, world=$(( NNODES * GPUS_PER_NODE )), tp=${TP} dp=${DP}) ==="

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --rdzv_id="$RDZV_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "$HERE/run_megatron.py" \
  --model "$MODEL" --tp "$TP" --dp "$DP" --mbs "$MBS" --gbs "$GBS" \
  --recompute "$RC" $DOPT --train-iters 8 --warmup 3 --out "$OUT"
