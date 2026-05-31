#!/usr/bin/env bash
# Per-case real Megatron-core micro-benchmark: one model x parallelism -> <out>.real.json.
# Run inside the localhost/mksit/syssim image on a GPU node. Launches torchrun with
# nproc_per_node = TP*DP. Args mirror examples/megatron/run_syssim.sh's CASES rows.
#   run_megatron.sh <model.yaml> <tp> <dp> <mbs> <gbs> <recompute:none|selective|full> <distopt:0|1> <out.json>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="$1"; TP="$2"; DP="$3"; MBS="$4"; GBS="$5"; RC="$6"; DISTOPT="$7"; OUT="$8"
DOPT=""; [ "$DISTOPT" = "1" ] && DOPT="--distributed-optimizer"
torchrun --nproc_per_node="$(( TP * DP ))" --nnodes=1 "$HERE/run_megatron.py" \
  --model "$MODEL" --tp "$TP" --dp "$DP" --mbs "$MBS" --gbs "$GBS" \
  --recompute "$RC" $DOPT --train-iters 8 --warmup 3 --out "$OUT"
