#!/usr/bin/env bash
# Run SysSim over the validation matrix. Run inside the localhost/mksit/syssim image on a GPU
# node (the tracer needs CUDA). The CASES array is the SINGLE SOURCE OF TRUTH for the matrix;
# run_megatron.py consumes the same rows so both sides use identical parallelism/batch settings.
# Each case is run with and WITHOUT activation recompute (suffix _rc / _norc).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
HW="$REPO/examples/configs/hardware/isambard_gh200_4gpu.yaml"
OUT="$REPO/docs/megatron_gh200_validation/results"
mkdir -p "$OUT"

# name  tp dp mbs gbs distopt   (gbs = dp*mbs -> one micro-batch per rank, no grad accumulation;
#                                distopt=1 enables ZeRO-1 optimizer-state sharding for DP, matching
#                                the real runs which use megatron's distributed optimizer.)
CASES=(
  "1gpu 1 1 1 1 0"
  "tp2  2 1 1 1 0"
  "tp4  4 1 1 1 0"
  "dp4  1 4 1 4 1"
)
RECOMPUTE=( "none" "full" )   # report both with and without activation recompute

for MODEL in llama3-8b qwen3-8b; do
  for c in "${CASES[@]}"; do
    read -r name tp dp mbs gbs distopt <<< "$c"
    for rc in "${RECOMPUTE[@]}"; do
      if [ "$rc" = "none" ]; then sfx="norc"; else sfx="rc"; fi
      echo "=== SysSim ${MODEL}/${name}_${sfx}  (tp=$tp dp=$dp mbs=$mbs gbs=$gbs recompute=$rc distopt=$distopt) ==="
      python "$HERE/_sim_case.py" \
        --model "$REPO/examples/megatron/${MODEL}.yaml" --hardware "$HW" \
        --tp "$tp" --dp "$dp" --mbs "$mbs" --gbs "$gbs" --recompute "$rc" \
        --distributed-optimizer "$distopt" \
        --out "$OUT/${MODEL}_${name}_${sfx}.sim.json"
    done
  done
done
echo "SysSim matrix done -> $OUT"
