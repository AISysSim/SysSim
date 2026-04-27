#!/bin/bash
# Pull the latest Blackhole profiling CSVs from the EPCC remote, then
# train one XGBoost efficiency model per operator that has at least
# `MIN_ROWS` valid rows. Safe to run repeatedly while a remote sweep is
# still appending data — the resume-friendly profile_operator path is
# the source of truth, not this script.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-bobbysong@tenstorrent-epcc.uaru-bull.ts.net}"
REMOTE_DATA="${REMOTE_DATA:-/tmp/syssim-prof/data}"
LOCAL_DATA="${LOCAL_DATA:-data/profiling}"
LOCAL_MODELS="${LOCAL_MODELS:-data/trained_models}"
PLATFORM="${PLATFORM:-tt_bh_p150b}"
MIN_ROWS="${MIN_ROWS:-200}"
PYTHON_BIN="${PYTHON_BIN:-.venv-train/bin/python}"

mkdir -p "$LOCAL_DATA" "$LOCAL_MODELS"

echo "[pull] rsync $REMOTE_HOST:$REMOTE_DATA/ -> $LOCAL_DATA/"
rsync -e ts-ssh -av --include='*.csv' --include='profiling_manifest.yaml' \
    --exclude='*' "$REMOTE_HOST:$REMOTE_DATA/" "$LOCAL_DATA/" \
    2>&1 | tail -5

for op in gemm attn rmsnorm silu; do
    csv="$LOCAL_DATA/${op}_${PLATFORM}_data.csv"
    if [ ! -s "$csv" ]; then
        echo "[skip] $op: $csv missing or empty"
        continue
    fi
    rows=$(($(wc -l < "$csv") - 1))
    valid=$($PYTHON_BIN - <<PY
import pandas as pd
df = pd.read_csv("$csv")
print((df["t_measured_ms"] > 0).sum())
PY
)
    echo "[$op] csv rows=$rows valid=$valid"
    if [ "$valid" -lt "$MIN_ROWS" ]; then
        echo "[skip] $op: $valid valid rows < MIN_ROWS=$MIN_ROWS"
        continue
    fi

    out="$LOCAL_MODELS/${op}_${PLATFORM}_xgb.pth"
    echo "[train] $op -> $out"
    $PYTHON_BIN -m syssim.compute.compute_cost_profiler \
        --operator "$op" \
        --data-path "$csv" \
        --output "$out" \
        --backend xgboost 2>&1 | tail -8
done

echo "[done] models in $LOCAL_MODELS:"
ls -la "$LOCAL_MODELS" 2>/dev/null | grep "${PLATFORM}_xgb.pth" || echo "  (none yet)"
