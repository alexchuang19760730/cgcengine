#!/bin/bash
# A/B: hot-pool auto-refresh on/off, prose 220 tok, 2 interleaved rounds.
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
PROF=$OUT/qwen36_top80.json

export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""
export TURBO_FIELDFARE_EXPERT_SLOTS=96
export TURBO_FIELDFARE_HOT_POOL=1
export TURBO_FIELDFARE_HOT_POOL_EXPERTS=80
export TURBO_FIELDFARE_HOT_POOL_PROFILE="$PROF"
export TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync

run_case() {
  local name=$1; shift
  env "$@" \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$OUT/msg_prose.json" \
    --max-new 220 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/rf_$name.log" 2>&1 || echo "$name exit=$?"
}

for round in 1 2; do
  run_case "r${round}_off"
  run_case "r${round}_on"  TURBO_FIELDFARE_HOT_POOL_REFRESH=1 TURBO_FIELDFARE_HOT_POOL_REFRESH_INTERVAL=100 TURBO_FIELDFARE_HOT_POOL_REFRESH_MIN_OVERLAP=0.75 TURBO_FIELDFARE_HOT_POOL_REFRESH_MAX_SWAPS=16
done
echo done
