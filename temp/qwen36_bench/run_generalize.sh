#!/bin/bash
# Profile generalization: pool80 built from code+prose, now evaluated on prose gen.
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench

export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""

run_case() {
  local name=$1; shift
  env "$@" \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$OUT/msg_prose.json" \
    --max-new 96 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/g_$name.log" 2>&1 || echo "$name exit=$?"
}

for round in 1 2; do
  run_case "r${round}_base" TURBO_FIELDFARE_EXPERT_SLOTS=96
  run_case "r${round}_p80"  TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top80.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
done
echo done
