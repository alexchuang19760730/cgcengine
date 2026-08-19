#!/bin/bash
# Small-budget interleaved A/B: the configs behind the 60-65% baseline.
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json

export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""

run_case() {
  local name=$1; shift
  env "$@" \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 96 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/s9_$name.log" 2>&1 || echo "$name exit=$?"
}

for round in 1 2; do
  run_case "r${round}_s48base" TURBO_FIELDFARE_EXPERT_SLOTS=48
  run_case "r${round}_s48p32"  TURBO_FIELDFARE_EXPERT_SLOTS=48 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=32 TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top64.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
  run_case "r${round}_s64base" TURBO_FIELDFARE_EXPERT_SLOTS=64
  run_case "r${round}_s64p48"  TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=48 TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top64.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
done
echo done
