#!/bin/bash
# qwen36-r3 memory Pareto: 48/64/96 slots x base/pool, r3 SELF profiles.
# Hot cache (Pareto methodology matches §13.57 r4 run; sync preload).
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r3.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
P32=$OUT/qwen36_r3_self_top32.json
P48=$OUT/qwen36_r3_self_top48.json
P80=$OUT/qwen36_r3_self_top80.json

export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""

run_case() {
  local name=$1; shift
  /usr/bin/time -l env "$@" \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/r3p_$name.log" 2> "$OUT/r3p_$name.time" || echo "$name exit=$?"
}

for round in 1 2; do
  run_case "r${round}_s48base" TURBO_FIELDFARE_EXPERT_SLOTS=48
  run_case "r${round}_s48p32"  TURBO_FIELDFARE_EXPERT_SLOTS=48 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=32 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P32 TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
  run_case "r${round}_s64base" TURBO_FIELDFARE_EXPERT_SLOTS=64
  run_case "r${round}_s64p48"  TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=48 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P48 TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
  run_case "r${round}_s96base" TURBO_FIELDFARE_EXPERT_SLOTS=96
  run_case "r${round}_s96p80"  TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P80 TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
done
echo "PARETO DONE $(date '+%H:%M:%S')"
