#!/bin/bash
# §13.94 乾淨窗口 4 輪交錯 64 tok HEAD_XSMEM on/off A/B
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r3.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
P80=$ROOT/prime-agent-worktrees/qwen36-r3.gturbo/profiles/top80_code.json
BASE_ENV="TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P80 TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_READ_WORKERS=8 TURBO_FIELDFARE_EXPERT_STATS=1 TURBO_FIELDFARE_GPU_TIMING=1"

run_case() {
  local name=$1 xsmem=$2
  env $BASE_ENV TURBO_FIELDFARE_Q36_HEAD_XSMEM=$xsmem \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 64 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/clean_$name.log" 2>&1 || echo "$name exit=$?"
  echo "done $name"
}

for round in 1 2 3 4; do
  run_case "r${round}_base" 0
  run_case "r${round}_xsmem" 1
done
echo ALLDONE > "$OUT/q36_head_clean.done"
