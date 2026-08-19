#!/bin/bash
# §13.99 lm_head/shared 4-bit A/B: r3 (fp16) vs r3q4 (4-bit) 128 tok 交錯 2 輪
set -uo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
BASE_ENV="TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_READ_WORKERS=8 TURBO_FIELDFARE_EXPERT_STATS=1 TURBO_FIELDFARE_GPU_TIMING=1"

run_case() {
  local name=$1 model=$2
  env $BASE_ENV TURBO_FIELDFARE_HOT_POOL_PROFILE="$ROOT/prime-agent-worktrees/$model/profiles/top80_code.json" \
    "$BIN" --model "$ROOT/prime-agent-worktrees/$model" --trust-receipt --messages-file "$MSG" \
    --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/q4ab_$name.log" 2>&1 || echo "$name exit=$?"
  echo "done $name"
}

for round in 1 2; do
  run_case "r${round}_r3" qwen36-r3.gturbo
  run_case "r${round}_q4" qwen36-r3q4.gturbo
done
echo ALLDONE > "$OUT/q36_q4ab.done"
