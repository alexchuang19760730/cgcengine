#!/bin/bash
# qwen36-r3 cold-cache A/B: base vs pool80 sync vs pool80 async.
# Flush = read 21GB qwen36-r4 (NOT r3) to evict r3 pages from 16GB RAM.
set -uo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r3.gturbo
FLUSH=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
P80=$OUT/qwen36_top80.json

export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""

flush_cache() {
  echo "FLUSH start $(date '+%H:%M:%S')"
  cat "$FLUSH"/model_weights.bin "$FLUSH"/packed_experts/* > /dev/null 2>&1
  echo "FLUSH done $(date '+%H:%M:%S')"
}

run_case() {
  local name=$1; shift
  /usr/bin/time -l env "$@" \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/r3_$name.log" 2> "$OUT/r3_$name.time" || echo "$name exit=$?"
}

for round in 1 2; do
  flush_cache
  run_case "r${round}_base" TURBO_FIELDFARE_EXPERT_SLOTS=96
  flush_cache
  run_case "r${round}_sync" TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P80 TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
  flush_cache
  run_case "r${round}_async" TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P80 TURBO_FIELDFARE_HOT_POOL_PRELOAD=async
done
echo ALLDONE
