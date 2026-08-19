#!/bin/bash
# qwen36-r3 cold-cache A/B: eager per-layer async pool preload vs lazy async
# vs sync vs base. r3 self top80 profile; flush = read 21GB r4 to evict r3 pages.
# Usage: run_eager_ab.sh <round>  (round = 1 or 2)
set -uo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r3.gturbo
FLUSH=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
P80=$OUT/qwen36_r3_self_top80.json
R=$1

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
    > "$OUT/e${R}_$name.log" 2> "$OUT/e${R}_$name.time" || echo "$name exit=$?"
}

flush_cache
run_case "base" TURBO_FIELDFARE_EXPERT_SLOTS=96
flush_cache
run_case "sync" TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P80 TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
flush_cache
run_case "lazy_async" TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P80 TURBO_FIELDFARE_HOT_POOL_PRELOAD=async
flush_cache
run_case "eager_async" TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P80 TURBO_FIELDFARE_HOT_POOL_PRELOAD=async TURBO_FIELDFARE_EAGER_POOL_OPEN=1
echo "ROUND$R DONE $(date '+%H:%M:%S')"
