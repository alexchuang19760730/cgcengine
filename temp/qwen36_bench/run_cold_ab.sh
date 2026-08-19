#!/bin/bash
# Cold page-cache A/B: flush (read 18GB r3) then run base vs pool80, 2 interleaved rounds.
# Purpose: quantify pinned-pool wall/tok/s value under production page-cache thrash.
set -uo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
R3=$ROOT/prime-agent-worktrees/qwen36-r3.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
P80=$OUT/qwen36_top80.json

export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""

flush_cache() {
  # Read ~18GB of r3 (NOT r4) to evict r4's file-backed pages from 16GB RAM.
  echo "FLUSH start $(date '+%H:%M:%S')"
  cat "$R3"/model_weights.bin "$R3"/packed_experts/* > /dev/null 2>&1
  echo "FLUSH done $(date '+%H:%M:%S')"
}

run_case() {
  local name=$1; shift
  /usr/bin/time -l env "$@" \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/cc_$name.log" 2> "$OUT/cc_$name.time" || echo "$name exit=$?"
}

for round in 1 2; do
  flush_cache
  run_case "r${round}_base" TURBO_FIELDFARE_EXPERT_SLOTS=96
  flush_cache
  run_case "r${round}_p80" TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$P80 TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
done
echo ALLDONE
