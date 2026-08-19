#!/bin/bash
# 交錯 A/B：base(64slots+pool64) vs spac+prefetch，各 2 輪 × 128 tok
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
export MTP_MODEL=""
BASE=(TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=64 TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top64.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_STATS=1 TURBO_FIELDFARE_GPU_TIMING=1)
run() {
  local name=$1; shift
  env "$@" "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/ab_$name.log" 2>"$OUT/ab_$name.err" || echo "$name exit=$?"
}
for r in 1 2; do
  run "base_r$r" "${BASE[@]}"
  run "pf_r$r" "${BASE[@]}" TURBO_FIELDFARE_MOE_SPAC=1 TURBO_FIELDFARE_MOE_SPAC_ASYNC=1 TURBO_FIELDFARE_MOE_SPAC_SWAPS=2 TURBO_FIELDFARE_MOE_SPAC_PREFETCH=1 TURBO_FIELDFARE_MOE_SPAC_PREFETCH_K=16
done
echo done
