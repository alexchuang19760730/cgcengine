#!/bin/bash
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
export TURBO_FIELDFARE_EXPERT_STATS=1
export TURBO_FIELDFARE_GPU_TIMING=1
export MTP_MODEL=""
BASE=(TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=64 TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top64.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync)
run() {
  local name=$1; shift
  env "$@" "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 64 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/pf_$name.log" 2>&1 || echo "$name exit=$?"
}
# 對拍：off vs on（同一 prompt 應逐 token 一致）
run off "${BASE[@]}"
run on "${BASE[@]}" TURBO_FIELDFARE_MOE_SPAC=1 TURBO_FIELDFARE_MOE_SPAC_PREFETCH=1 TURBO_FIELDFARE_MOE_SPAC_PREFETCH_K=16 TURBO_FIELDFARE_MOE_SPAC_ASYNC=1 TURBO_FIELDFARE_MOE_SPAC_SWAPS=2
echo "=== md5 比對 ==="
grep -oE '"text": "[^"]*"' pf_off.log | head -3
grep -oE '"text": "[^"]*"' pf_on.log | head -3
echo done
