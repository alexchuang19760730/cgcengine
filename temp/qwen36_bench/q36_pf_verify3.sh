#!/bin/bash
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
export MTP_MODEL=""
BASE=(TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=64 TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top64.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_STATS=1)
# 無 GPU_TIMING（乾淨輸出）
env "${BASE[@]}" "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" --max-new 64 --temperature 0 --repetition-penalty 1.0 --max-context 512 > "$OUT/v3_base.log" 2>"$OUT/v3_base.err"
env "${BASE[@]}" TURBO_FIELDFARE_MOE_SPAC=1 TURBO_FIELDFARE_MOE_SPAC_ASYNC=1 TURBO_FIELDFARE_MOE_SPAC_SWAPS=2 "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" --max-new 64 --temperature 0 --repetition-penalty 1.0 --max-context 512 > "$OUT/v3_spac.log" 2>"$OUT/v3_spac.err"
env "${BASE[@]}" TURBO_FIELDFARE_MOE_SPAC=1 TURBO_FIELDFARE_MOE_SPAC_ASYNC=1 TURBO_FIELDFARE_MOE_SPAC_SWAPS=2 TURBO_FIELDFARE_MOE_SPAC_PREFETCH=1 TURBO_FIELDFARE_MOE_SPAC_PREFETCH_K=16 "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" --max-new 64 --temperature 0 --repetition-penalty 1.0 --max-context 512 > "$OUT/v3_pf.log" 2>"$OUT/v3_pf.err"
echo DONE > "$OUT/v3.done"
