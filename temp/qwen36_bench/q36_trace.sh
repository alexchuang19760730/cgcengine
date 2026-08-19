#!/bin/bash
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r4.gturbo
OUT=$ROOT/temp/qwen36_bench
MSG=$OUT/msg_code.json
export MTP_MODEL=""
env TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=64 \
    TURBO_FIELDFARE_HOT_POOL_PROFILE=$OUT/qwen36_top64.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync \
    TURBO_FIELDFARE_EXPERT_TRACE=$OUT/trace_rt.csv \
    "$BIN" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$OUT/trace_run.log" 2>"$OUT/trace_run.err"
echo done
