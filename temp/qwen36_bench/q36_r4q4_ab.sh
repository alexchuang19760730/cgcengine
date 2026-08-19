#!/bin/bash
# q36_r4q4_ab.sh — r4 (fp16 head/shared) vs r4q4 (4-bit head/shared) 128 tok 交錯 A/B
# 生產設定（§13.58/13.61）：96 slots + pool80 + sync preload + READ_WORKERS=8
set -e
ROOT=/Users/alexchuang/Documents/flashkv0516
WT=$ROOT/prime-agent-worktrees
BIN=$WT/turbo-fieldfare/.build/arm64-apple-macosx/release/TurboFieldfareCLI
MSG=$ROOT/temp/qwen36_bench/msg_code.json
LOG=$ROOT/temp/qwen36_bench/r4q4_ab.log
: > "$LOG"

export TURBO_FIELDFARE_EXPERT_SLOTS=96
export TURBO_FIELDFARE_HOT_POOL=1
export TURBO_FIELDFARE_HOT_POOL_EXPERTS=80
export TURBO_FIELDFARE_HOT_POOL_PROFILE_SIZE=80
export TURBO_FIELDFARE_HOT_POOL_PROFILE=$WT/profiles/top80_code_prose.json
export TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
export TURBO_FIELDFARE_READ_WORKERS=8

run_one() {
  local name=$1 model=$2
  echo "=== $name ===" >> "$LOG"
  "$BIN" --model "$model" --trust-receipt \
    --messages-file "$MSG" --max-new 128 --temperature 0 \
    --repetition-penalty 1.0 --max-context 512 2>&1 | tail -1 >> "$LOG"
}

echo "A/B start: $(date '+%H:%M:%S') load=$(uptime | grep -oE 'load averages?: [0-9.]+' | grep -oE '[0-9.]+$')"
run_one "R1-r4    " "$WT/qwen36-r4.gturbo"
run_one "R2-r4q4  " "$WT/qwen36-r4q4.gturbo"
run_one "R3-r4    " "$WT/qwen36-r4.gturbo"
run_one "R4-r4q4  " "$WT/qwen36-r4q4.gturbo"
echo "A/B end:   $(date '+%H:%M:%S')"
