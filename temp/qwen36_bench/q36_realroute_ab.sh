#!/bin/bash
# q36_realroute_ab.sh — real-route prefetch (TURBO_FIELDFARE_Q36_REALROUTE_PREFETCH)
# on/off 交錯 A/B。§13.97 假設 wall 245→200ms；預期 scheduled=0（結構性 no-op）。
set -e
ROOT=/Users/alexchuang/Documents/flashkv0516
WT=$ROOT/prime-agent-worktrees
BIN=$WT/turbo-fieldfare/.build/arm64-apple-macosx/release/TurboFieldfareCLI
MSG=$ROOT/temp/qwen36_bench/msg_code.json
LOG=$ROOT/temp/qwen36_bench/realroute_ab.log
: > "$LOG"

export TURBO_FIELDFARE_EXPERT_SLOTS=96
export TURBO_FIELDFARE_HOT_POOL=1
export TURBO_FIELDFARE_HOT_POOL_EXPERTS=80
export TURBO_FIELDFARE_HOT_POOL_PROFILE_SIZE=80
export TURBO_FIELDFARE_HOT_POOL_PROFILE=$WT/profiles/top80_code_prose.json
export TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
export TURBO_FIELDFARE_READ_WORKERS=8

run_one() {
  local name=$1 prefetch=$2
  if [ "$prefetch" = "on" ]; then export TURBO_FIELDFARE_Q36_REALROUTE_PREFETCH=1; else unset TURBO_FIELDFARE_Q36_REALROUTE_PREFETCH; fi
  echo "=== $name ($prefetch) ===" >> "$LOG"
  "$BIN" --model "$WT/qwen36-r3q4.gturbo" --trust-receipt \
    --messages-file "$MSG" --max-new 96 --temperature 0 \
    --repetition-penalty 1.0 --max-context 512 2>&1 | tail -1 >> "$LOG"
}

echo "A/B start: $(date '+%H:%M:%S') load=$(uptime | grep -oE 'load averages?: [0-9.]+' | grep -oE '[0-9.]+$')"
run_one "R1-off" off
run_one "R2-on " on
run_one "R3-off" off
run_one "R4-on " on
echo "A/B end:   $(date '+%H:%M:%S')"
