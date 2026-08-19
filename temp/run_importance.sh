#!/bin/bash
# Layer-importance run: Q36IMP (per-layer RMS of attnOut/moeOut/hidden) + routing trace.
set -euo pipefail
ROOT=/Users/alexchuang/Documents/flashkv0516
BIN=$ROOT/prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=$ROOT/prime-agent-worktrees/qwen36-r3q4la_ga_e4.gturbo
OUT=$ROOT/temp/qwen36_bench
PROFILE=$MODEL/profiles/top112_code_prose.json
LOG=/tmp/q36_imp_run.log
TRACE=/tmp/q36_imp_trace.csv
rm -f "$TRACE" "$LOG"

export MTP_MODEL=""
export TURBO_FIELDFARE_EXPERT_SLOTS=96
export TURBO_FIELDFARE_HOT_POOL=1
export TURBO_FIELDFARE_HOT_POOL_EXPERTS=112
export TURBO_FIELDFARE_HOT_POOL_PROFILE="$PROFILE"
export TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
export TURBO_FIELDFARE_EXPERT_READ_WORKERS=8
export Q36_DUMP_LAYERS=1
export TURBO_FIELDFARE_EXPERT_TRACE="$TRACE"

"$BIN" --model "$MODEL" --trust-receipt \
  --messages-file "$OUT/msg_code_prose_mix.json" \
  --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 2048 \
  > "$LOG" 2>&1
echo "run done, exit=$?"
echo "Q36IMP tables: $(grep -c '^Q36IMP tokens=' "$LOG" || true)"
echo "trace rows: $(wc -l < "$TRACE")"
