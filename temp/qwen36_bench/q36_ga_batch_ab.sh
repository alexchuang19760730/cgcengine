#!/bin/bash
# GatedAttn batch 化 (M=B) TTFT A/B — off vs on, 2 輪交錯 (§13.122)
cd /Users/alexchuang/Documents/flashkv0516
BIN=prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/release/TurboFieldfareCLI
M=prime-agent-worktrees/qwen36-r3q4la_ga_e4.gturbo
MSG=temp/qwen36_bench/msg_code.json
BASE="TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=96 TURBO_FIELDFARE_HOT_POOL_PROFILE=$M/profiles/top96_code_prose.json TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_READ_WORKERS=8"
ORDER=(0 1 1 0)
R=1
for m in "${ORDER[@]}"; do
  if [ "$m" = "0" ]; then NAME=ga_off; EXTRA="TURBO_FIELDFARE_Q36_GATEDATTN_BATCH=0"; else NAME=ga_on; EXTRA="TURBO_FIELDFARE_Q36_GATEDATTN_BATCH=1"; fi
  echo "=== R$R $NAME ==="
  env $BASE $EXTRA "$BIN" --model "$M" --trust-receipt --messages-file "$MSG" --max-new 8 \
    --temperature 0 --repetition-penalty 1.0 --max-context 512 2>&1 | grep -a 'stop='
  R=$((R+1))
done
