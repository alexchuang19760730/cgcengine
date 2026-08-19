#!/bin/bash
# qwen36 reject-fix final A/B: off then on, 128 tok each, clean window
cd /Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare || exit 1
CLI=.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo
MTP=/Users/alexchuang/Documents/flashkv0516/temp/qwen36-mtp.gturbo
PROFILE=/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo/profiles/top64_code_prose.json
MSG=/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench/msg_code.json
OUT=/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench
BASE_ENV="TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=64 TURBO_FIELDFARE_HOT_POOL_PROFILE=$PROFILE TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_READ_WORKERS=8 TURBO_FIELDFARE_EXPERT_STATS=1 TURBO_FIELDFARE_TRUST_RECEIPT=1"
env $BASE_ENV "$CLI" --model "$MODEL" --trust-receipt --messages-file "$MSG" --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 > "$OUT/q36_final_off.log" 2>&1
env $BASE_ENV "$CLI" --model "$MODEL" --trust-receipt --mtp-model "$MTP" --messages-file "$MSG" --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 > "$OUT/q36_final_on.log" 2>&1
echo DONE > "$OUT/q36_final.done"
