#!/bin/bash
# qwen36 MTP real-hidden B=2 on/off A/B: off → on → off → on, msg_code, 128 tok
cd /Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare || exit 1
CLI=.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo
MTP=/Users/alexchuang/Documents/flashkv0516/temp/qwen36-mtp.gturbo
PROFILE=/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo/profiles/top64_code_prose.json
MSG=/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench/msg_code.json
BASE_ENV="TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=64 TURBO_FIELDFARE_HOT_POOL_PROFILE=$PROFILE TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_READ_WORKERS=8 TURBO_FIELDFARE_EXPERT_STATS=1 TURBO_FIELDFARE_TRUST_RECEIPT=1"

run_off() {
  local tag="$1" out="$2"
  env $BASE_ENV "$CLI" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$out" 2>&1
  echo "exit=$?"
}
run_on() {
  local tag="$1" out="$2"
  env $BASE_ENV "$CLI" --model "$MODEL" --trust-receipt --mtp-model "$MTP" \
    --messages-file "$MSG" --max-new 128 --temperature 0 --repetition-penalty 1.0 \
    --max-context 512 > "$out" 2>&1
  echo "exit=$?"
}

echo "=== R1 off ==="; run_off r1 /tmp/q36_mtp_r1.log
echo "=== R2 on ===";  run_on  r2 /tmp/q36_mtp_r2.log
echo "=== R3 off ==="; run_off r3 /tmp/q36_mtp_r3.log
echo "=== R4 on ===";  run_on  r4 /tmp/q36_mtp_r4.log
echo "ALL DONE"
