#!/bin/bash
# qwen36 WAKE_POLL_US A/B: baseline(未設定) vs 200, msg_prose, 128 tok, 4 輪交錯
cd /Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare || exit 1
CLI=.build/arm64-apple-macosx/debug/TurboFieldfareCLI
MODEL=/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo
PROFILE=/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo/profiles/top64_code_prose.json
MSG=/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench/msg_prose.json
BASE_ENV="TURBO_FIELDFARE_EXPERT_SLOTS=64 TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=64 TURBO_FIELDFARE_HOT_POOL_PROFILE=$PROFILE TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync TURBO_FIELDFARE_EXPERT_READ_WORKERS=8 TURBO_FIELDFARE_EXPERT_STATS=1 TURBO_FIELDFARE_TRUST_RECEIPT=1"

run_one() {
  local tag="$1" wakeval="$2" out="$3"
  local envs="$BASE_ENV"
  if [ "$wakeval" != "unset" ]; then envs="$envs TURBO_FIELDFARE_WAKE_POLL_US=$wakeval"; fi
  env $envs "$CLI" --model "$MODEL" --trust-receipt --messages-file "$MSG" \
    --max-new 128 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
    > "$out" 2>&1
  echo "exit=$?"
}

echo "=== R1 base (unset) ==="; run_one r1 unset /tmp/q36_wp_r1.log
echo "=== R2 poll200 ===";      run_one r2 200   /tmp/q36_wp_r2.log
echo "=== R3 base (unset) ==="; run_one r3 unset /tmp/q36_wp_r3.log
echo "=== R4 poll200 ===";      run_one r4 200   /tmp/q36_wp_r4.log
echo "ALL DONE"
