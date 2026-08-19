#!/bin/bash
# §13.101: r3q4 (linear_attn fp16) vs r3q4la (linear_attn all-q4) 128 tok 交錯 A/B
cd /Users/alexchuang/Documents/flashkv0516
BIN=prime-agent-worktrees/turbo-fieldfare/.build/arm64-apple-macosx/release/TurboFieldfareCLI
PROFILE=prime-agent-worktrees/qwen36-r3q4.gturbo/profiles/top80_mix.json
export TURBO_FIELDFARE_EXPERT_SLOTS=96 TURBO_FIELDFARE_HOT_POOL=1 \
       TURBO_FIELDFARE_HOT_POOL_EXPERTS=80 TURBO_FIELDFARE_HOT_POOL_PROFILE=$PROFILE \
       TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync
MSG=temp/qwen36_bench/msg_code.json
for R in 1 2 3 4; do
  if [ $((R % 2)) -eq 1 ]; then M=prime-agent-worktrees/qwen36-r3q4.gturbo; L=r3q4; else M=prime-agent-worktrees/qwen36-r3q4la.gturbo; L=r3q4la; fi
  echo "=== R$R $L ==="
  "$BIN" --model "$M" --trust-receipt --messages-file "$MSG" --max-new 128 \
    --temperature 0 --repetition-penalty 1.0 --max-context 512 2>&1 \
    | grep -E '\[stop=|prefill=' | tail -1
done
