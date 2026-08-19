#!/bin/bash
# EAGLE-2 chain-data collection: MTP fixed B=4, chain dump per prompt.
set -u
cd /Users/alexchuang/Documents/flashkv0516/turbo-fieldfare-github-official || exit 1

export TURBO_FIELDFARE_EXPERT_SLOTS=96
export TURBO_FIELDFARE_HOT_POOL=1
export TURBO_FIELDFARE_HOT_POOL_EXPERTS=64
export TURBO_FIELDFARE_HOT_POOL_PROFILE=/Users/alexchuang/Documents/flashkv0516/models/gemma4-r3.gturbo/profiles/top64_code.json
export TURBO_FIELDFARE_HOT_POOL_PRELOAD=async
export TURBO_FIELDFARE_EXPERT_READ_WORKERS=8
export TURBO_FIELDFARE_MTP_ADAPTIVE=0

M=/Users/alexchuang/Documents/flashkv0516/models/gemma4-r3.gturbo
H=/Users/alexchuang/Documents/flashkv0516/models/gemma-4-mtp-head
OUT=/Volumes/AlexZhuang/g4_mtp_train
PY=/Users/alexchuang/Documents/flashkv0516/temp/g4_eagle/prompts.json

python3 - <<'EOF'
import json
prompts = json.load(open('/Users/alexchuang/Documents/flashkv0516/temp/g4_eagle/prompts.json'))
for i, p in enumerate(prompts):
    fn = f"/Users/alexchuang/Documents/flashkv0516/temp/g4_eagle/p{i:02d}.json"
    json.dump([p], open(fn, "w"))
    print(fn)
EOF

for f in /Users/alexchuang/Documents/flashkv0516/temp/g4_eagle/p*.json; do
    base=$(basename "$f" .json)
    out="$OUT/chain_$base.bin"
    rm -f "$out"
    echo "=== $base ==="
    TURBO_FIELDFARE_MTP_CHAIN_DUMP="$out" \
    .build/arm64-apple-macosx/debug/TurboFieldfareCLI \
      --model "$M" --trust-receipt \
      --messages-file "$f" \
      --max-new 64 --temperature 0 --repetition-penalty 1.0 --max-context 512 \
      --mtp-model "$H" --mtp-max-draft 4 2>&1 | grep -oE 'tok/s=[0-9.]+' | tail -1
    ls -la "$out" 2>/dev/null | awk '{print "  chain file:", $5, "bytes"}'
done
echo "COLLECT DONE"
