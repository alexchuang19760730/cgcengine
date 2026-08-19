#!/bin/bash
# multitask_ab.sh — llama.cpp -ngl 混合 vs 全 offload × 多工壓力 矩陣
# 用法: bash temp/multitask_ab.sh <internal|external> <ngl> <pressure_gb>
set -u
BENCH=/Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/build/bin/llama-bench
INT=/Users/alexchuang/Documents/flashkv0516/models/gguf/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
EXT=/Volumes/AlexZhuang/gguf/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
MODEL=$([ "$1" = "external" ] && echo "$EXT" || echo "$INT")
NGL=$2
PRES=$3
TAG="$(basename $MODEL .gguf)_ngl${NGL}_pres${PRES}G"

# memory pressure holder (incompressible)
if [ "$PRES" != "0" ]; then
  python3 /Users/alexchuang/Documents/flashkv0516/temp/pressure_holder.py "$PRES" 600 > /tmp/pres_${TAG}.log 2>&1 &
  PRES_PID=$!
  sleep 8
fi

# memory sampler
python3 - "$TAG" <<'PYEOF' > /tmp/mem_${TAG}.log &
import time, subprocess, sys
tag = sys.argv[1]
end = time.time() + 300
while time.time() < end:
    out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().split('.')[0]
            if v.lstrip('-').isdigit():
                d[k.strip()] = int(v)
    free = (d.get('Pages free', 0) + d.get('Pages speculative', 0)) * 16384 / 16e9 * 100
    comp = d.get('Pages occupied by compressor', 0) * 16384 / 16e9
    print(f"free_pct={free:.1f}% compressor={comp:.2f}GB", flush=True)
    time.sleep(1)
PYEOF
SAM_PID=$!

echo "=== $TAG ==="
start=$(date +%s)
$BENCH -m "$MODEL" -p 128 -n 128 -r 1 -ngl $NGL -o md 2>/dev/null
rc=$?
echo "exit=$rc elapsed=$(( $(date +%s) - start ))s"
kill $SAM_PID 2>/dev/null
if [ "$PRES" != "0" ]; then kill $PRES_PID 2>/dev/null; fi
echo "--- mem log tail ---"
tail -2 /tmp/mem_${TAG}.log
echo
