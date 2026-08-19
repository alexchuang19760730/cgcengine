#!/bin/bash
# p0_final_bench.sh — 最終 P0 對照：qwen36 IQ2 (Metal -ngl99) vs gemma4 IQ3_S (CPU -ngl0)
# 每個 arm：RSS 峰值採樣 + free% 採樣
set -u
BENCH=/Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/build/bin/llama-bench
Q36=/Users/alexchuang/Documents/flashkv0516/models/gguf/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
G4=/Users/alexchuang/Documents/flashkv0516/models/gguf/gemma-4-26B-A4B-it-UD-IQ3_S.gguf

run_one() {
  local MODEL="$1" TAG="$2" NGL="$3"
  echo "############################################"
  echo "### $TAG  (-ngl $NGL, pp128/tg128)"
  echo "############################################"
  (python3 - "$TAG" <<'PYEOF' > "/tmp/p0_final_${TAG}.log" &
import time, subprocess, sys
tag = sys.argv[1]
end = time.time() + 400
peak_rss = 0
low_free = 100.0
n = 0
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
    low_free = min(low_free, free)
    r = subprocess.run(['ps', 'axo', 'pid,rss,comm'], capture_output=True, text=True).stdout
    for ln in r.splitlines():
        if 'llama-bench' in ln:
            parts = ln.split()
            if len(parts) >= 2 and parts[1].isdigit():
                peak_rss = max(peak_rss, int(parts[1]))
    n += 1
    if n % 10 == 1:
        print(f"free_pct={free:.1f}% peak_rss={peak_rss/1048576:.2f}GiB", flush=True)
    time.sleep(0.5)
PYEOF
) &
  SAM_PID=$!
  sleep 2
  start=$(date +%s)
  "$BENCH" -m "$MODEL" -p 128 -n 128 -r 1 -ngl "$NGL" -o md 2> "/tmp/p0_final_${TAG}_err.log"
  rc=$?
  echo "exit=$rc elapsed=$(( $(date +%s) - start ))s"
  sleep 1
  kill $SAM_PID 2>/dev/null
  wait $SAM_PID 2>/dev/null
  echo "--- mem: $(tail -1 "/tmp/p0_final_${TAG}.log") ---"
  echo
}

run_one "$Q36" qwen36_iq2_metal 99
run_one "$G4"  gemma4_iq3s_cpu  0
