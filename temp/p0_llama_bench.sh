#!/bin/bash
# p0_llama_bench.sh — P0 階段 llama.cpp bench：qwen36 IQ2 vs gemma4 IQ3_S
# prefill (pp128) / decode (tg128) / 記憶體使用（vm_stat 採樣 + 低水位熔斷）
# 注意：-p 512 會觸發 Metal alloc 失敗（res=-3，working set 11.45GB），用 pp128（已實證）
set -u
BENCH=/Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/build/bin/llama-bench
Q36=/Users/alexchuang/Documents/flashkv0516/models/gguf/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
G4=/Users/alexchuang/Documents/flashkv0516/models/gguf/gemma-4-26B-A4B-it-UD-IQ3_S.gguf
NGL=${NGL:-99}

run_one() {
  local MODEL="$1" TAG="$2"
  echo "############################################"
  echo "### $TAG  (-ngl $NGL, pp128/tg128)"
  echo "############################################"
  python3 - "$TAG" <<'PYEOF' > "/tmp/p0_mem_${TAG}.log" &
import time, subprocess, sys, os
tag = sys.argv[1]
end = time.time() + 420
low_since = None
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
    # 熔斷：free<0.5% 持續 30s（IQ2 proven 時 free 0.7% 也穩定，0.5% 以下才是真危險）
    if free < 0.5:
        if low_since is None: low_since = time.time()
        elif time.time() - low_since > 30:
            print("WATCHDOG: free%<0.5 for 30s, killing bench", flush=True)
            os.system("pkill -f llama-bench")
            break
    else:
        low_since = None
    time.sleep(1)
PYEOF
  SAM_PID=$!
  sleep 2
  start=$(date +%s)
  "$BENCH" -m "$MODEL" -p 128 -n 128 -r 1 -ngl "$NGL" -o md 2> "/tmp/p0_${TAG}_err.log"
  rc=$?
  echo "exit=$rc elapsed=$(( $(date +%s) - start ))s"
  kill $SAM_PID 2>/dev/null
  echo "--- mem log (每 10 筆取 1) ---"
  awk 'NR % 10 == 1' "/tmp/p0_mem_${TAG}.log" | tail -6
  echo
}

run_one "$Q36" qwen36_iq2
run_one "$G4"  gemma4_iq3s
