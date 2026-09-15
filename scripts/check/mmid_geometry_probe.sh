#!/bin/bash
# mmid_geometry_probe.sh — 把 mul_mat_id 的「幾何 vs ids」在兩條路徑上各印一次對打。
#
# 背景：expert cache 的兩條路徑在 src0（FFN expert 權重張量）上裝的是不同的 ne[2]：
#   - pool 模式  : ne02 = slots_per_layer（例如 71），ids 必須是 SLOT index（remap 之後）
#   - M2 slab 模式: ne02 = n_expert（256），        ids 必須是 RAW expert id
# ids 本身是由 llama-context.cpp 的 host hook 寫的，跟設定 ne02 的程式碼不同檔案、不同時機。
# 兩者一旦錯配，kernel 不會崩，而是去讀「真實存在但錯誤」的那個 expert —— 輸出看起來合理、
# logit 偏低、argmax 偏移，且資料層（pool bytes / pread / remap 值 / GPU 讀回）全部正確，
# 因此所有「資料面」的檢查都抓不到它。
#
# 這個腳本對同一個 prompt 跑若干條臂，各自蒐集 CGC-MMID 行。判讀方式：
#   * ids 相同、但同一個 id 的 fp（該 expert 前 4KB 的 FNV-1a）不同  -> pool 內容錯
#   * fp 相同、但 ids 不同                                          -> remap / hook 寫錯
#   * id_oob_vs_ne02 > 0（id >= ne02 或 < 0）                        -> 幾何與 ids 直接錯配，就是它
#
# 用法：
#   ./scripts/check/mmid_geometry_probe.sh                # 預設跑 gather + nocache
#   ./scripts/check/mmid_geometry_probe.sh gather nogather nocache
#   CGC_MMID_MV_DBG=3 ./scripts/check/mmid_geometry_probe.sh gather   # 印更多節點
#
# 產出：Backup/mmid_probe/<arm>_<ts>.log，以及同名的 .mmid（純 CGC-MMID 行）
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/src/llama.cpp/build/bin/llama-server"
MODEL="${CGC_PROBE_MODEL:-$ROOT/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf}"
OUTDIR="$ROOT/Backup/mmid_probe"
BUDGET_GB="${CGC_PROBE_POOL_GB:-8}"
PORT="${CGC_PROBE_PORT:-8091}"
DBG="${CGC_MMID_MV_DBG:-1}"
# 對照臂（nocache / nogather）沒有 bounded residency，必須讓 OS 可回收模型頁，
# 否則 13.66GB 模型在 16GB 機器上會直接 OOM。
LOAD_MODE="${CGC_PROBE_LOAD_MODE:-none}"
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
    ARMS=(gather nocache)
fi

mkdir -p "$OUTDIR"
[ -x "$BIN" ] || { echo "error: $BIN not built" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "error: model not found: $MODEL" >&2; exit 1; }

wait_health() {
    local port="$1" i
    for i in $(seq 1 240); do
        if curl -s -m 2 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; then
            return 0
        fi
        sleep 2
    done
    return 1
}

kill_server() {
    pkill -9 -f "build/bin/llama-server" 2>/dev/null || true
    sleep 3
}

run_arm() {
    local arm="$1"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log="$OUTDIR/${arm}_${ts}.log"
    local extra=()
    local budget=$(( BUDGET_GB * 1024 * 1024 * 1024 ))

    case "$arm" in
        gather)
            # 生產配置：bounded Metal pool + M2 prefill streaming
            extra=(LLAMA_EXPERT_CACHE_ALLOW_NGL=1
                   LLAMA_EXPERT_CACHE_DIRECT_IO=1
                   LLAMA_EXPERT_CACHE_MLOCK_DENSE=1
                   LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256
                   CGC_PREFILL_STREAM=1
                   CGC_GATHER_SLAB_CAP=256)
            ;;
        nogather)
            # 關 hook + skip_load（兩個變數一起關），全量 resident -> mmap
            LOAD_MODE=mmap
            extra=(LLAMA_EXPERT_CACHE_ALLOW_NGL=1
                   LLAMA_EXPERT_CACHE_DIRECT_IO=1
                   LLAMA_EXPERT_CACHE_MLOCK_DENSE=1
                   LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256
                   LLAMA_EXPERT_CACHE_NOGATHER=1)
            ;;
        nohook)
            # 第四臂：只關 hook，保留 skip_load（分離「串流 bytes 錯」與「remap/repoint 錯」）
            extra=(LLAMA_EXPERT_CACHE_ALLOW_NGL=1
                   LLAMA_EXPERT_CACHE_DIRECT_IO=1
                   LLAMA_EXPERT_CACHE_MLOCK_DENSE=1
                   LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256
                   LLAMA_EXPERT_CACHE_NOHOOK=1)
            LOAD_MODE=mmap
            ;;
        gatherslab)
            # 強制走 M2 whole-layer slab 路徑（大 prefill 的分支）。
            # 正常情況要 prompt 夠長（n_tokens > CGC_POOL_MAX_TOKENS）才會進這個分支，
            # 這裡把 pmax 壓到 1，200-token 的 prompt 就會走它 —— 省下長 prompt 的記憶體與時間。
            # 這條路的特徵：src0 ne02 = n_expert(256)、ids 是 RAW expert id（沒有 remap leaf）。
            # 因為 ids 是 raw，fp 可以直接跟 GGUF 比對，不需要 slot 對照表。
            extra=(LLAMA_EXPERT_CACHE_ALLOW_NGL=1
                   LLAMA_EXPERT_CACHE_DIRECT_IO=1
                   LLAMA_EXPERT_CACHE_MLOCK_DENSE=1
                   LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256
                   CGC_PREFILL_STREAM=1
                   CGC_GATHER_SLAB_CAP=256
                   CGC_POOL_MAX_TOKENS=1)
            ;;
        nocache)
            # 對照組：expert cache 完全關掉（全量 resident，用 mmap 讓 OS 可回收頁）
            budget=0
            LOAD_MODE=mmap
            extra=()
            ;;
        *)
            echo "error: unknown arm '$arm' (gather|nogather|nohook|nocache)" >&2
            return 1
            ;;
    esac

    echo "================================================================"
    echo " arm=$arm  budget=$budget  port=$PORT  dbg=$DBG"
    echo " log=$log"
    echo "================================================================"

    kill_server
    echo "  load-mode=$LOAD_MODE"
    env CGC_EXPERT_CACHE_BYTES="$budget" \
        CGC_MMID_MV_DBG="$DBG" \
        ${extra[@]+"${extra[@]}"} \
        "$BIN" -m "$MODEL" -expert-cache "$budget" -ngl 99 --load-mode "$LOAD_MODE" \
        -t 8 -c 2048 -np 1 --no-kv-unified -sps 0 \
        --host 127.0.0.1 --port "$PORT" --jinja \
        --reasoning off --reasoning-format none \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        > "$log" 2>&1 &
    local pid=$!

    if ! wait_health "$PORT"; then
        echo "  !! server did not become healthy (see $log)"
        tail -20 "$log"
        kill "$pid" 2>/dev/null
        kill_server
        return 1
    fi
    echo "  server up (pid $pid)"

    curl -s -m 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"巴黎是法國的首都。"}],"max_tokens":8,"temperature":0}' \
        > "$OUTDIR/${arm}_${ts}.resp.json" 2>&1
    echo "  request done"

    sleep 1
    kill "$pid" 2>/dev/null
    kill_server

    grep -a "CGC-MMID" "$log" > "$OUTDIR/${arm}_${ts}.mmid" || true
    local n
    n=$(wc -l < "$OUTDIR/${arm}_${ts}.mmid" | tr -d ' ')
    echo "  CGC-MMID lines: $n"
    if [ "$n" -gt 0 ]; then
        echo "  --- first 6 ---"
        head -6 "$OUTDIR/${arm}_${ts}.mmid" | sed 's/^/    /'
    fi
    echo "  oob count: $(grep -ac 'id_oob_vs_ne02=[1-9]' "$OUTDIR/${arm}_${ts}.mmid" || true)"
    echo "  argmax/step0: $(grep -ao 'argmax=[0-9]*' "$log" | head -1)"
}

for arm in "${ARMS[@]}"; do
    run_arm "$arm"
done
kill_server
echo ""
echo "Done. 比對："
echo "  diff <(grep -o 'ids ne=.*' A.mmid) <(grep -o 'ids ne=.*' B.mmid)"
