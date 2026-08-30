#!/bin/bash
# run_server.sh — llama-server 生產啟動器（Windows 夥伴 HTTP 測試用）
#
# 與 run_n30cache.sh 同源的防護 + 非 MTP 生產 env，包成單一 CLI。
# 2026-08-30 教訓制度化（見 release.html §4.5/§4.7）：
#   - 啟動前清殘留行程（kernel panic 根因 = 行程疊加；N30CACHE_NO_CLEAN=1 跳過）
#   - 啟動前記憶體水位檢查（free < 25% 拒跑——4GiB wired pool + 13GB 模型）
#   - log 寫持久路徑 Backup/cgc_logs/（/tmp 會被重開機清掉，§4.5 附帶損失）
#   - curl 測 localhost 必帶 --noproxy '*'（本地代理 7897 會攔 127.0.0.1 → 502 空回應）
#
# 用法：
#   ./scripts/run_server.sh                       # 預設 qwen36 非 MTP，port 8080
#   CGC_SERVER_PORT=9931 ./scripts/run_server.sh  # 換 port
#   伙伴（Windows/其他機器）：http://<Mac LAN IP>:8080/v1/chat/completions（OpenAI 相容）
#
# 鐵律：server 運行期間，本機禁止任何 13GB 級操作（llama-simple 對照/量化/HF 下載）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/src/llama.cpp/build/bin/llama-server"
Q36="$ROOT/models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

MODEL="${CGC_SERVER_MODEL:-$Q36}"
PORT="${CGC_SERVER_PORT:-8080}"
HOST_BIND="${CGC_SERVER_HOST:-0.0.0.0}"
CTX="${CGC_SERVER_CTX:-2048}"
BUDGET="${N30CACHE_BUDGET:-4294967296}"   # 4GiB expert pool（與生產線一致）
LOG_DIR="$ROOT/Backup/cgc_logs"
LOG="$LOG_DIR/llama_server_$(date +%Y%m%d_%H%M%S).log"

[ -x "$BIN" ] || { echo "error: llama-server 不存在：$BIN（cmake -DLLAMA_BUILD_SERVER=ON 後構建）" >&2; exit 1; }
if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    echo "" >&2
    echo "  GGUF 不進 git（>100MB）。從 Hugging Face 下載：" >&2
    echo "    hf download Alexchuang/cgcengine-models \"$(basename "$MODEL")\" --local-dir models/gguf" >&2
    echo "  下載後驗證：cd models/gguf && shasum -a 256 -c SHA256SUMS" >&2
    echo "  全部模型清單：models/gguf/MANIFEST.md" >&2
    exit 1
fi

# [防護 1] 清殘留（§4.5：殭屍 server 是 0000 退化與 kernel panic 的共同土壤）
# pattern 用「build/bin/llama-*」子字串：行程可能是絕對路徑或相對路徑啟動（sandbox 用相對），
# 絕對路徑 pattern 比對不到相對路徑行程 → port 衝突 → 新行程秒退（2026-08-30 實測踩過）。
if [ "${N30CACHE_NO_CLEAN:-0}" != 1 ]; then
    for pat in "build/bin/llama-server" "build/bin/llama-simple" "build/bin/llama-speculative-simple"; do
        pkill -9 -f "$pat" 2>/dev/null && echo "  [clean] killed stale $pat" || true
    done
    sleep 1
fi

# [防護 2] 記憶體水位（模型 13.2GB --no-mmap + 4GiB wired pool；低水位強制拒跑）
FREE_PCT=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/{print int($2)}')
if [ -n "${FREE_PCT:-}" ] && [ "$FREE_PCT" -lt 25 ]; then
    echo "error: 系統可用記憶體僅 ${FREE_PCT}%（<25%）——16GB 機上疊 13GB 模型會 kernel panic（§4.5）。關掉其他重工行程再跑。" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
ln -sf "$LOG" "$LOG_DIR/llama_server_latest.log"

# [防護 3] 單 slot + 非 unified KV：與 llama-simple 行為對齊（np=auto 的 4 slots 會把 context 切 512/流）
# -expert-cache 是單刮號參數（common args 註冊形式，--expert-cache 不認）
# [防護 5] -sps 0 禁用 KV slot LCP 復用（2026-08-30 晚間曾誤判為 0000 根因；後續實證推翻：
#   -sps 0 上線後 task 0 首請求照樣 iota。真正根因見下。保留 -sps 0 作為減少干擾變數的防護）。
# [防護 6 / 2026-08-30 0000 真正根因] 拿掉 CGC_OA_ASYNC（原此處設 1）：
#   async Metal split 下 ggml-alloc 會把 top-k ids buffer 回收給同 step 後續 tensor（gather
#   用的 iota/arange 索引）→ hook 快照讀到 iota(0..N) 線性序列 → 錯誤專家 → garbage
#   logits → 輸出 0000。log 證據：llama_server_20260830_210223.log（-sps 0 已生效、
#   無任何 slot reuse）task 0 pmax=37 起全層 iota。OA_ASYNC 的 +12.6% 速度不值得換正確性；
#   要恢復需在 C++ 端為 ids 張量加同步（OPEN 項）。
echo "[start] $MODEL  port=$PORT  ctx=$CTX  budget=${BUDGET}B"
echo "[log]   ${LOG}（tail -f 同路徑）"
CGC_EXPERT_CACHE_BYTES="$BUDGET" \
LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1 \
LLAMA_EXPERT_CACHE_WORKERS=8 \
CGC_WAKE_POLL_US=15 \
CGC_PREFETCH_SRC=hist \
CGC_EVICTED_RING=0 \
CGC_N_CB=8 \
CGC_GLU_FUSED_DOWN=1 \
CGC_WATCHDOG=1 \
"$BIN" -m "$MODEL" -expert-cache "$BUDGET" -ngl 99 --no-mmap -t 8 \
    -c "$CTX" -np 1 --no-kv-unified -sps 0 \
    --host "$HOST_BIND" --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!

# [防護 4] 健康輪詢（最多 120s；load 完成前 /health 不回）
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "<Mac IP>")
echo "[wait]  模型載入中（首次 ~1min）..."
for i in $(seq 1 60); do
    sleep 2
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "error: server 行程已退出——看 $LOG" >&2; exit 1; }
    if curl -s --noproxy '*' -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q "ok"; then
        echo ""
        echo "================ 連線卡（給夥伴） ================"
        echo "  Base URL   : http://$LAN_IP:$PORT/v1（OpenAI 相容）"
        echo "  測試       : curl --noproxy '*' http://127.0.0.1:$PORT/v1/models"
        echo "  Windows 伙伴 : 程式內直接指 http://$LAN_IP:$PORT/v1/chat/completions"
        echo "  注意       : 本機 curl 測 localhost 必帶 --noproxy '*'（代理 7897 攔截）"
        echo "  停止       : pkill -INT -f llama-server（或 kill ${SERVER_PID}）"
        echo "  非 MTP 基線: ~7 t/s（結構性；27 t/s 需 MTP，另跑 run_n30cache.sh --mtp）"
        echo "=================================================="
        echo ""
        echo "[run]    前景運行中（Ctrl+C = 優雅關閉）。log: tail -f $LOG"
        trap 'kill -INT "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; exit 0' INT TERM
        wait "$SERVER_PID"
        exit $?
    fi
done
echo "error: 120s 內 /health 未就緒——看 $LOG" >&2
exit 1
