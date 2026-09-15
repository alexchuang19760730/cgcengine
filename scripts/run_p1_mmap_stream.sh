#!/bin/bash
# run_p1_mmap_stream.sh — P1 Expert 流式加載（mmap-native streaming）測試啟動器
#
# 基於白皮書 P1：expert tensor 不縮小，走 mmap + Metal noCopy，identity slot_table。
# 與 run_server.sh 的差異：
#   - 去掉 --no-mmap（啟用 mmap）
#   - LLAMA_EXPERT_CACHE_MMAP_STREAM=1
#   - expert_cache_bytes 用於 resident budget（不再是 pool size）
#   - 關閉 MTP（先驗證基礎功能，再開 MTP）
#
# 用法：
#   ./scripts/run_p1_mmap_stream.sh              # 預設非 MTP，port 8090
#   ./scripts/run_p1_mmap_stream.sh --mtp        # 啟用 MTP
#   CGC_P1_PORT=9999 ./scripts/run_p1_mmap_stream.sh
#   CGC_P1_MODEL=/path/to/model.gguf ./scripts/run_p1_mmap_stream.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/src/llama.cpp/build/bin/llama-server"
MODEL_ROOT="${CGC_SERVER_MODEL_ROOT:-$ROOT/models/gguf}"
MODEL="${CGC_P1_MODEL:-$MODEL_ROOT/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf}"
PORT="${CGC_P1_PORT:-8090}"
CTX="${CGC_P1_CTX:-2048}"
NGL="${CGC_P1_NGL:-99}"
# P1: expert_cache_bytes 不再是 pool size，而是 resident budget hint。
# 設一個較小值（如 1GB），實際常駐由 OS page cache + madvise 管理。
BUDGET="${CGC_P1_BUDGET:-1073741824}"  # 1GiB
MTP=0
for arg in "$@"; do
    [ "$arg" = "--mtp" ] && MTP=1
done

# 清理殘留
pkill -9 -f "build/bin/llama-server" 2>/dev/null && echo "[clean] killed stale llama-server" || true
sleep 1

# 檢查二進制和模型
[ -x "$BIN" ] || { echo "error: llama-server 不存在：$BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "error: model not found：$MODEL"; exit 1; }

LOG_DIR="$ROOT/Backup/cgc_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/llama_server_p1_mmap_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG" "$LOG_DIR/llama_server_p1_latest.log"

echo "========================================"
echo " P1 mmap-streaming 測試啟動"
echo "========================================"
echo "  model:    $MODEL"
echo "  port:     $PORT"
echo "  ctx:      $CTX"
echo "  ngl:      $NGL"
echo "  budget:   $BUDGET bytes (resident hint)"
echo "  mmap:     enabled (no --no-mmap)"
echo "  mmap_stream: LLAMA_EXPERT_CACHE_MMAP_STREAM=1"
echo "  mtp:      $MTP"
echo "  log:      $LOG"
echo "========================================"

SERVER_ARGS=(
    -m "$MODEL"
    -expert-cache "$BUDGET"
    -ngl "$NGL"
    -t 8
    -c "$CTX"
    -np 1
    --no-kv-unified
    -sps 0
    --host 0.0.0.0
    --port "$PORT"
    --jinja
    --temp 0.4
    --top-k 0
    --top-p 0.8
)

# P1 關鍵環境變量
SERVER_ENV=(
    LLAMA_EXPERT_CACHE_MMAP_STREAM=1
    LLAMA_EXPERT_CACHE_ALLOW_NGL=1
    LLAMA_EXPERT_CACHE_RESIDENT_MB="${CGC_P1_RESIDENT_MB:-1024}"
    CGC_EXPERT_CACHE_BYTES="$BUDGET"
    # P1 模式下關閉 pool 相關優化（identity mapping 不需要）
    # [2026-09-16 值語意] CGC_DBUF / CGC_SPAC 原本的 reader 測「非空即開」——`0` 非空 ⇒ 這兩行
    # 寫的 OFF 其實是 ON，與上一行註解相反。reader 已改為值語意（cgc_env_on，見
    # llama-expert-cache.h），現在 `0` 才真的等於關。注意語意變更僅止於「值」：本腳本的
    # identity-mapping 路徑是否真的走到那兩個呼叫點（llama-context.cpp:5780 / 1942）尚未驗證，
    # 所以不要宣稱 P1 的行為一定變了。
    # CGC_SOFT_POOL_L0/L1 不受影響：它們走數值 parser（cgc_soft_pool_tier），0 本來就是 0。
    CGC_DBUF=0
    CGC_SPAC=0
    CGC_SOFT_POOL_L0=0
    CGC_SOFT_POOL_L1=0
)

if [ "$MTP" = "1" ]; then
    SERVER_ARGS+=(--spec-type draft-mtp --spec-draft-n-max 3)
    SERVER_ENV+=(CGC_VERIFY_DECODE=1 CGC_DRAFT_DECODE=1)
    echo "  MTP:      enabled (draft-mtp, n_max=3)"
fi

echo "[start] launching llama-server..."
env "${SERVER_ENV[@]}" "$BIN" "${SERVER_ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!

# 健康檢查
echo "[wait]  模型載入中（首次 ~1-2min，mmap 比 --no-mmap 快）..."
for i in $(seq 1 90); do
    sleep 2
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "error: server 已退出——看 $LOG"; tail -30 "$LOG"; exit 1; }
    if curl -s --noproxy '*' -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q "ok"; then
        echo ""
        echo "✅ P1 mmap-streaming server 就緒！"
        echo "  PID:  $SERVER_PID"
        echo "  URL:  http://127.0.0.1:$PORT/v1"
        echo "  Log:  tail -f $LOG"
        echo ""
        echo "  測試命令："
        echo "  curl --noproxy '*' http://127.0.0.1:$PORT/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"15+27=?\"}],\"max_tokens\":32}'"
        echo ""
        echo "  記憶體監控："
        echo "  top -pid $SERVER_PID -l 1 | grep PhysMem"
        echo "  vmmap $SERVER_PID | grep -E 'mmap|resident' | head -20"
        echo ""
        trap 'kill -INT "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; exit 0' INT TERM
        wait "$SERVER_PID"
        exit $?
    fi
    # 顯示進度
    if [ $((i % 5)) -eq 0 ]; then
        echo "  ...等待中 ($((i*2))s)，最後幾行 log："
        tail -3 "$LOG" 2>/dev/null | sed 's/^/    /'
    fi
done
echo "error: 180s 內 /health 未就緒——看 $LOG"
tail -50 "$LOG"
exit 1
