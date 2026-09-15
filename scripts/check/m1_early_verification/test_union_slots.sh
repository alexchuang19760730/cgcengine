#!/bin/bash
#
# M1 早期驗證測試 2：Union > Slots 串流路徑驗證
#
# 目的：驗證當 union > slots 時，是否能優雅降級到串流路徑，
# 而不是直接 FATAL abort。這是 pool size 可變的前提。
#
# 使用方式：
#     bash test_union_slots.sh
#
# 注意：
#     - 需要先關閉其他 llama-server 釋放內存
#     - 當前代碼（c22d75395）在 union > slots 時會 FATAL abort
#     - M1 需要實作串流路徑來解決這個問題
#
# 測試結果（2026-09-13）：
#     ❌ 當前代碼不支援，union > slots 時會 abort
#     錯誤訊息：llama_expert_cache: FATAL ensure_batch layer=%u: %zu distinct experts
#               exceed the %u usable pool slots and no fill is in flight — cannot assign; aborting
#

set -e

REPO_DIR="/Users/alexchuang/Documents/flashkv-devserver"
MODEL_PATH="${REPO_DIR}/models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
TEMPLATE_PATH="${REPO_DIR}/src/llama.cpp/models/templates/Qwen3-nothink-ChatML.jinja"
SERVER_BIN="${REPO_DIR}/src/llama.cpp/build/bin/llama-server"
LOG_FILE="/tmp/llama_server_union_slots_test.log"
PORT=8090

echo "============================================================"
echo "M1 早期驗證測試 2：Union > Slots 串流路徑驗證"
echo "============================================================"
echo ""

# 檢查二進制文件
if [ ! -f "$SERVER_BIN" ]; then
    echo "❌ llama-server 二進制不存在：$SERVER_BIN"
    echo "   請先編譯：cd $REPO_DIR/src/llama.cpp && cmake --build build --config Release"
    exit 1
fi

# 檢查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 模型文件不存在：$MODEL_PATH"
    exit 1
fi

# 檢查是否有其他 server 在運行
echo "=== 檢查現有 llama-server 進程 ==="
if pgrep -f "llama-server" > /dev/null; then
    echo "⚠️  檢測到其他 llama-server 進程："
    pgrep -fl "llama-server" | head -5
    echo ""
    echo "   建議先關閉其他 server 以釋放內存，否則小 pool server 可能無法啟動。"
    echo "   按 Ctrl+C 中止，或等待 5 秒繼續..."
    sleep 5
fi
echo ""

# 用 1GB pool 啟動 server（強制觸發 union > slots）
echo "=== 啟動 1GB pool server（port $PORT）==="
echo "   模型：$MODEL_PATH"
echo "   Pool：1GB（1073741824 bytes）"
echo "   日誌：$LOG_FILE"
echo ""

# 清理舊日誌
rm -f "$LOG_FILE"

# 啟動 server（背景）
nohup "$SERVER_BIN" \
    -m "$MODEL_PATH" \
    -expert-cache 1073741824 \
    -ngl 99 --load-mode none -t 4 -c 512 -np 1 \
    --no-kv-unified -sps 0 \
    --host 127.0.0.1 --port "$PORT" \
    --jinja --cache-type-k q8_0 --cache-type-v q8_0 \
    --chat-template-file "$TEMPLATE_PATH" \
    --reasoning off --reasoning-format none \
    --temp 0 --top-k 0 --top-p 1.0 \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "   Server PID: $SERVER_PID"
echo ""

# 等待 server 啟動
echo "=== 等待 server 啟動（最多 60 秒）==="
READY=0
for i in $(seq 1 30); do
    sleep 2
    if curl -s --noproxy '*' -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
        READY=1
        echo "   ✅ Server 就緒（等待 $((i*2)) 秒）"
        break
    fi
    # 檢查是否已經 crash
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "   ❌ Server 已 crash"
        break
    fi
    echo "   等待中... ($((i*2))s)"
done
echo ""

if [ "$READY" -eq 0 ]; then
    echo "❌ Server 未能就緒，檢查日誌："
    tail -30 "$LOG_FILE"
    echo ""
    echo "=== 檢查 FATAL/abort ==="
    grep -iE "(fatal|abort|union.*slot|exceed.*slot)" "$LOG_FILE" | head -10
    exit 1
fi

# 發送測試請求
echo "=== 發送測試請求 ==="
echo "   Prompt: What is the capital of Japan? Answer in one word."
echo ""

RESPONSE=$(curl -s --noproxy '*' -m 30 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role":"user","content":"What is the capital of Japan? Answer in one word."}],
        "max_tokens": 10,
        "temperature": 0
    }' 2>&1)

echo "   響應：$RESPONSE" | head -c 500
echo ""
echo ""

# 檢查 server 是否還活著
echo "=== 檢查 server 狀態 ==="
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "   ✅ Server 仍在運行"
else
    echo "   ❌ Server 已 crash（可能是 union > slots 導致 abort）"
fi
echo ""

# 檢查日誌中的 FATAL/abort
echo "=== 檢查日誌中的 FATAL/abort ==="
FATAL_COUNT=$(grep -ciE "(fatal|abort|union.*slot|exceed.*slot)" "$LOG_FILE" 2>/dev/null || echo 0)
if [ "$FATAL_COUNT" -gt 0 ]; then
    echo "   ❌ 檢測到 FATAL/abort（$FATAL_COUNT 次）："
    grep -iE "(fatal|abort|union.*slot|exceed.*slot)" "$LOG_FILE" | head -5
else
    echo "   ✅ 未檢測到 FATAL/abort"
fi
echo ""

# 清理
echo "=== 清理 ==="
kill "$SERVER_PID" 2>/dev/null || true
sleep 2
echo "   Server 已停止"
echo ""

# 總結
echo "============================================================"
echo "=== 總結 ==="
echo "============================================================"
echo ""
echo "當前狀態（c22d75395）："
echo "  - union > slots 時會 FATAL abort（llama-expert-cache.cpp line 798-803）"
echo "  - 沒有串流降級路徑"
echo ""
echo "M1 需要實作："
echo "  - 把 union-fit 改成 union-routable：union ≤ slots 或串流路徑被走過"
echo "  - 當 union > slots 時，優雅降級到串流路徑（從文件直接讀取 expert）"
echo "  - 驗證串流路徑與正常路徑的數值等價性（M1/M2）"
echo ""
echo "參考：ROADMAP_PREFILL250_DECODE25_2026-09-13.md M1 章節"
