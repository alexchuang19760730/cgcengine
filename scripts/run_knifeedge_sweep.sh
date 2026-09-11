#!/bin/bash
# run_knifeedge_sweep.sh — 4/6/8GB pool 抗刀鋒評測自動化
#
# 對每個 pool size：啟動 server → 等 health → 跑 flip_rate.py → 保存結果 → 殺 server
# 用法：./scripts/run_knifeedge_sweep.sh [4|6|8|all]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/src/llama.cpp/build/bin/llama-server"
MODEL="$ROOT/models/gguf/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
RESULT_DIR="$ROOT/Backup/knifeedge_results"
mkdir -p "$RESULT_DIR"

# 抗刀鋒參數
GREEDY_REPEATS=10       # 同題 greedy 重複 10 次（量化不確定性）
SEEDED_REPEATS=5        # 5 個固定 seed
TEMPERATURE=0.4         # 與 37/48 基線一致
PER_PROFILE=2           # 每 profile 2 題（快速版，全量改 8/11/6）
PROFILES="qa-zh,math,coding,reasoning"
MAX_TOKENS=64
TIMEOUT=180

POOL_GB="${1:-all}"
POOLS=()
if [ "$POOL_GB" = "all" ]; then
    POOLS=(4 6 8)
else
    POOLS=("$POOL_GB")
fi

kill_server() {
    pkill -9 -f "build/bin/llama-server" 2>/dev/null || true
    sleep 3
}

for GB in "${POOLS[@]}"; do
    BUDGET=$((GB * 1024 * 1024 * 1024))
    PORT=$((8090 + GB))
    LABEL="iq4xs_pool${GB}gb"
    OUTPUT="$RESULT_DIR/knifeedge_${LABEL}.json"
    LOG="$RESULT_DIR/server_${LABEL}.log"

    echo ""
    echo "========================================"
    echo " Pool ${GB}GB (budget=${BUDGET}, port=${PORT})"
    echo "========================================"

    kill_server
    echo "[1/4] 啟動 server..."
    env CGC_EXPERT_CACHE_BYTES="$BUDGET" \
        LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
        LLAMA_EXPERT_CACHE_DIRECT_IO=1 \
        LLAMA_EXPERT_CACHE_MLOCK_DENSE=1 \
        LLAMA_EXPERT_CACHE_LAYER_CAPS="40-40:256" \
        "$BIN" -m "$MODEL" -expert-cache "$BUDGET" -ngl 99 -t 8 -c 2048 \
        -np 1 --no-mmap --no-kv-unified -sps 0 \
        --host 127.0.0.1 --port "$PORT" \
        --jinja \
        --reasoning off --reasoning-format none \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        --spec-type draft-mtp --spec-draft-n-max 3 \
        --temp 0.4 --top-k 0 --top-p 0.8 > "$LOG" 2>&1 &
    SERVER_PID=$!

    echo "[2/4] 等待 server 就緒（最多 180s）..."
    READY=0
    for i in $(seq 1 90); do
        sleep 2
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  ❌ server 已退出，看 $LOG"
            tail -20 "$LOG"
            break
        fi
        if curl -s --noproxy '*' -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q "ok"; then
            READY=1
            echo "  ✅ server 就緒（$((i*2))s）"
            break
        fi
    done
    if [ "$READY" != "1" ]; then
        echo "  ❌ 超時未就緒，跳過 ${GB}GB"
        kill_server
        continue
    fi

    echo "[3/4] 跑抗刀鋒評測（10 greedy + 5 seeded × ${PER_PROFILE}題/profile）..."
    python3 "$ROOT/scripts/check/flip_rate.py" \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --model local \
        --label "$LABEL" \
        --profiles "$PROFILES" \
        --per-profile "$PER_PROFILE" \
        --greedy-repeats "$GREEDY_REPEATS" \
        --repeats "$SEEDED_REPEATS" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --timeout "$TIMEOUT" \
        --output "$OUTPUT" 2>&1 | tail -20

    echo "[4/4] 結果保存到 $OUTPUT"
    kill_server
    echo "  內存已釋放"
done

echo ""
echo "========================================"
echo " 全部完成！結果在 $RESULT_DIR/"
echo "========================================"
ls -lh "$RESULT_DIR"/knifeedge_*.json 2>/dev/null
echo ""
echo "對比分析："
echo "  python3 scripts/check/flip_rate.py --compare $RESULT_DIR/knifeedge_iq4xs_pool4gb.json $RESULT_DIR/knifeedge_iq4xs_pool6gb.json $RESULT_DIR/knifeedge_iq4xs_pool8gb.json"
