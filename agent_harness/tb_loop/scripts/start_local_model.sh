#!/bin/bash
# start_local_model.sh — Start local llama-server + configure tb_loop to use it
#
# Usage:
#   ./start_local_model.sh                    # Start Qwen3.6 on port 1234
#   ./start_local_model.sh --model qwen36     # Explicit model selection
#   ./start_local_model.sh --model gemma4     # Use gemma4 instead
#   ./start_local_model.sh --mtp              # Start with MTP (Qwen3.6 only)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# ★ E1 之後這支腳本位於 tb_loop/scripts/，所以 `ROOT` = tb_loop/（不是 harness 根）。
#   原本下面用 `$ROOT/../models/gguf/...` —— 搬遷前 ROOT=agent_harness/ 所以那正好是
#   repo 根的 models/；搬遷後 `$ROOT/..` 變成 agent_harness/ ⇒ 預設模型路徑指向
#   不存在的 `agent_harness/models/gguf/`。
#   失敗形態是**大聲但很晚**（不是靜默）：下面第 66 行有 `[ -f "$M" ] || exit 1`，
#   它會印 `model not found: …` 然後結束。準確地說：腳本的預設路徑壞了，而它自己會說出來
#   —— 所以這比另一支（`finetune_loopmoe.sh` 的同型問題，一半僥倖正確、因而靜默）容易發現。
#   為什麼一直沒被發現：本檔第 69/72 行的 `netstat -ano` / `taskkill` 是 Windows 專用，
#   它是給那條線用的腳本，而 E1 的搬遷檢查沒有區分「預設分支才走到的路徑」。
#   教訓：自帶錨點**不自動等於安全** —— 只有當「腳本」與「它要指的東西」一起搬時才安全。
#   判準不是「它有沒有自己算錨點」，是「算出來的路徑還存不存在」。
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
CGCROOT="$(cd "$REPO_ROOT/cgcengine" 2>/dev/null && pwd || echo "")"

MODEL="qwen36"
PORT="${PORT:-1234}"
MTP=0
NGL=""
BUDGET=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --mtp) MTP=1; shift ;;
        --ngl) NGL="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Find the best available server binary
# Priority: cgcengine fork build > pre-built b9553
if [ -n "$CGCROOT" ] && [ -f "$CGCROOT/src/llama.cpp/build/bin/llama-server.exe" ]; then
    BIN="$CGCROOT/src/llama.cpp/build/bin/llama-server.exe"
    echo "[cgcengine fork build]"
elif [ -f "/d/alex/toolchains/llama-b9553/llama-server.exe" ]; then
    BIN="/d/alex/toolchains/llama-b9553/llama-server.exe"
    echo "[pre-built b9553]"
else
    echo "error: no llama-server binary found" >&2
    echo "  Run: cd cgcengine && deploy-harmonyos/windows/build-windows.bat" >&2
    exit 1
fi

# Find model
case "$MODEL" in
    qwen36)
        M="${N30CACHE_Q36:-$REPO_ROOT/models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf}"
        [ -z "$NGL" ] && NGL=99
        ;;
    gemma4)
        M="${N30CACHE_G4:-$REPO_ROOT/models/gguf/gemma-4-26B-A4B-it-UD-IQ3_S.gguf}"
        [ -z "$NGL" ] && NGL=30
        ;;
    *) M="$MODEL"; [ -z "$NGL" ] && NGL=99 ;;
esac

[ -f "$M" ] || { echo "error: model not found: $M" >&2; exit 1; }

# Kill any existing llama-server on this port
existing_pid=$(netstat -ano 2>/dev/null | grep ":$PORT " | grep LISTENING | awk '{print $NF}' | head -1)
if [ -n "$existing_pid" ]; then
    echo "Killing existing process on port $PORT (PID $existing_pid)..."
    taskkill //F //PID "$existing_pid" 2>/dev/null || true
    sleep 2
fi

# CGC expert-cache env
export LLAMA_EXPERT_CACHE_ALLOW_NGL=1
export CGC_EXPERT_CACHE_BYTES="${BUDGET:-4294967296}"
export LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
export LLAMA_EXPERT_CACHE_WORKERS=8
export CGC_WAKE_POLL_US=15
export CGC_PREFETCH_SRC=hist
export CGC_EVICTED_RING=0
export CGC_OA_ASYNC=1
export CGC_N_CB=8
export CGC_GLU_FUSED_DOWN=1

echo "=== Starting llama-server ==="
echo "  Binary: $BIN"
echo "  Model:  $(basename "$M")"
echo "  Port:   $PORT"
echo "  NGL:    $NGL"
echo "  CGC expert-cache: ON (budget $(( ${BUDGET:-4294967296} / 1073741824 ))GiB)"
echo ""

# Start server in background
"$BIN" -m "$M" --host 0.0.0.0 --port "$PORT" \
    -ngl "$NGL" -c 2048 -t 8 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo -n "Waiting for server..."
for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo " READY!"
        break
    fi
    sleep 2
    echo -n "."
done

# Test endpoint
echo ""
echo "=== Testing API ==="
curl -s "http://127.0.0.1:$PORT/v1/models" 2>&1 | python -c "
import sys,json
try:
    data=json.load(sys.stdin)
    for m in data.get('data',[]):
        print(f'  Model: {m[\"id\"]}')
except: print('  (waiting...)')
" 2>/dev/null || true

# Update tb_loop config.env
CONFIG="$ROOT/config.env"
if [ -f "$CONFIG" ]; then
    echo ""
    echo "=== Updating tb_loop config.env ==="
    # Replace SFT_API_BASE_URL and SFT_API_KEY
    sed -i "s|^SFT_API_BASE_URL=.*|SFT_API_BASE_URL=\"http://127.0.0.1:$PORT/v1\"|" "$CONFIG"
    sed -i "s|^SFT_API_KEY=.*|SFT_API_KEY=\"local\"|" "$CONFIG"
    echo "  SFT_API_BASE_URL=http://127.0.0.1:$PORT/v1"
    echo "  SFT_API_KEY=local"
fi

echo ""
echo "=== Ready ==="
echo "Server running at http://127.0.0.1:$PORT"
echo "tb_loop config updated. You can now run:"
echo "  cd $ROOT && bash scripts/batch_run.sh batch_wsl"
echo ""
echo "To stop: kill $SERVER_PID (or taskkill //F //PID $SERVER_PID)"
