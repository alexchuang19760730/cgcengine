#!/bin/bash
# 原生 llama.cpp 基线启动 — 完全绕过 run_server.sh 的所有 CGC 环境变量
# 用途：归因测试 B/E 组，得到"量化+引擎"纯基线（无 expert cache / 无 MTP / 无 CGC 修改）
#
# 用法：
#   bash scripts/run_native_baseline.sh IQ3_XXS    # 用 IQ3_XXS 模型
#   bash scripts/run_native_baseline.sh IQ4_XS     # 用 IQ4_XS 模型
#   PORT=8090 bash scripts/run_native_baseline.sh IQ3_XXS
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/src/llama.cpp/build/bin/llama-server"
MODEL_ROOT="$ROOT/models/gguf"

QUANT="${1:-IQ3_XXS}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"
NGL="${NGL:-99}"

case "$QUANT" in
    IQ3_XXS|iq3_xxs|xxs)
        MODEL="$MODEL_ROOT/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
        LABEL="IQ3_XXS-native"
        ;;
    IQ4_XS|iq4_xs|iq4)
        MODEL="$MODEL_ROOT/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
        LABEL="IQ4_XS-native"
        ;;
    *)
        echo "未知量化: $QUANT（可选: IQ3_XXS / IQ4_XS）" >&2
        exit 1
        ;;
esac

if [ ! -x "$BIN" ]; then
    echo "error: llama-server 不存在: $BIN" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "error: 模型不存在: $MODEL" >&2
    exit 1
fi

echo "========================================"
echo "  原生基线启动（无 expert cache / 无 MTP / 无 CGC env）"
echo "========================================"
echo "  量化:    $QUANT"
echo "  模型:    $MODEL"
echo "  端口:    $PORT"
echo "  ctx:     $CTX"
echo "  ngl:     $NGL"
echo "  binary:  $BIN"
echo "========================================"
echo ""
echo "注意: 此模式下"
echo "  - 不传 -expert-cache → 完全禁用 expert cache"
echo "  - 不传 -draft / -draft-mtp → 禁用 MTP speculative decoding"
echo "  - 不设任何 CGC_* / LLAMA_EXPERT_CACHE_* 环境变量"
echo "  - 得到的是 llama.cpp + 量化的纯基线"
echo ""

# 干净环境：unset 所有 CGC 和 expert cache 相关变量
unset CGC_EXPERT_CACHE_BYTES
unset LLAMA_EXPERT_CACHE_ALLOW_NGL
unset LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0
unset LLAMA_EXPERT_CACHE_WORKERS
unset LLAMA_EXPERT_CACHE_LAYER_CAPS
unset CGC_WAKE_POLL_US CGC_PREFETCH_SRC CGC_EVICTED_RING
unset CGC_N_CB CGC_OA_ASYNC CGC_SERVER_AUTO_ANCHOR
unset CGC_VERIFY_DECODE CGC_DRAFT_DECODE CGC_NO_PREFETCH
unset CGC_DBUF CGC_SPAC CGC_SOFT_POOL_L0 CGC_SOFT_POOL_L1
unset CGC_LOOP_GUARD CGC_GLU_FUSED_DOWN CGC_WATCHDOG
unset CGC_FAST_COLD_MAX CGC_DRAFT_STRICT CGC_FAST_WAIT
unset CGC_PREV_TOKEN_PREFETCH CGC_DOWN_COMBINE

exec "$BIN" \
    -m "$MODEL" \
    -ngl "$NGL" \
    --no-mmap \
    -t 8 \
    -c "$CTX" \
    -np 1 \
    --no-kv-unified \
    -sps 0 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --jinja
