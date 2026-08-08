#!/bin/bash
# CGC PD 通用启动脚本 — Cloud (emit) 端
# 适用于任何 sglang 模型 (V4-Flash / Qwen3-VL / 其他)
#
# 用法:
#   CGC_EMIT_CUT=21 CGC_TRANSPORT=nixl bash launch_cgc_pd_emit.sh <model_path> <tp_size>
#
# 环境变量:
#   CGC_EMIT_CUT       - emit hidden_states 的层 (默认 21)
#   CGC_TRANSPORT      - 传输模式 nixl/tcp/file (默认 nixl)
#   CGC_TRANSPORT_TCP_PORT - TCP 基础端口 (默认 31000, 每_rank +1)
#   CGC_TRANSPORT_TCP_HOST - edge 连接地址 (VPC IP)

set -u

MODEL_PATH="${1:-/data/models/DeepSeek-V4-Flash-UD-IQ2}"
TP_SIZE="${2:-8}"
EMIT_CUT="${CGC_EMIT_CUT:-21}"
TRANSPORT="${CGC_TRANSPORT:-nixl}"

export CGC_EMIT_CUT="$EMIT_CUT"
export CGC_TRANSPORT="$TRANSPORT"
export CGC_TRANSPORT_TCP_PORT="${CGC_TRANSPORT_TCP_PORT:-31000}"
export CGC_TRANSPORT_TCP_HOST="${CGC_TRANSPORT_TCP_HOST:-172.30.132.117}"
export CGC_NIXL_CLOUD_HOST="${CGC_NIXL_CLOUD_HOST:-172.30.132.117}"
export PYTHONPATH="/root/flashkv0516:${PYTHONPATH:-}"

cd /root/flashkv0516

# 通过 --trust-remote-code + CGC_PD_AUTO_PATCH=1 触发通用 patch
# patch 在 cgc_pd_patch.py 的 patch_sglang_model() 中自动应用
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 --port 30001 \
  --device cuda --tp-size "$TP_SIZE" --ep-size 1 --nnodes 1 --node-rank 0 \
  --mem-fraction-static 0.82 \
  --max-running-requests 2 \
  --attention-backend flashinfer \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-custom-all-reduce \
  --watchdog-timeout 1800 \
  --trust-remote-code --skip-server-warmup
