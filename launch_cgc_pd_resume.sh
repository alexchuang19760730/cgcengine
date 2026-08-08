#!/bin/bash
# CGC PD 通用启动脚本 — Edge (resume) 端
# 适用于任何 sglang 模型 (V4-Flash / Qwen3-VL / 其他)
#
# 用法:
#   CGC_RESUME_FROM=21 CGC_TRANSPORT=nixl bash launch_cgc_pd_resume.sh <model_path> <tp_size>
#
# 环境变量:
#   CGC_RESUME_FROM    - 从哪层开始 resume (默认 21, = cloud 的 emit_cut)
#   CGC_TRANSPORT      - 传输模式 nixl/tcp/file (默认 nixl)
#   CGC_RESUME_TIMEOUT - recv 超时秒 (默认 120)

set -u

MODEL_PATH="${1:-/data/models/DeepSeek-V4-Flash-UD-IQ2}"
TP_SIZE="${2:-8}"
RESUME_FROM="${CGC_RESUME_FROM:-21}"
TRANSPORT="${CGC_TRANSPORT:-nixl}"

export CGC_RESUME_FROM="$RESUME_FROM"
export CGC_TRANSPORT="$TRANSPORT"
export CGC_TRANSPORT_TCP_PORT="${CGC_TRANSPORT_TCP_PORT:-31000}"
export CGC_TRANSPORT_TCP_HOST="${CGC_TRANSPORT_TCP_HOST:-172.30.132.117}"
export CGC_RESUME_TIMEOUT="${CGC_RESUME_TIMEOUT:-120}"
export PYTHONPATH="/root/flashkv0516:${PYTHONPATH:-}"

cd /root/flashkv0516

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
