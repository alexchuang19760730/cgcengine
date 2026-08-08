#!/usr/bin/env bash
# Host1 (39.106.118.206) — NORMAL single-instance + MTP (NEXTN) for MTP validation.
# No PD / no dp-attention. Validates the "edge decode + MTP draft" component of the end goal.
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:${PYTHONPATH:-}
# CGC custom kernels (same profile proven correct in single-node forward tests)
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_ENABLE_PREFILL_POOL=1
export CGC_ENABLE_GDS=1
export CGC_ENABLE_CQ4=1
cd /root/flashkv0516
exec python cgc_launch_dual_node.py \
  --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2 \
  --host 0.0.0.0 --port 30000 \
  --device cuda \
  --tp-size 8 --ep-size 1 --nnodes 1 --node-rank 0 \
  --context-length 16384 \
  --mem-fraction-static 0.82 \
  --max-running-requests 4 \
  --max-total-tokens 32768 \
  --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.125 \
  --speculative-algorithm NEXTN \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-custom-all-reduce \
  --trust-remote-code --skip-server-warmup
