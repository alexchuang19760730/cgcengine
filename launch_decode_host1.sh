#!/bin/bash
# Phase 1 — Host1 (edge) = DECODE-only instance + MTP (NEXTN), native SGLang PD (nixl).
# Single node, tp8, NO dp-attention, NO cross-node dp. CGC kernels on (matches broken run config).
# Decode fetches KV from prefill (Host2 private 172.30.132.117:8998) via per-request bootstrap_host.
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
# Default UCX (eRDMA) needed for VRAM memory registration; cu12 conflict removed so no segfault.
export PYTHONPATH=/root/flashkv0516/ComputeGraphCompiler-main:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:/root/flashkv0516:${PYTHONPATH:-}
exec python /root/flashkv0516/cgc_launch_dual_node.py \
  --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2 \
  --host 0.0.0.0 --port 30000 \
  --device cuda --tp-size 8 --ep-size 1 --nnodes 1 --node-rank 0 \
  --context-length 16384 --mem-fraction-static 0.82 \
  --max-running-requests 4 --max-total-tokens 32768 --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.125 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --speculative-algorithm NEXTN \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-custom-all-reduce \
  --trust-remote-code --skip-server-warmup
