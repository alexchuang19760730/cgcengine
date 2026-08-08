#!/bin/bash
# Phase 2 M1v2 — Host1 (edge) = RESUME instance over TCP transport.
# Pulls layer-CGC_RESUME_FROM hidden_states from the cloud's per-rank TCP
# server (reached via SSH tunnel: CGC_TRANSPORT_TCP_HOST=127.0.0.1 ->
# Host2:31000+rank) and runs cut+1..end + lm_head to produce tokens.
# Decode side: --disable-cuda-graph REMOVED to enable full cuda-graph and
# measure decode tok/s (target 50). --disable-piecewise-cuda-graph kept
# (use full, non-piecewise graph for decode).
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516/ComputeGraphCompiler-main:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:/root/flashkv0516:${PYTHONPATH:-}
# CGC custom kernels (profile proven correct in single-node forward tests)
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_ENABLE_PREFILL_POOL=1
export CGC_ENABLE_GDS=1
export CGC_ENABLE_CQ4=1
# --- M1v2 resume (edge side, TCP transport, cuda-graph ON) ---
export CGC_RESUME_FROM="${CGC_RESUME_FROM:-21}"
export CGC_HANDOFF_PATH="${CGC_HANDOFF_PATH:-/data/cgc_handoff.pt}"
export CGC_TRANSPORT="${CGC_TRANSPORT:-tcp}"
export CGC_TRANSPORT_TCP_HOST="${CGC_TRANSPORT_TCP_HOST:-127.0.0.1}"
export CGC_TRANSPORT_TCP_PORT="${CGC_TRANSPORT_TCP_PORT:-31000}"
cd /root/flashkv0516
exec python cgc_launch_dual_node.py \
  --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2 \
  --host 0.0.0.0 --port 30000 \
  --device cuda --tp-size 8 --ep-size 1 --nnodes 1 --node-rank 0 \
  --context-length 16384 --mem-fraction-static 0.82 \
  --max-running-requests 2 --max-total-tokens 32768 --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.125 \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-custom-all-reduce \
  --watchdog-timeout 1800 \
  --trust-remote-code --skip-server-warmup
