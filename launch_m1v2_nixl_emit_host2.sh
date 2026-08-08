#!/bin/bash
# Phase 2 M1v2 — Host2 (cloud) = EMIT instance over NIXL zero-copy transport.
# Streams layer-CGC_EMIT_CUT hidden_states to the edge (Host1) via NIXL
# READ: each rank runs a nixl_agent listening on CGC_TRANSPORT_TCP_PORT+rank
# (default 31000..31007) on the VPC IP 172.30.132.117; the edge pulls the
# tensor straight from cloud VRAM into edge VRAM (no host bounce, no disk).
# Falls back to the per-rank TCP channel if NIXL is unavailable.
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516/ComputeGraphCompiler-main:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:/root/flashkv0516:${PYTHONPATH:-}
# keep Ray / torch temp off the nearly-full root fs (Host2 has /data2 free)
export RAY_TMPDIR=/data2/ray
export TMPDIR=/data2/ray
# CGC custom kernels (profile proven correct in single-node forward tests)
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_ENABLE_PREFILL_POOL=1
export CGC_ENABLE_GDS=1
export CGC_ENABLE_CQ4=1
# --- M1v2 emit (cloud side, NIXL zero-copy transport) ---
export CGC_EMIT_CUT="${CGC_EMIT_CUT:-21}"
export CGC_HANDOFF_PATH="${CGC_HANDOFF_PATH:-/data/cgc_handoff.pt}"
export CGC_TRANSPORT="${CGC_TRANSPORT:-nixl}"
export CGC_TRANSPORT_TCP_PORT="${CGC_TRANSPORT_TCP_PORT:-31000}"
export CGC_TRANSPORT_TCP_HOST="${CGC_TRANSPORT_TCP_HOST:-172.30.132.117}"
export CGC_NIXL_CLOUD_HOST="${CGC_NIXL_CLOUD_HOST:-172.30.132.117}"
cd /root/flashkv0516
exec python cgc_launch_dual_node.py \
  --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2 \
  --host 0.0.0.0 --port 30001 \
  --device cuda --tp-size 8 --ep-size 1 --nnodes 1 --node-rank 0 \
  --context-length 16384 --mem-fraction-static 0.82 \
  --max-running-requests 2 --max-total-tokens 32768 --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.125 \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-custom-all-reduce \
  --watchdog-timeout 1800 \
  --trust-remote-code --skip-server-warmup
