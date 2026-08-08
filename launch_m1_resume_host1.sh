#!/bin/bash
# Phase 2 M1v1 — Host1 (edge) = RESUME instance.
# Plain single-node instance (NO MTP for M1v1 degenerate baseline). At request
# time it loads /data/cgc_handoff.pt.rank{N} (emitted by Host2) and resumes the
# model from CGC_RESUME_FROM, producing the same first token the cloud produced.
# NOTE: degenerate baseline (cut=last layer 42) runs ZERO decoder layers on the
# edge, so the request MUST use max_tokens=1 (no KV cache exists on the edge).
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516/ComputeGraphCompiler-main:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:/root/flashkv0516:${PYTHONPATH:-}
# CGC custom kernels (same profile proven correct in single-node forward tests)
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_ENABLE_PREFILL_POOL=1
export CGC_ENABLE_GDS=1
export CGC_ENABLE_CQ4=1
# --- M1 resume (edge side) ---
export CGC_RESUME_FROM="${CGC_RESUME_FROM:-42}"
export CGC_HANDOFF_PATH="${CGC_HANDOFF_PATH:-/data/cgc_handoff.pt}"
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
