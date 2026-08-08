#!/bin/bash
# Phase 2 M1v1 — Host2 (cloud) = EMIT instance.
# Plain single-node instance (NO native nixl PD, NO MTP). At CGC_EMIT_CUT it
# captures hidden_states to /data/cgc_handoff.pt.rank{N} (one file per tp rank)
# and ALSO returns its normal output (baseline token). The edge (Host1) later
# loads those files and resumes from the cut layer.
# Degenerate baseline: CGC_EMIT_CUT = last layer (42) -> edge only runs lm_head.
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516/ComputeGraphCompiler-main:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:/root/flashkv0516:${PYTHONPATH:-}
# CGC custom kernels (same profile proven correct in single-node forward tests)
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_ENABLE_PREFILL_POOL=1
export CGC_ENABLE_GDS=1
export CGC_ENABLE_CQ4=1
# --- M1 emit (cloud side) ---
export CGC_EMIT_CUT="${CGC_EMIT_CUT:-42}"
export CGC_HANDOFF_PATH="${CGC_HANDOFF_PATH:-/data/cgc_handoff.pt}"
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
