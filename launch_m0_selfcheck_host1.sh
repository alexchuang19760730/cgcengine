#!/usr/bin/env bash
# Host1 (39.106.118.206) — M0 in-process resume self-check (CGC Phase 2).
# Single instance, NO PD, NO MTP. The model's forward runs a baseline forward
# (capturing hidden_states at CGC_SELFCHECK_CUT) then a resume forward
# (finished_layer=cut) and logs [CGC_SELFCHECK] PASS/FAIL. One cut per request;
# pass a comma list (e.g. 21,42) and send that many requests.
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:${PYTHONPATH:-}
# CGC custom kernels (same profile proven correct in single-node forward tests)
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_ENABLE_PREFILL_POOL=1
export CGC_ENABLE_GDS=1
export CGC_ENABLE_CQ4=1
# --- M0 self-check ---
export CGC_SELFCHECK_CUT="${CGC_SELFCHECK_CUT:-21,42}"
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
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-custom-all-reduce \
  --trust-remote-code --skip-server-warmup
