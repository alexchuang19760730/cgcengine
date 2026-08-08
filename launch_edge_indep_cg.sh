#!/bin/bash
# Edge INDEPENDENT decode (no cloud handoff, no resume) + cuda-graph ON.
# Purpose: verify whether the edge's c4_v2.cuh:322 "expected 1 but got 2"
# cuda-graph crash is fixed by the instrumentation host-sync patches
# (same patches that fixed the cloud). If capture succeeds here, c4_v2 was
# a cascade of the instrumentation sync, not a real kernel capture bug.
# Independent (no CGC_RESUME_FROM / CGC_TRANSPORT) to isolate the graph question.
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516/ComputeGraphCompiler-main:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:/root/flashkv0516:${PYTHONPATH:-}
export RAY_TMPDIR=/data/ray
export TMPDIR=/data/ray
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_ENABLE_PREFILL_POOL=1
export CGC_ENABLE_GDS=1
export CGC_ENABLE_CQ4=1
cd /root/flashkv0516
exec python cgc_launch_dual_node.py \
  --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2 \
  --host 0.0.0.0 --port 30000 \
  --device cuda --tp-size 8 --ep-size 1 --nnodes 1 --node-rank 0 \
  --context-length 16384 --mem-fraction-static 0.82 \
  --max-running-requests 2 --max-total-tokens 32768 --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.125 \
  --cuda-graph-max-bs 2 \
  --disable-piecewise-cuda-graph --disable-custom-all-reduce \
  --watchdog-timeout 1800 \
  --trust-remote-code --skip-server-warmup
