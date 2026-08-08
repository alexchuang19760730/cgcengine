#!/bin/bash
# Edge NIXL resume: Host2 sglang (clean, cuda-graph-capable) + CGC OFF (native,
# cgc_engine overlay garbles on Host2 sglang) + --disable-cuda-graph (so emit
# callback fires every forward for per-step handoff) + nixl + resume_from=21.
# Pulls cloud hidden_states via NIXL (metadata-refresh fix in transport).
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516/ComputeGraphCompiler-main:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:/root/flashkv0516:${PYTHONPATH:-}
export RAY_TMPDIR=/data/ray
export TMPDIR=/data/ray
export CGC_ENABLE_ORTHO_KDA=0
export CGC_ENABLE_RSWA=0
export CGC_ENABLE_PREFILL_POOL=0
export CGC_ENABLE_GDS=0
export CGC_ENABLE_CQ4=0
# --- M1v2 resume (edge side, NIXL zero-copy transport) ---
export CGC_RESUME_FROM="${CGC_RESUME_FROM:-21}"
export CGC_TRANSPORT="${CGC_TRANSPORT:-nixl}"
export CGC_TRANSPORT_TCP_PORT="${CGC_TRANSPORT_TCP_PORT:-31000}"
export CGC_TRANSPORT_TCP_HOST="${CGC_TRANSPORT_TCP_HOST:-172.30.132.117}"
export CGC_NIXL_CLOUD_HOST="${CGC_NIXL_CLOUD_HOST:-172.30.132.117}"
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
