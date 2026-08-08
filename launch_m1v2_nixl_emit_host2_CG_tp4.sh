#!/bin/bash
# Variant TP4: cuda-graph ON + TP4 EP1 (single-node, GPU0-3 same NUMA).
# vs CG variant: tp-size 8 -> 4. Goal: measure TP4 decode tok/s vs TP8 36.6
# baseline (allreduce shrinks from 8-card cross-NUMA SYS to 4-card same-NUMA NODE).
# mem: TP4 doubles per-card weights (~42GB); 72GB card @0.82 static should fit.
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516/ComputeGraphCompiler-main:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:/root/flashkv0516:${PYTHONPATH:-}
export RAY_TMPDIR=/data2/ray
export TMPDIR=/data2/ray
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_ENABLE_PREFILL_POOL=1
export CGC_ENABLE_GDS=1
export CGC_ENABLE_CQ4=1
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
  --device cuda --tp-size 4 --ep-size 1 --nnodes 1 --node-rank 0 \
  --context-length 16384 --mem-fraction-static 0.82 \
  --max-running-requests 2 --max-total-tokens 32768 --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.125 \
  --cuda-graph-max-bs 2 \
  --disable-piecewise-cuda-graph --disable-custom-all-reduce \
  --watchdog-timeout 1800 \
  --trust-remote-code --skip-server-warmup
