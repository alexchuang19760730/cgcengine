#!/bin/bash
# Edge INDEPENDENT decode + cuda-graph ON + CGC OFF (native) + MTP speculative.
# Path to 50 tok/s: native 37 tok/s * MTP ~2x = ~74. CGC kernels disabled
# (they garble output under the Host2-sglang overlay on Host1).
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
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 --speculative-num-draft-tokens 2 --speculative-eagle-topk 1 \
  --watchdog-timeout 1800 \
  --trust-remote-code --skip-server-warmup
