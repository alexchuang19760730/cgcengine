#!/usr/bin/env bash
set -u
export REPO=/root/flashkv0516
export PYTHONPATH="$REPO/ComputeGraphCompiler-main:$REPO/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:$REPO"
source "$REPO/.venv_deepep_ssp/bin/activate"

# All CGC custom kernels OFF -> plain sglang (torch) path, most comparable to the reference.
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_ENABLE_PREFILL_POOL=1
export CGC_ENABLE_GDS=1
export CGC_ENABLE_CQ4=1

# Diagnostic dump: capture per-layer post-block residual for the exact reference prompt ids.
export SGLANG_DSV4_DUMP_HS=1
export SGLANG_DSV4_DUMP_HS_IDS="0,128803,671,6102,294,8760,344,128804,128822"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
rm -f /data/fork_hs.pt
cd "$REPO"
exec python "$REPO/cgc_launch_dual_node.py" \
  --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2 \
  --host 127.0.0.1 --port 30000 --device cuda \
  --tp-size 8 --ep-size 1 \
  --nnodes 1 --node-rank 0 \
  --context-length 16384 --mem-fraction-static 0.82 \
  --max-running-requests 2 --max-total-tokens 32768 \
  --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.125 \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-custom-all-reduce \
  --trust-remote-code --skip-server-warmup
