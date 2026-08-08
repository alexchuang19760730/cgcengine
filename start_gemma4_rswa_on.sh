#!/bin/bash
# Host1: sglang + Gemma 4 26B-A4B + MTP (steps=5, topk=2) + R-SWA patch
# R-SWA: caps full-attention decode to reference(4) + window(128) = 132 tokens

export TMPDIR=/data/tmp
export HF_HOME=/data/hf_cache
export FLASHINFER_DISABLE_VERSION_CHECK=1

# CGC 注入
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_RSWA_WINDOW_SIZE=128
export CGC_ORTHO_BASE_DIM=128

# R-SWA Triton Backend Patch: cap attention length
export RSWA_MAX_ATTN_LEN=132

cd /root/flashkv0516

exec /data/venv_gemma4/bin/python3 -m sglang.launch_server \
  --model-path /data/models/gemma-4-26b-a4b-it \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /data/models/gemma-4-26b-a4b-it-assistant \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 2 \
  --speculative-num-draft-tokens 8 \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30001 \
  --trust-remote-code \
  --mem-fraction-static 0.55 \
  --cuda-graph-max-bs 4
