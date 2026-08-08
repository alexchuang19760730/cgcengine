#!/bin/bash
# Host1: sglang + Gemma 4 26B-A4B + MTP (num-steps=5, topk=2) + SWA ratio=0.5
# Test: more SWA layers, less full attention memory

export TMPDIR=/data/tmp
export HF_HOME=/data/hf_cache
export FLASHINFER_DISABLE_VERSION_CHECK=1

# CGC env vars (set but R-SWA not patched — testing native SWA tuning)
export CGC_ENABLE_ORTHO_KDA=0
export CGC_ENABLE_RSWA=0

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
  --cuda-graph-max-bs 4 \
  --swa-full-tokens-ratio 0.5
