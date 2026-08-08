#!/bin/bash
# Host1: sglang 0.5.16 + Gemma 4 26B-A4B + MTP (num-steps=5, topk=2) + cuda-graph + CGC
# 修改: speculative-num-steps 3→5, eagle-topk 1→2, speculative-num-draft-tokens 3→8

export TMPDIR=/data/tmp
export HF_HOME=/data/hf_cache
export FLASHINFER_DISABLE_VERSION_CHECK=1

# CGC 注入 (OrthoKDA + R-SWA)
export CGC_ENABLE_ORTHO_KDA=1
export CGC_ENABLE_RSWA=1
export CGC_RSWA_WINDOW_SIZE=128
export CGC_ORTHO_BASE_DIM=128

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
