#!/bin/bash
# Host1: sglang 0.5.16 + Gemma 4 26B-A4B + MTP + cuda-graph + CGC (R-SWA + OrthoKDA)

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
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30001 \
  --trust-remote-code \
  --mem-fraction-static 0.60 \
  --cuda-graph-max-bs 4
