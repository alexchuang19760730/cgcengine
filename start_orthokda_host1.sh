#!/bin/bash
# Host1: sglang + Gemma 4 26B-A4B + OrthoKDA (no MTP, no cuda-graph)
# Uses sitecustomize.py to auto-load patch before sglang imports

export TMPDIR=/data/tmp
export HF_HOME=/data/hf_cache
export FLASHINFER_DISABLE_VERSION_CHECK=1

# OrthoKDA config
export ORTHO_KDA_ENABLED=1
export ORTHO_REF_LEN=4
export ORTHO_WINDOW_SIZE=128
export ORTHO_BASE_DIM=128

# Disable R-SWA clamp
export RSWA_MAX_ATTN_LEN=0

# CGC env vars (dead flags on standard sglang)
export CGC_ENABLE_ORTHO_KDA=0
export CGC_ENABLE_RSWA=0

# Python path: sitecustomize.py + orthokda_patch_loader.py are here
export PYTHONPATH=/root/flashkv0516/rswaengine/python:/root/flashkv0516:$PYTHONPATH

cd /root/flashkv0516

# Kill existing sglang (use PID file, not pattern match)
if [ -f /tmp/sglang_orthokda.pid ]; then
  kill -9 $(cat /tmp/sglang_orthokda.pid) 2>/dev/null || true
  rm -f /tmp/sglang_orthokda.pid
fi
sleep 2

# Launch with standard sglang entry point (sitecustomize.py auto-loads patch)
exec /data/venv_gemma4/bin/python3 -m sglang.launch_server \
  --model-path /data/models/gemma-4-26b-a4b-it \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30001 \
  --trust-remote-code \
  --mem-fraction-static 0.55 \
  --cuda-graph-max-bs 1
