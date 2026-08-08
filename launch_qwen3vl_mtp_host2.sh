#!/bin/bash
# Cloud Qwen3-VL-30B-A3B + EAGLE3 MTP (fa3 attention, 无 dsv4 约束, 解 V4-Flash MTP 阻塞)
# 验证: acceptance length + tok/s (对比 V4-Flash accept 0.28 / 20.2 tok/s)
set -u
source /root/flashkv0516/.venv_deepep_ssp/bin/activate
export PYTHONPATH=/root/flashkv0516/ComputeGraphCompiler-main:/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python:/root/flashkv0516:${PYTHONPATH:-}
export RAY_TMPDIR=/data2/ray
export TMPDIR=/data2/ray
cd /root/flashkv0516
exec python -m sglang.launch_server \
  --model-path /data2/models/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 --port 30001 \
  --tp 8 --ep-size 1 --nnodes 1 --node-rank 0 \
  --trust-remote-code \
  --context-length 32768 --mem-fraction-static 0.82 \
  --max-running-requests 2 --cuda-graph-max-bs 2 \
  --attention-backend flashinfer --mm-attention-backend triton_attn \
  --speculative-algorithm EAGLE3 \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-path /data2/models/EAGLE3-Qwen3-30B-draft \
  --watchdog-timeout 1800 \
  --skip-server-warmup
