#!/bin/bash
cd /Users/alexchuang/Documents/flashkv0516
echo "Starting download at $(date)" > /tmp/huihui_mirror.log
curl -L --retry 99 --retry-delay 3 --connect-timeout 15 --max-time 0 \
  -o models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf \
  "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf" \
  2>&1 >> /tmp/huihui_mirror.log
echo "Finished at $(date) with exit $?" >> /tmp/huihui_mirror.log
