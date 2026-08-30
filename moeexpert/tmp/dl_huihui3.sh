#!/bin/bash
# Download from hf-mirror.com (no proxy needed)
set -e
cd /Users/alexchuang/Documents/flashkv0516
OUT="models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
URL="https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"

echo "=== Starting $(date) ===" 
# Remove old partial file
rm -f "$OUT"

for i in $(seq 1 50); do
    echo "Attempt $i at $(date)"
    curl -fSL --retry 3 --retry-delay 2 --connect-timeout 15 --max-time 600 \
        -o "$OUT" "$URL" && echo "SUCCESS" && exit 0
    echo "Failed, retrying..."
    sleep 2
done
echo "FAILED after 50 attempts"
