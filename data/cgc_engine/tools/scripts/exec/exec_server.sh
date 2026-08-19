#!/bin/bash
set -e
export all_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

echo "=== 1. 同步代碼到服務器 ==="
rsync -avz --exclude '__pycache__' --exclude '.git' --exclude '*.pyc' \
    /Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/ \
    gs01@10.100.200.65:/home/gs01/MagiCompiler/

echo "=== 2. 在服務器上執行真實測試 ==="
ssh gs01@10.100.200.65 << 'ENDSSH'
    export all_proxy=http://127.0.0.1:7897
    export http_proxy=http://127.0.0.1:7897
    export https_proxy=http://127.0.0.1:7897
    cd /home/gs01/MagiCompiler
    chmod +x server/run_real_benchmark.sh
    python3 server/real_cgc_vllm_benchmark.py
ENDSSH
