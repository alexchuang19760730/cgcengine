#!/bin/bash
set -e

export all_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

echo "=== 驗證 cuDNN ==="
ssh gs01@10.100.200.65 "ldconfig -p | grep cudnn"

echo ""
echo "=== 測試 PyTorch CUDA ==="
ssh gs01@10.100.200.65 "python3 -c 'import torch; print(\"PyTorch:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available()); print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")'"

echo ""
echo "=== 執行真實基準測試 ==="
ssh gs01@10.100.200.65 "cd /home/gs01/MagiCompiler && python3 server/real_cgc_vllm_benchmark.py"
