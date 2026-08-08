#!/bin/bash
set -e

export all_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

cat << 'SCRIPT' | ssh -o StrictHostKeyChecking=no gs01@10.100.200.65 bash
export all_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

echo "=== 驗證 cuDNN ==="
ldconfig -p | grep cudnn || echo "cuDNN 未找到"

echo ""
echo "=== 測試 PyTorch CUDA ==="
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

echo ""
echo "=== 執行真實基準測試 ==="
cd /home/gs01/MagiCompiler
python3 server/real_cgc_vllm_benchmark.py
SCRIPT
