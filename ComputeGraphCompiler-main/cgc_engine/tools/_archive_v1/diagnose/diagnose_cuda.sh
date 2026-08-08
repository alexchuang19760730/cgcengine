#!/bin/bash
set -e

export all_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

echo "=== 完整診斷服務器 CUDA/GPU 環境 ==="
ssh gs01@10.100.200.65 << 'ENDSSH'
    export all_proxy=http://127.0.0.1:7897
    export http_proxy=http://127.0.0.1:7897
    export https_proxy=http://127.0.0.1:7897

    echo "--- 1. 檢查 NVIDIA GPU ---"
    nvidia-smi || echo "nvidia-smi 失敗"

    echo ""
    echo "--- 2. 檢查 CUDA 版本 ---"
    nvcc --version 2>/dev/null || echo "nvcc 未找到"
    cat /usr/local/cuda/version.txt 2>/dev/null || echo "CUDA version.txt 未找到"

    echo ""
    echo "--- 3. 檢查 cuDNN ---"
    find /usr -name "libcudnn*" 2>/dev/null | head -10
    ldconfig -p | grep cudnn 2>/dev/null || echo "ldconfig 無 cuDNN"

    echo ""
    echo "--- 4. 檢查 PyTorch ---"
    pip3 show torch 2>/dev/null | grep Version || echo "PyTorch 未安裝"
    python3 -c "import torch; print('PyTorch:', torch.__version__)" 2>&1

    echo ""
    echo "--- 5. 嘗試修復 cuDNN ---"
    echo "y" | sudo apt install --reinstall libcudnn9 libcudnn9-dev 2>&1 || echo "需要手動處理 sudo"

    echo ""
    echo "--- 6. 驗證修復 ---"
    python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
ENDSSH
