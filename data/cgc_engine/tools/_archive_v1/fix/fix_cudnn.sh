#!/bin/bash
set -e

export all_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

echo "=== 在服務器上安裝 cuDNN ==="
ssh gs01@10.100.200.65 << 'ENDSSH'
    export all_proxy=http://127.0.0.1:7897
    export http_proxy=http://127.0.0.1:7897
    export https_proxy=http://127.0.0.1:7897

    echo "檢查 CUDA 版本..."
    nvcc --version 2>/dev/null || echo "CUDA 未安裝"

    echo "安裝 cuDNN..."
    sudo apt update
    sudo apt install -y libcudnn9 libcudnn9-dev

    echo "驗證 cuDNN..."
    python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
ENDSSH
