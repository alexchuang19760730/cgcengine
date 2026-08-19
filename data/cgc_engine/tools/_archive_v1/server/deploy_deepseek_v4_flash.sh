#!/bin/bash
set -e

echo "============================================================"
echo "  DeepSeek V4 Flash + vLLM + CGC Engine 部署腳本"
echo "============================================================"

WORK_DIR="/home/gs01"
cd $WORK_DIR

echo "[1/7] 更新系統依賴..."
sudo apt update -y && sudo apt install -y \
    git python3-pip python3-venv nvme-cli \
    libnuma-dev libaio-dev libspdk-dev \
    libnccl2 libnccl-dev

echo "[2/7] 創建 Python 虛擬環境..."
python3 -m venv cgc_vllm_env
source cgc_vllm_env/bin/activate
pip install --upgrade pip setuptools wheel

echo "[3/7] 克隆 vLLM (DeepSeek V4 Flash 支援版)..."
if [ ! -d "vllm" ]; then
    git clone --depth 1 https://github.com/vllm-project/vllm.git
    cd vllm
else
    cd vllm
    git pull
fi

echo "[4/7] 安裝 vLLM 基礎依賴..."
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install nccl-cu12 cuGraph-cu12 pylibspdk

echo "[5/7] 克隆 MagiCompiler (CGC Engine)..."
cd $WORK_DIR
if [ ! -d "MagiCompiler" ]; then
    echo "請確保 MagiCompiler 已從本機同步到服務器"
else
    cd MagiCompiler
    pip install -e .
fi

echo "[6/7] SPDK NVMe 配置..."
sudo nvme list
sudo modprobe nvme
echo "啟用 SPDK NVMe 零拷貝..."
sudo hugepages --setup 1024

echo "[7/7] 完成部署！"
echo "============================================================"
echo "  下一步: 執行 benchmark 進行效能測試"
echo "============================================================"
