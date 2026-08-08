#!/bin/bash
set -e
echo "============================================================"
echo "  🚀 CGC Engine + DeepSeek V4 Flash + vLLM 一鍵基準測試"
echo "============================================================"

WORK_DIR="/home/gs01"
cd $WORK_DIR

echo "[步驟 1/5] 初始化環境..."
if [ -d "cgc_vllm_env" ]; then
    source cgc_vllm_env/bin/activate
    echo "虛擬環境已激活"
fi

echo "[步驟 2/5] 配置環境變數..."
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export SPDK_MAX_QUEUES=16
export KDA_ENABLED=1

echo "[步驟 3/5] 設定雙端 GPU/PD 分離..."
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "偵測到 GPU 數量: $NUM_GPUS"

echo "[步驟 4/5] 啟動 Harness Agent 執行基準測試..."
cd MagiCompiler
python3 server/cgc_vllm_benchmark.py

echo "[步驟 5/5] 完成！查看報告"
ls -la deepseek_v4_benchmark_report.json

echo ""
echo "============================================================"
echo "  ✅ 全部基準測試完成！"
echo "============================================================"
