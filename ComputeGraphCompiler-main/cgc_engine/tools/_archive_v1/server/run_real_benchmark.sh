#!/bin/bash
set -e
echo "============================================================"
echo "  🚀 真實 GPU 效能基準測試 - 100% 無模擬數據"
echo "  DeepSeek V4 Flash + vLLM + CGC Engine"
echo "============================================================"

WORK_DIR="/home/gs01/MagiCompiler"
cd $WORK_DIR

if [ -d "cgc_vllm_env" ]; then
    source cgc_vllm_env/bin/activate
fi

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=0

echo "[檢查 CUDA 環境]"
python3 -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"

echo ""
echo "[執行真實基準測試]"
python3 server/real_cgc_vllm_benchmark.py

echo ""
echo "============================================================"
echo "  ✅ 真實測試完成！查看 real_deepseek_v4_benchmark_report.json"
echo "============================================================"
