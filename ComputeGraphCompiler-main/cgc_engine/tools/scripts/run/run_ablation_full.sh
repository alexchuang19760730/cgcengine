#!/bin/bash
# 消融测试启动脚本 - 完整版本
# 在服务器上执行: bash run_ablation_full.sh

echo "=========================================="
echo "消融测试：完整方案对比"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/gs01/MagiCompiler-main:$PYTHONPATH

# 进入项目目录
cd /home/gs01/MagiCompiler-main

# 检查环境
echo ""
echo "[1] 环境检查"
echo "------------"
python3 --version
nvcc --version | head -3
echo "GPU数量: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"

# 运行完整消融测试
echo ""
echo "[2] 开始完整消融测试"
echo "-------------------"
echo "测试项:"
echo "  1. 原生vLLM推理（单GPU）"
echo "  2. PD分离并行（双GPU）"
echo "  3. CUDAGraph优化"
echo "  4. OrthoKDA固定KV缓存"
echo "  5. 完整栈优化"
echo ""

python3 ablation_test_full.py

echo ""
echo "=========================================="
echo "测试完成!"
echo "=========================================="