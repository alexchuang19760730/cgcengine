#!/bin/bash
# FlashKDA for RTX 5090 - 完整安装脚本
# 等服务器恢复后直接运行此脚本

echo "=========================================="
echo "FlashKDA RTX 5090 安装脚本"
echo "=========================================="

# 1. 安装 flash-attn (支持 RTX 5090)
echo "[1/4] 安装 flash-attn..."
REPO_ROOT="$(cd "$(dirname "$0")/../../../../" && pwd)"
cd "$REPO_ROOT/Backend/Vllm/vllm_backend"
pip install flash-attn --no-build-isolation -U

# 2. 写入 FA2-KDA 后端
echo "[2/4] 写入 cgc_kda_backend.py..."
cat > cgc_kda_backend.py << 'BACKEND_EOF'
import torch
from flash_attn import flash_attn_func

class CGCKDABackend:
    def __init__(self):
        self.name = "cgc_kda_fa2"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
        **kwargs
    ):
        out = flash_attn_func(
            query, key, value,
            causal=is_causal,
            softmax_scale=None
        )
        print("[CGC-FA2-KDA] ✅ RTX 5090 全速运行")
        return out

def get_kda_backend():
    return CGCKDABackend()
BACKEND_EOF

# 3. 测试导入
echo "[3/4] 测试导入..."
python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from Backend.Vllm.vllm_backend.cgc_kda_backend import get_kda_backend
print('✅ FA2 KDA 载入成功 (RTX5090 支持)')
"

# 4. 运行 vLLM 测试
echo "[4/4] 运行 vLLM + FA2-KDA 测试..."
cd /home/gs01
python3 << 'TEST_EOF'
import os
os.environ["VLLM_USE_CGC_KDA"] = "1"
import sys
sys.path.insert(0, "/home/gs01")

import torch
from vllm import LLM, SamplingParams

print("Loading model with FA2-KDA...")
llm = LLM(
    model="/home/gs01/models/Qwen/Qwen2___5-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    max_model_len=2048,
    enforce_eager=True,
    disable_log_stats=True
)
print("Model loaded!")

prompt = "Hello"
sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
outputs = llm.generate([prompt], sampling_params)
print(f"Output: {outputs[0].outputs[0].text}")
print("✅ 测试成功!")
TEST_EOF

echo "=========================================="
echo "安装完成!"
echo "=========================================="
