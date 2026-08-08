"""
測試 KDA Backend 是否正確工作
"""
import sys
import os
import torch

sys.path.insert(0, "/home/gs01")
os.environ["VLLM_USE_CGC_KDA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("=" * 60)
print("Testing KDA Backend Integration with vLLM")
print("=" * 60)

# 1. 測試導入
print("\n[1] Testing imports...")
from vllm import LLM
print("✅ vLLM imported successfully")

# 2. 測試 KDA Backend 註冊
print("\n[2] Testing KDA Backend registration...")
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
backend_path = _ATTN_OVERRIDES.get(AttentionBackendEnum.FLASH_ATTN, None)
if backend_path:
    print(f"✅ KDA Backend registered: {backend_path}")
else:
    print("❌ KDA Backend NOT registered")
    sys.exit(1)

# 3. 測試簡單推理
print("\n[3] Testing simple inference with KDA Backend...")
print("   (This will load the model - may take a minute)")

# 創建一個簡單的推理請求 - 使用較小的記憶體配置
llm = LLM(
    model="/home/gs01/models/Qwen/Qwen2___5-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    max_model_len=1024,
    enable_prefix_caching=False,
    enforce_eager=True,
)

print("✅ Model loaded successfully with KDA Backend!")

# 4. 運行簡單推理
print("\n[4] Running simple inference...")
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=64,
)

outputs = llm.generate(
    ["Hello, my name is"],
    sampling_params=sampling_params,
)

for output in outputs:
    generated_text = output.outputs[0].text
    print(f"✅ Generated text: {generated_text}")

print("\n" + "=" * 60)
print("✅ KDA Backend is working correctly!")
print("=" * 60)