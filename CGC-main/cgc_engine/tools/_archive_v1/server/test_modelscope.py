#!/usr/bin/env python3
"""
Test ModelScope DeepSeek models availability
"""

import os
import sys

sys.path.insert(0, "/home/gs01/.local/lib/python3.10/site-packages")

from modelscope import snapshot_download

print("=" * 70)
print("  🔍 ModelScope 模型可用性測試")
print("=" * 70)

model_names = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3-Flash",
    "deepseek-ai/DeepSeek-V4",
    "deepseek-ai/DeepSeek-V4-Flash",
    "qwen/Qwen2.5-7B-Instruct",
]

for model in model_names:
    try:
        print(f"\n  正在檢查模型: {model}")
        cache_dir = "/home/gs01/modelscope_test"
        os.makedirs(cache_dir, exist_ok=True)
        
        model_dir = snapshot_download(
            model,
            cache_dir=cache_dir,
            revision='master',
            ignore_file_pattern=['*.bin', '*.safetensors', '*.pth', '*.pt'],
        )
        print(f"  ✅ 模型存在！目錄: {model_dir}")
        print(f"     找到模型，可以繼續下載完整權重")
        break
    except Exception as e:
        print(f"  ❌ 模型不可用: {str(e)[:150]}")

print("\n" + "=" * 70)
print("  測試完成")
print("=" * 70)
