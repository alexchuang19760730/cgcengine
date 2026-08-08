#!/usr/bin/env python3
"""
從 ModelScope 下載 DeepSeek-V4-Flash
"""

import os
import sys

sys.path.insert(0, "/home/gs01/.local/lib/python3.10/site-packages")

from modelscope import snapshot_download

print("=" * 70)
print("  🚀 從 ModelScope 下載 DeepSeek-V4-Flash")
print("=" * 70)

model_name = "deepseek-ai/DeepSeek-V4-Flash"
local_dir = "/home/gs01/models/DeepSeek-V4-Flash"

os.makedirs(local_dir, exist_ok=True)

print(f"\n  模型: {model_name}")
print(f"  輸出: {local_dir}")
print("\n  開始下載... (可能需要 10-30 分鐘)")

model_dir = snapshot_download(
    model_name,
    local_dir=local_dir,
    revision="master",
)

print(f"\n  ✅ 下載完成！模型位於: {model_dir}")
print("\n  模型內容:")
os.system(f"ls -lh {model_dir} | head -30")
