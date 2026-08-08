#!/usr/bin/env python3
"""
DeepSeek-V4-Flash 完全乾淨下載
完全繞開默認 ~/models 路徑，直接下載到 13T 數據盤
"""
import os
import sys

os.environ["MODELSCOPE_CACHE"] = "/mnt/data/gs01_models/.modelscope_cache"

from modelscope import snapshot_download

print("=" * 70)
print("  🔥 DeepSeek-V4-Flash 完全乾淨下載")
print("=" * 70)
print(f"  MODELSCOPE_CACHE: {os.environ['MODELSCOPE_CACHE']}")

model_dir = snapshot_download(
    "deepseek-ai/DeepSeek-V4-Flash",
    local_dir="/mnt/data/gs01_models/DeepSeek-V4-Flash",
)

print(f"\n✅ 模型下載完成！")
print(f"  模型路徑: {model_dir}")

total_size = 0
count = 0
for root, dirs, files in os.walk(model_dir):
    for f in files:
        fp = os.path.join(root, f)
        total_size += os.path.getsize(fp)
        count += 1

print(f"  文件數量: {count}")
print(f"  總大小: {total_size / (1024**3):.2f} GB")
