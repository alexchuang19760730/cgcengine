#!/usr/bin/env python3
"""
從 ModelScope 下載 DeepSeek-V4-Flash 模型 - 直接模式
下載位置: /mnt/data/gs01_models/DeepSeek-V4-Flash
"""
import os
import sys

print("=" * 70)
print("  🔥 開始下載 DeepSeek-V4-Flash")
print("=" * 70)

try:
    from modelscope import snapshot_download
    print("  ✅ ModelScope 載入成功")
except Exception as e:
    print(f"  ❌ ModelScope 載入失敗: {e}")
    print("  重新安裝 ModelScope...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
    from modelscope import snapshot_download

print("  開始下載中...")
model_dir = snapshot_download(
    "deepseek-ai/DeepSeek-V4-Flash",
    local_dir="/mnt/data/gs01_models/DeepSeek-V4-Flash",
)
print(f"✅ 模型下載完成: {model_dir}")
total_size = sum(
    os.path.getsize(os.path.join(root, f))
    for root, dirs, files in os.walk(model_dir)
    for f in files
) / (1024 ** 3)
print(f"模型總大小: {total_size:.2f} GB")
