#!/usr/bin/env python3
"""
Test ModelScope DeepSeek V4 Flash Download
"""

import os
import sys
import time

print("=" * 70)
print("  🔥 Testing ModelScope DeepSeek V4 Flash Download")
print("=" * 70)

from modelscope import snapshot_download

model_names = [
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V3-Flash",
    "deepseek-ai/DeepSeek-V2.5",
    "qwen/Qwen2.5-7B-Instruct"
]

for model_name in model_names:
    print(f"\nTrying: {model_name}")
    try:
        print("  Downloading from ModelScope...")
        model_dir = snapshot_download(
            model_name,
            cache_dir="/home/gs01/modelscope",
            revision='master'
        )
        print(f"  ✅ Success! Path: {model_dir}")
        print(f"  Files: {os.listdir(model_dir)[:10]}")
        break
    except Exception as e:
        print(f"  ❌ Failed: {type(e).__name__}: {str(e)[:100]}")

print("\nDone.")
