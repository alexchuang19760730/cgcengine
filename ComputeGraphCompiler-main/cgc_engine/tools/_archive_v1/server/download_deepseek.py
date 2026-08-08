#!/usr/bin/env python3
"""
從 ModelScope 下載 DeepSeek-V4-Flash 模型
下載位置: /mnt/data/gs01_models/DeepSeek-V4-Flash
"""

import os
import sys
from modelscope import snapshot_download

model_dir = snapshot_download(
    "deepseek-ai/DeepSeek-V4-Flash",
    local_dir="/mnt/data/gs01_models/DeepSeek-V4-Flash",
)
print(f"✅ 模型下載完成: {model_dir}")
print(f"模型大小: {sum(os.path.getsize(os.path.join(root, f)) for root, dirs, files in os.walk(model_dir) for f in files) / 1024**3:.2f} GB")
