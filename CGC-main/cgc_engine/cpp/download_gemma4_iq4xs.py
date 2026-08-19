#!/usr/bin/env python3
"""Download gemma-4-26B-A4B-it-heretic-IQ4_XS.gguf using HF Hub Python API.

14.2 GB file - may take a while.
"""

import os
import sys

# Set mirror for faster download in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

repo_id = "mradermacher/gemma-4-26B-A4B-it-heretic-GGUF"
filename = "gemma-4-26B-A4B-it-heretic-IQ4_XS.gguf"
local_dir = r"D:\alex\flashkv0516\models\gemma4_gguf"

try:
    print(f"Downloading {filename}")
    print(f"  From: {repo_id}")
    print(f"  To: {local_dir}")
    print(f"  Mirror: {os.environ['HF_ENDPOINT']}")
    print(f"  Expected size: ~14.2 GB")
    print()
    
    local_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"\n✅ Download complete!")
    print(f"   Path: {local_file_path}")
    print(f"   Size: {os.path.getsize(local_file_path):,} bytes")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
