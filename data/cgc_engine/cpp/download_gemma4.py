#!/usr/bin/env python3
"""Download gemma-4-26B-A4B-it-heretic-IQ4_XS.gguf using HF Hub Python API."""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

repo_id = "mradermacher/gemma-4-26B-A4B-it-heretic-GGUF"
filename = "gemma-4-26B-A4B-it-heretic-IQ4_XS.gguf"
local_dir = r"D:\alex\flashkv0516\models"

try:
    print(f"Downloading {filename} to {local_dir}...")
    local_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"Download complete! Path: {local_file_path}")
except Exception as e:
    print(f"Download failed: {e}")