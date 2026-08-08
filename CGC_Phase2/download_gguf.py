#!/usr/bin/env python3
"""Download GGUF model from HuggingFace."""
import os
import sys

# Enable hf_transfer for faster downloads
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

repo_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
filename = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5-1.5b-instruct-fp16.gguf"
local_dir = sys.argv[3] if len(sys.argv) > 3 else "/Users/alexchuang/models/gguf/"

print(f"Downloading {filename} from {repo_id}...", flush=True)

from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
)

print(f"Done: {path}", flush=True)
print(f"Size: {os.path.getsize(path) / 1e9:.2f} GB", flush=True)
