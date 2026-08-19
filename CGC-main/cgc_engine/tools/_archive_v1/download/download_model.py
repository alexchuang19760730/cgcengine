#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/gs01')

from modelscope import snapshot_download
import os

model_dir = '/home/gs01/models/Qwen2.5-7B-Instruct'
os.makedirs(model_dir, exist_ok=True)

print(f"Downloading Qwen2.5-7B-Instruct from ModelScope to {model_dir}...")
model_dir = snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct',
    cache_dir='/home/gs01/models',
    revision='master'
)
print(f"Model downloaded to: {model_dir}")
print(f"Files: {os.listdir(model_dir)[:10]}...")
