#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/gs01')

from modelscope import snapshot_download
import os

cache_dir = '/home/gs01/models'
model_id = 'Qwen/Qwen2.5-7B-Instruct'

print(f"Downloading {model_id} from ModelScope...")
try:
    model_dir = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        revision='master'
    )
    print(f"SUCCESS! Model downloaded to: {model_dir}")

    # List downloaded files
    print("\nDownloaded files:")
    for f in os.listdir(model_dir):
        size = os.path.getsize(os.path.join(model_dir, f))
        print(f"  {f}: {size/1024/1024:.2f} MB")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
