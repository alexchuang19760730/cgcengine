#!/usr/bin/env python3
"""Download gemma-4-26B-A4B-it-heretic-IQ4_XS.gguf using huggingface-cli."""

import os
import subprocess
import sys

# Set mirror for faster download in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

repo_id = "mradermacher/gemma-4-26B-A4B-it-heretic-GGUF"
filename = "gemma-4-26B-A4B-it-heretic-IQ4_XS.gguf"
local_dir = r"D:\alex\flashkv0516\models\gemma4_gguf"

# Try using huggingface-cli directly
cmd = [
    sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
    "download", repo_id,
    "--include", filename,
    "--local-dir", local_dir
]

print(f"Running: {' '.join(cmd)}")
print(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
print(f"Expected size: ~14.2 GB")
print()

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print("Return code:", result.returncode)
except subprocess.TimeoutExpired:
    print("Download timed out after 10 minutes (still running in background)")
    print("Check the huggingface-cli process for progress.")
except Exception as e:
    print(f"Error: {e}")
