#!/usr/bin/env python3
"""Truly detach a download process."""
import os, sys

URL = "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
OUT = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
SCRIPT = "/Users/alexchuang/Documents/flashkv0516/moeexpert/dl_both.py"

# First fork
pid = os.fork()
if pid > 0:
    print(f"Parent: child pid={pid}")
    sys.exit(0)

# Child: create new session
os.setsid()

# Second fork to fully detach
pid2 = os.fork()
if pid2 > 0:
    os._exit(0)

# Grandchild: redirect stdio
with open("/tmp/huihui_dl2.log", "w") as f:
    os.dup2(f.fileno(), 0)
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)

os.execvp(sys.executable, [sys.executable, SCRIPT])
