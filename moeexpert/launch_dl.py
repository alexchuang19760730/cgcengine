#!/usr/bin/env python3
"""Launch curl download in a detached subprocess."""
import subprocess, os

OUT = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
URL = "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"

# Remove old partial
if os.path.exists(OUT):
    os.remove(OUT)

# Fork a detached process
pid = os.fork()
if pid == 0:
    # Child: redirect stdout/stderr to log
    with open("/tmp/huihui_curl.log", "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    os.execvp("curl", [
        "curl", "-fSL", "--retry", "99", "--retry-delay", "3",
        "--connect-timeout", "15", "--max-time", "0",
        "-o", OUT, URL
    ])
else:
    print(f"Download child PID={pid}")
