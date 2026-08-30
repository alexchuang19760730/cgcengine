#!/usr/bin/env python3
"""Download GGUF with keep-alive + small chunks for unstable proxy."""
import requests
import os
import time
import sys

URL = "https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
OUTPUT = "models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
PROXIES = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
TOTAL = 11950000000
MAX_RETRIES = 100

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Connection": "keep-alive",
})

for attempt in range(MAX_RETRIES):
    existing = os.path.getsize(OUTPUT) if os.path.exists(OUTPUT) else 0
    if existing >= TOTAL:
        print(f"\nDone! {existing/1e9:.2f}GB")
        sys.exit(0)
    
    pct = existing / TOTAL * 100
    print(f"[{attempt+1}] Resume from {existing/1e9:.2f}GB ({pct:.1f}%)")
    sys.stdout.flush()
    
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
    try:
        r = session.get(URL, headers=headers, proxies=PROXIES, stream=True, timeout=120)
        mode = "ab" if existing > 0 and r.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        dl_total = int(r.headers.get("content-length", 0)) + existing
        with open(OUTPUT, mode) as f:
            for chunk in r.iter_content(chunk_size=256*1024):  # 256KB chunks
                if chunk:
                    f.write(chunk)
                    existing += len(chunk)
                    if existing % (100*1024*1024) < 256*1024:  # print every 100MB
                        print(f"  {existing/1e9:.2f}GB ({existing/TOTAL*100:.1f}%)")
                        sys.stdout.flush()
        print(f"[{attempt+1}] Done: {existing/1e9:.2f}GB")
    except Exception as e:
        print(f"[{attempt+1}] Failed: {e}")
        time.sleep(3)

print("Failed after all retries")
sys.exit(1)
