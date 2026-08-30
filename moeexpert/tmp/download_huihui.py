#!/usr/bin/env python3
"""Download Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf with auto-retry."""
import requests, os, sys, time

URL = "https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
PROXY = "http://127.0.0.1:7897"
OUT = "models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
CHUNK = 2 * 1024 * 1024  # 2MB chunks
MAX_RETRIES = 99

proxies = {"https": PROXY, "http": PROXY}

for attempt in range(MAX_RETRIES):
    current = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    print(f"[attempt {attempt+1}] Resuming from {current/1e9:.2f} GB", flush=True)

    headers = {}
    mode = "wb"
    if current > 0:
        headers["Range"] = f"bytes={current}-"
    
    try:
        r = requests.get(URL, proxies=proxies, headers=headers, stream=True, timeout=60)
        if current > 0 and r.status_code == 206:
            mode = "ab"
            total = int(r.headers.get("content-length", 0)) + current
        else:
            mode = "wb"
            current = 0
            total = int(r.headers.get("content-length", 0))
        
        print(f"  Status: {r.status_code}, total: {total/1e9:.2f} GB", flush=True)
        start = time.time()
        last_print = 0
        
        with open(OUT, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                current += len(chunk)
                elapsed = time.time() - start
                if current - last_print >= 100 * 1024 * 1024:  # print every 100MB
                    last_print = current
                    pct = current / total * 100 if total else 0
                    speed = (current - (current - len(chunk) * max(1, current // CHUNK))) / max(elapsed, 0.01)
                    print(f"  {current/1e9:.2f}/{total/1e9:.2f} GB ({pct:.1f}%) {elapsed:.0f}s", flush=True)
        
        print(f"DONE: {current/1e9:.2f} GB in {time.time()-start:.0f}s")
        sys.exit(0)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        time.sleep(3)
        continue

print("FAILED after max retries")
sys.exit(1)
