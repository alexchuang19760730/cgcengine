#!/usr/bin/env python3
"""Download with requests - direct, no fork."""
import requests, os, sys, time

URL = "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
OUT = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
LOG = "/tmp/huihui_progress.log"

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        f.flush()

for attempt in range(100):
    current = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    log(f"attempt={attempt+1} resume_from={current/1e6:.0f}MB")
    
    headers = {"Range": f"bytes={current}-"} if current > 0 else {}
    try:
        r = requests.get(URL, headers=headers, stream=True, timeout=60)
        if current > 0 and r.status_code == 206:
            total = int(r.headers.get("content-length", 0)) + current
            mode = "ab"
        else:
            total = int(r.headers.get("content-length", 0))
            current = 0
            mode = "wb"
        
        log(f"status={r.status_code} total={total/1e9:.2f}GB")
        t0 = time.time()
        
        with open(OUT, mode) as f:
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                f.write(chunk)
                current += len(chunk)
                now = time.time()
                if now - t0 > 0 and (now - t0) % 30 < 4:
                    pct = current / total * 100 if total else 0
                    log(f"progress {current/1e9:.2f}/{total/1e9:.2f}GB ({pct:.0f}%) {now-t0:.0f}s")
                    time.sleep(0.1)  # avoid duplicate prints
        
        log(f"DONE {current/1e6:.0f}MB")
        print(f"DONE {current/1e6:.0f}MB", flush=True)
        sys.exit(0)
    except Exception as e:
        log(f"ERR: {e}")
        time.sleep(3)

log("FAILED")
sys.exit(1)
