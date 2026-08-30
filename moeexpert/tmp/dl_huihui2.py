#!/usr/bin/env python3
import requests, os, sys, time

URL = "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
OUT = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
CHUNK = 4 * 1024 * 1024

for attempt in range(100):
    current = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    print(f"[{attempt+1}] resume from {current/1e6:.0f} MB", flush=True)
    
    headers = {"Range": f"bytes={current}-"} if current > 0 else {}
    try:
        r = requests.get(URL, headers=headers, stream=True, timeout=30)
        if current > 0 and r.status_code == 206:
            total = int(r.headers.get("content-length", 0)) + current
            mode = "ab"
        else:
            total = int(r.headers.get("content-length", 0))
            current = 0
            mode = "wb"
        
        print(f"  status={r.status_code} total={total/1e9:.2f}GB", flush=True)
        t0 = time.time()
        last_t = t0
        
        with open(OUT, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                current += len(chunk)
                now = time.time()
                if now - last_t >= 30:
                    last_t = now
                    pct = current / total * 100 if total else 0
                    print(f"  {current/1e9:.2f}/{total/1e9:.2f} GB ({pct:.0f}%) {now-t0:.0f}s", flush=True)
        
        print(f"DONE {current/1e6:.0f} MB", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"  ERR: {e}", flush=True)
        time.sleep(2)
