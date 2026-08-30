#!/usr/bin/env python3
"""Download with aggressive retry."""
import requests, os, sys, time

URL = "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
OUT = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

for attempt in range(200):
    current = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    print(f"[{attempt+1}] {current/1e9:.2f}GB", flush=True)
    
    headers = {}
    if current > 0:
        headers["Range"] = f"bytes={current}-"
    
    try:
        r = session.get(URL, headers=headers, stream=True, timeout=(10, 30))
        
        if current > 0 and r.status_code == 206:
            mode = "ab"
            total = int(r.headers.get("content-length", 0)) + current
        elif r.status_code == 200:
            mode = "wb"
            current = 0
            total = int(r.headers.get("content-length", 0))
        else:
            print(f"  HTTP {r.status_code}, retrying...", flush=True)
            time.sleep(2)
            continue
        
        print(f"  total={total/1e9:.2f}GB mode={mode}", flush=True)
        t0 = time.time()
        last_print = t0
        chunk_count = 0
        
        with open(OUT, mode) as f:
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                f.write(chunk)
                current += len(chunk)
                chunk_count += 1
                now = time.time()
                
                # Print every 200MB or every 60 seconds
                if current % (200*1024*1024) < 4*1024*1024 or (now - last_print) > 60:
                    last_print = now
                    pct = current / total * 100 if total else 0
                    speed = current / (now - t0) / 1024 / 1024 if now > t0 else 0
                    print(f"  {current/1e9:.2f}/{total/1e9:.2f}GB ({pct:.0f}%) {speed:.1f}MB/s", flush=True)
        
        print(f"DONE! {current/1e6:.0f}MB", flush=True)
        sys.exit(0)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, 
            requests.exceptions.ChunkedEncodingError, ConnectionResetError) as e:
        print(f"  ERR: {type(e).__name__}: {e}", flush=True)
        time.sleep(1)
    except Exception as e:
        print(f"  ERR: {e}", flush=True)
        time.sleep(2)

print("FAILED after 200 attempts")
sys.exit(1)
