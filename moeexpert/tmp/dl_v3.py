#!/usr/bin/env python3
"""Download using urllib3 with explicit short timeouts."""
import urllib3, os, sys, time

URL = "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
OUT = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"

# Remove partial file
if os.path.exists(OUT):
    os.remove(OUT)

http = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=10, read=15),
    retries=urllib3.Retry(total=99, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
)

total = None
t0 = time.time()
downloaded = 0

for attempt in range(100):
    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
    
    try:
        r = http.request("GET", URL, headers=headers, preload_content=False)
        
        if downloaded > 0 and r.status == 206:
            content_length = int(r.headers.get("content-length", 0))
            total = content_length + downloaded
            mode = "ab"
        elif r.status == 200:
            content_length = int(r.headers.get("content-length", 0))
            total = content_length
            downloaded = 0
            mode = "wb"
        else:
            print(f"[{attempt+1}] HTTP {r.status}", flush=True)
            time.sleep(2)
            continue
        
        print(f"[{attempt+1}] {downloaded/1e9:.2f}/{total/1e9:.2f}GB", flush=True)
        t0 = time.time()
        
        with open(OUT, mode) as f:
            while True:
                chunk = r.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                now = time.time()
                elapsed = now - t0
                if elapsed > 0 and downloaded % (100*1024*1024) < 4*1024*1024:
                    pct = downloaded / total * 100 if total else 0
                    speed = downloaded / elapsed / 1024 / 1024
                    print(f"  {downloaded/1e9:.2f}/{total/1e9:.2f}GB ({pct:.0f}%) {speed:.0f}MB/s", flush=True)
        
        print(f"DONE! {downloaded/1e6:.0f}MB", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[{attempt+1}] ERR: {type(e).__name__}", flush=True)
        time.sleep(1)
        r.close() if 'r' in dir() else None

print("FAILED", flush=True)
sys.exit(1)
