#!/usr/bin/env python3
"""Download two models in parallel using threads."""
import requests, os, sys, time, threading

MODELS = [
    {
        "url": "https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf",
        "out": "/Users/alexchuang/Documents/flashkv0516/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf",
        "name": "Dense-teacher",
    },
    {
        "url": "https://hf-mirror.com/scima/Qwen3.8-Whittle-MoE-27B-A17.8B-GGUF-all-quants/resolve/main/Whittle-MoE-27B-A18B-v2.1-Q3_K_S.gguf",
        "out": "/Users/alexchuang/Documents/flashkv0516/models/gguf/Whittle-MoE-27B-A18B-v2.1-Q3_K_S.gguf",
        "name": "MoE-inference",
    },
]

def download(model):
    name = model["name"]
    url = model["url"]
    out = model["out"]
    
    for attempt in range(100):
        current = os.path.getsize(out) if os.path.exists(out) else 0
        print(f"[{name}] attempt {attempt+1}, resume from {current/1e6:.0f}MB", flush=True)
        
        headers = {"Range": f"bytes={current}-"} if current > 0 else {}
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=(10, 30))
            if current > 0 and r.status_code == 206:
                mode = "ab"
                total = int(r.headers.get("content-length", 0)) + current
            elif r.status_code == 200:
                mode = "wb"
                current = 0
                total = int(r.headers.get("content-length", 0))
            else:
                print(f"[{name}] HTTP {r.status_code}", flush=True)
                time.sleep(2)
                continue
            
            print(f"[{name}] total={total/1e9:.2f}GB mode={mode}", flush=True)
            t0 = time.time()
            
            with open(out, mode) as f:
                for chunk in r.iter_content(chunk_size=4*1024*1024):
                    f.write(chunk)
                    current += len(chunk)
                    now = time.time()
                    if now - t0 > 0 and (current % (200*1024*1024) < 4*1024*1024):
                        pct = current / total * 100 if total else 0
                        speed = current / (now - t0) / 1024 / 1024
                        print(f"[{name}] {current/1e9:.2f}/{total/1e9:.2f}GB ({pct:.0f}%) {speed:.0f}MB/s", flush=True)
            
            print(f"[{name}] DONE! {current/1e6:.0f}MB", flush=True)
            return True
        except Exception as e:
            print(f"[{name}] ERR: {type(e).__name__}", flush=True)
            time.sleep(2)
    
    print(f"[{name}] FAILED", flush=True)
    return False

threads = []
for m in MODELS:
    t = threading.Thread(target=download, args=(m,), daemon=False)
    t.start()
    threads.append(t)
    time.sleep(1)  # stagger starts

for t in threads:
    t.join()

print("ALL DONE", flush=True)
