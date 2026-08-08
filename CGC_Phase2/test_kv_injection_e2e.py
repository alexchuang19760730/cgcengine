#!/usr/bin/env python3
"""KV 注入端到端测试: Mac MLX forward P=8 → emit hidden+KV → cloud resume → 检查注入。

直接调 edge_first_proxy._layer_split_stream, 绕过路由逻辑, 固定 P=8。
前置: SSH 隧道 31010 + 30001 已建, cloud sglang resume 模式已启动。
"""
import os, sys, time

# 环境变量必须在 import edge_first_proxy 之前设置
os.environ.setdefault("EDGE_LOCAL_MODEL_PATH", "/Users/alexchuang/models/Qwen3-VL-2B-bf16")
os.environ.setdefault("EDGE_LOCAL_NUM_LAYERS", "28")
os.environ.setdefault("EDGE_LOCAL_KV_HEADS", "8")
os.environ.setdefault("EDGE_LOCAL_KV_HEAD_DIM", "128")
os.environ.setdefault("CGC_MAC_EMIT_CLOUD_HOST", "127.0.0.1")
os.environ.setdefault("CGC_MAC_EMIT_PORT", "31010")
os.environ.setdefault("CGC_KV_DIAG", "1")

REPO = "/Users/alexchuang/Documents/flashkv0516"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from app.servers.edge_first_proxy import _layer_split_stream

P = 8
body = {
    "model": "Qwen3-VL-2B-Instruct",
    "messages": [{"role": "user", "content": "France is"}],
    "max_tokens": 5,
    "stream": True,
}
cloud_endpoint = "http://127.0.0.1:30001/v1/chat/completions"
cloud_payload = dict(body)
headers = {"Content-Type": "application/json"}

print(f"[test] P={P} model=Qwen3-VL-2B-BF16 prompt='France is'", flush=True)
print(f"[test] cloud_endpoint={cloud_endpoint}", flush=True)
print(f"[test] mac_emit → 127.0.0.1:31010 (tunnel→Host2)", flush=True)
print(f"[test] CGC_KV_DIAG=1 (cloud will log kv_layers type/len)", flush=True)
print("-" * 60, flush=True)

t0 = time.time()
chunks = []
try:
    for chunk in _layer_split_stream(body, P, cloud_endpoint, cloud_payload, headers):
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        chunks.append(chunk)
        print(chunk, end="", flush=True)
except Exception as e:
    print(f"\n[test] ERROR: {e!r}", flush=True)
    import traceback
    traceback.print_exc()

elapsed = time.time() - t0
print(f"\n{'-' * 60}")
print(f"[test] done in {elapsed:.1f}s, {len(chunks)} chunks")
print(f"[test] 检查 cloud 日志: grep 'CGC_KV_INJECT\\|CGC_KV_DIAG\\|CGC_RESUME.*kv_layers' /tmp/sglang_resume_2b.log")
