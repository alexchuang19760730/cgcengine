#!/usr/bin/env python3
"""layer-split PD TTFT 时间分解: 端侧P层 + 传输 + 云端resume + 首 token。"""
import os, sys, time, json, urllib.request

os.environ.setdefault("EDGE_LOCAL_MODEL_PATH", "/Users/alexchuang/models/Qwen3-VL-2B-bf16")
os.environ.setdefault("EDGE_LOCAL_NUM_LAYERS", "28")
os.environ.setdefault("EDGE_LOCAL_KV_HEADS", "8")
os.environ.setdefault("EDGE_LOCAL_KV_HEAD_DIM", "128")
os.environ.setdefault("CGC_MAC_EMIT_CLOUD_HOST", "127.0.0.1")
os.environ.setdefault("CGC_MAC_EMIT_PORT", "31010")

REPO = "/Users/alexchuang/Documents/flashkv0516"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from app.servers.edge_first_proxy import (
    _load_local_mlx_first_p_layers, _mlx_forward_first_p_layers,
    _get_mac_emit_transport, _cloud_stream,
)

P = 8
messages = [{"role": "user", "content": "France is"}]
cloud_endpoint = "http://127.0.0.1:30001/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# === t0: 开始 ===
t0 = time.time()

# === 端侧: MLX forward P 层 ===
model_tuple = _load_local_mlx_first_p_layers(P)
t_load = time.time()
print(f"[t] 模型加载: {t_load-t0:.3f}s", flush=True)

hidden_P, residual_P, input_ids, seq_len, kv_layers = _mlx_forward_first_p_layers(
    model_tuple[0], model_tuple[1], messages, P
)
t_mac_fwd = time.time()
print(f"[t] 端侧 forward P={P} 层: {t_mac_fwd-t_load:.3f}s (seq={seq_len}, KV={sum(1 for x in kv_layers if x is not None)}/{len(kv_layers)})", flush=True)

# === 端侧: emit (序列化 + SSH 传输) ===
transport = _get_mac_emit_transport()
import hashlib
request_id = hashlib.sha1(f"{time.time()}.{seq_len}.{P}".encode()).hexdigest()[:12]
payload = {
    "finished_layer": P,
    "hidden_states": hidden_P,
    "residual": residual_P,
    "step": 0,
    "request_id": request_id,
    "input_ids": input_ids,
    "seq_len": seq_len,
    "kv_layers": kv_layers,
}
transport.send(0, 0, payload)
t_emit = time.time()
print(f"[t] emit (序列化+SSH传输 ~{521328//1024}KB): {t_emit-t_mac_fwd:.3f}s", flush=True)

# === 云端: POST /v1/chat/completions (触发 resume forward) ===
cloud_payload = {
    "model": "Qwen3-VL-2B-Instruct",
    "messages": messages,
    "max_tokens": 5,
    "stream": True,
}
t_post = time.time()

# === 首 token 到达 ===
first_token_time = None
first_token_content = ""
chunks = []
for chunk in _cloud_stream(cloud_endpoint, cloud_payload, headers):
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="replace")
    chunks.append(chunk)
    if first_token_time is None and "content" in chunk and chunk.strip():
        # 解析首 token
        for line in chunk.split("\n"):
            if line.startswith("data: ") and "content" in line:
                try:
                    d = json.loads(line[6:])
                    c = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if c:
                        first_token_time = time.time()
                        first_token_content = c
                        break
                except:
                    pass
    if first_token_time:
        break  # 首 token 到达, 不需要等全部

if first_token_time is None:
    first_token_time = time.time()

t_end = time.time()

print(f"[t] cloud POST→首 token: {first_token_time-t_post:.3f}s (含 SSH 传输 + cloud recv + KV注入 + resume forward + 首 decode)", flush=True)
print(f"[t] 首 token: '{first_token_content}'", flush=True)
print(f"[t] === 总 TTFT: {first_token_time-t0:.3f}s ===", flush=True)
print(f"[t] 分解: 端侧P层={t_mac_fwd-t_load:.3f}s + emit={t_emit-t_mac_fwd:.3f}s + cloud={first_token_time-t_post:.3f}s + 模型加载={t_load-t0:.3f}s", flush=True)
