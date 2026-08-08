#!/usr/bin/env python3
"""Mac MLX decode forward: 用 KV cache 做 1-token decode, 只 forward 前 P 层。

这是 layer-split decode 的 Mac 端核心: cloud prefill → Mac decode P 层 → emit hidden_P → cloud。
"""
import os, sys, time
import torch

os.environ.setdefault("EDGE_LOCAL_MODEL_PATH", "/Users/alexchuang/models/Qwen3-VL-2B-bf16")
os.environ.setdefault("EDGE_LOCAL_NUM_LAYERS", "28")
os.environ.setdefault("EDGE_LOCAL_KV_HEADS", "8")
os.environ.setdefault("EDGE_LOCAL_KV_HEAD_DIM", "128")

REPO = "/Users/alexchuang/Documents/flashkv0516"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask


def mac_mlx_decode_forward(model, tokenizer, messages, P=6, max_tokens=10):
    """Mac MLX decode: prefill 全部层 → decode P 层/token → 输出 hidden_P 序列。

    返回: list of (hidden_P_tensor, token_text) per decode step。
    hidden_P 是第 P 层输出 (torch tensor, 供 emit 给 cloud)。
    """
    import mlx.nn as nn

    # === Prefill (Mac 本地, 全部层) ===
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # 对齐 sglang: 过滤 special tokens
    input_ids = [t for t in input_ids if t not in (151644, 151645)]

    # 获取模型结构
    lang = getattr(model, "language_model", None)
    inner = getattr(lang, "model", lang) if lang else getattr(model, "model", model)
    embed = inner.embed_tokens
    layers = inner.layers
    norm = getattr(inner, "norm", None)
    lm_head = getattr(model, "lm_head", getattr(inner, "lm_head", None))

    print(f"[mac-decode] model: {type(model).__name__}, layers={len(layers)}, P={P}", flush=True)
    print(f"[mac-decode] input_ids len={len(input_ids)} (filtered)", flush=True)

    # Prefill forward (全部层, 初始化 KV cache)
    t0 = time.monotonic()
    from mlx_lm.models.base import KVCache
    kv_cache = [KVCache() for _ in range(len(layers))]

    h = embed(mx.array([input_ids]))
    mask = create_attention_mask(h, None)

    for i, layer in enumerate(layers):
        h = layer(h, mask, cache=kv_cache[i])

    # Norm + lm_head → 首 token
    if norm is not None:
        h = norm(h)
    logits = lm_head(h) if lm_head else h @ embed.weight.T
    first_token = int(mx.argmax(logits[:, -1, :]).item())
    first_text = tokenizer.decode([first_token])

    t_prefill = time.monotonic() - t0
    print(f"[mac-decode] prefill: {t_prefill*1000:.0f}ms, first_token='{first_text}' ({first_token})", flush=True)

    # === Decode loop (每 token: forward P 层 → hidden_P → 全部层 → token) ===
    results = [(None, first_text)]
    current_token = first_token

    for step in range(max_tokens - 1):
        t_step = time.monotonic()

        # Embed new token
        h = embed(mx.array([[current_token]]))

        # Forward P 层 (用 KV cache)
        for i in range(min(P, len(layers))):
            h = layers[i](h, None, cache=kv_cache[i])

        # hidden_P = 第 P 层输出 (转 torch for emit)
        hidden_P = torch.from_numpy(np.array(h)).clone()

        # 继续 forward 剩余层 (Mac 本地完成, 获取 token)
        for i in range(P, len(layers)):
            h = layers[i](h, None, cache=kv_cache[i])

        if norm is not None:
            h = norm(h)
        logits = lm_head(h) if lm_head else h @ embed.weight.T
        next_token = int(mx.argmax(logits[:, -1, :]).item())
        next_text = tokenizer.decode([next_token])

        t_token = time.monotonic() - t_step
        results.append((hidden_P, next_text))
        current_token = next_token

        print(f"  step {step}: token='{next_text}' ({t_token*1000:.0f}ms) "
              f"hidden_P shape={tuple(hidden_P.shape)}", flush=True)

    t_total = time.monotonic() - t0
    print(f"\n[mac-decode] done: {len(results)} tokens in {t_total:.1f}s "
          f"= {len(results)/t_total:.1f} tok/s", flush=True)
    return results


if __name__ == "__main__":
    import numpy as np

    print("=== Mac MLX Decode Forward Test ===")
    model_path = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"
    print(f"Loading {model_path}...")
    model, tokenizer = load(model_path)

    messages = [{"role": "user", "content": "Write a short story about a cat"}]
    results = mac_mlx_decode_forward(model, tokenizer, messages, P=6, max_tokens=10)

    print(f"\nOutput: {' '.join(r[1] for r in results)}")
