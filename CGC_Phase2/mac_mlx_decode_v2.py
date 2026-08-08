#!/usr/bin/env python3
"""Mac MLX decode forward v2: 用 mlx_lm stream_generate + monkey-patch 捕获 hidden_P。

stream_generate 内部管理 KV cache (prompt_cache), 每次 decode 自动复用。
monkey-patch layers[P-1].__call__ 捕获第 P 层输出 (hidden_P), 供 emit 给 cloud。
"""
import os, sys, time
import torch
import numpy as np

os.environ.setdefault("EDGE_LOCAL_MODEL_PATH", "/Users/alexchuang/models/Qwen3-VL-2B-bf16")

REPO = "/Users/alexchuang/Documents/flashkv0516"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mlx.core as mx
from mlx_lm import load, stream_generate


def mac_mlx_decode_with_hidden(model, tokenizer, messages, P=6, max_tokens=10):
    """Mac MLX decode: prefill + decode, 每步输出 (hidden_P, token)。

    用 stream_generate 管理 KV cache, monkey-patch layers[P-1] 捕获 hidden_P。
    """
    # 找到 layers 列表
    lang = getattr(model, "language_model", None)
    inner = getattr(lang, "model", lang) if lang else getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError(f"无法找到 layers: model type={type(model).__name__}")

    print(f"[mac-decode] model={type(model).__name__}, layers={len(layers)}, P={P}", flush=True)

    # Monkey-patch: 用 wrapper 替换 layers[P-1], 捕获 hidden_P
    # (不能 patch forward/__call__ — MLX Module 的 dunder 在类上查找)
    captured_hidden = [None]
    target_idx = min(P - 1, len(layers) - 1)
    _orig_layer = layers[target_idx]

    class _LayerCaptureWrapper:
        """透明 wrapper: 调原始 layer, 捕获输出 (存 MLX array, 延迟转 torch)。"""
        def __init__(self, layer):
            self._layer = layer
        def __call__(self, *args, **kwargs):
            result = self._layer(*args, **kwargs)
            hs = result[0] if isinstance(result, (tuple, list)) else result
            if hs is not None and hasattr(hs, 'shape'):
                # 存 MLX array (eval 确保 computed), 延迟转 torch
                mx.eval(hs)
                captured_hidden[0] = hs
            return result
        def __getattr__(self, name):
            return getattr(self._layer, name)

    layers[target_idx] = _LayerCaptureWrapper(_orig_layer)
    print(f"[mac-decode] wrapped layer[{target_idx}] for hidden_P capture", flush=True)

    # 准备 prompt
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 过滤 special tokens (对齐 sglang)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [t for t in input_ids if t not in (151644, 151645)]
    print(f"[mac-decode] prompt tokens: {len(input_ids)} (filtered)", flush=True)

    # stream_generate (内部管理 KV cache)
    results = []
    t0 = time.monotonic()

    for n, response in enumerate(stream_generate(
        model, tokenizer, mx.array(input_ids), max_tokens=max_tokens
    )):
        token_text = response.text
        token_id = response.token
        hidden_P = captured_hidden[0]

        t_token = time.monotonic() - t0
        if n == 0:
            print(f"[mac-decode] prefill+first token: {t_token*1000:.0f}ms, "
                  f"token='{token_text}' ({token_id})", flush=True)
        else:
            dt = t_token - sum(r[2] for r in results)
            print(f"  step {n}: token='{token_text}' ({token_id}) "
                  f"dt={dt*1000:.0f}ms hidden_P={'captured' if hidden_P is not None else 'None'}",
                  flush=True)

        results.append((hidden_P, token_text, t_token))
        captured_hidden[0] = None  # 重置, 等下次 capture

    # 恢复原 layer
    layers[target_idx] = _orig_layer

    t_total = time.monotonic() - t0
    n_tokens = len(results)
    if n_tokens > 1:
        decode_time = results[-1][2] - results[0][2]
        decode_tps = (n_tokens - 1) / decode_time if decode_time > 0 else 0
    else:
        decode_tps = 0

    print(f"\n[mac-decode] done: {n_tokens} tokens in {t_total:.1f}s", flush=True)
    print(f"[mac-decode] prefill: {results[0][2]*1000:.0f}ms, "
          f"decode: {decode_tps:.1f} tok/s ({n_tokens-1} tokens)", flush=True)
    print(f"[mac-decode] output: {''.join(r[1] for r in results)}", flush=True)
    return results


if __name__ == "__main__":
    print("=== Mac MLX Decode + hidden_P Capture Test ===\n")
    model_path = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"
    print(f"Loading {model_path}...")
    t0 = time.time()
    model, tokenizer = load(model_path)
    print(f"Loaded in {time.time()-t0:.1f}s\n")

    messages = [{"role": "user", "content": "Write a short story about a cat"}]
    results = mac_mlx_decode_with_hidden(model, tokenizer, messages, P=6, max_tokens=10)

    # 验证 hidden_P
    hps = [r[0] for r in results if r[0] is not None]
    print(f"\nhidden_P captured: {len(hps)}/{len(results)} steps")
    if hps:
        # MLX array → numpy → torch (延迟转换)
        import mlx.core as mx
        hp_np = np.array(hps[0].astype(mx.float32))
        hp_torch = torch.from_numpy(hp_np).clone()
        print(f"hidden_P shape: {tuple(hp_torch.shape)}, dtype: {hp_torch.dtype}")
