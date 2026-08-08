"""MTP Draft Model (MLX) — 独立运行的投机 draft model (4bit 量化)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch


class MTPRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        x_f32 = x.astype(mx.float32)
        var = mx.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        return (self.weight * (x_f32 * mx.rsqrt(var + self.eps))).astype(x.dtype)


class MTPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(hidden_size, inner, bias=False)
        self.o_proj = nn.Linear(inner, hidden_size, bias=False)

    def __call__(self, x, cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if cache is not None and k.shape[2] > T:
            attn = attn[:, :, -1:, :]
        else:
            mask = mx.triu(mx.full((T, T), -1e9), k=1)
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


class MTPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPDraftModel(nn.Module):
    """MTP Draft Model — 独立运行, 用 embed 代替 hidden_states."""

    def __init__(self, hidden_size=2048, vocab_size=151936, num_heads=16,
                 head_dim=128, intermediate_size=5632):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm1 = MTPRMSNorm(hidden_size)
        self.attn = MTPAttention(hidden_size, num_heads, head_dim)
        self.norm2 = MTPRMSNorm(hidden_size)
        self.mlp = MTPMLP(hidden_size, intermediate_size)
        self.norm_out = MTPRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # mlx_lm 需要的属性
        self.layers = [self.attn]  # 只包含 attention (KV cache 用)
        self.model_type = "mtp_draft"
        self.args = type("Args", (), {
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_hidden_layers": 1,
        })()

    def __call__(self, tokens, cache=None):
        hidden = self.embed_tokens(tokens)
        x = mx.concatenate([hidden, hidden], axis=-1)
        x = self.proj(x)
        x = self.norm1(x)
        # cache 是 list[KVCache], 取第一个
        attn_cache = cache[0] if isinstance(cache, list) else cache
        x = x + self.attn(x, attn_cache)
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = self.norm_out(x)
        return self.lm_head(x)


def load_mtp_draft(checkpoint_path: str, target_path: str, quantize: bool = True):
    """加载 MTP draft model."""
    print(f"[load] MTP checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)

    print(f"[load] Target model: {target_path}")
    from mlx_lm import load as mlx_load
    target, tokenizer = mlx_load(target_path)

    # 获取 target 的 embed + lm_head
    target_lm = target.language_model
    target_embed = target_lm.model.embed_tokens  # language_model.model.embed_tokens
    embed_w = target_embed.weight  # [151936, 2048]
    # lm_head: Qwen3-VL 可能 tied 或在 language_model
    lm_head_w = embed_w  # tied weights (默认假设)

    # 检查是否有独立 lm_head
    for name, p in nn.utils.tree_flatten(target.parameters()):
        if "lm_head" in name:
            lm_head_w = p
            print(f"  found lm_head: {name}")
            break

    print(f"  embed: {embed_w.shape}, lm_head: {lm_head_w.shape}")

    # 创建 model
    print("[load] Creating MTPDraftModel...")
    model = MTPDraftModel()

    # 加载 MTP head 权重 (用直接赋值)
    print("[load] Loading MTP head weights...")
    loaded = 0
    for name, p in sd.items():
        if "lm_head" in name or "embed" in name:
            continue
        w_np = p.float().numpy()
        # checkpoint 的 weight shape: [out, in] (PyTorch)
        # MLX Linear weight: [out, in] (相同)

        # 映射到 model 属性
        parts = name.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        attr = parts[-1]

        # 检查 shape
        current = getattr(obj, attr)
        if hasattr(current, "shape"):
            if current.shape != tuple(w_np.shape):
                if len(w_np.shape) == 2 and current.shape == (w_np.shape[1], w_np.shape[0]):
                    w_np = w_np.T
        setattr(obj, attr, mx.array(w_np))
        loaded += 1
    print(f"  MTP weights: {loaded}")

    # 设置 embed + lm_head (从 target 复制)
    print("[load] Setting embed_tokens + lm_head from target...")
    model.embed_tokens.weight = embed_w
    model.lm_head.weight = lm_head_w

    # 4bit 量化
    if quantize:
        print("[load] 4bit quantization (embed + lm_head)...")
        try:
            nn.quantize(model, bits=4, group_size=64)
            print("  quantization done (全模型)")
        except Exception as e:
            print(f"  quantization failed: {e}")

    del target
    return model, tokenizer


def bench(model, tokenizer, prompt, max_tokens=20, num_draft=0, label=""):
    from mlx_lm import stream_generate
    import time

    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")

    # warmup
    try:
        if num_draft > 0:
            list(stream_generate(model, tokenizer, prompt, max_tokens=1,
                                 draft_model=model, num_draft_tokens=num_draft))
        else:
            list(stream_generate(model, tokenizer, prompt, max_tokens=1))
    except Exception as e:
        print(f"  warmup error: {e}")
        return

    t0 = time.time()
    tokens = []
    draft_count = 0
    total = 0
    t_first = None

    try:
        if num_draft > 0:
            gen = stream_generate(model, tokenizer, prompt, max_tokens=max_tokens,
                                  draft_model=model, num_draft_tokens=num_draft)
        else:
            gen = stream_generate(model, tokenizer, prompt, max_tokens=max_tokens)

        for resp in gen:
            tokens.append(resp.token)
            total += 1
            if hasattr(resp, "from_draft") and resp.from_draft:
                draft_count += 1
            if len(tokens) == 1:
                t_first = time.time()
    except Exception as e:
        print(f"  generate error: {e}")
        return

    t_end = time.time()
    if t_first is None:
        t_first = t0
    dt = t_end - t_first
    nd = len(tokens) - 1
    if nd <= 0 or dt <= 0:
        print("  no tokens")
        return

    ar = draft_count / total if total else 0
    tps = nd / dt
    print(f"  TTFT: {1000*(t_first-t0):.0f}ms")
    print(f"  Decode: {tps:.1f} tok/s ({nd} tok / {dt:.2f}s)")
    print(f"  Accept: {ar:.1%} ({draft_count}/{total})")
    print(f"  Output: {tokenizer.decode(tokens[:30])}")


if __name__ == "__main__":
    model, tokenizer = load_mtp_draft(
        "/tmp/mtp_head_final.pt",
        "/Users/alexchuang/models/Qwen3-VL-2B-bf16",
        quantize=True,
    )

    prompt = "Write a short story about a cat"

    # Baseline
    bench(model, tokenizer, prompt, max_tokens=20, num_draft=0, label="Baseline (无投机)")

    # Spec N=4
    bench(model, tokenizer, prompt, max_tokens=20, num_draft=4, label="投机 N=4")

    # Spec N=10
    bench(model, tokenizer, prompt, max_tokens=20, num_draft=10, label="投机 N=10")
