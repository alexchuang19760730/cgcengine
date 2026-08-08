"""MTP Draft Model (MLX) — 独立运行的投机 draft model.

包装 MTP head 为 mlx_lm 兼容的 nn.Module:
- embed_tokens (4bit 量化, 从 target 复用)
- MTP transformer 层 (proj + attention + MLP)
- lm_head (4bit 量化)

forward: token_ids → embed → MTP layer → lm_head → logits

注意: 训练时用 target hidden_states, 推理时用 embed 代替。
      accept rate 可能下降, 需实测。
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch


class MTPRMSNorm(nn.Module):
    """RMSNorm (对齐 Qwen3)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        x_f32 = x.astype(mx.float32)
        var = mx.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        x_normed = x_f32 * mx.rsqrt(var + self.eps)
        return (self.weight * x_normed).astype(x.dtype)


class MTPAttention(nn.Module):
    """Multi-head attention with RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(inner, hidden_size, bias=False)

    def __call__(self, x, cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        # causal mask
        if cache is not None and k.shape[2] > T:
            # decode: 只 attend 最后一个
            attn = attn[:, :, -1:, :]
        else:
            # prefill: causal
            mask = mx.triu(mx.full((T, T), -1e9), k=1)
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v  # [B, H, T, D]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


class MTPMLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(nn.silu(gate) * up)


class MTPDraftModel(nn.Module):
    """MTP Draft Model for speculative decoding.

    独立运行: token_ids → embed → MTP layer → lm_head → logits
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        vocab_size: int = 151936,
        num_heads: int = 16,
        head_dim: int = 128,
        intermediate_size: int = 5632,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # embed_tokens (4bit 量化, 从 target 复用)
        # 延迟初始化 (从 checkpoint 加载)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # MTP head 层
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm1 = MTPRMSNorm(hidden_size)
        self.attn = MTPAttention(hidden_size, num_heads, head_dim)
        self.norm2 = MTPRMSNorm(hidden_size)
        self.mlp = MTPMLP(hidden_size, intermediate_size)
        self.norm3 = MTPRMSNorm(hidden_size)

        # lm_head (4bit 量化, 从 target 复用)
        # 延迟初始化 (从 checkpoint 加载)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 层信息 (mlx_lm 需要)
        self.layers = [self]  # 让 mlx_lm 能识别

    def __call__(self, tokens, cache=None):
        """forward: token_ids → logits.

        Args:
            tokens: [B, T] token ids
            cache: KV cache (可选)

        Returns:
            logits: [B, T, vocab_size]
        """
        # embed
        hidden = self.embed_tokens(tokens)  # [B, T, hidden]

        # MTP forward: concat(hidden, embed) → proj → norm → attn → norm → mlp → norm → lm_head
        # 训练时: concat(hidden_states, embed(token))
        # 独立运行时: 用 embed 代替 hidden_states → concat(embed, embed)
        x = mx.concatenate([hidden, hidden], axis=-1)  # [B, T, 2*hidden]
        x = self.proj(x)  # [B, T, hidden]
        x = self.norm1(x)
        x = x + self.attn(x, cache)
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = self.norm3(x)

        # lm_head
        logits = self.lm_head(x)  # [B, T, vocab]
        return logits


def load_mtp_draft_model(
    checkpoint_path: str,
    target_model_path: str,
    quantize_4bit: bool = True,
) -> tuple[MTPDraftModel, object]:
    """加载 MTP draft model + tokenizer.

    Args:
        checkpoint_path: MTP head checkpoint (.pt)
        target_model_path: target model path (用于 embed/lm_head + tokenizer)
        quantize_4bit: 是否 4bit 量化 embed/lm_head

    Returns:
        (model, tokenizer)
    """
    print(f"[load] Loading MTP checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)

    print(f"[load] Loading tokenizer from: {target_model_path}")
    from mlx_lm import load as mlx_load
    target_model, tokenizer = mlx_load(target_model_path)

    # 创建 MTP draft model
    print("[load] Creating MTPDraftModel...")
    model = MTPDraftModel()

    # 加载 MTP head 权重
    print("[load] Loading MTP head weights...")
    mtp_keys_loaded = 0
    for name, p in sd.items():
        if "lm_head" in name or "embed" in name:
            continue  # 跳过, 用 target 的
        # 映射 key
        # checkpoint: proj.weight, mlp.gate_proj.weight, etc.
        # model: proj.weight, mlp.gate_proj.weight, etc.
        mlx_name = name
        if hasattr(model, mlx_name.split(".")[0]):
            try:
                param = getattr(model, mlx_name.split(".")[0])
                for part in mlx_name.split(".")[1:]:
                    param = getattr(param, part)
                # 检查 shape
                w_np = p.float().numpy()
                if param.shape != tuple(w_np.shape):
                    # 可能需要 transpose
                    if len(w_np.shape) == 2 and param.shape == (w_np.shape[1], w_np.shape[0]):
                        w_np = w_np.T
                param.update(mx.array(w_np))
                mtp_keys_loaded += 1
            except (AttributeError, ValueError) as e:
                print(f"  skip {name}: {e}")

    print(f"  MTP weights loaded: {mtp_keys_loaded}")

    # 从 target 复制 embed_tokens + lm_head
    print("[load] Copying embed_tokens + lm_head from target...")

    # target 结构: model.language_model.embed_tokens, model.language_model.lm_head
    target_lang = getattr(target_model, "language_model", target_model.model)
    target_embed = getattr(target_lang, "embed_tokens", None)
    target_lm_head = getattr(target_lang, "lm_head", None) or getattr(target_model, "lm_head", None)

    if target_embed is not None:
        # 复制 embed weight
        embed_w = target_embed.weight
        if quantize_4bit:
            print("  [quantize] embed_tokens 4bit...")
            # mlx_lm 的量化用 quantize_model
            # 简化: 直接复制 (后续可量化)
            model.embed_tokens.weight = embed_w
        else:
            model.embed_tokens.weight = embed_w

    if target_lm_head is not None:
        lm_head_w = target_lm_head.weight
        if quantize_4bit:
            print("  [quantize] lm_head 4bit...")
            # 简化: 直接复制, 后续用 nn.quantize
            model.lm_head.weight = lm_head_w
        else:
            model.lm_head.weight = lm_head_w

    # 4bit 量化 (用 mlx.nn.quantize)
    if quantize_4bit:
        print("[load] Applying 4bit quantization to embed + lm_head...")
        try:
            # 只量化 embed + lm_head
            nn.quantize(model.embed_tokens, bits=4, group_size=64)
            nn.quantize(model.lm_head, bits=4, group_size=64)
            print("  quantization done")
        except Exception as e:
            print(f"  quantization failed (fallback to bf16): {e}")

    # 释放 target model
    del target_model

    return model, tokenizer


def benchmark_spec_decode(
    model: MTPDraftModel,
    tokenizer,
    prompt: str,
    max_tokens: int = 30,
    num_draft: int = 4,
):
    """测试投机 decode."""
    from mlx_lm import stream_generate
    from mlx_lm.cache_prompt import make_prompt_cache

    print(f"\n[bench] Prompt: {prompt[:50]}...")
    print(f"[bench] max_tokens={max_tokens}, num_draft={num_draft}")

    # warmup
    try:
        list(stream_generate(
            model, tokenizer, prompt, max_tokens=1, draft_model=model, num_draft_tokens=num_draft
        ))
    except Exception as e:
        print(f"  warmup error: {e}")
        return

    # benchmark
    import time
    t0 = time.time()
    tokens = []
    draft_count = 0
    total_count = 0
    t_first = None

    try:
        for resp in stream_generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            draft_model=model,
            num_draft_tokens=num_draft,
        ):
            tokens.append(resp.token)
            total_count += 1
            if resp.from_draft:
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
        print("  no tokens generated")
        return

    ar = draft_count / total_count if total_count else 0
    tps = nd / dt

    print(f"\n[result]")
    print(f"  TTFT: {1000*(t_first-t0):.0f}ms")
    print(f"  Decode: {nd} tokens in {dt:.2f}s = {tps:.1f} tok/s")
    print(f"  Draft accepted: {draft_count}/{total_count} = {ar:.1%}")
    print(f"  Output: {tokenizer.decode(tokens[:30])}")


if __name__ == "__main__":
    import sys

    checkpoint = "/tmp/mtp_head_final.pt"
    target_path = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"

    if not Path(checkpoint).exists():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        sys.exit(1)

    if not Path(target_path).exists():
        print(f"ERROR: target model not found: {target_path}")
        sys.exit(1)

    # 1. 加载 MTP draft model
    model, tokenizer = load_mtp_draft_model(
        checkpoint, target_path, quantize_4bit=True
    )

    # 2. baseline (无投机)
    print("\n" + "="*60)
    print("Baseline (无投机)")
    print("="*60)
    benchmark_spec_decode(model, tokenizer, "Write a short story about a cat", max_tokens=20, num_draft=0)

    # 3. 投机 decode (N=4)
    print("\n" + "="*60)
    print("投机 decode N=4")
    print("="*60)
    benchmark_spec_decode(model, tokenizer, "Write a short story about a cat", max_tokens=20, num_draft=4)

    # 4. 投机 decode (N=10)
    print("\n" + "="*60)
    print("投机 decode N=10")
    print("="*60)
    benchmark_spec_decode(model, tokenizer, "Write a short story about a cat", max_tokens=20, num_draft=10)
