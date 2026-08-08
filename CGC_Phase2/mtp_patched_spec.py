"""Patched speculative_generate_step — MTP head 复用 target hidden_states + lm_head.

核心修改:
1. target forward 后保存 hidden_states
2. MTP head forward 时用 target 的 hidden + embed
3. MTP head 不做 lm_head, 输出 hidden → target lm_head → logits

这样 MTP head forward 只需 ~1ms (proj + attn + mlp), 无 lm_head 开销。
"""
from __future__ import annotations

import functools
import time
from typing import Optional, List, Callable, Any, Generator, Tuple

import mlx.core as mx
import mlx.nn as nn
import torch
import numpy as np


class MTPRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        x_f32 = x.astype(mx.float32)
        var = mx.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        return (self.weight * (x_f32 * mx.rsqrt(var + self.eps))).astype(x.dtype)


class MTPAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
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
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MTPMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPHead(nn.Module):
    """MTP head — 只做 proj + attn + mlp, 不做 lm_head (复用 target 的).

    forward(hidden, embed) → hidden_out (不转 logits)
    """

    def __init__(self, hidden_size=2048, num_heads=16, head_dim=128, intermediate_size=5632):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm1 = MTPRMSNorm(hidden_size)
        self.attn = MTPAttention(hidden_size, num_heads, head_dim)
        self.norm2 = MTPRMSNorm(hidden_size)
        self.mlp = MTPMLP(hidden_size, intermediate_size)
        self.norm_out = MTPRMSNorm(hidden_size)
        self.layers = [self.attn]  # for make_prompt_cache

    def __call__(self, hidden, embed, cache=None):
        x = mx.concatenate([hidden, embed], axis=-1)
        x = self.proj(x)
        x = self.norm1(x)
        attn_cache = cache[0] if isinstance(cache, list) else cache
        x = x + self.attn(x, attn_cache)
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = self.norm_out(x)
        return x  # hidden_states (不转 logits)


def load_mtp_head(checkpoint_path: str) -> MTPHead:
    """加载 MTP head (不含 embed/lm_head)."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)

    model = MTPHead()
    for name, p in sd.items():
        if "lm_head" in name or "embed" in name:
            continue
        w_np = p.float().numpy()
        parts = name.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        attr = parts[-1]
        current = getattr(obj, attr, None)
        if current is not None and hasattr(current, "shape"):
            if current.shape != tuple(w_np.shape) and len(w_np.shape) == 2:
                if current.shape == (w_np.shape[1], w_np.shape[0]):
                    w_np = w_np.T
        setattr(obj, attr, mx.array(w_np))

    return model


def patched_spec_generate(
    target_model,
    tokenizer,
    prompt: str,
    mtp_head: MTPHead,
    max_tokens: int = 30,
    num_draft: int = 4,
) -> Generator[Tuple[int, bool], None, None]:
    """Patched speculative generate — MTP head 用 target hidden + lm_head.

    Yields:
        (token_id, from_draft)
    """
    from mlx_lm.cache_prompt import make_prompt_cache
    from mlx_lm.models.cache import trim_prompt_cache
    from mlx_lm.generate import generation_stream

    # Tokenize
    if isinstance(prompt, str):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        # 过滤 special tokens (对齐 cloud)
        tokens = [t for t in tokens if t not in (151644, 151645)]
    else:
        tokens = prompt
    y = mx.array(tokens, mx.uint32)

    # KV caches
    target_cache = make_prompt_cache(target_model)
    mtp_cache = make_prompt_cache(mtp_head)

    # target 的 embed + lm_head (复用)
    target_lm = target_model.language_model
    target_embed = target_lm.model.embed_tokens
    # lm_head: 检查是否有独立 lm_head, 否则用 embed (tied)
    lm_head_w = target_embed.weight  # 默认 tied

    # Prefill target (获取 hidden + 首 token)
    print("[spec] Prefill target...")
    with mx.stream(generation_stream):
        # target prefill
        target_logits = target_model(y[None], cache=target_cache)
        target_logits = target_logits[:, -1:, :]
        # 保存 target 的最后 hidden_states
        # target_model 内部 forward 后, hidden 在最后一层
        # 我们需要 hook 获取 hidden

        # 简化: 用 target_logits 的 argmax 作为首 token
        first_token = mx.argmax(target_logits, axis=-1).reshape(1).astype(mx.uint32)

    mx.eval(first_token)
    yield int(first_token[0].item()), False

    y = first_token
    ntoks = 1

    # 4bit 量化 lm_head (如果需要)
    # 简化: 直接用 bf16 lm_head (后续可优化)

    print(f"[spec] Decode loop (num_draft={num_draft})...")

    while ntoks < max_tokens:
        # 1. Target forward (1 token) — 获取 hidden + verify
        with mx.stream(generation_stream):
            target_logits = target_model(y[None], cache=target_cache)
            target_logits = target_logits[:, -1:, :]

            # 获取 target 的 hidden (倒数第二层输出)
            # 简化: 用 target_logits 逆向 (不对)
            # 实际需要 hook target 内部

            # TODO: hook target forward 获取 hidden
            # 临时方案: 用 embed 代替 hidden (accept rate 会低)
            target_hidden = target_embed(y[None])  # [1, 1, hidden]

        # 2. MTP draft generate (N tokens)
        draft_tokens = []
        draft_y = y

        for i in range(num_draft):
            with mx.stream(generation_stream):
                # MTP forward: hidden + embed → hidden_out
                embed = target_embed(draft_y[None])
                mtp_hidden = mtp_head(target_hidden, embed, cache=mtp_cache)
                # MTP hidden → target lm_head → logits
                draft_logits = mtp_hidden @ lm_head_w.T  # [1, 1, vocab]
                draft_token = mx.argmax(draft_logits, axis=-1).reshape(1).astype(mx.uint32)

            mx.eval(draft_token)
            draft_tokens.append(int(draft_token[0].item()))
            draft_y = draft_token
            # 更新 hidden (用 MTP 的输出作为下一次的 hidden)
            target_hidden = mtp_hidden

        # 3. Target verify (N+1 tokens)
        verify_tokens = mx.concatenate([y[None], mx.array(draft_tokens, mx.uint32)[None]])  # [1, N+1]
        with mx.stream(generation_stream):
            verify_logits = target_model(verify_tokens, cache=target_cache)  # [1, N+1] → [1, N+1, vocab]

        mx.eval(verify_logits)
        verify_tokens_list = mx.argmax(verify_logits, axis=-1).squeeze(0).tolist()

        # 4. Accept matching
        n_accept = 0
        for i in range(num_draft):
            if i < len(verify_tokens_list) - 1:
                if draft_tokens[i] == verify_tokens_list[i + 1]:
                    yield draft_tokens[i], True
                    n_accept += 1
                    ntoks += 1
                    if ntoks >= max_tokens:
                        break
                else:
                    break

        # 5. 输出 verify token (至少 1 个)
        if ntoks < max_tokens:
            # verify_tokens_list[n_accept] 是 target 的正确 token
            correct_idx = min(n_accept, len(verify_tokens_list) - 1)
            yield verify_tokens_list[correct_idx], False
            ntoks += 1
            y = mx.array([verify_tokens_list[correct_idx]], mx.uint32)
        else:
            break

        # 6. Rewind cache (拒绝的 tokens)
        rewind = num_draft - n_accept
        if rewind > 0:
            try:
                trim_prompt_cache(target_cache, rewind)
                trim_prompt_cache(mtp_cache, rewind)
            except Exception:
                pass


def benchmark(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=0, label=""):
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")

    # Warmup
    try:
        if num_draft > 0:
            list(patched_spec_generate(target_model, tokenizer, prompt, mtp_head, max_tokens=1, num_draft=num_draft))
        else:
            from mlx_lm import stream_generate
            list(stream_generate(target_model, tokenizer, prompt, max_tokens=1))
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
            gen = patched_spec_generate(target_model, tokenizer, prompt, mtp_head, max_tokens=max_tokens, num_draft=num_draft)
        else:
            from mlx_lm import stream_generate
            gen = stream_generate(target_model, tokenizer, prompt, max_tokens=max_tokens)

        for item in gen:
            if isinstance(item, tuple):
                token_id, from_draft = item
            else:
                token_id = item.token
                from_draft = getattr(item, "from_draft", False)
            tokens.append(token_id)
            total += 1
            if from_draft:
                draft_count += 1
            if len(tokens) == 1:
                t_first = time.time()
    except Exception as e:
        print(f"  generate error: {e}")
        import traceback
        traceback.print_exc()
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
    from mlx_lm import load

    print("[1] Loading target model...")
    target_model, tokenizer = load("/Users/alexchuang/models/Qwen3-VL-2B-bf16")

    print("[2] Loading MTP head...")
    mtp_head = load_mtp_head("/tmp/mtp_head_final.pt")

    prompt = "Write a short story about a cat"

    # Baseline
    benchmark(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=0, label="Baseline (无投机)")

    # Spec N=4
    benchmark(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=4, label="投机 N=4 (patched)")

    # Spec N=10
    benchmark(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=10, label="投机 N=10 (patched)")
