"""MTP patched spec v3 — 直接调用 target 内部模块获取 hidden.

Qwen3Model.__call__ 返回 self.norm(h) = hidden_states (lm_head 前).
直接调用 target_model.language_model.model(y, cache) 获取 hidden.
"""
from __future__ import annotations

import time
from typing import Generator, Tuple

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
    def __init__(self, hidden_size=2048, num_heads=16, head_dim=128, intermediate_size=5632):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm1 = MTPRMSNorm(hidden_size)
        self.attn = MTPAttention(hidden_size, num_heads, head_dim)
        self.norm2 = MTPRMSNorm(hidden_size)
        self.mlp = MTPMLP(hidden_size, intermediate_size)
        self.norm_out = MTPRMSNorm(hidden_size)
        self.layers = [self.attn]

    def __call__(self, hidden, embed, cache=None):
        x = mx.concatenate([hidden, embed], axis=-1)
        x = self.proj(x)
        x = self.norm1(x)
        attn_cache = cache[0] if isinstance(cache, list) else cache
        x = x + self.attn(x, attn_cache)
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = self.norm_out(x)
        return x


def load_mtp_head(checkpoint_path: str) -> MTPHead:
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
    """Patched spec — 直接调用 target 内部获取 hidden."""
    from mlx_lm.cache_prompt import make_prompt_cache
    from mlx_lm.models.cache import trim_prompt_cache
    from mlx_lm.generate import generation_stream
    from mlx_lm.models.qwen3 import create_attention_mask

    def target_forward_raw(model, inputs, cache):
        """手动 forward, 返回 norm 后 hidden (和 PyTorch hidden_states[-1] 一致)."""
        h = model.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(model.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(model.layers, cache):
            h = layer(h, mask, c)
        return model.norm(h)  # norm 后! (和 PyTorch hidden_states[-1] 一致)

    # Tokenize
    if isinstance(prompt, str):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [t for t in tokens if t not in (151644, 151645)]
    else:
        tokens = prompt
    y = mx.array(tokens, mx.uint32)

    # KV caches
    target_cache = make_prompt_cache(target_model)
    # MTP head 不用 cache (训练时是单 token self-attention, 用 cache 会不一致)

    # target 内部模块
    target_lm = target_model.language_model
    target_inner = target_lm.model  # Qwen3Model (返回 hidden)
    target_embed = target_inner.embed_tokens
    lm_head_w = target_embed.weight  # tied

    # Prefill: 用 target_inner 获取 norm 后 hidden + logits
    print("[spec] Prefill...")
    with mx.stream(generation_stream):
        # target_inner 返回 self.norm(h) = norm 后 hidden (和 PyTorch hidden_states[-1] 一致)
        hidden = target_inner(y[None], cache=target_cache)  # [1, seq_len, hidden]
        target_hidden = hidden[:, -1:, :]  # [1, 1, hidden] norm 后
        # hidden 已 norm 后, 直接 lm_head
        target_logits = target_hidden @ lm_head_w.T  # [1, 1, vocab]
        first_token = mx.argmax(target_logits, axis=-1).reshape(1).astype(mx.uint32)

    mx.eval(first_token, target_hidden)
    # 调试: 打印 hidden 值 (对比独立测试)
    print(f"  [debug prefill] hidden[:5]={np.array(target_hidden[0,0,:5].astype(mx.float32))}")
    print(f"  [debug prefill] expected=[-0.44, 1.49, 1.17, -0.5, 0.51]")
    yield int(first_token[0].item()), False

    y = first_token  # 1D [1]
    ntoks = 1

    print(f"[spec] Decode loop (N={num_draft})...")

    while ntoks < max_tokens:
        # 1. MTP draft generate (N tokens)
        draft_tokens = []
        draft_y = y
        current_hidden = target_hidden  # [1, 1, hidden]

        for i in range(num_draft):
            with mx.stream(generation_stream):
                embed = target_embed(draft_y[None])  # [1, 1, hidden]
                mtp_hidden = mtp_head(current_hidden, embed, cache=None)  # 不用 cache
                draft_logits = mtp_hidden @ lm_head_w.T
                draft_token = mx.argmax(draft_logits, axis=-1).reshape(1).astype(mx.uint32)
            mx.eval(draft_token)
            draft_tokens.append(int(draft_token[0].item()))
            draft_y = draft_token
            current_hidden = mtp_hidden

        # 2. Target verify (N+1 tokens, 用 target_inner 获取 hidden + logits)
        all_tokens = mx.concatenate([y[None], mx.array(draft_tokens, mx.uint32)[None]], axis=1)  # [1, N+1]
        with mx.stream(generation_stream):
            verify_hidden = target_inner(all_tokens, cache=target_cache)  # [1, N+1, hidden] norm 后
            verify_logits = verify_hidden @ lm_head_w.T  # [1, N+1, vocab]

        mx.eval(verify_logits)
        verify_argmax = mx.argmax(verify_logits, axis=-1).squeeze(0).tolist()

        # 调试: 打印 draft vs verify
        if ntoks <= 5:
            print(f"  [debug] y={int(y[0].item())}, draft={draft_tokens[:4]}, verify={verify_argmax[:5]}")
            # 打印 MTP logits top-5
            embed0 = target_embed(y[None])
            mtp_h0 = mtp_head(target_hidden, embed0, cache=None)
            mtp_logits0 = mtp_h0 @ lm_head_w.T
            top5 = mx.argsort(-mtp_logits0[0, 0])[:5].tolist()
            print(f"  [debug] MTP top-5: {top5}, target pick: {verify_argmax[0]}")

        # 3. Accept matching
        # verify_logits[i] 预测位置 i 的 next token
        # verify_argmax[0] = target 对 y 的预测 (应该 = draft_tokens[0] 如果正确)
        # verify_argmax[i] = target 对 draft_tokens[i-1] 的预测 (应该 = draft_tokens[i])
        n_accept = 0
        for i in range(num_draft):
            if i < len(verify_argmax):
                if draft_tokens[i] == verify_argmax[i]:
                    yield draft_tokens[i], True
                    n_accept += 1
                    ntoks += 1
                    if ntoks >= max_tokens:
                        break
                else:
                    break

        # 4. 输出 target 正确 token (verify_argmax[n_accept] 是 target 对最后 accept 位置的预测)
        if ntoks < max_tokens:
            correct_idx = min(n_accept, len(verify_argmax) - 1)
            correct_token = verify_argmax[correct_idx]
            yield correct_token, False
            ntoks += 1
            y = mx.array([correct_token], mx.uint32)
            # 更新 hidden (用 verify 的 correct_idx 位置的 hidden, norm 后)
            target_hidden = verify_hidden[:, correct_idx:correct_idx+1, :]
        else:
            break

        # 5. Rewind cache (verify forward 了 N+1 tokens, 只保留 1+n_accept)
        rewind = num_draft - n_accept
        if rewind > 0:
            try:
                trim_prompt_cache(target_cache, rewind)
            except Exception as e:
                print(f"  [warn] trim failed: {e}")


def bench(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=0, label=""):
    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")

    # warmup
    try:
        if num_draft > 0:
            list(patched_spec_generate(target_model, tokenizer, prompt, mtp_head, max_tokens=3, num_draft=num_draft))
        else:
            from mlx_lm import stream_generate
            list(stream_generate(target_model, tokenizer, prompt, max_tokens=1))
    except Exception as e:
        print(f"  warmup error: {e}")
        import traceback; traceback.print_exc()
        return None

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
        print(f"  error: {e}")
        import traceback; traceback.print_exc()
        return None

    t_end = time.time()
    if t_first is None:
        t_first = t0
    dt = t_end - t_first
    nd = len(tokens) - 1
    if nd <= 0 or dt <= 0:
        print("  no tokens")
        return None

    ar = draft_count / total if total else 0
    tps = nd / dt
    print(f"  TTFT: {1000*(t_first-t0):.0f}ms")
    print(f"  Decode: {tps:.1f} tok/s")
    print(f"  Accept: {ar:.1%} ({draft_count}/{total})")
    print(f"  Output: {tokenizer.decode(tokens[:30])}")
    return {"tps": tps, "ar": ar}


if __name__ == "__main__":
    from mlx_lm import load

    print("[1] Loading target model...")
    target_model, tokenizer = load("/Users/alexchuang/models/Qwen3-VL-2B-bf16")

    print("[2] Loading MTP head...")
    mtp_head = load_mtp_head("/tmp/mtp_head_final.pt")

    prompt = "Write a short story about a cat"

    # Baseline
    b = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=0, label="Baseline (无投机)")
    baseline_tps = b["tps"] if b else 26

    # Spec N=4
    r4 = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=4, label="投机 N=4 (MTP + 真 hidden)")
    if r4:
        print(f"  加速: {r4['tps']/baseline_tps:.2f}x")

    # Spec N=10
    r10 = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=10, label="投机 N=10 (MTP + 真 hidden)")
    if r10:
        print(f"  加速: {r10['tps']/baseline_tps:.2f}x")

    # 汇总
    print(f"\n{'='*55}")
    print("汇总")
    print(f"{'='*55}")
    print(f"Baseline: {baseline_tps:.1f} tok/s")
    if r4:
        print(f"N=4: {r4['tps']:.1f} tok/s ({r4['tps']/baseline_tps:.2f}x, accept {r4['ar']:.0%})")
    if r10:
        print(f"N=10: {r10['tps']:.1f} tok/s ({r10['tps']/baseline_tps:.2f}x, accept {r10['ar']:.0%})")
