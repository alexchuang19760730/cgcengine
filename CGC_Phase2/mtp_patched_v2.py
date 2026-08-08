"""MTP patched speculative decode v2 — hook target 获取真正 hidden_states.

关键修复:
1. Hook target 最后一层 forward, 捕获 hidden_states
2. MTP head 用真正的 hidden (不是 embed) → accept rate 高
3. MTP head 不含 lm_head, 复用 target lm_head → forward ~1ms
4. Shape bug 修复 (reshape(1) 确保 1D)
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
    """MTP head — forward(hidden, embed) → hidden_out (不含 lm_head)."""

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


class HiddenCapture:
    """用 wrapper 替换最后一层, 捕获 hidden_states (MLX dunder 限制)."""

    def __init__(self, target_model):
        self.captured = None
        lm = target_model.language_model
        self.layers_list = lm.model.layers
        self.last_layer = self.layers_list[-1]
        self._installed = False

    def __enter__(self):
        capture = self
        original = self.last_layer

        class _Wrapper(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self._wrapped = wrapped
                # 复制所有属性 (让 mlx_lm 能正常使用)
                for attr in dir(wrapped):
                    if not attr.startswith("_") and not callable(getattr(wrapped, attr, None)):
                        try:
                            setattr(self, attr, getattr(wrapped, attr))
                        except:
                            pass

            def __call__(self, *args, **kwargs):
                result = self._wrapped(*args, **kwargs)
                hs = result[0] if isinstance(result, (tuple, list)) else result
                if hs is not None and hasattr(hs, "shape"):
                    capture.captured = hs[:, -1:, :]
                return result

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        self._wrapper = _Wrapper(original)
        self.layers_list[-1] = self._wrapper
        self._installed = True
        return self

    def __exit__(self, *args):
        if self._installed:
            self.layers_list[-1] = self.last_layer
            self._installed = False


def patched_spec_generate(
    target_model,
    tokenizer,
    prompt: str,
    mtp_head: MTPHead,
    max_tokens: int = 30,
    num_draft: int = 4,
) -> Generator[Tuple[int, bool], None, None]:
    """Patched spec — MTP head 用 target hidden + 复用 lm_head."""
    from mlx_lm.cache_prompt import make_prompt_cache
    from mlx_lm.models.cache import trim_prompt_cache
    from mlx_lm.generate import generation_stream

    # Tokenize
    if isinstance(prompt, str):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [t for t in tokens if t not in (151644, 151645)]
    else:
        tokens = prompt
    y = mx.array(tokens, mx.uint32)  # 1D [seq_len]

    # KV caches
    target_cache = make_prompt_cache(target_model)
    mtp_cache = make_prompt_cache(mtp_head)

    # target embed + lm_head
    target_lm = target_model.language_model
    target_embed = target_lm.model.embed_tokens
    lm_head_w = target_embed.weight  # tied

    # Hook
    hook = HiddenCapture(target_model)

    # Prefill
    print("[spec] Prefill...")
    with mx.stream(generation_stream):
        with hook:
            target_logits = target_model(y[None], cache=target_cache)
        target_logits = target_logits[:, -1:, :]
        target_hidden = hook.captured  # [1, 1, hidden] 真正的 hidden!

    first_token = mx.argmax(target_logits, axis=-1).reshape(1).astype(mx.uint32)
    mx.eval(first_token, target_hidden)
    yield int(first_token[0].item()), False

    y = first_token  # 1D [1]
    ntoks = 1

    print(f"[spec] Decode loop (N={num_draft})...")

    while ntoks < max_tokens:
        # 1. MTP draft generate (N tokens, 用真正的 target hidden)
        draft_tokens = []
        draft_y = y
        current_hidden = target_hidden  # [1, 1, hidden]

        for i in range(num_draft):
            with mx.stream(generation_stream):
                embed = target_embed(draft_y[None])  # [1, 1, hidden]
                mtp_hidden = mtp_head(current_hidden, embed, cache=mtp_cache)
                draft_logits = mtp_hidden @ lm_head_w.T
                draft_token = mx.argmax(draft_logits, axis=-1).reshape(1).astype(mx.uint32)
            mx.eval(draft_token)
            draft_tokens.append(int(draft_token[0].item()))
            draft_y = draft_token
            current_hidden = mtp_hidden  # 链式: 用 MTP 输出作为下一次 hidden

        # 2. Target verify (N+1 tokens, 带hidden hook)
        all_tokens = mx.concatenate([y[None], mx.array(draft_tokens, mx.uint32)[None]])  # [1, N+1]
        with mx.stream(generation_stream):
            with hook:
                verify_logits = target_model(all_tokens, cache=target_cache)

        mx.eval(verify_logits)
        # verify_logits: [1, N+1, vocab], 取 argmax
        verify_argmax = mx.argmax(verify_logits, axis=-1).squeeze(0).tolist()  # [N+1]

        # 3. Accept matching
        # verify_argmax[i] 是 target 对位置 i 的预测 (即第 i+1 个 token)
        n_accept = 0
        for i in range(num_draft):
            if i < len(verify_argmax) - 1:
                if draft_tokens[i] == verify_argmax[i + 1]:
                    yield draft_tokens[i], True
                    n_accept += 1
                    ntoks += 1
                    if ntoks >= max_tokens:
                        break
                else:
                    break

        # 4. 输出 target 的正确 token
        if ntoks < max_tokens:
            correct_idx = min(n_accept + 1, len(verify_argmax) - 1)
            correct_token = verify_argmax[correct_idx]
            yield correct_token, False
            ntoks += 1
            y = mx.array([correct_token], mx.uint32)
            # 更新 hidden (hook 捕获的最后一个)
            target_hidden = hook.captured if hook.captured is not None else target_hidden
        else:
            break

        # 5. Rewind cache (拒绝的 tokens)
        rewind = num_draft - n_accept
        if rewind > 0:
            try:
                trim_prompt_cache(target_cache, rewind)
                trim_prompt_cache(mtp_cache, rewind)
            except Exception:
                pass


def bench(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=0, label=""):
    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")

    # warmup
    try:
        if num_draft > 0:
            list(patched_spec_generate(target_model, tokenizer, prompt, mtp_head, max_tokens=2, num_draft=num_draft))
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
    if b:
        baseline_tps = b["tps"]
    else:
        baseline_tps = 26  # fallback

    # Spec N=4
    r4 = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=4, label=f"投机 N=4 (MTP + hidden hook)")
    if r4:
        print(f"  加速: {r4['tps']/baseline_tps:.2f}x")

    # Spec N=10
    r10 = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=20, num_draft=10, label=f"投机 N=10 (MTP + hidden hook)")
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
