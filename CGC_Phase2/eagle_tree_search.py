"""EAGLE tree search 投机 decode — top-k tree draft + target verify.

简化版 EAGLE (深度 2, flat verify):
  1. Draft forward → top-k 候选
  2. 对每个候选, draft forward → next (depth 2)
  3. Target flat verify (一次 forward 所有 tree 节点)
  4. 找最长匹配路径

对比标准 speculative (链式 draft):
  - 链式: draft 1 条路径 N 步, accept 取决于 argmax 匹配
  - EAGLE: draft k 条路径 (top-k), accept 取决于 top-k 包含 target
"""
from __future__ import annotations

import time
from typing import Generator, Tuple, Optional
import mlx.core as mx
import mlx.nn as nn


def eagle_tree_generate(
    target_model,
    tokenizer,
    draft_model,
    prompt: str,
    max_tokens: int = 30,
    top_k: int = 4,
    tree_depth: int = 2,
) -> Generator[Tuple[int, bool], None, None]:
    """EAGLE tree search speculative decode.

    Args:
        target_model: target LLM (Qwen3-VL-2B)
        draft_model: draft LLM (0.5B 4bit)
        top_k: 每个 position 的候选数
        tree_depth: tree 深度 (1=只 top-k, 2=top-k + 每候选 1 步)
    """
    from mlx_lm.cache_prompt import make_prompt_cache

    # Tokenize
    if isinstance(prompt, str):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [t for t in tokens if t not in (151644, 151645)]
    else:
        tokens = prompt
    y = mx.array(tokens, mx.uint32)

    # KV caches
    target_cache = make_prompt_cache(target_model)
    draft_cache = make_prompt_cache(draft_model)

    # Prefill both models
    target_logits = target_model(y[None], cache=target_cache)
    draft_model(y[None], cache=draft_cache)

    # First token (from target)
    first_token = mx.argmax(target_logits[0, -1])
    mx.eval(first_token)
    first_token_id = int(first_token.item())
    yield first_token_id, False

    current_token = mx.array([[first_token_id]], mx.uint32)  # [1, 1] 2D
    ntoks = 1

    while ntoks < max_tokens:
        # === 1. Draft forward current_token → top-k 候选 ===
        draft_logits = draft_model(current_token, cache=draft_cache)
        draft_last = draft_logits[0, -1]  # [vocab]
        top_k_tokens = mx.argsort(-draft_last)[:top_k]  # [k]
        mx.eval(top_k_tokens)
        top_k_list = [int(t.item()) for t in top_k_tokens]

        if tree_depth == 1:
            # === 深度 1: 只 top-k, target verify 1 token ===
            target_logits = target_model(current_token, cache=target_cache)
            target_next = int(mx.argmax(target_logits[0, -1]).item())

            if target_next in top_k_list:
                yield target_next, True
            else:
                yield target_next, False

            current_token = mx.array([[target_next]], mx.uint32)
            ntoks += 1

        else:
            # === 深度 2: top-k + 每候选 1 步链式 ===
            # 构造 tree: k 条路径, 每条 2 token [ci, ci_next]
            # 简化: 用 draft cache forward 每个 ci (会污染 cache, 但后续 trim)
            chains = []
            for k_idx in range(top_k):
                ci = top_k_tokens[k_idx]
                ci_val = int(ci.item())
                ci_arr = mx.array([[ci_val]], mx.uint32)
                # draft forward ci → ci_next (argmax)
                ci_logits = draft_model(ci_arr, cache=draft_cache)
                ci_next = mx.argmax(ci_logits[0, -1])
                mx.eval(ci_next)
                chains.append([ci_val, int(ci_next.item())])

            # === 2. Target flat verify ===
            # forward [current_token, c1, c1_next, c2, c2_next, ...]
            # 简化: flat forward (非 tree attention, 候选互相可见)
            verify_tokens = [int(current_token[0, 0].item())]
            for chain in chains:
                verify_tokens.extend(chain)

            verify_arr = mx.array([verify_tokens], mx.uint32)  # [1, 1+2k]
            verify_logits = target_model(verify_arr, cache=target_cache)
            verify_argmax = mx.argmax(verify_logits[0], axis=-1)  # [1+2k]
            mx.eval(verify_argmax)
            verify_list = [int(t) for t in verify_argmax]

            # verify_list[0] = target 对 current_token 的 next 预测
            # verify_list[1] = target 对 c1 的 next 预测
            # verify_list[2] = target 对 c1_next 的 next 预测
            # ...
            target_next = verify_list[0]

            # === 3. 找最长匹配路径 ===
            # chains[i] = [ci, ci_next]
            # 验证: ci == target_next? (target 对 current_token 的预测)
            # 如果 ci == target_next, 进一步验证 ci_next == verify_list[1+2i] (target 对 ci 的预测)

            best_chain = None
            best_len = 0
            matched_i = -1
            for i, chain in enumerate(chains):
                ci, ci_next = chain
                if ci == target_next:
                    matched_i = i
                    # 第一步匹配! 检查第二步
                    target_ci_next = verify_list[1 + 2 * i]  # target 对 ci 的 next 预测
                    if ci_next == target_ci_next:
                        # 两步都匹配!
                        best_len = 2
                        best_chain = chain
                    else:
                        best_len = 1
                        best_chain = [ci, target_ci_next]  # 用 target 的 ci_next
                    break  # 只接受第一个匹配的候选

            if best_chain and best_len > 0:
                # Accept!
                for j in range(best_len):
                    yield best_chain[j], True
                    ntoks += 1
                    if ntoks >= max_tokens:
                        break

                # next token = target 对最后 accept token 的预测
                # verify_list[1 + 2*matched_i + best_len - 1]
                next_idx = 1 + 2 * matched_i + best_len - 1
                if next_idx < len(verify_list):
                    next_t = verify_list[next_idx]
                else:
                    next_t = best_chain[-1]
                current_token = mx.array([[next_t]], mx.uint32)
            else:
                # Reject, 用 target_next
                yield target_next, False
                ntoks += 1
                current_token = mx.array([[target_next]], mx.uint32)

            # === 4. Trim caches ===
            # target verify forward 了 1+2k tokens, 只保留 1+best_len
            # draft forward 了 k tokens (每个候选 1 步), 需要全部 trim
            rewind_target = (1 + 2 * top_k) - (1 + best_len)
            if rewind_target > 0:
                try:
                    from mlx_lm.models.cache import trim_prompt_cache
                    trim_prompt_cache(target_cache, rewind_target)
                except Exception:
                    pass

            # draft cache: forward 了 k candidates, 需要 trim k
            try:
                from mlx_lm.models.cache import trim_prompt_cache
                trim_prompt_cache(draft_cache, top_k)
            except Exception:
                pass


def bench_eagle(target_model, tokenizer, draft_model, prompt, max_tokens=30,
                 top_k=4, tree_depth=2, label=""):
    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")

    # Warmup
    try:
        list(eagle_tree_generate(target_model, tokenizer, draft_model, prompt,
                                  max_tokens=3, top_k=top_k, tree_depth=tree_depth))
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
        gen = eagle_tree_generate(target_model, tokenizer, draft_model, prompt,
                                   max_tokens=max_tokens, top_k=top_k, tree_depth=tree_depth)
        for token_id, from_draft in gen:
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
