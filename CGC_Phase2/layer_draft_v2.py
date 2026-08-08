"""Layer draft 投机 decode — 高效版 (只加载一次模型).

方案:
- target: Qwen3-VL-2B (28 层) 完整
- draft: 同模型前 N 层 (浅拷贝, 共享 embed/lm_head)
- mlx_lm 原生 draft_model 接口
"""
from __future__ import annotations

import copy
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate


def make_draft(target_model, num_layers: int):
    """从 target 浅拷贝创建前 N 层 draft (共享 embed/lm_head 权重)."""
    draft = copy.copy(target_model)
    draft.language_model = copy.copy(target_model.language_model)
    draft.language_model.model = copy.copy(target_model.language_model.model)
    # 只保留前 N 层 (引用, 不复制权重)
    draft.language_model.model.layers = target_model.language_model.model.layers[:num_layers]
    return draft


def bench(model, tokenizer, prompt, max_tokens, num_draft=0, draft_model=None, label=""):
    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")

    # warmup
    try:
        kwargs = {"max_tokens": 1}
        if draft_model is not None and num_draft > 0:
            kwargs["draft_model"] = draft_model
            kwargs["num_draft_tokens"] = num_draft
        list(stream_generate(model, tokenizer, prompt, **kwargs))
    except Exception as e:
        print(f"  warmup error: {e}")
        return None

    t0 = time.time()
    tokens = []
    draft_count = 0
    total = 0
    t_first = None

    try:
        kwargs = {"max_tokens": max_tokens}
        if draft_model is not None and num_draft > 0:
            kwargs["draft_model"] = draft_model
            kwargs["num_draft_tokens"] = num_draft
        for resp in stream_generate(model, tokenizer, prompt, **kwargs):
            tokens.append(resp.token)
            total += 1
            if getattr(resp, "from_draft", False):
                draft_count += 1
            if len(tokens) == 1:
                t_first = time.time()
    except Exception as e:
        print(f"  error: {e}")
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
    if draft_model is not None:
        print(f"  Accept: {ar:.1%} ({draft_count}/{total})")
    print(f"  Output: {tokenizer.decode(tokens[:30])}")
    return {"ttft": 1000*(t_first-t0), "tps": tps, "ar": ar}


if __name__ == "__main__":
    model_path = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"
    prompt = "Write a short story about a cat"

    print("[load] Loading target model...")
    target, tokenizer = load(model_path)
    total_layers = len(target.language_model.model.layers)
    print(f"  {total_layers} layers")

    # Baseline
    baseline = bench(target, tokenizer, prompt, max_tokens=20, label="Baseline (28 层, 无投机)")

    if baseline is None:
        print("Baseline failed, exit")
        exit(1)

    # 测试不同 draft 层数
    results = {}
    for n_layers in [1, 2, 4]:
        print(f"\n[load] Creating {n_layers}-layer draft...")
        draft = make_draft(target, n_layers)

        result = bench(
            target, tokenizer, prompt, max_tokens=20,
            num_draft=4, draft_model=draft,
            label=f"Draft {n_layers} 层 + 投机 N=4"
        )
        if result:
            result["boost"] = result["tps"] / baseline["tps"]
            print(f"  加速: {result['boost']:.2f}x")
            results[n_layers] = result

        del draft
        mx.clear_cache()

    # 汇总
    print(f"\n{'='*55}")
    print("汇总")
    print(f"{'='*55}")
    print(f"Baseline: {baseline['tps']:.1f} tok/s")
    for n, r in results.items():
        print(f"Draft {n} 层: {r['tps']:.1f} tok/s ({r['boost']:.2f}x, accept {r['ar']:.0%})")
