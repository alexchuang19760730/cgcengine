"""用 target 前 N 层作为 draft model — 同模型同分布, accept rate 高.

方案:
- target: Qwen3-VL-2B (28 层)
- draft: 同模型前 2 层 (2/28 = 7% 计算)
- forward: ~2.7ms (2/28 × 38ms)
- accept rate: 预期高 (同模型同分布)

mlx_lm 原生 draft_model 接口, 无需 patch.
"""
from __future__ import annotations

import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.cache_prompt import make_prompt_cache


def create_layer_draft(target_model, num_layers: int = 2):
    """从 target 创建前 N 层 draft model.

    复制 target 的 embed + 前 N 层 + lm_head, 作为一个独立模型.
    """
    # 获取 language_model
    lm = target_model.language_model

    # 保存原始 layers
    original_layers = lm.layers

    # 截断为前 N 层
    lm.layers = original_layers[:num_layers]

    print(f"[draft] Created {num_layers}-layer draft (from {len(original_layers)} layers)")
    print(f"[draft] Draft params: {sum(p.size for _, p in nn.utils.tree_flatten(lm.parameters())) / 1e6:.0f}M")

    return target_model  # 返回修改后的 model (前 N 层)


def benchmark_layer_draft(model_path: str, prompt: str, max_tokens: int = 20):
    """测试 layer draft 投机 decode."""
    print(f"\n{'='*60}")
    print(f"Layer Draft Benchmark")
    print(f"{'='*60}")

    # 1. 加载完整 target
    print("[1] Loading target model (28 layers)...")
    target_model, tokenizer = load(model_path)
    original_layers = target_model.language_model.layers
    print(f"    Target: {len(original_layers)} layers")

    # 2. Baseline (无投机, 28 层)
    print("\n[2] Baseline (28 layers, 无投机)...")
    t0 = time.time()
    tokens = []
    t_first = None
    for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=max_tokens):
        tokens.append(resp.token)
        if len(tokens) == 1:
            t_first = time.time()
    t_end = time.time()
    dt = t_end - t_first
    nd = len(tokens) - 1
    baseline_tps = nd / dt if dt > 0 else 0
    baseline_ttft = 1000 * (t_first - t0)
    print(f"    TTFT: {baseline_ttft:.0f}ms")
    print(f"    Decode: {baseline_tps:.1f} tok/s")
    print(f"    Output: {tokenizer.decode(tokens[:30])}")

    # 3. 测试不同层数的 draft
    for num_draft_layers in [1, 2, 4, 6]:
        print(f"\n[{3+num_draft_layers}] Draft = 前 {num_draft_layers} 层 + 投机 N=4...")

        # 重新加载 target (恢复完整 layers)
        target_model2, _ = load(model_path)

        # 创建 draft (前 N 层)
        draft_model, _ = load(model_path)
        draft_model.language_model.layers = draft_model.language_model.layers[:num_draft_layers]
        print(f"    Draft: {num_draft_layers} layers")

        # Benchmark
        t0 = time.time()
        tokens = []
        draft_count = 0
        total = 0
        t_first = None

        try:
            for resp in stream_generate(
                target_model2, tokenizer, prompt,
                max_tokens=max_tokens,
                draft_model=draft_model,
                num_draft_tokens=4,
            ):
                tokens.append(resp.token)
                total += 1
                if getattr(resp, "from_draft", False):
                    draft_count += 1
                if len(tokens) == 1:
                    t_first = time.time()
        except Exception as e:
            print(f"    Error: {e}")
            continue

        t_end = time.time()
        if t_first is None:
            t_first = t0
        dt = t_end - t_first
        nd = len(tokens) - 1
        if nd <= 0 or dt <= 0:
            print("    No tokens")
            continue

        tps = nd / dt
        ar = draft_count / total if total else 0
        boost = tps / baseline_tps if baseline_tps > 0 else 0

        print(f"    TTFT: {1000*(t_first-t0):.0f}ms")
        print(f"    Decode: {tps:.1f} tok/s ({boost:.1f}x)")
        print(f"    Accept: {ar:.1%} ({draft_count}/{total})")
        print(f"    Output: {tokenizer.decode(tokens[:30])}")

        # 释放
        del target_model2, draft_model
        mx.clear_cache()


if __name__ == "__main__":
    model_path = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"
    prompt = "Write a short story about a cat"

    benchmark_layer_draft(model_path, prompt, max_tokens=20)
