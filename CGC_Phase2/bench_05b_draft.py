"""0.5B 4bit draft + Qwen3-VL-2B target 投机 decode bench.

用 mlx_lm 原生 speculative_generate (draft_model 参数).
不依赖 target hidden, 避免 prefill/decode/extend 不一致问题.
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from mlx_lm import load, stream_generate

    target_path = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"
    draft_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    print(f"[1] Loading target: {target_path}")
    target_model, tokenizer = load(target_path)
    print(f"  target vocab: {getattr(target_model, 'vocab_size', '?')}")

    print(f"[2] Loading draft (4bit): {draft_path}")
    try:
        draft_model, draft_tokenizer = load(draft_path)
        print(f"  draft loaded, vocab: {getattr(draft_model, 'vocab_size', '?')}")
    except Exception as e:
        print(f"  draft load failed: {e}")
        print("  trying alternative: mlx-community/Qwen2.5-0.5B-Instruct-8bit")
        try:
            draft_model, draft_tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-8bit")
            print(f"  8bit draft loaded")
        except Exception as e2:
            print(f"  8bit also failed: {e2}")
            return

    prompts = [
        "Write a short story about a cat",
        "Explain how photosynthesis works in simple terms",
        "What are the benefits of exercise?",
    ]

    all_results = {}

    for prompt in prompts:
        print(f"\n{'#'*60}")
        print(f"# Prompt: {prompt[:50]}")
        print(f"{'#'*60}")

        # Baseline (无投机)
        print(f"\n--- Baseline ---")
        try:
            list(stream_generate(target_model, tokenizer, prompt, max_tokens=1))  # warmup
        except Exception as e:
            print(f"  warmup error: {e}")

        t0 = time.time()
        tokens = []
        t_first = None
        try:
            for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=30):
                tokens.append(resp.token)
                if len(tokens) == 1:
                    t_first = time.time()
        except Exception as e:
            print(f"  error: {e}")

        t_end = time.time()
        if t_first and len(tokens) > 1:
            dt = t_end - t_first
            nd = len(tokens) - 1
            baseline_tps = nd / dt
            print(f"  TTFT: {1000*(t_first-t0):.0f}ms, Decode: {baseline_tps:.1f} tok/s")
        else:
            baseline_tps = 26
            print(f"  no tokens, using default baseline {baseline_tps}")

        # Speculative decode (N=4)
        for N in [2, 4, 8]:
            print(f"\n--- Speculative N={N} ---")
            try:
                list(stream_generate(target_model, tokenizer, prompt, max_tokens=3,
                                     draft_model=draft_model, num_draft_tokens=N))  # warmup
            except Exception as e:
                print(f"  warmup error: {e}")
                continue

            t0 = time.time()
            tokens = []
            draft_count = 0
            total = 0
            t_first = None
            try:
                for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=30,
                                             draft_model=draft_model, num_draft_tokens=N):
                    tokens.append(resp.token)
                    total += 1
                    if hasattr(resp, "from_draft") and resp.from_draft:
                        draft_count += 1
                    if len(tokens) == 1:
                        t_first = time.time()
            except Exception as e:
                print(f"  error: {e}")
                continue

            t_end = time.time()
            if t_first and len(tokens) > 1:
                dt = t_end - t_first
                nd = len(tokens) - 1
                tps = nd / dt
                ar = draft_count / total if total else 0
                print(f"  TTFT: {1000*(t_first-t0):.0f}ms, Decode: {tps:.1f} tok/s, Accept: {ar:.0%} ({draft_count}/{total})")
                print(f"  加速: {tps/baseline_tps:.2f}x")
                print(f"  Output: {tokenizer.decode(tokens[:20])}")

                if prompt[:30] not in all_results:
                    all_results[prompt[:30]] = {"baseline": baseline_tps}
                all_results[prompt[:30]][f"n{N}"] = {"tps": tps, "ar": ar}

    # 汇总
    print(f"\n{'='*60}")
    print("汇总 (0.5B 4bit draft)")
    print(f"{'='*60}")
    for prompt, res in all_results.items():
        print(f"\n  Prompt: {prompt}...")
        print(f"    Baseline: {res['baseline']:.1f} tok/s")
        for key in ["n2", "n4", "n8"]:
            if key in res:
                r = res[key]
                print(f"    {key.upper()}: {r['tps']:.1f} tok/s ({r['tps']/res['baseline']:.2f}x, accept {r['ar']:.0%})")


if __name__ == "__main__":
    main()
