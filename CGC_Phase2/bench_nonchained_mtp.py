"""Non-chained MTP 投机 decode 实测 — 验证泛化 accept 和 chain degradation.

Non-chained MTP 泛化好 (新 prompt single-token 33-67%), 但链式退化严重.
N=1 (不链式) 预期 accept ~50%, N>1 预期退化.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mtp_patched_v3 import load_mtp_head, patched_spec_generate, bench


def main():
    from mlx_lm import load

    checkpoint = "/tmp/mtp_head_final.pt"
    target_path = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"

    print(f"[1] Loading target model: {target_path}")
    target_model, tokenizer = load(target_path)

    print(f"[2] Loading non-chained MTP head: {checkpoint}")
    mtp_head = load_mtp_head(checkpoint)
    print(f"  MTP head loaded")

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

        # Baseline
        b = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=30, num_draft=0, label="Baseline (无投机)")
        baseline_tps = b["tps"] if b else 26

        results = {"baseline": baseline_tps}

        # N=1 (不链式, 无 chain degradation)
        r1 = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=30, num_draft=1, label="投机 N=1 (不链式)")
        if r1:
            print(f"  加速: {r1['tps']/baseline_tps:.2f}x")
            results["n1"] = r1

        # N=2 (链式 1 步)
        r2 = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=30, num_draft=2, label="投机 N=2 (链式 1 步)")
        if r2:
            print(f"  加速: {r2['tps']/baseline_tps:.2f}x")
            results["n2"] = r2

        # N=4 (链式 3 步, 看 degradation)
        r4 = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=30, num_draft=4, label="投机 N=4 (链式 3 步)")
        if r4:
            print(f"  加速: {r4['tps']/baseline_tps:.2f}x")
            results["n4"] = r4

        all_results[prompt[:30]] = results

    # 汇总
    print(f"\n{'='*60}")
    print("汇总 (non-chained MTP)")
    print(f"{'='*60}")
    for prompt, res in all_results.items():
        print(f"\n  Prompt: {prompt}...")
        print(f"    Baseline: {res['baseline']:.1f} tok/s")
        for key, label in [("n1", "N=1"), ("n2", "N=2"), ("n4", "N=4")]:
            if key in res:
                r = res[key]
                print(f"    {label}: {r['tps']:.1f} tok/s ({r['tps']/res['baseline']:.2f}x, accept {r['ar']:.0%})")


if __name__ == "__main__":
    main()
