"""Chained MTP head 投机 decode 实测 — 验证 50.9% accept rate 在 Mac MLX 上是否成立.

复用 mtp_patched_v3 的 patched_spec_generate (target 真 hidden + 链式 MTP draft).
checkpoint: /tmp/mtp_head_chained.pt (50K 多卡训练, 训练 accept 50.9%)
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mtp_patched_v3 import load_mtp_head, patched_spec_generate, bench


def main():
    from mlx_lm import load

    checkpoint = "/tmp/mtp_head_chained_slim.pt"
    target_path = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"

    print(f"[1] Loading target model: {target_path}")
    target_model, tokenizer = load(target_path)

    print(f"[2] Loading chained MTP head: {checkpoint}")
    if not os.path.exists(checkpoint):
        print(f"  ERROR: checkpoint not found! Download first.")
        sys.exit(1)
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

        # Spec N=4 (跟训练 num_chain=4 一致)
        r4 = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=30, num_draft=4, label="投机 N=4 (chained MTP)")
        if r4:
            print(f"  加速: {r4['tps']/baseline_tps:.2f}x")

        # Spec N=10
        r10 = bench(target_model, tokenizer, mtp_head, prompt, max_tokens=30, num_draft=10, label="投机 N=10 (chained MTP)")
        if r10:
            print(f"  加速: {r10['tps']/baseline_tps:.2f}x")

        all_results[prompt[:30]] = {
            "baseline": baseline_tps,
            "n4": r4,
            "n10": r10,
        }

    # 汇总
    print(f"\n{'='*60}")
    print("汇总 (chained MTP 50K 多卡训练)")
    print(f"{'='*60}")
    for prompt, res in all_results.items():
        print(f"\n  Prompt: {prompt}...")
        print(f"    Baseline: {res['baseline']:.1f} tok/s")
        if res["n4"]:
            print(f"    N=4:  {res['n4']['tps']:.1f} tok/s ({res['n4']['tps']/res['baseline']:.2f}x, accept {res['n4']['ar']:.0%})")
        if res["n10"]:
            print(f"    N=10: {res['n10']['tps']:.1f} tok/s ({res['n10']['tps']/res['baseline']:.2f}x, accept {res['n10']['ar']:.0%})")


if __name__ == "__main__":
    main()
