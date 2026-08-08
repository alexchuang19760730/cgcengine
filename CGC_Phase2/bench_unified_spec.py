"""统一投机 decode bench — chain | eagle 模式切换 + backend option.

用法:
  python bench_unified_spec.py --mode eagle --backend mlx
  python bench_unified_spec.py --mode chain --backend mlx
  python bench_unified_spec.py --mode both --backend mlx  # 对比两种

Backend (当前只实现 mlx):
  mlx: Mac MLX (Qwen3-VL-2B target + 0.5B 4bit draft)
  pytorch: GPU PyTorch (待实现)
  sglang: sglang server (待实现)
"""
from __future__ import annotations

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_models_mlx():
    """加载 MLX 后端的 target + draft 模型."""
    from mlx_lm import load
    target_model, tokenizer = load("/Users/alexchuang/models/Qwen3-VL-2B-bf16")
    draft_model, _ = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    return target_model, tokenizer, draft_model


def bench_chain_mlx(target_model, tokenizer, draft_model, prompt, max_tokens=30, N=8):
    """Chain 模式: mlx_lm 原生 speculative (链式 draft)."""
    from mlx_lm import stream_generate

    # warmup
    try:
        list(stream_generate(target_model, tokenizer, prompt, max_tokens=3,
                             draft_model=draft_model, num_draft_tokens=N))
    except Exception:
        pass

    t0 = time.time()
    tokens = []
    draft_count = 0
    total = 0
    t_first = None

    for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=max_tokens,
                                draft_model=draft_model, num_draft_tokens=N):
        tokens.append(resp.token)
        total += 1
        if getattr(resp, "from_draft", False):
            draft_count += 1
        if len(tokens) == 1:
            t_first = time.time()

    t_end = time.time()
    if t_first is None:
        t_first = t0
    dt = t_end - t_first
    nd = len(tokens) - 1
    if nd <= 0 or dt <= 0:
        return None
    return {"tps": nd/dt, "ar": draft_count/total if total else 0, "tokens": len(tokens)}


def bench_eagle_mlx(target_model, tokenizer, draft_model, prompt, max_tokens=30, top_k=4, tree_depth=2):
    """Eagle 模式: EAGLE tree search."""
    from eagle_tree_search import bench_eagle
    return bench_eagle(target_model, tokenizer, draft_model, prompt,
                       max_tokens=max_tokens, top_k=top_k, tree_depth=tree_depth,
                       label=f"EAGLE (k={top_k}, d={tree_depth})")


def bench_baseline_mlx(target_model, tokenizer, prompt, max_tokens=30):
    """Baseline: 无投机."""
    from mlx_lm import stream_generate
    try:
        list(stream_generate(target_model, tokenizer, prompt, max_tokens=1))
    except Exception:
        pass
    t0 = time.time()
    tokens = []
    t_first = None
    for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=max_tokens):
        tokens.append(resp.token)
        if len(tokens) == 1:
            t_first = time.time()
    t_end = time.time()
    if t_first is None:
        t_first = t0
    dt = t_end - t_first
    nd = len(tokens) - 1
    if nd <= 0 or dt <= 0:
        return None
    return {"tps": nd/dt, "ar": 0, "tokens": len(tokens)}


def main():
    parser = argparse.ArgumentParser(description="统一投机 decode bench")
    parser.add_argument("--mode", default="both", choices=["chain", "eagle", "both", "baseline"],
                        help="chain=链式draft, eagle=tree search, both=对比, baseline=无投机")
    parser.add_argument("--backend", default="mlx", choices=["mlx", "pytorch", "sglang"],
                        help="后端 (当前只实现 mlx)")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=4, help="EAGLE top-k 候选数")
    parser.add_argument("--tree-depth", type=int, default=2, help="EAGLE tree 深度")
    parser.add_argument("--chain-n", type=int, default=16, help="chain 模式 num_draft_tokens (N=16 最优)")
    args = parser.parse_args()

    print(f"[config] mode={args.mode}, backend={args.backend}")
    if args.backend != "mlx":
        print(f"  backend {args.backend} 待实现, 使用 mlx")
    print(f"  chain_n={args.chain_n}, eagle top_k={args.top_k}, tree_depth={args.tree_depth}")

    # 加载模型
    print("\n[1] Loading models...")
    target_model, tokenizer, draft_model = load_models_mlx()
    print("  target: Qwen3-VL-2B-bf16")
    print("  draft: Qwen2.5-0.5B-Instruct-4bit")

    prompts = [
        "Write a short story about a cat",
        "Explain how photosynthesis works in simple terms",
        "What are the benefits of exercise?",
    ]

    all_results = {}

    for prompt in prompts:
        print(f"\n{'#'*60}")
        print(f"# {prompt[:50]}")
        print(f"{'#'*60}")

        results = {}

        # Baseline
        if args.mode in ("both", "baseline"):
            b = bench_baseline_mlx(target_model, tokenizer, prompt, args.max_tokens)
            if b:
                print(f"  Baseline: {b['tps']:.1f} tok/s")
                results["baseline"] = b

        # Chain
        if args.mode in ("both", "chain"):
            print(f"\n--- Chain (N={args.chain_n}) ---")
            c = bench_chain_mlx(target_model, tokenizer, draft_model, prompt,
                                args.max_tokens, N=args.chain_n)
            if c:
                baseline_tps = results.get("baseline", {}).get("tps", 26)
                print(f"  Decode: {c['tps']:.1f} tok/s, Accept: {c['ar']:.0%}, 加速: {c['tps']/baseline_tps:.2f}x")
                results["chain"] = c

        # Eagle
        if args.mode in ("both", "eagle"):
            print(f"\n--- Eagle (k={args.top_k}, d={args.tree_depth}) ---")
            e = bench_eagle_mlx(target_model, tokenizer, draft_model, prompt,
                                args.max_tokens, args.top_k, args.tree_depth)
            if e:
                baseline_tps = results.get("baseline", {}).get("tps", 26)
                print(f"  加速: {e['tps']/baseline_tps:.2f}x")
                results["eagle"] = e

        all_results[prompt[:30]] = results

    # 汇总
    print(f"\n{'='*60}")
    print(f"汇总 (mode={args.mode}, backend={args.backend})")
    print(f"{'='*60}")
    for prompt, res in all_results.items():
        print(f"\n  {prompt}...")
        baseline = res.get("baseline", {}).get("tps", 26)
        if "baseline" in res:
            print(f"    Baseline: {res['baseline']['tps']:.1f} tok/s")
        if "chain" in res:
            c = res["chain"]
            print(f"    Chain:    {c['tps']:.1f} tok/s ({c['tps']/baseline:.2f}x, accept {c['ar']:.0%})")
        if "eagle" in res:
            e = res["eagle"]
            print(f"    Eagle:    {e['tps']:.1f} tok/s ({e['tps']/baseline:.2f}x, accept {e['ar']:.0%})")


if __name__ == "__main__":
    main()
