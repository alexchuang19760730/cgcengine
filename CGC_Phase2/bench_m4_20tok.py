#!/usr/bin/env python3
"""M4 16GB 量化基线 tok/s 测量 — gemma4-int2 / qwen36-2bit.

目标: 确认两个模型在 stock mlx_lm 下能跑多远 (20 tok/s 目标的基线),
      以及 16GB 内存里权重 + KV cache 的实际占用。

用法:
    python3.13 bench_m4_20tok.py --model gemma4
    python3.13 bench_m4_20tok.py --model qwen36
    python3.13 bench_m4_20tok.py --model both

指标:
    - load time / 权重 nbytes / peak memory
    - TTFT (prefill + 首 token)
    - decode tok/s (首 token 之后)
    - 2-bit 量化下生成的前 40 tokens (抽样检查输出是否可用)
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

MODELS = {
    "gemma4": "/Users/alexchuang/Documents/flashkv0516/models/mlx/gemma4-int2",
    "qwen36": "/Users/alexchuang/Documents/flashkv0516/models/mlx/qwen36-2bit",
}

PROMPTS = {
    "gemma4": "Explain what recursion is in computer science, step by step.",
    "qwen36": "Write a Python function that computes fibonacci numbers with memoization.",
}

DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog. Continue:"


def human(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f}GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f}MB"
    return f"{n / (1 << 10):.1f}KB"


def bench_one(name: str, path: str, max_tokens: int = 64, prompt: str | None = None,
              n_trials: int = 2) -> dict:
    print(f"\n{'=' * 64}\n[{name}] loading {path}", flush=True)
    t0 = time.monotonic()
    model, tokenizer = load(path)
    t_load = time.monotonic() - t0

    weights_bytes = int(model.nbytes) if hasattr(model, "nbytes") else 0
    active_mem = mx.get_active_memory()
    print(f"[{name}] loaded in {t_load:.1f}s | weights={human(weights_bytes)} "
          f"| active mem={human(active_mem)}", flush=True)

    p = prompt or PROMPTS.get(name, DEFAULT_PROMPT)
    input_ids = tokenizer.encode(p)
    if len(input_ids) > 128:
        input_ids = input_ids[:128]
    prompt_tokens = len(input_ids)
    print(f"[{name}] prompt={prompt_tokens} tokens | max_tokens={max_tokens}", flush=True)

    ttfts: list[float] = []
    decode_rates: list[float] = []
    sample_out = ""

    for trial in range(n_trials):
        t_first = None
        t_last = None
        n_gen = 0
        out_text = []
        t_start = time.monotonic()
        try:
            for resp in stream_generate(model, tokenizer, mx.array(input_ids),
                                        max_tokens=max_tokens):
                out_text.append(resp.text)
                n_gen += 1
                if n_gen == 1:
                    t_first = time.monotonic()
                t_last = time.monotonic()
        except Exception as e:
            print(f"[{name}] trial {trial} FAILED: {e}", flush=True)
            continue

        if t_first is None or t_last is None or n_gen <= 1:
            continue
        ttft = t_first - t_start
        dec = (t_last - t_first) / (n_gen - 1)
        rate = (n_gen - 1) / (t_last - t_first) if (t_last - t_first) > 0 else 0.0
        ttfts.append(ttft)
        decode_rates.append(rate)
        print(f"[{name}] trial {trial}: TTFT={ttft * 1000:.0f}ms | "
              f"decode={rate:.1f} tok/s ({n_gen - 1} tokens, "
              f"{dec * 1000:.1f} ms/tok)", flush=True)
        if trial == 0:
            sample_out = "".join(out_text)[:400]

    peak = mx.get_peak_memory()
    print(f"[{name}] peak memory={human(peak)}", flush=True)

    result = {
        "model": name,
        "load_s": round(t_load, 1),
        "weights": human(weights_bytes),
        "active_mem": human(active_mem),
        "peak_mem": human(peak),
        "prompt_tokens": prompt_tokens,
        "ttft_ms": round(min(ttfts) * 1000) if ttfts else None,
        "decode_tps": round(max(decode_rates), 1) if decode_rates else None,
        "sample": sample_out,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gemma4", "qwen36", "both"], default="both")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--trials", type=int, default=2)
    args = ap.parse_args()

    targets = ["gemma4", "qwen36"] if args.model == "both" else [args.model]
    results = {}
    for name in targets:
        results[name] = bench_one(name, MODELS[name], args.max_tokens,
                                  args.prompt, args.trials)

    print(f"\n{'=' * 64}\nSUMMARY (best of {args.trials}):")
    for name, r in results.items():
        print(f"  {name:8s} TTFT={r['ttft_ms']}ms decode={r['decode_tps']} tok/s "
              f"peak={r['peak_mem']}")


if __name__ == "__main__":
    main()
