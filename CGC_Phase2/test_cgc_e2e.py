"""End-to-end verify loop test: CGC IR Dispatcher vs MLX vs PyTorch CPU.

Tests the full verify loop with llama.cpp + MTP head, comparing three backends:
1. CGC IR Dispatcher (IR-driven Metal execution)
2. MLX Metal (direct MLX forward)
3. PyTorch CPU (reference)

Measures: accept rate, TPS, draft time, verify time.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "/Users/alexchuang/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf"
V2_CKPT = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_output/qwen25_llamacpp_v2/mtp_head_qwen25-0.5b_decode.pt"
EMBED = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_train_data/qwen25_llamacpp/embed_head.pt"

PROMPTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The capital of France is",
    "import numpy as np\n\nx = np.array([1, 2, 3])\nprint(x.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if",
]


def run_verify_loop(backend: str, num_draft: int = 4, max_tokens: int = 40):
    """Run verify loop with specified backend."""
    from mtp_verify_loop import MTPVerifyLoop

    use_mlx = (backend == "mlx")
    use_cgc_ir = (backend == "cgc_ir")

    print(f"\n{'='*60}")
    print(f"Backend: {backend.upper()} | nd={num_draft} | max_tokens={max_tokens}")
    print(f"{'='*60}")

    loop = MTPVerifyLoop(
        model_path=MODEL,
        mtp_checkpoint=V2_CKPT,
        hidden_size=896,
        vocab_size=151936,
        num_heads=14,
        head_dim=64,
        intermediate_size=4864,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
        use_ngram_fallback=True,
        embed_head_path=EMBED,
        use_mlx=use_mlx,
        use_cgc_ir=use_cgc_ir,
    )

    results = []
    for prompt in PROMPTS:
        print(f"\n  Prompt: {prompt[:60]}...")
        loop.prefill(prompt)

        tokens = []
        t0 = time.time()
        for token_id, from_draft in loop.generate(max_tokens=max_tokens, num_draft=num_draft):
            tokens.append(token_id)
        elapsed = time.time() - t0

        stats = loop.stats
        results.append({
            "prompt": prompt[:40],
            "tokens": len(tokens),
            "accept_rate": stats.accept_rate,
            "avg_accept_len": stats.avg_accept_len,
            "tps": stats.tps,
            "draft_ms": stats.draft_ms_total / max(stats.total_rounds, 1),
            "verify_ms": stats.verify_ms_total / max(stats.total_rounds, 1),
            "elapsed": elapsed,
        })

        print(f"  Tokens: {len(tokens)}, Accept: {stats.accept_rate:.1%}, "
              f"AvgAccept: {stats.avg_accept_len:.2f}, TPS: {stats.tps:.1f}")

    # Summary
    print(f"\n  --- {backend.upper()} Summary ---")
    avg_accept = sum(r["accept_rate"] for r in results) / len(results)
    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_draft = sum(r["draft_ms"] for r in results) / len(results)
    avg_verify = sum(r["verify_ms"] for r in results) / len(results)
    print(f"  Avg accept rate: {avg_accept:.1%}")
    print(f"  Avg TPS: {avg_tps:.1f}")
    print(f"  Avg draft time: {avg_draft:.1f}ms/round")
    print(f"  Avg verify time: {avg_verify:.1f}ms/round")

    # Cleanup
    del loop
    import gc
    gc.collect()

    return {"backend": backend, "accept_rate": avg_accept, "tps": avg_tps,
            "draft_ms": avg_draft, "verify_ms": avg_verify}


def main():
    print("CGC IR Dispatcher End-to-End Verify Loop Test")
    print(f"Model: {MODEL}")
    print(f"Checkpoint: {V2_CKPT}")
    print(f"Prompts: {len(PROMPTS)}")

    all_results = []

    # Test 1: CGC IR Dispatcher
    try:
        r = run_verify_loop("cgc_ir", num_draft=4, max_tokens=40)
        all_results.append(r)
    except Exception as e:
        print(f"  CGC IR failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: MLX Metal
    try:
        r = run_verify_loop("mlx", num_draft=4, max_tokens=40)
        all_results.append(r)
    except Exception as e:
        print(f"  MLX failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: PyTorch CPU (baseline)
    try:
        r = run_verify_loop("pytorch", num_draft=4, max_tokens=40)
        all_results.append(r)
    except Exception as e:
        print(f"  PyTorch failed: {e}")
        import traceback
        traceback.print_exc()

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Backend':<12} {'Accept':>8} {'TPS':>8} {'Draft(ms)':>10} {'Verify(ms)':>11}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['backend']:<12} {r['accept_rate']:>7.1%} {r['tps']:>8.1f} "
              f"{r['draft_ms']:>10.1f} {r['verify_ms']:>11.1f}")


if __name__ == "__main__":
    main()
