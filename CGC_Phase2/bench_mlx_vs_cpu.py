#!/usr/bin/env python3
"""Compare verify loop performance: PyTorch CPU vs MLX Metal."""
import os, sys, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [REPO_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mtp_verify_loop import MTPVerifyLoop

MODEL = "/Users/alexchuang/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf"
CKPT = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_output/qwen25_llamacpp/mtp_head_qwen25-0.5b_decode.pt"
EMBED = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_train_data/qwen25_llamacpp/embed_head.pt"

PROMPTS = [
    "Write a Python function to check if a number is prime:",
    "def binary_search(arr, target):\n    ",
    "class LinkedList:\n    def __init__(self):\n        ",
    "import numpy as np\n\ndef matrix_multiply(a, b):\n    ",
    "async def fetch_data(url):\n    ",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ",
    "What is the time complexity of quicksort?",
    "Explain the difference between TCP and UDP.",
    "The main advantage of using a linked list over an array is",
    "In object-oriented programming, encapsulation means",
]

def run_bench(label, use_mlx, num_drafts=[1, 4]):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    loop = MTPVerifyLoop(
        model_path=MODEL,
        mtp_checkpoint=CKPT,
        hidden_size=896, vocab_size=151936,
        num_heads=14, head_dim=64, intermediate_size=4864,
        n_gpu_layers=-1, n_ctx=2048, verbose=False,
        use_ngram_fallback=True, embed_head_path=EMBED,
        use_mlx=use_mlx,
    )

    for nd in num_drafts:
        total_accept = 0
        total_draft = 0
        total_tokens = 0
        total_time = 0
        total_rounds = 0
        total_draft_ms = 0

        for prompt in PROMPTS:
            try:
                result = loop.bench(prompt, max_tokens=30, num_draft=nd, label=f"nd={nd}")
                if result:
                    s = result["stats"]
                    total_accept += s.accepted_tokens
                    total_draft += s.draft_tokens
                    total_tokens += s.total_tokens
                    total_rounds += s.total_rounds
                    total_time += (s.prefill_ms + s.draft_ms_total + s.verify_ms_total) / 1000
                    total_draft_ms += s.draft_ms_total
            except Exception as e:
                print(f"  Error: {e}")

        if total_draft > 0:
            avg_accept = total_accept / total_draft
            avg_tps = total_tokens / max(total_time, 0.001)
            avg_draft_per_round = total_draft_ms / max(total_rounds, 1)
            print(f"  nd={nd}: accept={avg_accept:.1%} ({total_accept}/{total_draft}), "
                  f"tps={avg_tps:.1f}, avg_draft_time={avg_draft_per_round:.1f}ms/round")

    # Baseline
    baseline_tps = []
    for prompt in PROMPTS[:5]:
        t0 = time.time()
        loop.llm.create_completion(prompt, max_tokens=30, temperature=0, top_p=1)
        dt = time.time() - t0
        baseline_tps.append(30 / dt)
    print(f"  Baseline: {np.mean(baseline_tps):.1f} tok/s")

    del loop

def main():
    # Test with MLX Metal
    run_bench("MLX Metal GPU", use_mlx=True)

    # Test with PyTorch CPU (for comparison)
    run_bench("PyTorch CPU", use_mlx=False)

if __name__ == "__main__":
    main()
