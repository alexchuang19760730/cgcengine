#!/usr/bin/env python3
"""
llama.cpp Benchmark (Local Mac)

測量原生 llama.cpp 的性能
需要先編譯 llama.cpp 並準備 GGUF 模型
"""

import time
import subprocess
import os
import glob

MODEL_PATHS = [
    "/Users/alexchuang/LLM/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "/Users/alexchuang/LLM/models/llama-2-7b-chat.Q4_K_M.gguf",
    "./models/llama-2-7b-chat.Q4_K_M.gguf",
]

CONTEXT_SIZES = [256, 512, 1024, 2048, 4096]
MAX_TOKENS = 50
PROMPTS = ["Hello, how are you?", "What is AI?"]

def find_model():
    """找第一個存在的模型"""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None

def get_memory_mb():
    """macOS 記憶體使用"""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024 / 1024
    except:
        return 0

def run_llama_cpp(model_path, prompt, ctx_size, num_tokens):
    """用 llama.cpp 跑推理"""
    cmd = [
        "./llama.cpp/main",
        "-m", model_path,
        "-p", prompt,
        "-n", str(num_tokens),
        "-c", str(ctx_size),
        "--no-display-prompt",
        "--temp", "0.7",
    ]

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        return None, elapsed

    output = result.stdout.strip()
    return output, elapsed

def benchmark_native_llama():
    """測量原生 llama.cpp"""
    model = find_model()
    if model is None:
        print("錯誤: 找不到 GGUF 模型")
        print("請下載一個 GGUF 模型，例如:")
        print("  huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf")
        return None

    print(f"使用模型: {model}")

    if not os.path.exists("./llama.cpp/main"):
        print("錯誤: llama.cpp 未編譯")
        print("請先編譯 llama.cpp:")
        print("  cd llama.cpp && mkdir build && cd build && cmake .. && make")
        return None

    print("\n" + "=" * 80)
    print("Native llama.cpp Benchmark")
    print("=" * 80)

    results = []

    for ctx_size in CONTEXT_SIZES:
        print(f"\n  [Context: {ctx_size}]")

        for prompt in PROMPTS:
            output, elapsed = run_llama_cpp(model, prompt, ctx_size, MAX_TOKENS)

            if output:
                prompt_tokens = len(prompt.split())
                generated = len(output.split())
                decode_speed = generated / elapsed if elapsed > 0 else 0
                print(f"    {prompt_tokens}→{generated} tokens: {elapsed:.2f}s, {decode_speed:.1f} tok/s")

                results.append({
                    "context": ctx_size,
                    "prompt_tokens": prompt_tokens,
                    "generated": generated,
                    "elapsed": elapsed,
                    "decode_speed": decode_speed,
                })

    return results

def print_summary(results):
    """打印結果"""
    if not results:
        return

    print("\n" + "=" * 80)
    print("llama.cpp RESULTS")
    print("=" * 80)

    from collections import defaultdict
    by_context = defaultdict(list)

    for r in results:
        by_context[r["context"]].append(r)

    print(f"\n{'Context':>8} | {'Decode Speed':>15} | {'Tokens':>10}")
    print("-" * 40)

    for ctx in sorted(by_context.keys()):
        ctx_results = by_context[ctx]
        avg_speed = sum(r["decode_speed"] for r in ctx_results) / len(ctx_results)
        total_tokens = sum(r["generated"] for r in ctx_results)
        print(f"{ctx:>8} | {avg_speed:>14.1f} tok/s | {total_tokens:>10}")

def main():
    print("=" * 80)
    print("llama.cpp Benchmark (Local Mac)")
    print("=" * 80)

    results = benchmark_native_llama()

    if results:
        print_summary(results)
    else:
        print("\n無法運行 benchmark，請確保:")
        print("1. llama.cpp 已編譯 (./llama.cpp/main 存在)")
        print("2. GGUF 模型已下載")

if __name__ == "__main__":
    main()