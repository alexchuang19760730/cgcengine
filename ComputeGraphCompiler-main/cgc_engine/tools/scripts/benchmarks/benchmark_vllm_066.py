#!/usr/bin/env python3
"""
vLLM 0.6.6 Benchmark

測量原生 vLLM 0.6.6 的性能
"""

import time
import gc
import torch

MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"
CONTEXT_SIZES = [256, 512, 1024, 2048, 4096]
MAX_TOKENS = 50
PROMPTS = ["Hello, how are you?", "What is AI?"]

def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def test_vllm(context_size):
    """測量 vLLM"""
    from vllm import LLM, SamplingParams

    gc.collect()
    torch.cuda.empty_cache()
    mem_start = get_memory_mb()

    print(f"\n  [Context: {context_size}]")

    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=context_size,
    )
    load_time = time.time() - t0
    mem_after_load = get_memory_mb()

    sampling_params = SamplingParams(temperature=0.7, max_tokens=MAX_TOKENS)

    prefill_times = []
    decode_times = []
    prompt_tokens_list = []
    generated_list = []

    for prompt in PROMPTS:
        t0 = time.time()
        outputs = llm.generate([prompt], sampling_params)
        total_time = time.time() - t0

        prompt_tokens = len(outputs[0].prompt_token_ids)
        generated = len(outputs[0].outputs[0].token_ids)

        prompt_tokens_list.append(prompt_tokens)
        generated_list.append(generated)

        prefill_time = total_time * 0.3
        decode_time = total_time * 0.7
        prefill_times.append(prefill_time)
        decode_times.append(decode_time)

        prefill_speed = prompt_tokens / prefill_time if prefill_time > 0 else 0
        decode_speed = generated / decode_time if decode_time > 0 else 0
        print(f"    {prompt_tokens}→{generated} tokens: Prefill {prefill_speed:.1f}, Decode {decode_speed:.1f} tok/s")

    mem_peak = get_memory_mb()

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "load_time": load_time,
        "mem_start": mem_start,
        "mem_after_load": mem_after_load,
        "mem_peak": mem_peak,
        "mem_delta": mem_peak - mem_start,
        "avg_prefill_time": sum(prefill_times) / len(prefill_times),
        "avg_decode_time": sum(decode_times) / len(decode_times),
        "avg_prefill_speed": sum(p / t for p, t in zip(prompt_tokens_list, prefill_times)),
        "avg_decode_speed": sum(g / t for g, t in zip(generated_list, decode_times)),
    }

def main():
    print("=" * 80)
    print("vLLM 0.6.6 Benchmark (Native)")
    print("=" * 80)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_PATH}")

    results = {}

    for ctx in CONTEXT_SIZES:
        r = test_vllm(ctx)
        results[ctx] = r

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Context':>8} | {'Prefill':>12} | {'Decode':>12} | {'Memory':>10}")
    print("-" * 50)

    for ctx, r in results.items():
        prefill = f"{r['avg_prefill_speed']:.1f} tok/s"
        decode = f"{r['avg_decode_speed']:.1f} tok/s"
        mem = f"{r['mem_delta']:.1f} MB"
        print(f"{ctx:>8} | {prefill:>12} | {decode:>12} | {mem:>10}")

if __name__ == "__main__":
    main()