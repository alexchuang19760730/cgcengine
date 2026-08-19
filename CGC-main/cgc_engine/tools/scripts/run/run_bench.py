import os, time, torch, gc, json
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"
RESULTS = []

configs = [
    ("native", 512, 64, 4),
    ("kda", 512, 64, 4),
    ("native", 1024, 128, 4),
    ("kda", 1024, 128, 4),
]

for backend, input_len, output_len, batch_size in configs:
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)
    if backend == "kda":
        os.environ["VLLM_ATTENTION_BACKEND"] = "cgc_kda_fa2"

    print(f"\n=== {backend.upper()} (input={input_len}, output={output_len}, batch={batch_size}) ===")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=input_len + output_len,
        enforce_eager=True,
    )

    prompts = [f"Test prompt {i}." * (input_len // 50) for i in range(batch_size)]
    sampling_params = SamplingParams(max_tokens=output_len, temperature=0.8)

    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000

    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tp = tokens / (elapsed / 1000)

    print(f"  Time: {elapsed:.2f}ms, Tokens: {tokens}, Throughput: {tp:.2f} tok/s")
    RESULTS.append({
        "backend": backend,
        "input": input_len,
        "output": output_len,
        "batch": batch_size,
        "time_ms": elapsed,
        "tokens": tokens,
        "tp": tp,
    })

    del llm
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(f"\n{'Config':<30} {'Native (tok/s)':<18} {'KDA (tok/s)':<18} {'Speedup':<10}")
print("-" * 75)

for input_len in [512, 1024]:
    n = next((r for r in RESULTS if r["backend"] == "native" and r["input"] == input_len), None)
    k = next((r for r in RESULTS if r["backend"] == "kda" and r["input"] == input_len), None)
    if n and k:
        speedup = k["tp"] / n["tp"]
        print(f"input={input_len}, output={n['output']}, batch={n['batch']}  {n['tp']:<18.2f} {k['tp']:<18.2f} {speedup:<10.2f}x")

with open("benchmark_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(f"\nResults saved to: benchmark_results.json")