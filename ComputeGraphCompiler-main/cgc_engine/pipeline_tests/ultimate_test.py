#!/usr/bin/env python3
"""
终极完整对比脚本！vLLM vs vLLM+KDA！
包括 Prefill Decode 和 Memory！
"""
import os
import sys
import time
import json
import torch
import glob
from pathlib import Path

# 配置
MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"
CONTEXT_LENGTHS = [256, 512, 1024, 2048]
OUTPUT_LENGTH = 32
GPU_MEM_UTIL = 0.7
RESULTS_FILE = "ultimate_comparison_results.json"


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_stats():
    if not torch.cuda.is_available():
        return {"peak_gb": 0.0, "current_gb": 0.0}
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    current = torch.cuda.memory_allocated() / (1024 ** 3)
    return {"peak_gb": peak, "current_gb": current}


def read_kda_stats_from_files():
    """从子进程文件读取 KDA 统计信息"""
    stats_files = glob.glob("/tmp/kda_stats_*.json")
    all_stats = []
    
    for fpath in stats_files:
        try:
            with open(fpath, "r") as f:
                stats = json.load(f)
                all_stats.append(stats)
            os.remove(fpath)
        except:
            pass
    
    if not all_stats:
        return {"peak_gb": 0.0, "avg_gb": 0.0, "call_count": 0}
    
    peak_gb = max(s.get("peak_gb", 0.0) for s in all_stats)
    avg_gb = sum(s.get("avg_gb", 0.0) for s in all_stats) / len(all_stats)
    call_count = sum(s.get("call_count", 0) for s in all_stats)
    
    return {
        "peak_gb": peak_gb,
        "avg_gb": avg_gb,
        "call_count": call_count,
        "prefill": sum(s.get("prefill", 0) for s in all_stats),
        "decode": sum(s.get("decode", 0) for s in all_stats),
    }


def generate_prompt(length):
    return "Hello, " * (length // 6) + "What is your name?"


def run_single_test(llm, sampling_params, context_len):
    clear_gpu_cache()
    prompt = generate_prompt(context_len)

    start = time.time()
    outputs = llm.generate(
        [prompt],
        sampling_params=sampling_params
    )
    total_time = time.time() - start

    num_generated_tokens = len(outputs[0].outputs[0].token_ids)
    total_tokens = len(outputs[0].prompt_token_ids) + num_generated_tokens

    stats = get_gpu_stats()
    
    # 读取子进程的 KDA 统计
    kda_stats = read_kda_stats_from_files()
    if kda_stats["peak_gb"] > 0:
        stats["peak_gb"] = max(stats["peak_gb"], kda_stats["peak_gb"])
        stats["kda_call_count"] = kda_stats["call_count"]
        stats["kda_prefill"] = kda_stats["prefill"]
        stats["kda_decode"] = kda_stats["decode"]

    result = {
        "context_len": context_len,
        "total_time": total_time,
        "prompt_tokens": len(outputs[0].prompt_token_ids),
        "generated_tokens": num_generated_tokens,
        "total_tokens": total_tokens,
        "throughput": total_tokens / total_time,
        "peak_gb": stats["peak_gb"],
        "current_gb": stats["current_gb"],
        **{k: v for k, v in stats.items() if k.startswith("kda_")},
    }
    return result


def run_single_label(label, kda_enabled):
    print("\n" + "=" * 120)
    print(f"🏆 {label} {'(KDA Enabled)' if kda_enabled else '(Baseline)'}")
    print("=" * 120)

    if kda_enabled:
        os.environ["VLLM_USE_CGC_KDA"] = "1"
        print(f"✅ Environment variable set: VLLM_USE_CGC_KDA=1")
    else:
        os.environ.pop("VLLM_USE_CGC_KDA", None)

    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if kda_enabled:
        print(f"🚀 Importing cgc_kda_backend...")
        try:
            import Backend.Vllm.vllm_backend.cgc_kda_backend
            print(f"✅ cgc_kda_backend imported!")
        except Exception as e:
            print(f"❌ Import failed: {e}")
            import traceback
            traceback.print_exc()

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=OUTPUT_LENGTH,
    )

    print(f"🚀 Initializing LLM...")
    start_init = time.time()

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
    )

    init_time = time.time() - start_init
    print(f"✅ LLM initialized in {init_time:.2f}s!")

    all_results = []

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n📌 Testing context length: {ctx_len}")
        res = run_single_test(llm, sampling_params, ctx_len)
        all_results.append(res)
        print(f"✅ Done: {res}")

    del llm
    clear_gpu_cache()

    return all_results


def main():
    print("=" * 120)
    print("🎯 终极完整对比测试：vLLM Baseline vs vLLM+KDA")
    print("=" * 120)

    results = {}

    baseline_results = run_single_label("vLLM Baseline", kda_enabled=False)
    results["baseline"] = baseline_results

    kda_results = run_single_label("vLLM + KDA", kda_enabled=True)
    results["kda"] = kda_results

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 120)
    print("📊 最终完整对比")
    print("=" * 120)

    print(f"{'Context':<10} | {'Baseline Tok/s':<20} | {'Baseline Mem (GB)':<20} | {'KDA Tok/s':<20} | {'KDA Mem (GB)':<20}")
    print("-" * 120)

    for b, k in zip(baseline_results, kda_results):
        ctx = b['context_len']
        b_tps = b['throughput']
        b_mem = b['peak_gb']
        k_tps = k['throughput']
        k_mem = k['peak_gb']
        print(f"{ctx:<10} | {b_tps:<20.1f} | {b_mem:<20.2f} | {k_tps:<20.1f} | {k_mem:<20.2f}")

    print("\n" + "=" * 120)
    print(f"✅ 结果已保存到: {RESULTS_FILE}")
    print("=" * 120)


if __name__ == "__main__":
    main()
