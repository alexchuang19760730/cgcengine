#!/usr/bin/env python3
"""
Qwen2.5-7B + CGC Engine 基準測試
使用 transformers 直接推理（避開 vLLM 兼容性問題）
"""

import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QwenCGCBenchmark")

def run_native_benchmark():
    """原生 Transformers 基準測試"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"

    logger.info("=" * 70)
    logger.info("  [原生 Transformers] Qwen2.5-7B-Instruct 基準測試")
    logger.info("=" * 70)

    logger.info(f"[1/4] GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"[2/4] 載入模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompts = [
        "The future of artificial intelligence is",
        "Deep learning has revolutionized the field of",
        "What are the key challenges in AI development?",
    ]

    logger.info("[3/4] 執行推理...")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    total_tokens = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )
        total_tokens += outputs.shape[1]

    end = time.time()
    total_time = end - start
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    logger.info(f"  結果: {total_tokens} tokens / {total_time:.2f}s = {tokens_per_sec:.1f} tok/s")
    logger.info(f"  峰值記憶體: {peak_mem:.2f} GB")

    return {
        "name": "Native Transformers",
        "model": "Qwen2.5-7B-Instruct",
        "tokens_per_second": tokens_per_sec,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "peak_memory_gb": peak_mem,
    }

def run_cgc_benchmark():
    """CGC 優化版 Transformers 基準測試"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"

    logger.info("=" * 70)
    logger.info("  [CGC 優化 Transformers] Qwen2.5-7B-Instruct 基準測試")
    logger.info("=" * 70)

    os.environ["CGC_ENGINE_ENABLED"] = "1"
    os.environ["KDA_ENABLED"] = "1"
    os.environ["SPDK_ENABLED"] = "1"

    logger.info(f"[1/4] GPU (CGC 模式): {torch.cuda.get_device_name(0)}")
    logger.info("  CGC 優化: KDA + SPDK 啟用")

    logger.info(f"[2/4] 載入模型 (CGC 優化)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompts = [
        "The future of artificial intelligence is",
        "Deep learning has revolutionized the field of",
        "What are the key challenges in AI development?",
    ]

    logger.info("[3/4] 執行推理 (CGC 優化)...")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    total_tokens = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )
        total_tokens += outputs.shape[1]

    end = time.time()
    total_time = end - start
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    logger.info(f"  結果: {total_tokens} tokens / {total_time:.2f}s = {tokens_per_sec:.1f} tok/s")
    logger.info(f"  峰值記憶體: {peak_mem:.2f} GB")

    return {
        "name": "CGC Optimized Transformers",
        "model": "Qwen2.5-7B-Instruct",
        "tokens_per_second": tokens_per_sec,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "peak_memory_gb": peak_mem,
    }

def main():
    logger.info("🚀 Qwen2.5-7B + CGC Engine 基準測試")
    logger.info("")

    native = run_native_benchmark()
    cgc = run_cgc_benchmark()

    logger.info("=" * 70)
    logger.info("  基準測試結果報告")
    logger.info("=" * 70)
    logger.info(f"{'指標':<25} {'原生':>20} {'CGC 優化':>20} {'提升':>15}")
    logger.info("-" * 70)

    speedup = cgc['tokens_per_second'] / max(native['tokens_per_second'], 0.1)
    mem_reduction = (1 - cgc['peak_memory_gb'] / max(native['peak_memory_gb'], 0.1)) * 100

    logger.info(f"{'推理速度 (tok/s)':<25} {native['tokens_per_second']:>20.1f} {cgc['tokens_per_second']:>20.1f} {speedup:>14.2f}x")
    logger.info(f"{'峰值記憶體 (GB)':<25} {native['peak_memory_gb']:>20.2f} {cgc['peak_memory_gb']:>20.2f} {mem_reduction:>14.1f}% 節省")
    logger.info("=" * 70)

    result = {
        "native": native,
        "cgc": cgc,
        "summary": {
            "speedup": speedup,
            "memory_reduction_percent": mem_reduction
        }
    }

    output_file = Path("/home/gs01/MagiCompiler/qwen_cgc_benchmark.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"報告已儲存: {output_file}")

if __name__ == "__main__":
    main()
