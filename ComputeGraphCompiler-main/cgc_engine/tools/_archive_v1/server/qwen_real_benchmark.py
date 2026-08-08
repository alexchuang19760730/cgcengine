#!/usr/bin/env python3
"""
使用 transformers 直接進行真實 LLM 推理測試
"""

import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TransformersBenchmark")

def run_benchmark():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"

    logger.info("🚀 開始 Qwen2.5-7B-Instruct 真實推理基準測試 (Transformers)")

    logger.info("[1/5] 檢查 GPU 環境...")
    assert torch.cuda.is_available(), "CUDA 不可用"
    device = torch.device("cuda:0")
    logger.info(f"  使用 GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"[2/5] 載入 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"[3/5] 載入模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("[4/5] 執行推理基準測試...")

    test_prompts = [
        "The future of artificial intelligence is",
        "Deep learning has revolutionized the field of",
        "What are the key challenges in training large language models?",
    ]

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    all_tokens = 0
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
        all_tokens += outputs.shape[1]

    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_sec = all_tokens / total_time if total_time > 0 else 0

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    logger.info(f"  總生成 tokens: {all_tokens}")
    logger.info(f"  總耗時: {total_time:.2f}s")
    logger.info(f"  生成速度: {tokens_per_sec:.1f} tokens/s")
    logger.info(f"  峰值 GPU 記憶體: {peak_memory:.2f} GB")

    result = {
        "model": "Qwen2.5-7B-Instruct",
        "method": "transformers",
        "gpu": torch.cuda.get_device_name(0),
        "total_time_s": total_time,
        "total_tokens": all_tokens,
        "tokens_per_second": tokens_per_sec,
        "peak_gpu_memory_gb": peak_memory,
    }

    output_file = Path("/home/gs01/MagiCompiler/qwen2.5_7b_real_report.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 70)
    logger.info("  Qwen2.5-7B-Instruct 真實推理效能報告 (Transformers)")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"推理速度: {tokens_per_sec:.1f} tokens/s")
    logger.info(f"峰值 GPU 記憶體: {peak_memory:.2f} GB")
    logger.info("=" * 70)

    return result

if __name__ == "__main__":
    run_benchmark()
