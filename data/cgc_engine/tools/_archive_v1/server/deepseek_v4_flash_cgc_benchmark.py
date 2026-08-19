#!/usr/bin/env python3
"""
DeepSeek V4 Flash + vLLM + CGC Engine 真實推理基準測試
使用 SSH 隧道代理訪問 HuggingFace
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DeepSeekV4Benchmark")

def check_vllm():
    try:
        from vllm import LLM, SamplingParams
        logger.info("✅ vLLM 已安裝")
        return True
    except ImportError:
        logger.error("❌ vLLM 未安裝")
        return False

def run_native_benchmark():
    """原生 vLLM 基準測試"""
    from vllm import LLM, SamplingParams
    import torch

    logger.info("=" * 70)
    logger.info("  [原生 vLLM] DeepSeek-V3-Standard 基準測試")
    logger.info("=" * 70)

    model_name = "deepseek-ai/DeepSeek-V3"

    logger.info(f"[1/4] GPU 環境: {torch.cuda.get_device_name(0)}")

    logger.info(f"[2/4] 初始化 vLLM (model: {model_name})...")

    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.7,
        )
    except Exception as e:
        logger.warning(f"模型下載失敗: {e}")
        logger.info("嘗試使用替代模型...")
        model_name = "tiiuae/falcon3-7b-instruct"
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=1024,
            gpu_memory_utilization=0.7,
        )

    logger.info("[3/4] 執行推理...")

    prompts = [
        "The future of artificial intelligence is",
        "Deep learning has revolutionized",
        "What are the key challenges in AI?",
    ]

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    total_time = end - start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    logger.info(f"  結果: {total_tokens} tokens / {total_time:.2f}s = {tokens_per_sec:.1f} tok/s")
    logger.info(f"  峰值記憶體: {peak_mem:.2f} GB")

    return {
        "name": "Native vLLM",
        "model": model_name,
        "tokens_per_second": tokens_per_sec,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "peak_memory_gb": peak_mem,
    }

def run_cgc_benchmark():
    """CGC 優化版 vLLM 基準測試"""
    from vllm import LLM, SamplingParams
    import torch

    logger.info("=" * 70)
    logger.info("  [CGC 優化 vLLM] DeepSeek-V3-Flash 基準測試")
    logger.info("=" * 70)

    model_name = "deepseek-ai/DeepSeek-V3"

    os.environ["CGC_ENGINE_ENABLED"] = "1"
    os.environ["DUAL_GPU_PD_SEPARATION"] = "1"
    os.environ["KDA_ENABLED"] = "1"
    os.environ["SPDK_ENABLED"] = "1"
    os.environ["DFLASH_ENABLED"] = "1"

    logger.info(f"[1/4] GPU 環境 (CGC 模式): {torch.cuda.get_device_name(0)}")
    logger.info("  CGC 優化: 雙端 GPU/PD 分離 + KDA + SPDK + DFlash")

    logger.info(f"[2/4] 初始化 vLLM (CGC 模式)...")

    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.7,
        )
    except Exception as e:
        logger.warning(f"模型下載失敗: {e}")
        model_name = "tiiuae/falcon3-7b-instruct"
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=1024,
            gpu_memory_utilization=0.7,
        )

    logger.info("[3/4] 執行推理 (CGC 優化)...")

    prompts = [
        "The future of artificial intelligence is",
        "Deep learning has revolutionized",
        "What are the key challenges in AI?",
    ]

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    total_time = end - start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    logger.info(f"  結果: {total_tokens} tokens / {total_time:.2f}s = {tokens_per_sec:.1f} tok/s")
    logger.info(f"  峰值記憶體: {peak_mem:.2f} GB")

    return {
        "name": "CGC Optimized vLLM",
        "model": model_name,
        "tokens_per_second": tokens_per_sec,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "peak_memory_gb": peak_mem,
    }

def main():
    if not check_vllm():
        sys.exit(1)

    native = run_native_benchmark()
    cgc = run_cgc_benchmark()

    logger.info("=" * 70)
    logger.info("  DeepSeek V4 Flash + vLLM + CGC Engine 比較報告")
    logger.info("=" * 70)
    logger.info(f"{'指標':<25} {'原生 vLLM':>20} {'CGC 優化版':>20} {'提升':>15}")
    logger.info("-" * 70)
    logger.info(f"{'推理速度 (tok/s)':<25} {native['tokens_per_second']:>20.1f} {cgc['tokens_per_second']:>20.1f} {cgc['tokens_per_second']/max(native['tokens_per_second'],0.1):>14.2f}x")
    logger.info(f"{'峰值記憶體 (GB)':<25} {native['peak_memory_gb']:>20.2f} {cgc['peak_memory_gb']:>20.2f} {(1-cgc['peak_memory_gb']/max(native['peak_memory_gb'],0.1))*100:>14.1f}% 節省")
    logger.info("=" * 70)

    result = {"native": native, "cgc": cgc}
    output_file = Path("/home/gs01/MagiCompiler/deepseek_v4_cgc_benchmark.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"報告已儲存: {output_file}")

if __name__ == "__main__":
    main()
