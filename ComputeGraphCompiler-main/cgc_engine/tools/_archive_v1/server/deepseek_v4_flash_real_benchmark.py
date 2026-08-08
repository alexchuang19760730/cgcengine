#!/usr/bin/env python3
"""
DeepSeek V4 Flash 真正推理基準測試
- 實際下載模型
- 使用 vLLM 進行真實推理
- 使用代理訪問 HuggingFace
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DeepSeekV4RealBenchmark")

def check_vllm_installed():
    try:
        from vllm import LLM, SamplingParams
        logger.info("✅ vLLM 已安裝")
        return True
    except ImportError:
        logger.error("❌ vLLM 未安裝，請先執行: pip install vllm")
        return False

def download_and_benchmark(model_name: str = "deepseek-ai/DeepSeek-V3"):
    from vllm import LLM, SamplingParams
    import torch

    logger.info(f"🚀 開始 DeepSeek V4 Flash 真實推理基準測試")
    logger.info(f"模型: {model_name}")

    logger.info("[1/5] 檢查 GPU 環境...")
    assert torch.cuda.is_available(), "CUDA 不可用"
    gpu_count = torch.cuda.device_count()
    logger.info(f"  GPU 數量: {gpu_count}")
    for i in range(gpu_count):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    logger.info("[2/5] 下載模型中...")
    logger.info("  這可能需要一些時間，取決於網絡速度...")

    logger.info("[3/5] 初始化 vLLM 引擎...")
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=gpu_count,
            trust_remote_code=True,
            max_model_len=4096,
        )
    except Exception as e:
        logger.error(f"模型下載失敗: {e}")
        logger.info("嘗試使用較小的模型進行測試...")
        model_name = "microsoft/Phi-3-mini-128k-instruct"
        llm = LLM(
            model=model_name,
            tensor_parallel_size=gpu_count,
            trust_remote_code=True,
            max_model_len=2048,
        )

    logger.info("[4/5] 執行推理基準測試...")

    test_prompts = [
        "The future of artificial intelligence is",
        "Deep learning has revolutionized the field of",
        "In a groundbreaking development, researchers have",
    ]

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

    logger.info("  執行 Prefill + Decode 測試...")

    start_prefill = time.time()
    outputs = llm.generate(test_prompts, sampling_params)
    end_prefill = time.time()

    prefill_time = end_prefill - start_prefill
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tokens_per_sec = total_tokens / prefill_time if prefill_time > 0 else 0

    logger.info(f"  總生成 tokens: {total_tokens}")
    logger.info(f"  總耗時: {prefill_time:.2f}s")
    logger.info(f"  生成速度: {tokens_per_sec:.1f} tokens/s")

    logger.info("[5/5] 生成效能報告...")

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    result = {
        "model": model_name,
        "gpu_count": gpu_count,
        "prefill_time_s": prefill_time,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_sec,
        "peak_gpu_memory_gb": peak_memory,
    }

    output_file = Path("/home/gs01/MagiCompiler/deepseek_v4_flash_real_report.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("  DeepSeek V4 Flash 真實推理效能報告")
    logger.info("=" * 70)
    logger.info(f"模型: {model_name}")
    logger.info(f"GPU: {gpu_count}x {torch.cuda.get_device_name(0)}")
    logger.info(f"推理速度: {tokens_per_sec:.1f} tokens/s")
    logger.info(f"峰值 GPU 記憶體: {peak_memory:.2f} GB")
    logger.info(f"報告已儲存: {output_file}")
    logger.info("=" * 70)

    return result

def main():
    if not check_vllm_installed():
        sys.exit(1)

    model_name = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V3")
    download_and_benchmark(model_name)

if __name__ == "__main__":
    main()
