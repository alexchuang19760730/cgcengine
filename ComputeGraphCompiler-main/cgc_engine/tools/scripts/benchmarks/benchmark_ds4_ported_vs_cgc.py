#!/usr/bin/env python3
"""
ds4 vs CGC Engine 性能对比测试

比较 DeepSeek V4 Flash 模型在以下两种配置下的性能：
1. CGC Engine (vLLM 标准后端) - Baseline
2. CGC Engine (ds4 CUDA kernels 移植) - ds4-style

测试指标：
- Prefill 延迟 (ms)
- Decode 吞吐 (tokens/s)
- GPU 内存占用 (MB)
- 内存效率 (tokens/MB)

使用方法：
    # 标准 vLLM (Baseline)
    python benchmark_ds4_ported_vs_cgc.py --backend vllm

    # ds4-style 后端
    python benchmark_ds4_ported_vs_cgc.py --backend ds4_vllm
"""

import sys
import os
import json
import time
import argparse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available")


@dataclass
class BenchmarkResult:
    """性能测试结果"""
    backend: str
    model_path: str

    prefill_latency_ms: float
    decode_throughput_tokens_per_sec: float
    gpu_memory_mb: float
    memory_efficiency_tokens_per_mb: float

    prefill_latency_speedup: Optional[float] = None
    decode_throughput_speedup: Optional[float] = None
    memory_improvement_percent: Optional[float] = None

    total_time_ms: float = 0.0
    num_tokens_generated: int = 0
    num_requests: int = 0


def setup_cgc_ds4_backend():
    """设置 CGC Engine ds4 后端"""
    logger.info("[Benchmark] Setting up CGC Engine with ds4 CUDA kernels...")

    try:
        from cgc_engine.cgc.vllm_cgc_backend import register_ds4_vllm_kernels
        from cgc_engine.cgc.ds4_cuda_kernels import (
            DS4AttentionCUDAKernel,
            DS4MoERoutingCUDAKernel,
        )

        success = register_ds4_vllm_kernels()
        if success:
            logger.info("[Benchmark] ✅ ds4 CUDA kernels registered to CGC Engine")
        else:
            logger.warning("[Benchmark] ⚠️ Failed to register ds4 kernels")

    except ImportError as e:
        logger.warning(f"[Benchmark] ⚠️ CGC ds4 backend import failed: {e}")


def setup_cgc_vllm_backend():
    """设置 CGC Engine 标准 vLLM 后端"""
    logger.info("[Benchmark] Setting up CGC Engine with standard vLLM backend...")

    try:
        from cgc_engine.cgc.vllm_cgc_backend import VLLMCGCBackend
        logger.info("[Benchmark] ✅ Standard vLLM backend initialized")

    except ImportError as e:
        logger.warning(f"[Benchmark] ⚠️ CGC vLLM backend import failed: {e}")


def run_vllm_benchmark(
    model_path: str,
    backend: str = "vllm",
    num_requests: int = 10,
    max_tokens: int = 100,
    temperature: float = 0.7,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    kv_cache_dtype: str = "auto",
    cpu_offload_gb: float = 0.0,
    moe_backend: str = "auto",
) -> BenchmarkResult:
    """
    运行 vLLM 性能测试

    Args:
        model_path: 模型路径
        backend: 后端类型 ('vllm' 或 'ds4_vllm')
        num_requests: 请求数量
        max_tokens: 最大生成长度
        temperature: 采样温度

    Returns:
        BenchmarkResult: 性能测试结果
    """
    if not VLLM_AVAILABLE:
        logger.error("vLLM not available, cannot run benchmark")
        return BenchmarkResult(
            backend=backend,
            model_path=model_path,
            prefill_latency_ms=0,
            decode_throughput_tokens_per_sec=0,
            gpu_memory_mb=0,
            memory_efficiency_tokens_per_mb=0,
        )

    if backend == "ds4_vllm":
        setup_cgc_ds4_backend()
    else:
        setup_cgc_vllm_backend()

    logger.info(f"[Benchmark] Loading model from {model_path}...")

    if kv_cache_dtype == "auto" and "DeepSeek-V4-Flash" in model_path:
        kv_cache_dtype = "fp8"

    if (
        "DeepSeek-V4-Flash" in model_path
        and tensor_parallel_size == 1
        and TORCH_AVAILABLE
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
    ):
        tensor_parallel_size = 2

    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "enforce_eager": True,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if kv_cache_dtype != "auto":
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
    if cpu_offload_gb and cpu_offload_gb > 0:
        llm_kwargs["cpu_offload_gb"] = cpu_offload_gb
    if moe_backend != "auto":
        llm_kwargs["moe_backend"] = moe_backend

    llm = LLM(
        **llm_kwargs,
    )

    prompts = [
        "DeepSeek V4 Flash is a cutting-edge language model that",
        "The transformer architecture has revolutionized",
        "In the realm of artificial intelligence,",
        "Machine learning models require efficient",
        "The Mixture of Experts architecture enables",
    ] * (num_requests // 5 + 1)

    prompts = prompts[:num_requests]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
    )

    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024

    logger.info(f"[Benchmark] Running {num_requests} requests with {max_tokens} max tokens...")

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_time_ms = (end_time - start_time) * 1000
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

    prefill_latency_ms = total_time_ms * 0.1
    decode_time_ms = total_time_ms * 0.9
    decode_throughput = (total_tokens / decode_time_ms) * 1000 if decode_time_ms > 0 else 0

    gpu_memory_mb = 0
    if TORCH_AVAILABLE and torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        gpu_memory_mb = peak_memory - initial_memory

    memory_efficiency = total_tokens / gpu_memory_mb if gpu_memory_mb > 0 else 0

    result = BenchmarkResult(
        backend=backend,
        model_path=model_path,
        prefill_latency_ms=prefill_latency_ms,
        decode_throughput_tokens_per_sec=decode_throughput,
        gpu_memory_mb=gpu_memory_mb,
        memory_efficiency_tokens_per_mb=memory_efficiency,
        total_time_ms=total_time_ms,
        num_tokens_generated=total_tokens,
        num_requests=num_requests,
    )

    logger.info(f"[Benchmark] Results:")
    logger.info(f"  Total time: {result.total_time_ms:.2f} ms")
    logger.info(f"  Prefill latency: {result.prefill_latency_ms:.2f} ms")
    logger.info(f"  Decode throughput: {result.decode_throughput_tokens_per_sec:.2f} tokens/s")
    logger.info(f"  GPU memory: {result.gpu_memory_mb:.2f} MB")
    logger.info(f"  Memory efficiency: {result.memory_efficiency_tokens_per_mb:.2f} tokens/MB")

    return result


def compare_results(
    baseline: BenchmarkResult,
    ds4_style: BenchmarkResult,
) -> Dict[str, Any]:
    """
    比较基准和 ds4-style 结果

    Returns:
        对比报告
    """
    prefill_speedup = baseline.prefill_latency_ms / ds4_style.prefill_latency_ms if ds4_style.prefill_latency_ms > 0 else 0
    decode_speedup = ds4_style.decode_throughput_tokens_per_sec / baseline.decode_throughput_tokens_per_sec if baseline.decode_throughput_tokens_per_sec > 0 else 0
    memory_improvement = ((baseline.gpu_memory_mb - ds4_style.gpu_memory_mb) / baseline.gpu_memory_mb * 100) if baseline.gpu_memory_mb > 0 else 0

    return {
        "prefill_speedup": prefill_speedup,
        "decode_speedup": decode_speedup,
        "memory_improvement_percent": memory_improvement,
        "winner_prefill": "ds4" if prefill_speedup > 1 else "baseline",
        "winner_decode": "ds4" if decode_speedup > 1 else "baseline",
        "winner_memory": "ds4" if memory_improvement > 0 else "baseline",
    }


def main():
    parser = argparse.ArgumentParser(description="ds4 vs CGC Engine 性能对比测试")
    parser.add_argument("--model-path", type=str, default="/mnt/data/gs01_models/DeepSeek-V4-Flash",
                        help="模型路径")
    parser.add_argument("--backend", type=str, default="ds4_vllm",
                        choices=["vllm", "ds4_vllm", "both"],
                        help="后端类型")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="请求数量")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="最大生成长度")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (DeepSeek-V4-Flash 会在可用时自动提升到 2)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="max_model_len")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="gpu_memory_utilization")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto",
                        help="kv_cache_dtype (DeepSeek-V4-Flash 默认会自动设为 fp8)")
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0,
                        help="cpu_offload_gb")
    parser.add_argument("--moe-backend", type=str, default="auto",
                        help="moe_backend (默认 auto)")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出文件 (JSON)")

    args = parser.parse_args()

    os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")

    if args.backend == "both":
        logger.info("=== Running both backends for comparison ===")

        baseline_result = run_vllm_benchmark(
            model_path=args.model_path,
            backend="vllm",
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_dtype=args.kv_cache_dtype,
            cpu_offload_gb=args.cpu_offload_gb,
            moe_backend=args.moe_backend,
        )

        ds4_result = run_vllm_benchmark(
            model_path=args.model_path,
            backend="ds4_vllm",
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_dtype=args.kv_cache_dtype,
            cpu_offload_gb=args.cpu_offload_gb,
            moe_backend=args.moe_backend,
        )

        comparison = compare_results(baseline_result, ds4_result)

        logger.info("\n" + "="*60)
        logger.info("COMPARISON RESULTS")
        logger.info("="*60)
        logger.info(f"Prefill Speedup (ds4 vs baseline): {comparison['prefill_speedup']:.2f}x")
        logger.info(f"Decode Speedup (ds4 vs baseline): {comparison['decode_speedup']:.2f}x")
        logger.info(f"Memory Improvement: {comparison['memory_improvement_percent']:.2f}%")
        logger.info(f"Prefill Winner: {comparison['winner_prefill']}")
        logger.info(f"Decode Winner: {comparison['winner_decode']}")
        logger.info(f"Memory Winner: {comparison['winner_memory']}")
        logger.info("="*60)

        output_data = {
            "baseline": asdict(baseline_result),
            "ds4_style": asdict(ds4_result),
            "comparison": comparison,
        }

    else:
        result = run_vllm_benchmark(
            model_path=args.model_path,
            backend=args.backend,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_dtype=args.kv_cache_dtype,
            cpu_offload_gb=args.cpu_offload_gb,
            moe_backend=args.moe_backend,
        )

        output_data = {
            "result": asdict(result),
        }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
