# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
Performance Comparator - Step7: 自动化性能对比框架
原生模型 vs 优化后模型 全方位性能对比
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import time
import logging
import gc

logger = logging.getLogger(__name__)


class PerformanceComparator:
    """自动化性能对比器"""
    
    @classmethod
    def compare(
        cls,
        native_model: nn.Module,
        optimized_model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 10,
        warmup_runs: int = 3,
        seed: int = 0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
        """主入口: 对比原生与优化模型性能"""
        logger.info("[PerformanceComparator] 📊 开始性能对比...")
        
        # 清空缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 原生模型基准
        native_perf = cls._benchmark_model(
            native_model,
            input_shape,
            num_runs,
            warmup_runs,
            label="原生模型",
            seed=seed,
        )
        
        # 优化后模型基准
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        opt_perf = cls._benchmark_model(
            optimized_model,
            input_shape,
            num_runs,
            warmup_runs,
            label="优化后模型",
            seed=seed,
        )
        
        speedup_ratio = native_perf["avg_time_ms"] / max(opt_perf["avg_time_ms"], 1e-9)
        memory_saving_ratio = native_perf.get("peak_memory_mb", 1024) / max(opt_perf.get("peak_memory_mb", 512), 1)
        
        # 输出对比结果
        logger.info("=" * 60)
        logger.info("📈 性能对比总结:")
        logger.info(f"  原生耗时: {native_perf['avg_time_ms']:.2f} ms")
        logger.info(f"  优化耗时: {opt_perf['avg_time_ms']:.2f} ms")
        logger.info(f"  ⚡ 加速比: {speedup_ratio:.2f}x")
        logger.info(f"  💾 显存节省: {memory_saving_ratio:.2f}x")
        logger.info("=" * 60)
        
        return native_perf, opt_perf, speedup_ratio, memory_saving_ratio
    
    @classmethod
    def _benchmark_model(
        cls,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int,
        warmup_runs: int,
        label: str,
        seed: int,
    ) -> Dict[str, Any]:
        """基准测试单个模型"""
        if getattr(model, "is_mlx_model", False):
            import mlx.core as mx

            if seed is not None:
                mx.random.seed(int(seed))

            model_dtype = getattr(model, "mlx_dtype", mx.float16)
            dummy_input = mx.random.normal(tuple(input_shape), dtype=model_dtype)

            logger.info(f"  [{label}] 预热 {warmup_runs} 次...")
            for _ in range(warmup_runs):
                out = model(dummy_input)
                mx.eval(out)

            logger.info(f"  [{label}] 正式运行 {num_runs} 次...")
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                out = model(dummy_input)
                mx.eval(out)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time_s = sum(times) / len(times)
            avg_time_ms = avg_time_s * 1000
            tokens_per_sec = input_shape[1] / avg_time_s

            perf = {
                "avg_time_ms": avg_time_ms,
                "min_time_ms": min(times) * 1000,
                "max_time_ms": max(times) * 1000,
                "tokens_per_sec": tokens_per_sec,
                "peak_memory_mb": 0,
                "input_shape": input_shape,
            }

            logger.info(f"  [{label}] 平均耗时: {avg_time_ms:.2f} ms")
            logger.info(f"  [{label}] Token/s: {tokens_per_sec:.2f}")
            return perf

        model.eval()
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        model_device = None
        model_dtype = None
        try:
            p = next(model.parameters())
            model_device = p.device
            model_dtype = p.dtype
        except StopIteration:
            try:
                b = next(model.buffers())
                model_device = b.device
                model_dtype = b.dtype
            except StopIteration:
                model_device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
                model_dtype = torch.float32

        dummy_input = torch.randn(*input_shape, device=model_device, dtype=model_dtype)
        
        # 记录峰值内存
        peak_memory_mb = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        logger.info(f"  [{label}] 预热 {warmup_runs} 次...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass
        
        # 正式计时
        logger.info(f"  [{label}] 正式运行 {num_runs} 次...")
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
                elapsed = time.perf_counter() - start
                times.append(elapsed)
        
        # 统计
        avg_time_s = sum(times) / len(times)
        avg_time_ms = avg_time_s * 1000
        tokens_per_sec = input_shape[1] / avg_time_s
        
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        perf = {
            "avg_time_ms": avg_time_ms,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "tokens_per_sec": tokens_per_sec,
            "peak_memory_mb": peak_memory_mb,
            "input_shape": input_shape,
        }
        
        logger.info(f"  [{label}] 平均耗时: {avg_time_ms:.2f} ms")
        logger.info(f"  [{label}] Token/s: {tokens_per_sec:.2f}")
        
        return perf
