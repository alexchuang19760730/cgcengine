# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Auto Benchmarking Script - 吞吐量/延迟压测

功能:
- 吞吐量测试 (Throughput)
- 延迟测试 (Latency)
- 并发测试 (Concurrency)
- 显存测试 (Memory)
- 生成压测报告

Usage:
    python -m cgc_engine.cgc.benchmark
"""

import torch
import time
import json
import argparse
import statistics
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import uuid

try:
    from .cgc_opcodes import CGC_OP_CODES
    from .cgc_simd_executor import CGCExecutor
    from .flashkda_integration import FlashKDALayer, FLASHKDA_AVAILABLE
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """压测结果"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_sec: float
    throughput: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_memory_mb: float
    peak_memory_mb: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_duration_sec": self.total_duration_sec,
            "throughput": self.throughput,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "avg_memory_mb": self.avg_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkConfig:
    """压测配置"""
    batch_size: int = 1
    seq_len: int = 1024
    num_heads: int = 32
    head_dim: int = 128
    hidden_dim: int = 4096
    num_layers: int = 24
    warmup_iterations: int = 10
    test_iterations: int = 100
    num_threads: int = 1
    enable_flashkda: bool = True
    enable_cgc: bool = True
    enable_profiling: bool = True
    memory_test: bool = True


def get_memory_usage() -> float:
    """获取当前 GPU 显存使用 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_memory() -> float:
    """获取峰值 GPU 显存使用 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


class CGCBenchmark:
    """
    CGC Benchmark 测试器
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.cgc_exec: Optional[CGCExecutor] = None
        self.flashkda: Optional[FlashKDALayer] = None

        self._init_components()

    def _init_components(self):
        """初始化组件"""
        if CGC_AVAILABLE:
            self.cgc_exec = CGCExecutor(enable_profiling=self.config.enable_profiling)
            if FLASHKDA_AVAILABLE and self.config.enable_flashkda:
                self.flashkda = FlashKDALayer()
            print(f"[Benchmark] CGC available: {CGC_AVAILABLE}, FlashKDA: {FLASHKDA_AVAILABLE}")
        else:
            print("[Benchmark] CGC not available, using PyTorch native")

    def _create_tensors(self, batch_size: int, seq_len: int, hidden_dim: int, device: str = "cuda"):
        """创建测试张量"""
        return {
            "q": torch.randn(batch_size, seq_len, self.config.num_heads, self.config.head_dim, device=device),
            "k": torch.randn(batch_size, seq_len, self.config.num_heads, self.config.head_dim, device=device),
            "v": torch.randn(batch_size, seq_len, self.config.num_heads, self.config.head_dim, device=device),
        }

    def benchmark_kda_forward(self) -> BenchmarkResult:
        """KDA Forward 压测"""
        print("\n[Benchmark] KDA Forward...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        tensors = self._create_tensors(self.config.batch_size, self.config.seq_len, self.config.hidden_dim)

        for _ in range(self.config.warmup_iterations):
            if self.flashkda:
                self.flashkda(tensors["q"], tensors["k"], tensors["v"])
            else:
                torch.nn.functional.scaled_dot_product_attention(
                    tensors["q"], tensors["k"], tensors["v"]
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        memory_samples = []

        start_time = time.time()

        for _ in range(self.config.test_iterations):
            iter_start = time.time()

            if self.flashkda:
                out, _ = self.flashkda(tensors["q"], tensors["k"], tensors["v"])
            else:
                out = torch.nn.functional.scaled_dot_product_attention(
                    tensors["q"], tensors["k"], tensors["v"]
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency_ms = (time.time() - iter_start) * 1000
            latencies.append(latency_ms)

            if self.config.memory_test and torch.cuda.is_available():
                memory_samples.append(get_memory_usage())

        total_duration = time.time() - start_time

        latencies.sort()
        n = len(latencies)

        result = BenchmarkResult(
            test_name="kda_forward",
            total_requests=self.config.test_iterations,
            successful_requests=self.config.test_iterations,
            failed_requests=0,
            total_duration_sec=total_duration,
            throughput=self.config.test_iterations / total_duration,
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=latencies[int(n * 0.50)],
            p95_latency_ms=latencies[int(n * 0.95)],
            p99_latency_ms=latencies[int(n * 0.99)],
            avg_memory_mb=statistics.mean(memory_samples) if memory_samples else 0,
            peak_memory_mb=get_peak_memory() if torch.cuda.is_available() else 0,
        )

        print(f"[Benchmark] KDA Forward: {result.throughput:.2f} req/s, "
              f"avg latency: {result.avg_latency_ms:.3f} ms, "
              f"p99: {result.p99_latency_ms:.3f} ms")

        return result

    def benchmark_sdpa(self) -> BenchmarkResult:
        """SDPA 压测"""
        print("\n[Benchmark] SDPA...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        tensors = self._create_tensors(self.config.batch_size, self.config.seq_len, self.config.hidden_dim)

        for _ in range(self.config.warmup_iterations):
            torch.nn.functional.scaled_dot_product_attention(
                tensors["q"], tensors["k"], tensors["v"]
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        memory_samples = []

        start_time = time.time()

        for _ in range(self.config.test_iterations):
            iter_start = time.time()

            out = torch.nn.functional.scaled_dot_product_attention(
                tensors["q"], tensors["k"], tensors["v"]
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency_ms = (time.time() - iter_start) * 1000
            latencies.append(latency_ms)

            if self.config.memory_test and torch.cuda.is_available():
                memory_samples.append(get_memory_usage())

        total_duration = time.time() - start_time

        latencies.sort()
        n = len(latencies)

        result = BenchmarkResult(
            test_name="sdpa",
            total_requests=self.config.test_iterations,
            successful_requests=self.config.test_iterations,
            failed_requests=0,
            total_duration_sec=total_duration,
            throughput=self.config.test_iterations / total_duration,
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=latencies[int(n * 0.50)],
            p95_latency_ms=latencies[int(n * 0.95)],
            p99_latency_ms=latencies[int(n * 0.99)],
            avg_memory_mb=statistics.mean(memory_samples) if memory_samples else 0,
            peak_memory_mb=get_peak_memory() if torch.cuda.is_available() else 0,
        )

        print(f"[Benchmark] SDPA: {result.throughput:.2f} req/s, "
              f"avg latency: {result.avg_latency_ms:.3f} ms")

        return result

    def benchmark_concurrent(self, num_concurrent: int = 4) -> BenchmarkResult:
        """并发压测"""
        print(f"\n[Benchmark] Concurrent ({num_concurrent} threads)...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        latencies = []
        memory_samples = []
        success_count = 0
        fail_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal success_count, fail_count

            tensors = self._create_tensors(
                self.config.batch_size,
                self.config.seq_len,
                self.config.hidden_dim,
            )

            iter_start = time.time()

            try:
                if self.flashkda:
                    self.flashkda(tensors["q"], tensors["k"], tensors["v"])
                else:
                    torch.nn.functional.scaled_dot_product_attention(
                        tensors["q"], tensors["k"], tensors["v"]
                    )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                latency_ms = (time.time() - iter_start) * 1000

                with lock:
                    latencies.append(latency_ms)
                    success_count += 1
                    if torch.cuda.is_available():
                        memory_samples.append(get_memory_usage())

            except Exception as e:
                with lock:
                    fail_count += 1
                    print(f"[Benchmark] Worker error: {e}")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(worker)
                for _ in range(self.config.test_iterations)
            ]
            for f in as_completed(futures):
                pass

        total_duration = time.time() - start_time

        latencies.sort()
        n = len(latencies) if latencies else 1

        result = BenchmarkResult(
            test_name=f"concurrent_{num_concurrent}",
            total_requests=self.config.test_iterations,
            successful_requests=success_count,
            failed_requests=fail_count,
            total_duration_sec=total_duration,
            throughput=self.config.test_iterations / total_duration,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            p50_latency_ms=latencies[int(n * 0.50)] if latencies else 0,
            p95_latency_ms=latencies[int(n * 0.95)] if latencies else 0,
            p99_latency_ms=latencies[int(n * 0.99)] if latencies else 0,
            avg_memory_mb=statistics.mean(memory_samples) if memory_samples else 0,
            peak_memory_mb=get_peak_memory() if torch.cuda.is_available() else 0,
        )

        print(f"[Benchmark] Concurrent: {result.throughput:.2f} req/s")

        return result

    def benchmark_full_model(self) -> BenchmarkResult:
        """完整模型压测"""
        print("\n[Benchmark] Full Model Forward...")

        if not CGC_AVAILABLE:
            print("[Benchmark] CGC not available, skipping full model test")
            return None

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        hidden_states = torch.randn(
            self.config.batch_size,
            self.config.seq_len,
            self.config.hidden_dim,
            device="cuda",
        )

        for _ in range(self.config.warmup_iterations):
            x = hidden_states
            for _ in range(self.config.num_layers):
                x = torch.nn.functional.rms_norm(x, (self.config.hidden_dim,))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        memory_samples = []

        start_time = time.time()

        for _ in range(self.config.test_iterations):
            iter_start = time.time()

            x = hidden_states
            for _ in range(self.config.num_layers):
                x = torch.nn.functional.rms_norm(x, (self.config.hidden_dim,))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency_ms = (time.time() - iter_start) * 1000
            latencies.append(latency_ms)

            if self.config.memory_test and torch.cuda.is_available():
                memory_samples.append(get_memory_usage())

        total_duration = time.time() - start_time

        latencies.sort()
        n = len(latencies)

        result = BenchmarkResult(
            test_name="full_model",
            total_requests=self.config.test_iterations,
            successful_requests=self.config.test_iterations,
            failed_requests=0,
            total_duration_sec=total_duration,
            throughput=self.config.test_iterations / total_duration,
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=latencies[int(n * 0.50)],
            p95_latency_ms=latencies[int(n * 0.95)],
            p99_latency_ms=latencies[int(n * 0.99)],
            avg_memory_mb=statistics.mean(memory_samples) if memory_samples else 0,
            peak_memory_mb=get_peak_memory() if torch.cuda.is_available() else 0,
        )

        print(f"[Benchmark] Full Model: {result.throughput:.2f} req/s, "
              f"latency: {result.avg_latency_ms:.3f} ms")

        return result

    def run_all(self) -> List[BenchmarkResult]:
        """运行所有测试"""
        print("=" * 60)
        print("CGC + FlashKDA Benchmark Suite")
        print("=" * 60)
        print(f"Config: batch={self.config.batch_size}, seq_len={self.config.seq_len}, "
              f"heads={self.config.num_heads}, head_dim={self.config.head_dim}")

        self.results = []

        if self.config.enable_flashkda and FLASHKDA_AVAILABLE:
            result = self.benchmark_kda_forward()
            if result:
                self.results.append(result)

        result = self.benchmark_sdpa()
        if result:
            self.results.append(result)

        for num_threads in [1, 2, 4, 8]:
            if num_threads <= (torch.cuda.device_count() if torch.cuda.is_available() else 1):
                result = self.benchmark_concurrent(num_threads)
                if result:
                    self.results.append(result)

        if CGC_AVAILABLE:
            result = self.benchmark_full_model()
            if result:
                self.results.append(result)

        return self.results

    def save_results(self, path: str = "benchmark_results.json"):
        """保存压测结果"""
        output = {
            "config": {
                "batch_size": self.config.batch_size,
                "seq_len": self.config.seq_len,
                "num_heads": self.config.num_heads,
                "head_dim": self.config.head_dim,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "test_iterations": self.config.test_iterations,
            },
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_tests": len(self.results),
                "fastest_test": max(self.results, key=lambda r: r.throughput).test_name if self.results else None,
                "lowest_latency": min(self.results, key=lambda r: r.avg_latency_ms).test_name if self.results else None,
            }
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n[Benchmark] Results saved to {path}")

    def print_summary(self):
        """打印结果摘要"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        print(f"{'Test':<20} {'Throughput':<15} {'Avg Latency':<15} {'P99 Latency':<15} {'Peak Memory':<15}")
        print("-" * 80)

        for result in self.results:
            print(f"{result.test_name:<20} "
                  f"{result.throughput:<15.2f} "
                  f"{result.avg_latency_ms:<15.3f} ms "
                  f"{result.p99_latency_ms:<15.3f} ms "
                  f"{result.peak_memory_mb:<15.2f} MB")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="CGC + FlashKDA Benchmark")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--no-flashkda", action="store_true")
    parser.add_argument("--no-memory", action="store_true")

    args = parser.parse_args()

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        test_iterations=args.iterations,
        warmup_iterations=args.warmup,
        enable_flashkda=not args.no_flashkda,
        memory_test=not args.no_memory,
    )

    benchmark = CGCBenchmark(config)
    benchmark.run_all()
    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
