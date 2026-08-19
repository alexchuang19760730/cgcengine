#!/usr/bin/env python3
"""
Harness Agent 驱动的 Multi-Batch Prefill (MTP) 机制验证
"""

import time
import logging
import torch
from typing import List
from dataclasses import dataclass

try:
    from cgc_engine.cgc.multi_batch_prefill import (
        MultiBatchPrefillScheduler,
        PrefillRequest,
    )
    MTP_AVAILABLE = True
except ImportError:
    MTP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MTPBenchmarkResult:
    """MTP 基准测试结果"""
    backend: str
    batch_size: int
    seq_len: int
    throughput: float  # tokens/s
    avg_latency: float  # ms
    max_memory: float  # MB
    optimization_gain: float = 0.0


class MTPHarnessAgent:
    def __init__(self, device: str = "cpu"):
        self.device = str(device)

    def _create_requests(self, batch_size: int, seq_len: int) -> List["PrefillRequest"]:
        requests: List[PrefillRequest] = []
        for i in range(batch_size):
            input_ids = torch.randint(0, 32000, (seq_len,), device="cpu")
            req = PrefillRequest(
                request_id=f"req_{i}",
                input_ids=input_ids,
                block_ids=list(range(max(1, seq_len // 32))),
                priority=0,
            )
            requests.append(req)
        return requests

    def run_mtp(self, batch_size: int, seq_len: int, num_runs: int = 5) -> MTPBenchmarkResult:
        scheduler = MultiBatchPrefillScheduler(max_batch_size=batch_size, max_seq_len=seq_len)

        warmup = self._create_requests(batch_size, seq_len)
        for r in warmup:
            scheduler.add_request(r)
        _ = scheduler.execute_batch()

        latencies: List[float] = []
        start_time = time.time()
        for _i in range(num_runs):
            scheduler.clear()
            reqs = self._create_requests(batch_size, seq_len)
            for r in reqs:
                scheduler.add_request(r)
            run_start = time.time()
            _ = scheduler.execute_batch()
            latencies.append((time.time() - run_start) * 1000)
        total_time = time.time() - start_time

        avg_latency = sum(latencies) / len(latencies)
        total_tokens = batch_size * seq_len * num_runs
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        max_memory = (batch_size * seq_len * 8) / (1024 * 1024)

        return MTPBenchmarkResult(
            backend="MultiBatchPrefillScheduler",
            batch_size=batch_size,
            seq_len=seq_len,
            throughput=throughput,
            avg_latency=avg_latency,
            max_memory=max_memory,
            optimization_gain=1.0,
        )


def main():
    """主函数"""
    print("="*80)
    print("🤖 Harness Agent - Multi-Batch Prefill (MTP) 验证")
    print("="*80)

    if not MTP_AVAILABLE:
        print("⚠ MultiBatchPrefillScheduler 不可用")
        return

    agent = MTPHarnessAgent(device="cpu")
    batch_sizes = [1, 4, 8]
    seq_lens = [128, 256, 512]

    print("\n{:<28} {:<8} {:<10} {:<15} {:<15}".format("Backend", "Batch", "SeqLen", "Throughput", "Latency(ms)"))
    print("-" * 80)
    for bs in batch_sizes:
        for sl in seq_lens:
            res = agent.run_mtp(bs, sl)
            print("{:<28} {:<8} {:<10} {:<15.2f} {:<15.2f}".format(res.backend, res.batch_size, res.seq_len, res.throughput, res.avg_latency))

    print("\n" + "="*80)
    print("🎉 MTP 验证完成！")
    print("="*80)


if __name__ == "__main__":
    main()
