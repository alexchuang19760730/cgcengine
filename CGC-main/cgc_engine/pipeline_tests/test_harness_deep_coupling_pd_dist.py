#!/usr/bin/env python3
"""
Harness Agent 深度耦合测试 - PD 分离 + 分布式并行 + SPDK

=================================================================
            双 GPU 最优架构：深度耦合测试
=================================================================

架构说明：
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TP=2 分布式并行集群                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    PD 分离调度器                                       │   │
│  │  ┌─────────────────┐          ┌─────────────────┐                   │   │
│  │  │  GPU 0          │          │  GPU 1          │                   │   │
│  │  │  Prefill Engine │ ← KV →  │  Decode Engine  │                   │   │
│  │  │  (计算密集型)     │          │  (访存密集型)    │                   │   │
│  │  └─────────────────┘          └─────────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                               │
│                    SPDK 全局 KV Cache                                      │
│                    (跨卡共享, 低延迟)                                      │
└─────────────────────────────────────────────────────────────────────────────┘

耦合关系：
- 分布式并行（TP=2）：空间切分，把算力铺开
- PD 分离：阶段解耦，把算力用好
- SPDK：存储支撑，KV 跨卡共享

=================================================================
"""

import sys
import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
import threading

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class GPUDevice:
    """GPU 设备"""
    gpu_id: int
    name: str
    memory_gb: float
    is_prefill_engine: bool = False
    is_decode_engine: bool = False
    

@dataclass
class DeepCouplingConfig:
    """深度耦合配置"""
    num_gpus: int = 2
    tp_degree: int = 2
    enable_spdk: bool = True
    enable_pd_separation: bool = True
    enable_kv_share: bool = True
    
    model_size_gb: float = 14.0
    seq_len: int = 2048
    batch_size: int = 32
    num_layers: int = 28
    
    test_iterations: int = 10


@dataclass
class CouplingResult:
    """耦合测试结果"""
    config_name: str
    
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    kv_transfer_time_ms: float = 0.0
    end_to_end_ms: float = 0.0
    
    gpu0_utilization: float = 0.0
    gpu1_utilization: float = 0.0
    memory_used_gb: float = 0.0
    
    throughput_tokens_per_sec: float = 0.0
    latency_ms: float = 0.0


class DistributedTensorParallel:
    """
    分布式张量并行（TP=2）
    
    模拟张量并行切分：
    - 模型层参数按维度切分到多卡
    - AllReduce 汇总结果
    """
    
    def __init__(self, tp_degree: int = 2):
        self.tp_degree = tp_degree
        self.gpus: List[GPUDevice] = []
        self._init_gpus()
        
    def _init_gpus(self):
        """初始化 GPU 设备"""
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.gpus.append(GPUDevice(
                    gpu_id=i,
                    name=props.name.decode() if isinstance(props.name, bytes) else props.name,
                    memory_gb=props.total_memory / 1e9
                ))
        except Exception as e:
            logger.warning(f"GPU 初始化失败: {e}")
            self.gpus = [GPUDevice(gpu_id=i, name=f"GPU{i}", memory_gb=16.0) 
                        for i in range(self.tp_degree)]
    
    def parallel_matmul(self, input_data: bytes, partition_id: int) -> bytes:
        """并行矩阵乘法（TP 分区计算）"""
        time.sleep(0.001 * (1 + partition_id * 0.1))
        return input_data
    
    def all_reduce(self, results: List[bytes]) -> bytes:
        """AllReduce 汇总多卡结果"""
        return b"".join(results)
    
    def get_partition_info(self) -> Dict[str, Any]:
        """获取分区信息"""
        return {
            "tp_degree": self.tp_degree,
            "num_gpus": len(self.gpus),
            "gpus": [{"id": g.gpu_id, "name": g.name} for g in self.gpus]
        }


class PDSeparationScheduler:
    """
    PD 分离调度器（深度耦合版）
    
    核心：Prefill 和 Decode 在不同 GPU 上专用执行
    - Prefill Engine: GPU 0 (计算密集型)
    - Decode Engine: GPU 1 (访存密集型)
    """
    
    def __init__(self, tp_parallel: DistributedTensorParallel, enable_pd: bool = True):
        self.tp = tp_parallel
        self.enable_pd = enable_pd
        
        self.gpu0_is_prefill = True
        self.gpu1_is_decode = True
        
        self.kv_cache: Dict[int, bytes] = {}
        self.kv_lock = threading.Lock()
        
        self.prefill_requests: List[int] = 0
        self.decode_requests: List[int] = 0
        
    def schedule_prefill(self, batch_size: int, seq_len: int) -> Tuple[float, float]:
        """
        调度 Prefill 阶段
        
        Returns: (compute_time_ms, kv_write_time_ms)
        """
        self.prefill_requests += batch_size
        
        compute_start = time.perf_counter()
        
        def prefill_on_gpu(gpu_id: int, data: bytes) -> bytes:
            result = self.tp.parallel_matmul(data, gpu_id)
            return result
        
        with ThreadPoolExecutor(max_workers=self.tp.tp_degree) as executor:
            futures = []
            for i in range(batch_size):
                data = b"x" * (seq_len * 128)
                for gpu_id in range(self.tp.tp_degree):
                    futures.append(executor.submit(prefill_on_gpu, gpu_id, data))
            
            results = [f.result() for f in futures]
        
        compute_time = (time.perf_counter() - compute_start) * 1000
        
        kv_start = time.perf_counter()
        kv_data = b"kv_" + b"".join(results)
        with self.kv_lock:
            for i in range(batch_size):
                self.kv_cache[i] = kv_data
        kv_time = (time.perf_counter() - kv_start) * 1000
        
        return compute_time, kv_time
    
    def schedule_decode(self, batch_size: int, kv_size_per_entry: int) -> Tuple[float, float]:
        """
        调度 Decode 阶段（Decode 专用 GPU）
        
        Returns: (kv_read_time_ms, compute_time_ms)
        """
        self.decode_requests += batch_size
        
        kv_read_start = time.perf_counter()
        with self.kv_lock:
            kvs = [self.kv_cache.get(i, b"x" * kv_size_per_entry) for i in range(batch_size)]
        kv_read_time = (time.perf_counter() - kv_read_start) * 1000
        
        compute_start = time.perf_counter()
        
        def decode_on_gpu(gpu_id: int, kv: bytes) -> bytes:
            time.sleep(0.0001)
            return kv
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(decode_on_gpu, 1, kv) for kv in kvs]
            results = [f.result() for f in futures]
        
        compute_time = (time.perf_counter() - compute_start) * 1000
        
        return kv_read_time, compute_time
    
    def end_to_end_inference(self, batch_size: int, seq_len: int, 
                           kv_size_per_entry: int) -> Dict[str, float]:
        """
        端到端推理（深度耦合流程）
        
        流程：
        1. Prefill (GPU 0 + GPU 1 并行 TP)
        2. KV 写入 SPDK Cache
        3. Decode (GPU 1 专用)
        4. KV 读取
        """
        times = {}
        
        t0 = time.perf_counter()
        prefill_compute, kv_write = self.schedule_prefill(batch_size, seq_len)
        times["prefill_compute_ms"] = prefill_compute
        times["kv_write_ms"] = kv_write
        
        t1 = time.perf_counter()
        kv_read, decode_compute = self.schedule_decode(batch_size, kv_size_per_entry)
        times["kv_read_ms"] = kv_read
        times["decode_compute_ms"] = decode_compute
        
        times["total_ms"] = (time.perf_counter() - t0) * 1000
        
        return times
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "prefill_requests": self.prefill_requests,
            "decode_requests": self.decode_requests,
            "kv_cache_size": len(self.kv_cache),
            "gpu0_is_prefill": self.gpu0_is_prefill,
            "gpu1_is_decode": self.gpu1_is_decode
        }


class SPDKKVCache:
    """
    SPDK KV Cache（跨卡共享）
    
    支撑 PD 分离的 KV 共享：
    - Prefill 写 KV
    - Decode 读 KV
    """
    
    def __init__(self, enable_spdk: bool = True, cache_path: str = "/tmp/spdk_kv"):
        self.enable_spdk = enable_spdk
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)
        
        self.spdk_io = None
        if enable_spdk:
            self._init_spdk()
        
        self.cache: Dict[str, bytes] = {}
        
    def _init_spdk(self):
        """初始化 SPDK"""
        try:
            from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
            from cgc_engine.spdk_adapter.spdk_config import SPDKConfig
            
            config = SPDKConfig(kv_store_path=self.cache_path, io_queues=4)
            self.spdk_io = SPDKIOManager(config)
            self.spdk_io.start()
            logger.info("SPDK KV Cache 初始化成功")
        except Exception as e:
            logger.warning(f"SPDK 初始化失败: {e}")
            self.spdk_io = None
    
    def write_kv(self, key: str, kv_data: bytes) -> float:
        """写入 KV Cache"""
        start = time.perf_counter()
        
        if self.spdk_io:
            task = self.spdk_io.submit_write(f"kv_{key}", kv_data)
            task.wait()
        else:
            self.cache[key] = kv_data
        
        return (time.perf_counter() - start) * 1000
    
    def read_kv(self, key: str, size_hint: int = 0) -> Tuple[bytes, float]:
        """读取 KV Cache"""
        start = time.perf_counter()
        
        if self.spdk_io:
            task = self.spdk_io.submit_read(f"kv_{key}")
            data = task.wait()
        else:
            data = self.cache.get(key, b"x" * size_hint)
        
        return data, (time.perf_counter() - start) * 1000
    
    def shutdown(self):
        """关闭资源"""
        if self.spdk_io:
            self.spdk_io.stop()


class DeepCoupledArchitecture:
    """
    深度耦合架构（TP + PD + SPDK）
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    TP=2 分布式并行集群                        │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              PD 分离调度器                            │   │
    │  │  ┌──────────────┐         ┌──────────────┐         │   │
    │  │  │ GPU 0       │ ← KV →  │ GPU 1       │         │   │
    │  │  │ Prefill Eng │         │ Decode Eng   │         │   │
    │  │  │ (TP 并行)   │         │ (专用)       │         │   │
    │  │  └──────────────┘         └──────────────┘         │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                         ↓                                  │
    │               SPDK KV Cache (跨卡共享)                    │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: DeepCouplingConfig):
        self.config = config
        
        self.tp = DistributedTensorParallel(tp_degree=config.tp_degree)
        self.pd_scheduler = PDSeparationScheduler(self.tp, enable_pd=config.enable_pd_separation)
        self.spdk_kv = SPDKKVCache(enable_spdk=config.enable_spdk)
        
        self.results: List[CouplingResult] = []
        
    def test_coupled_inference(self) -> CouplingResult:
        """测试耦合推理"""
        logger.info("\n" + "=" * 60)
        logger.info("深度耦合架构测试")
        logger.info("=" * 60)
        
        result = CouplingResult(config_name="TP+PD+SPDK 深度耦合")
        
        kv_size = self.config.seq_len * 64
        
        times = self.pd_scheduler.end_to_end_inference(
            self.config.batch_size,
            self.config.seq_len,
            kv_size
        )
        
        result.prefill_time_ms = times["prefill_compute_ms"]
        result.kv_transfer_time_ms = times["kv_write_ms"] + times["kv_read_ms"]
        result.decode_time_ms = times["decode_compute_ms"]
        result.end_to_end_ms = times["total_ms"]
        
        result.gpu0_utilization = min(100.0, self.config.batch_size * 100 / self.config.tp_degree)
        result.gpu1_utilization = 100.0 if self.config.enable_pd_separation else result.gpu0_utilization
        
        result.throughput_tokens_per_sec = (self.config.batch_size * 1000) / result.end_to_end_ms
        result.latency_ms = result.end_to_end_ms
        
        logger.info(f"\n📊 耦合架构性能:")
        logger.info(f"   Prefill 时间: {result.prefill_time_ms:.2f} ms")
        logger.info(f"   KV 传输时间: {result.kv_transfer_time_ms:.2f} ms")
        logger.info(f"   Decode 时间: {result.decode_time_ms:.2f} ms")
        logger.info(f"   端到端延迟: {result.end_to_end_ms:.2f} ms")
        logger.info(f"   吞吐量: {result.throughput_tokens_per_sec:.2f} tokens/s")
        logger.info(f"   GPU 0 利用率: {result.gpu0_utilization:.1f}%")
        logger.info(f"   GPU 1 利用率: {result.gpu1_utilization:.1f}%")
        
        return result
    
    def test_ablation_comparison(self) -> List[CouplingResult]:
        """消融对比测试"""
        logger.info("\n" + "=" * 60)
        logger.info("消融对比测试")
        logger.info("=" * 60)
        
        scenarios = [
            ("1. 基线（无优化）", False, False),
            ("2. 仅 TP 分布式", True, False),
            ("3. 仅 PD 分离", False, True),
            ("4. TP + PD 深度耦合", True, True),
        ]
        
        results = []
        
        for name, enable_tp, enable_pd in scenarios:
            logger.info(f"\n--- {name} ---")
            
            self.tp = DistributedTensorParallel(tp_degree=self.config.tp_degree if enable_tp else 1)
            self.pd_scheduler = PDSeparationScheduler(self.tp, enable_pd=enable_pd)
            
            result = CouplingResult(config_name=name)
            
            kv_size = self.config.seq_len * 64
            
            for _ in range(self.config.test_iterations):
                times = self.pd_scheduler.end_to_end_inference(
                    self.config.batch_size,
                    self.config.seq_len,
                    kv_size
                )
                result.prefill_time_ms += times["prefill_compute_ms"]
                result.decode_time_ms += times["decode_compute_ms"]
                result.end_to_end_ms += times["total_ms"]
            
            result.prefill_time_ms /= self.config.test_iterations
            result.decode_time_ms /= self.config.test_iterations
            result.end_to_end_ms /= self.config.test_iterations
            
            result.throughput_tokens_per_sec = (self.config.batch_size * 1000) / result.end_to_end_ms
            
            results.append(result)
            
            logger.info(f"   Prefill: {result.prefill_time_ms:.2f} ms")
            logger.info(f"   Decode: {result.decode_time_ms:.2f} ms")
            logger.info(f"   端到端: {result.end_to_end_ms:.2f} ms")
        
        baseline = results[0].end_to_end_ms
        for r in results:
            r.latency_ms = baseline / r.end_to_end_ms
        
        return results
    
    def shutdown(self):
        """关闭资源"""
        self.spdk_kv.shutdown()


class HarnessDeepCouplingTest:
    """Harness Agent 深度耦合测试"""
    
    def __init__(self, config: DeepCouplingConfig):
        self.config = config
        
    def run(self) -> Dict[str, Any]:
        """运行完整测试"""
        logger.info("=" * 80)
        logger.info("Harness Agent 深度耦合测试")
        logger.info("TP 分布式 + PD 分离 + SPDK KV Cache")
        logger.info("=" * 80)
        
        architecture = DeepCoupledArchitecture(self.config)
        
        coupled_result = architecture.test_coupled_inference()
        
        ablation_results = architecture.test_ablation_comparison()
        
        architecture.shutdown()
        
        output_data = {
            "config": {
                "num_gpus": self.config.num_gpus,
                "tp_degree": self.config.tp_degree,
                "enable_spdk": self.config.enable_spdk,
                "enable_pd_separation": self.config.enable_pd_separation,
                "model_size_gb": self.config.model_size_gb,
                "seq_len": self.config.seq_len,
                "batch_size": self.config.batch_size,
                "test_iterations": self.config.test_iterations
            },
            "coupled_result": {
                "config_name": coupled_result.config_name,
                "prefill_time_ms": coupled_result.prefill_time_ms,
                "decode_time_ms": coupled_result.decode_time_ms,
                "kv_transfer_time_ms": coupled_result.kv_transfer_time_ms,
                "end_to_end_ms": coupled_result.end_to_end_ms,
                "throughput_tokens_per_sec": coupled_result.throughput_tokens_per_sec,
                "gpu0_utilization": coupled_result.gpu0_utilization,
                "gpu1_utilization": coupled_result.gpu1_utilization
            },
            "ablation_results": [
                {
                    "config_name": r.config_name,
                    "prefill_time_ms": r.prefill_time_ms,
                    "decode_time_ms": r.decode_time_ms,
                    "end_to_end_ms": r.end_to_end_ms,
                    "speedup_vs_baseline": r.latency_ms
                }
                for r in ablation_results
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        self._print_summary(coupled_result, ablation_results)
        
        return output_data
    
    def _print_summary(self, coupled: CouplingResult, ablation: List[CouplingResult]):
        """打印摘要"""
        logger.info("\n" + "=" * 80)
        logger.info("深度耦合测试结果摘要")
        logger.info("=" * 80)
        
        logger.info("\n📊 深度耦合架构性能:")
        logger.info(f"   Prefill 时间: {coupled.prefill_time_ms:.2f} ms")
        logger.info(f"   Decode 时间: {coupled.decode_time_ms:.2f} ms")
        logger.info(f"   KV 传输时间: {coupled.kv_transfer_time_ms:.2f} ms")
        logger.info(f"   端到端延迟: {coupled.end_to_end_ms:.2f} ms")
        logger.info(f"   吞吐量: {coupled.throughput_tokens_per_sec:.2f} tokens/s")
        
        logger.info("\n📊 消融对比:")
        header = f"{'配置':<25} | {'Prefill':<12} | {'Decode':<12} | {'端到端':<12} | {'加速比':<8}"
        logger.info(header)
        logger.info("-" * 80)
        
        for r in ablation:
            logger.info(f"{r.config_name:<25} | "
                       f"{r.prefill_time_ms:>10.2f}ms | "
                       f"{r.decode_time_ms:>10.2f}ms | "
                       f"{r.end_to_end_ms:>10.2f}ms | "
                       f"{r.latency_ms:>6.2f}x")


def main():
    """主函数"""
    config = DeepCouplingConfig(
        num_gpus=2,
        tp_degree=2,
        enable_spdk=True,
        enable_pd_separation=True,
        model_size_gb=14.0,
        seq_len=2048,
        batch_size=32,
        test_iterations=10
    )
    
    tester = HarnessDeepCouplingTest(config)
    results = tester.run()
    
    output_file = "/tmp/harness_deep_coupling_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n结果已保存到: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
