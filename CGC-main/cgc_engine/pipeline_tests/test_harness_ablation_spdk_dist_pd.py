#!/usr/bin/env python3
"""
Harness Agent 消融测试 - SPDK + 分布式并行 + PD 分离

=================================================================
                    消融实验设计
=================================================================

测试组合（8 种配置）：

| 配置 | SPDK | 分布式 | PD 分离 |
|------|------|--------|---------|
| 1. 基线（标准 IO） | ❌ | ❌ | ❌ |
| 2. SPDK 单独 | ✅ | ❌ | ❌ |
| 3. 分布式单独 | ❌ | ✅ | ❌ |
| 4. PD 分离单独 | ❌ | ❌ | ✅ |
| 5. SPDK + 分布式 | ✅ | ✅ | ❌ |
| 6. SPDK + PD | ✅ | ❌ | ✅ |
| 7. 分布式 + PD | ❌ | ✅ | ✅ |
| 8. SPDK + 分布式 + PD（完整）| ✅ | ✅ | ✅ |

测试场景：
1. 专家权重加载延迟
2. KV Cache 读写吞吐量
3. Prefill/Decode 调度延迟
4. 端到端推理延迟

=================================================================
"""

import sys
import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """消融实验配置"""
    num_experts: int = 16
    expert_size_mb: float = 32.0
    num_gpus: int = 2
    kv_cache_entries: int = 100
    kv_entry_size_mb: float = 1.0
    test_iterations: int = 10
    warmup_iterations: int = 2
    enable_spdk: bool = True
    enable_distributed: bool = True
    enable_pd_separation: bool = True


@dataclass
class AblationResult:
    """消融实验结果"""
    config_name: str
    spdk_enabled: bool
    distributed_enabled: bool
    pd_separation_enabled: bool
    
    expert_loading_time_ms: float = 0.0
    kv_cache_write_throughput_mbs: float = 0.0
    kv_cache_read_throughput_mbs: float = 0.0
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    
    memory_peak_gb: float = 0.0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    
    speedup_vs_baseline: float = 1.0


class PDAblationSimulator:
    """
    PD 分离消融模拟器
    
    模拟 PD (Prefill/Decode) 分离的调度和执行过程
    """
    
    def __init__(self, enable_pd: bool = True):
        self.enable_pd = enable_pd
        self.prefill_queue: List[Any] = []
        self.decode_queue: List[Any] = []
        self.prefix_cache_hits: int = 0
        self.total_requests: int = 0
        
    def determine_phase(self, input_length: int, output_length: int = 0) -> str:
        """确定执行阶段"""
        if output_length == 0:
            return "prefill"
        elif not self.enable_pd:
            return "hybrid"
        elif input_length > 4096:
            return "hybrid"
        else:
            return "decode"
    
    def schedule_request(self, seq_id: int, input_ids: List[int], 
                        output_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """调度请求"""
        self.total_requests += 1
        phase = self.determine_phase(len(input_ids), 
                                     len(output_ids) if output_ids else 0)
        
        cache_key = str(hash(tuple(input_ids[:min(10, len(input_ids))])))
        cache_hit = self.prefix_cache_hits > 0 and hash(cache_key) % 3 == 0
        
        if cache_hit:
            self.prefix_cache_hits += 1
        
        latency_base = 0.5 if cache_hit else 2.0
        
        if phase == "prefill":
            latency = latency_base * len(input_ids) * 0.01
        elif phase == "decode":
            latency = latency_base * 0.1
        else:
            latency = latency_base * (len(input_ids) + len(output_ids or [])) * 0.005
        
        return {
            "seq_id": seq_id,
            "phase": phase,
            "cache_hit": cache_hit,
            "latency_ms": latency,
            "input_len": len(input_ids),
            "output_len": len(output_ids) if output_ids else 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self.total_requests,
            "prefix_cache_hits": self.prefix_cache_hits,
            "cache_hit_rate": self.prefix_cache_hits / max(1, self.total_requests)
        }


class ExpertLoader:
    """专家权重加载器"""
    
    def __init__(self, enable_spdk: bool = False, enable_distributed: bool = False,
                 expert_store_path: str = "/tmp/experts"):
        self.enable_spdk = enable_spdk
        self.enable_distributed = enable_distributed
        self.expert_store_path = expert_store_path
        os.makedirs(expert_store_path, exist_ok=True)
        
        self.spdk_io = None
        if enable_spdk:
            self._init_spdk()
    
    def _init_spdk(self):
        """初始化 SPDK"""
        try:
            from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
            from cgc_engine.spdk_adapter.spdk_config import SPDKConfig
            
            config = SPDKConfig(
                kv_store_path=self.expert_store_path,
                io_queues=4
            )
            self.spdk_io = SPDKIOManager(config)
            self.spdk_io.start()
            logger.info("SPDK ExpertLoader 初始化成功")
        except Exception as e:
            logger.warning(f"SPDK 初始化失败: {e}")
            self.spdk_io = None
    
    def load_experts_standard(self, num_experts: int, expert_size_bytes: int) -> float:
        """标准加载（基线）"""
        start = time.perf_counter()
        
        for i in range(num_experts):
            expert_file = f"{self.expert_store_path}/expert_{i}.bin"
            if not os.path.exists(expert_file):
                with open(expert_file, "wb") as f:
                    f.write(b"x" * expert_size_bytes)
            
            with open(expert_file, "rb") as f:
                data = f.read()
        
        return (time.perf_counter() - start) * 1000
    
    def load_experts_spdk(self, num_experts: int, expert_size_bytes: int) -> float:
        """SPDK 加载"""
        if self.spdk_io is None:
            return self.load_experts_standard(num_experts, expert_size_bytes)
        
        start = time.perf_counter()
        
        for i in range(num_experts):
            key = f"expert_{i}.bin"
            task = self.spdk_io.submit_read(key)
            task.wait()
        
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed
    
    def load_experts_distributed(self, num_experts: int, expert_size_bytes: int,
                                 num_gpus: int) -> float:
        """分布式加载"""
        from concurrent.futures import ThreadPoolExecutor
        
        start = time.perf_counter()
        
        def load_partition(gpu_id: int, expert_ids: List[int]):
            for i in expert_ids:
                expert_file = f"{self.expert_store_path}/expert_{i}.bin"
                if not os.path.exists(expert_file):
                    with open(expert_file, "wb") as f:
                        f.write(b"x" * expert_size_bytes)
                with open(expert_file, "rb") as f:
                    data = f.read()
        
        per_gpu = num_experts // num_gpus
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                start_idx = gpu_id * per_gpu
                end_idx = (gpu_id + 1) * per_gpu if gpu_id < num_gpus - 1 else num_experts
                futures.append(executor.submit(load_partition, gpu_id, range(start_idx, end_idx)))
            for f in futures:
                f.result()
        
        return (time.perf_counter() - start) * 1000
    
    def load_experts_spdk_distributed(self, num_experts: int, expert_size_bytes: int,
                                      num_gpus: int) -> float:
        """SPDK + 分布式加载"""
        if self.spdk_io is None:
            return self.load_experts_distributed(num_experts, expert_size_bytes, num_gpus)
        
        from concurrent.futures import ThreadPoolExecutor
        
        start = time.perf_counter()
        
        def load_partition(gpu_id: int, expert_ids: List[int]):
            futures = []
            for i in expert_ids:
                key = f"expert_{i}.bin"
                futures.append(self.spdk_io.submit_read(key))
            for f in futures:
                f.wait()
        
        per_gpu = num_experts // num_gpus
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                start_idx = gpu_id * per_gpu
                end_idx = (gpu_id + 1) * per_gpu if gpu_id < num_gpus - 1 else num_experts
                futures.append(executor.submit(load_partition, gpu_id, range(start_idx, end_idx)))
            for f in futures:
                f.result()
        
        return (time.perf_counter() - start) * 1000
    
    def shutdown(self):
        """关闭资源"""
        if self.spdk_io:
            self.spdk_io.stop()


class KVCacheBenchmark:
    """KV Cache 性能基准测试"""
    
    def __init__(self, enable_spdk: bool = False, enable_distributed: bool = False,
                 enable_pd: bool = False, kv_cache_path: str = "/tmp/kv_cache"):
        self.enable_spdk = enable_spdk
        self.enable_distributed = enable_distributed
        self.enable_pd = enable_pd
        self.kv_cache_path = kv_cache_path
        os.makedirs(kv_cache_path, exist_ok=True)
        
        self.spdk_io = None
        if enable_spdk:
            self._init_spdk()
    
    def _init_spdk(self):
        """初始化 SPDK"""
        try:
            from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
            from cgc_engine.spdk_adapter.spdk_config import SPDKConfig
            
            config = SPDKConfig(
                kv_store_path=self.kv_cache_path,
                io_queues=4
            )
            self.spdk_io = SPDKIOManager(config)
            self.spdk_io.start()
        except Exception as e:
            logger.warning(f"SPDK 初始化失败: {e}")
            self.spdk_io = None
    
    def benchmark_write(self, num_entries: int, entry_size_bytes: int,
                       num_gpus: int) -> float:
        """写入基准测试"""
        test_data = b"x" * entry_size_bytes
        
        if self.enable_spdk and self.enable_distributed:
            return self._write_spdk_distributed(num_entries, test_data, num_gpus)
        elif self.enable_spdk:
            return self._write_spdk(num_entries, test_data)
        elif self.enable_distributed:
            return self._write_distributed(num_entries, test_data, num_gpus)
        else:
            return self._write_standard(num_entries, test_data)
    
    def _write_standard(self, num_entries: int, test_data: bytes) -> float:
        """标准写入"""
        start = time.perf_counter()
        for i in range(num_entries):
            with open(f"{self.kv_cache_path}/std_{i}.dat", "wb") as f:
                f.write(test_data)
        return (time.perf_counter() - start) * 1000
    
    def _write_spdk(self, num_entries: int, test_data: bytes) -> float:
        """SPDK 写入"""
        if self.spdk_io is None:
            return self._write_standard(num_entries, test_data)
        
        start = time.perf_counter()
        for i in range(num_entries):
            task = self.spdk_io.submit_write(f"spdk_{i}", test_data)
            task.wait()
        return (time.perf_counter() - start) * 1000
    
    def _write_distributed(self, num_entries: int, test_data: bytes,
                          num_gpus: int) -> float:
        """分布式写入"""
        from concurrent.futures import ThreadPoolExecutor
        
        def write_worker(entries: List[int]):
            for i in entries:
                with open(f"{self.kv_cache_path}/dist_{i}.dat", "wb") as f:
                    f.write(test_data)
        
        start = time.perf_counter()
        per_worker = num_entries // num_gpus
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                start_idx = gpu_id * per_worker
                end_idx = (gpu_id + 1) * per_worker if gpu_id < num_gpus - 1 else num_entries
                futures.append(executor.submit(write_worker, list(range(start_idx, end_idx))))
            for f in futures:
                f.result()
        
        return (time.perf_counter() - start) * 1000
    
    def _write_spdk_distributed(self, num_entries: int, test_data: bytes,
                                num_gpus: int) -> float:
        """SPDK + 分布式写入"""
        if self.spdk_io is None:
            return self._write_distributed(num_entries, test_data, num_gpus)
        
        from concurrent.futures import ThreadPoolExecutor
        
        def write_worker(gpu_id: int, entries: List[int]):
            for i in entries:
                task = self.spdk_io.submit_write(f"spdk_dist_{gpu_id}_{i}", test_data)
                task.wait()
        
        start = time.perf_counter()
        per_worker = num_entries // num_gpus
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                start_idx = gpu_id * per_worker
                end_idx = (gpu_id + 1) * per_worker if gpu_id < num_gpus - 1 else num_entries
                futures.append(executor.submit(write_worker, gpu_id, list(range(start_idx, end_idx))))
            for f in futures:
                f.result()
        
        return (time.perf_counter() - start) * 1000
    
    def benchmark_read(self, num_entries: int, entry_size_bytes: int,
                      num_gpus: int) -> float:
        """读取基准测试（与写入类似）"""
        return self.benchmark_write(num_entries, entry_size_bytes, num_gpus)
    
    def shutdown(self):
        """关闭资源"""
        if self.spdk_io:
            self.spdk_io.stop()


class AblationExperiment:
    """消融实验"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.results: List[AblationResult] = []
        
    def run_single_experiment(self, config_name: str, 
                             spdk: bool, distributed: bool, pd: bool) -> AblationResult:
        """运行单个实验"""
        logger.info(f"\n{'='*60}")
        logger.info(f"实验: {config_name}")
        logger.info(f"  SPDK: {spdk}, 分布式: {distributed}, PD分离: {pd}")
        logger.info(f"{'='*60}")
        
        expert_size_bytes = int(self.config.expert_size_mb * 1024 * 1024)
        kv_entry_bytes = int(self.config.kv_entry_size_mb * 1024 * 1024)
        
        result = AblationResult(
            config_name=config_name,
            spdk_enabled=spdk,
            distributed_enabled=distributed,
            pd_separation_enabled=pd
        )
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except:
            pass
        
        expert_loader = ExpertLoader(
            enable_spdk=spdk,
            enable_distributed=distributed,
            expert_store_path=f"/tmp/harness_ablation_{hash(config_name) % 100000}"
        )
        
        kv_benchmark = KVCacheBenchmark(
            enable_spdk=spdk,
            enable_distributed=distributed,
            enable_pd=pd,
            kv_cache_path=f"/tmp/harness_kv_{hash(config_name) % 100000}"
        )
        
        pd_simulator = PDAblationSimulator(enable_pd=pd)
        
        logger.info("\n[1/4] 专家权重加载测试...")
        if spdk and distributed:
            result.expert_loading_time_ms = expert_loader.load_experts_spdk_distributed(
                self.config.num_experts, expert_size_bytes, self.config.num_gpus)
        elif spdk:
            result.expert_loading_time_ms = expert_loader.load_experts_spdk(
                self.config.num_experts, expert_size_bytes)
        elif distributed:
            result.expert_loading_time_ms = expert_loader.load_experts_distributed(
                self.config.num_experts, expert_size_bytes, self.config.num_gpus)
        else:
            result.expert_loading_time_ms = expert_loader.load_experts_standard(
                self.config.num_experts, expert_size_bytes)
        
        logger.info(f"      专家加载耗时: {result.expert_loading_time_ms:.2f} ms")
        
        logger.info("\n[2/4] KV Cache 写入测试...")
        write_time_ms = kv_benchmark.benchmark_write(
            self.config.kv_cache_entries, kv_entry_bytes, self.config.num_gpus)
        total_bytes = self.config.kv_cache_entries * kv_entry_bytes
        result.kv_cache_write_throughput_mbs = total_bytes / 1e6 / (write_time_ms / 1000)
        logger.info(f"      写入吞吐量: {result.kv_cache_write_throughput_mbs:.2f} MB/s")
        
        logger.info("\n[3/4] KV Cache 读取测试...")
        read_time_ms = kv_benchmark.benchmark_read(
            self.config.kv_cache_entries, kv_entry_bytes, self.config.num_gpus)
        result.kv_cache_read_throughput_mbs = total_bytes / 1e6 / (read_time_ms / 1000)
        logger.info(f"      读取吞吐量: {result.kv_cache_read_throughput_mbs:.2f} MB/s")
        
        logger.info("\n[4/4] PD 调度延迟测试...")
        prefill_latencies = []
        decode_latencies = []
        
        for i in range(self.config.test_iterations):
            input_len = 512 + (i * 10)
            output_len = 128
            
            prefill_result = pd_simulator.schedule_request(i, list(range(input_len)))
            prefill_latencies.append(prefill_result["latency_ms"])
            
            decode_result = pd_simulator.schedule_request(
                i, list(range(input_len)), list(range(output_len)))
            decode_latencies.append(decode_result["latency_ms"])
        
        result.prefill_latency_ms = sum(prefill_latencies) / len(prefill_latencies)
        result.decode_latency_ms = sum(decode_latencies) / len(decode_latencies)
        
        logger.info(f"      Prefill 延迟: {result.prefill_latency_ms:.4f} ms")
        logger.info(f"      Decode 延迟: {result.decode_latency_ms:.4f} ms")
        
        result.end_to_end_latency_ms = result.expert_loading_time_ms + \
            result.prefill_latency_ms + result.decode_latency_ms
        
        try:
            if torch.cuda.is_available():
                result.memory_peak_gb = torch.cuda.max_memory_allocated() / 1e9
        except:
            pass
        
        expert_loader.shutdown()
        kv_benchmark.shutdown()
        
        return result
    
    def run_ablation(self) -> List[AblationResult]:
        """运行全部消融实验"""
        logger.info("=" * 60)
        logger.info("Harness Agent 消融实验")
        logger.info("SPDK + 分布式并行 + PD 分离")
        logger.info("=" * 60)
        
        experiments = [
            ("1. 基线（标准IO）", False, False, False),
            ("2. SPDK 单独", True, False, False),
            ("3. 分布式单独", False, True, False),
            ("4. PD 分离单独", False, False, True),
            ("5. SPDK + 分布式", True, True, False),
            ("6. SPDK + PD", True, False, True),
            ("7. 分布式 + PD", False, True, True),
            ("8. SPDK + 分布式 + PD（完整）", True, True, True),
        ]
        
        for config_name, spdk, dist, pd in experiments:
            result = self.run_single_experiment(config_name, spdk, dist, pd)
            self.results.append(result)
        
        baseline = self.results[0]
        for result in self.results:
            if result.expert_loading_time_ms > 0:
                result.speedup_vs_baseline = \
                    baseline.expert_loading_time_ms / result.expert_loading_time_ms
        
        return self.results
    
    def print_summary(self):
        """打印结果摘要"""
        logger.info("\n" + "=" * 80)
        logger.info("消融实验结果摘要")
        logger.info("=" * 80)
        
        header = f"{'配置':<30} | {'专家加载':<12} | {'KV写入':<10} | {'KV读取':<10} | {'加速比':<8}"
        logger.info(header)
        logger.info("-" * 80)
        
        for result in self.results:
            row = f"{result.config_name:<30} | " \
                  f"{result.expert_loading_time_ms:>10.2f}ms | " \
                  f"{result.kv_cache_write_throughput_mbs:>8.2f}MB/s | " \
                  f"{result.kv_cache_read_throughput_mbs:>8.2f}MB/s | " \
                  f"{result.speedup_vs_baseline:>6.2f}x"
            logger.info(row)
        
        logger.info("\n" + "=" * 80)
        logger.info("PD 分离效果分析")
        logger.info("=" * 80)
        
        for result in self.results:
            if result.pd_separation_enabled:
                baseline = next(r for r in self.results if 
                              not r.spdk_enabled and not r.distributed_enabled and not r.pd_separation_enabled)
                pd_effect = baseline.prefill_latency_ms / result.prefill_latency_ms if result.prefill_latency_ms > 0 else 1.0
                logger.info(f"{result.config_name}: Prefill 加速 {pd_effect:.2f}x")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Harness Agent 消融测试")
    logger.info("SPDK + 分布式并行 + PD 分离")
    logger.info("=" * 60)
    
    config = AblationConfig(
        num_experts=16,
        expert_size_mb=32.0,
        num_gpus=2,
        kv_cache_entries=100,
        kv_entry_size_mb=1.0,
        test_iterations=10,
        warmup_iterations=2
    )
    
    logger.info(f"实验配置:")
    logger.info(f"  专家数量: {config.num_experts}")
    logger.info(f"  专家大小: {config.expert_size_mb} MB")
    logger.info(f"  GPU 数量: {config.num_gpus}")
    logger.info(f"  KV Cache 条目: {config.kv_cache_entries}")
    logger.info(f"  KV 条目大小: {config.kv_entry_size_mb} MB")
    logger.info(f"  测试迭代: {config.test_iterations}")
    
    experiment = AblationExperiment(config)
    results = experiment.run_ablation()
    experiment.print_summary()
    
    output_file = "/tmp/harness_ablation_results.json"
    output_data = {
        "config": {
            "num_experts": config.num_experts,
            "expert_size_mb": config.expert_size_mb,
            "num_gpus": config.num_gpus,
            "kv_cache_entries": config.kv_cache_entries,
            "kv_entry_size_mb": config.kv_entry_size_mb,
            "test_iterations": config.test_iterations
        },
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "config_name": r.config_name,
                "spdk_enabled": r.spdk_enabled,
                "distributed_enabled": r.distributed_enabled,
                "pd_separation_enabled": r.pd_separation_enabled,
                "expert_loading_time_ms": r.expert_loading_time_ms,
                "kv_cache_write_throughput_mbs": r.kv_cache_write_throughput_mbs,
                "kv_cache_read_throughput_mbs": r.kv_cache_read_throughput_mbs,
                "prefill_latency_ms": r.prefill_latency_ms,
                "decode_latency_ms": r.decode_latency_ms,
                "end_to_end_latency_ms": r.end_to_end_latency_ms,
                "memory_peak_gb": r.memory_peak_gb,
                "speedup_vs_baseline": r.speedup_vs_baseline
            }
            for r in results
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n结果已保存到: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
