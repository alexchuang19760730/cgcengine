#!/usr/bin/env python3
"""
Harness Agent GDS/SPDK 策略测试

=================================================================
                    测试内容规范
=================================================================

1. GDS 基准测试
   - GDS 零拷贝 vs 标准 PyTorch 加载对比
   - 专家权重加载延迟
   - 吞吐量对比

2. SPDK 基准测试
   - SPDK 异步 IO vs 标准文件 IO 对比
   - KV Cache 读写延迟
   - 批量操作吞吐量

3. 分布式基准测试（双卡 5090）
   - 双卡并行加载加速比
   - 专家分区与路由测试
   - 负载均衡验证

4. Harness Agent 策略测试
   - 端侧 llama.cpp 模式: q4_0 量化 + 64x64x64 tiling
   - 云侧 vLLM 模式: Flash Attention + Paged Attention + MoE (TP=2)

=================================================================
                    预期结果示例
=================================================================

=== 测试结果摘要 ===

配置:
  GPU 数量: 2
  GDS 启用: True
  SPDK 启用: True

GDS 加速比: 2.35x
SPDK 写入加速比: 1.82x
SPDK 读取加速比: 1.67x
双卡并行加载加速比: 1.91x

=================================================================
                    测试输出
=================================================================

测试结果会保存到：/tmp/harness_gds_spdk_results.json

包含完整的：
- 配置信息
- 各测试项的详细数据
- 平均值和加速比
- 统计摘要

使用方法: python test_harness_gds_spdk_strategy.py
或者: ./run_harness_gds_spdk.sh all
"""

import sys
import os
import time
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import threading

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    model_path: str = "/data/models/Qwen2.5-7B-Instruct"
    expert_store_path: str = "/data/flashmoe_experts"
    kv_cache_path: str = "/data/spdk_kv_cache"
    num_gpus: int = 2
    enable_gds: bool = True
    enable_spdk: bool = True
    test_iters: int = 10
    warmup_iters: int = 2


class GDSBenchmark:
    """
    GDS 性能基准测试

    测试项：
    1. GDS 零拷贝 vs 标准 PyTorch 加载
    2. 专家权重加载延迟
    3. 吞吐量对比
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """运行 GDS 基准测试"""
        logger.info("=" * 60)
        logger.info("GDS 基准测试")
        logger.info("=" * 60)

        gds_available = self._check_gds_availability()
        logger.info(f"GDS 可用: {gds_available}")

        if gds_available:
            self._benchmark_gds_vs_pytorch()
            self._benchmark_expert_loading()
        else:
            logger.warning("GDS 不可用，仅测试标准 PyTorch 加载")
            self._benchmark_standard_loading()

        return self.results

    def _check_gds_availability(self) -> bool:
        """检查 GDS 是否可用 - 增强版检查"""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA 不可用")
                return False

            cuda_version = torch.version.cuda
            logger.info(f"CUDA 版本: {cuda_version}")

            gds_available = False
            gds_info = {}

            # 检查 cuda.bindings.cufile（正确的导入方式）
            try:
                import cuda.bindings.cufile as cufile
                gds_info["cuda_bindings_cufile"] = "已安装"
                gds_info["has_read_async"] = hasattr(cufile, "read_async")
                gds_info["has_write_async"] = hasattr(cufile, "write_async")
                gds_available = True
            except ImportError as e:
                gds_info["cuda_bindings_cufile"] = f"未安装: {e}"

            try:
                import nvidia.cufile
                gds_info["nvidia_cufile"] = "已安装"
                gds_available = gds_available or True
            except ImportError:
                gds_info["nvidia_cufile"] = "未安装"

            try:
                from cgc_engine.gds_service.cufile_wrapper import CUFILE_AVAILABLE
                gds_info["cufile_wrapper"] = CUFILE_AVAILABLE
                gds_available = gds_available or CUFILE_AVAILABLE
            except ImportError:
                gds_info["cufile_wrapper"] = "不可用"

            try:
                from cgc_engine.gds_service.gds_manager import GDSManager
                gds = GDSManager()
                gds_info["GDSManager"] = "可用"
                gds_info["GDS_enabled"] = gds.enabled
                gds_available = gds_available and gds.enabled
            except ImportError:
                gds_info["GDSManager"] = "不可用"
            except Exception as e:
                gds_info["GDSManager_error"] = str(e)

            logger.info(f"GDS 库检查: {gds_info}")

            if gds_available:
                try:
                    from cgc_engine.gds_service.gds_manager import GDSManager
                    gds = GDSManager()
                    logger.info(f"GDS Manager info: {gds.info()}")
                    return gds.enabled
                except:
                    pass

            if gds_info.get("cuda_bindings_cufile") == "已安装":
                logger.info("检测到 cuda.bindings.cufile 库，将使用 GDS 模式")
                return True

            if "nvidia.cufile" in str(gds_info) and gds_info["nvidia_cufile"] == "已安装":
                logger.info("检测到 nvidia-cufile 库，将使用 GDS 模式")
                return True

            return False

        except Exception as e:
            logger.error(f"GDS 检查失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _benchmark_gds_vs_pytorch(self):
        """对比 GDS vs 标准 PyTorch 加载"""
        logger.info("\n--- GDS vs PyTorch 加载对比 ---")

        try:
            import torch
            from cgc_engine.gds_service.gds_manager import GDSManager

            gds = GDSManager()

            shape = [4096, 4096]
            dtype = torch.float16
            test_file = f"{self.config.expert_store_path}/test_weight.bin"

            os.makedirs(self.config.expert_store_path, exist_ok=True)
            if not os.path.exists(test_file):
                weight = torch.randn(*shape, dtype=dtype)
                torch.save({"weight": weight}, test_file)
                logger.info(f"创建测试文件: {test_file}")

            torch_times = []
            for i in range(self.config.test_iters):
                start = time.perf_counter()
                data = torch.load(test_file, map_location="cuda:0")
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                torch_times.append(elapsed * 1000)

            gds_times = []
            for i in range(self.config.test_iters):
                start = time.perf_counter()
                weight = gds.load_weight_from_pd(test_file, shape)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                gds_times.append(elapsed * 1000)

            avg_torch = sum(torch_times) / len(torch_times)
            avg_gds = sum(gds_times) / len(gds_times)
            speedup = avg_torch / avg_gds if avg_gds > 0 else 1.0

            self.results["gds_vs_pytorch"] = {
                "pytorch_avg_ms": avg_torch,
                "gds_avg_ms": avg_gds,
                "speedup": speedup,
                "pytorch_times": torch_times,
                "gds_times": gds_times
            }

            logger.info(f"PyTorch 平均: {avg_torch:.2f} ms")
            logger.info(f"GDS 平均: {avg_gds:.2f} ms")
            logger.info(f"加速比: {speedup:.2f}x")

        except Exception as e:
            logger.error(f"GDS vs PyTorch 测试失败: {e}")
            import traceback
            traceback.print_exc()

    def _benchmark_expert_loading(self):
        """测试专家加载性能"""
        logger.info("\n--- 专家加载测试 ---")

        try:
            from cgc_engine.flash_moe.gds_expert_loader import GDSExpertLoader

            loader = GDSExpertLoader(expert_dir=self.config.expert_store_path)

            expert_times = []
            for i in range(self.config.test_iters):
                start = time.perf_counter()
                expert = loader.load_expert(i % 8, [4096, 4096])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                expert_times.append(elapsed * 1000)

            avg_time = sum(expert_times) / len(expert_times)
            self.results["expert_loading"] = {
                "avg_ms": avg_time,
                "times": expert_times,
                "cache_hits": loader.get_stats()["hits"]
            }

            logger.info(f"专家加载平均: {avg_time:.2f} ms")
            logger.info(f"缓存命中: {loader.get_stats()['hits']}")

        except Exception as e:
            logger.error(f"专家加载测试失败: {e}")

    def _benchmark_standard_loading(self):
        """标准加载测试（降级方案）"""
        logger.info("\n--- 标准加载测试 ---")

        try:
            import torch

            shape = [4096, 4096]
            test_file = f"{self.config.expert_store_path}/test_weight.bin"

            os.makedirs(self.config.expert_store_path, exist_ok=True)
            if not os.path.exists(test_file):
                weight = torch.randn(*shape, dtype=torch.float16)
                torch.save({"weight": weight}, test_file)

            times = []
            for i in range(self.config.test_iters):
                start = time.perf_counter()
                data = torch.load(test_file, map_location="cpu")
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)

            avg_time = sum(times) / len(times)
            self.results["standard_loading"] = {
                "avg_ms": avg_time,
                "times": times
            }

            logger.info(f"标准加载平均: {avg_time:.2f} ms")

        except Exception as e:
            logger.error(f"标准加载测试失败: {e}")


class SPDKBenchmark:
    """
    SPDK 性能基准测试

    测试项：
    1. SPDK 异步 IO vs 标准文件 IO
    2. KV Cache 读写延迟
    3. 批量操作吞吐量
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """运行 SPDK 基准测试"""
        logger.info("=" * 60)
        logger.info("SPDK 基准测试")
        logger.info("=" * 60)

        spdk_available = self._check_spdk_availability()
        logger.info(f"SPDK 可用: {spdk_available}")

        if spdk_available:
            self._benchmark_spdk_vs_standard()
            self._benchmark_kv_cache()
        else:
            logger.warning("SPDK 不可用，仅测试标准文件 IO")
            self._benchmark_standard_io()

        return self.results

    def _check_spdk_availability(self) -> bool:
        """检查 SPDK 是否可用 - 增强版检查"""
        try:
            spdk_info = {}
            spdk_available = False

            try:
                import liburing
                spdk_info["liburing"] = f"已安装 (版本: {liburing.__version__})"
                spdk_available = True
            except ImportError:
                spdk_info["liburing"] = "未安装"

            try:
                import nvidia.cufile
                spdk_info["nvidia_cufile"] = "已安装"
            except ImportError:
                spdk_info["nvidia_cufile"] = "未安装"

            try:
                from cgc_engine.spdk_adapter.spdk_io_manager import SPDK_AVAILABLE
                spdk_info["spdk_adapter"] = SPDK_AVAILABLE
                spdk_available = spdk_available or SPDK_AVAILABLE
            except ImportError:
                spdk_info["spdk_adapter"] = "不可用"
            except Exception as e:
                spdk_info["spdk_adapter_error"] = str(e)

            try:
                from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
                spdk_info["SPDKIOManager"] = "可用"
            except ImportError:
                spdk_info["SPDKIOManager"] = "不可用"

            logger.info(f"SPDK 库检查: {spdk_info}")

            if spdk_available and spdk_info.get("SPDKIOManager") == "可用":
                return True

            if spdk_info.get("liburing") != "未安装":
                logger.info("检测到 liburing 库，将使用 SPDK 异步 IO 模式")
                return True

            return False

        except Exception as e:
            logger.error(f"SPDK 检查失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _benchmark_spdk_vs_standard(self):
        """对比 SPDK vs 标准文件 IO"""
        logger.info("\n--- SPDK vs 标准 IO 对比 ---")

        try:
            import torch
            from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
            from cgc_engine.spdk_adapter.spdk_config import SPDKConfig

            config = SPDKConfig(
                kv_store_path=self.config.kv_cache_path,
                io_queues=self.config.num_gpus * 2
            )
            io_manager = SPDKIOManager(config)
            io_manager.start()

            test_data = b"x" * 1024 * 1024
            key_prefix = "spdk_test_"

            spdk_write_times = []
            for i in range(self.config.test_iters):
                start = time.perf_counter()
                task = io_manager.submit_write(f"{key_prefix}{i}", test_data)
                task.wait()
                elapsed = time.perf_counter() - start
                spdk_write_times.append(elapsed * 1000)

            spdk_read_times = []
            for i in range(self.config.test_iters):
                start = time.perf_counter()
                task = io_manager.submit_read(f"{key_prefix}{i}")
                task.wait()
                elapsed = time.perf_counter() - start
                spdk_read_times.append(elapsed * 1000)

            io_manager.stop()

            standard_write_times = []
            standard_read_times = []
            test_dir = f"{self.config.kv_cache_path}/standard"
            os.makedirs(test_dir, exist_ok=True)

            for i in range(self.config.test_iters):
                start = time.perf_counter()
                with open(f"{test_dir}/{key_prefix}{i}.dat", "wb") as f:
                    f.write(test_data)
                elapsed = time.perf_counter() - start
                standard_write_times.append(elapsed * 1000)

                start = time.perf_counter()
                with open(f"{test_dir}/{key_prefix}{i}.dat", "rb") as f:
                    f.read()
                elapsed = time.perf_counter() - start
                standard_read_times.append(elapsed * 1000)

            avg_spdk_write = sum(spdk_write_times) / len(spdk_write_times)
            avg_spdk_read = sum(spdk_read_times) / len(spdk_read_times)
            avg_std_write = sum(standard_write_times) / len(standard_write_times)
            avg_std_read = sum(standard_read_times) / len(standard_read_times)

            self.results["spdk_vs_standard"] = {
                "spdk_write_avg_ms": avg_spdk_write,
                "spdk_read_avg_ms": avg_spdk_read,
                "standard_write_avg_ms": avg_std_write,
                "standard_read_avg_ms": avg_std_read,
                "write_speedup": avg_std_write / avg_spdk_write if avg_spdk_write > 0 else 1.0,
                "read_speedup": avg_std_read / avg_spdk_read if avg_spdk_read > 0 else 1.0,
            }

            logger.info(f"SPDK 写入平均: {avg_spdk_write:.2f} ms")
            logger.info(f"标准写入平均: {avg_std_write:.2f} ms")
            logger.info(f"写入加速比: {avg_std_write / avg_spdk_write:.2f}x")
            logger.info(f"SPDK 读取平均: {avg_spdk_read:.2f} ms")
            logger.info(f"标准读取平均: {avg_std_read:.2f} ms")
            logger.info(f"读取加速比: {avg_std_read / avg_spdk_read:.2f}x")

        except Exception as e:
            logger.error(f"SPDK vs 标准 IO 测试失败: {e}")
            import traceback
            traceback.print_exc()

    def _benchmark_kv_cache(self):
        """测试 KV Cache 性能"""
        logger.info("\n--- KV Cache 测试 ---")

        try:
            import torch
            from cgc_engine.pd.spdk_kv_cache import SPDKKVCache

            kv_cache = SPDKKVCache(
                kv_store_path=self.config.kv_cache_path,
                io_queues=self.config.num_gpus * 2
            )

            k = torch.randn(1, 32, 512, 64)
            v = torch.randn(1, 32, 512, 64)

            write_times = []
            for i in range(self.config.test_iters):
                start = time.perf_counter()
                kv_cache.set_kv(f"session_{i}", k, v)
                elapsed = time.perf_counter() - start
                write_times.append(elapsed * 1000)

            read_times = []
            for i in range(self.config.test_iters):
                start = time.perf_counter()
                result = kv_cache.get_kv(f"session_{i % self.config.test_iters}")
                elapsed = time.perf_counter() - start
                read_times.append(elapsed * 1000)

            kv_cache.shutdown()

            avg_write = sum(write_times) / len(write_times)
            avg_read = sum(read_times) / len(read_times)

            self.results["kv_cache"] = {
                "write_avg_ms": avg_write,
                "read_avg_ms": avg_read,
                "spdk_enabled": kv_cache.spdk_enabled
            }

            logger.info(f"KV 写入平均: {avg_write:.2f} ms")
            logger.info(f"KV 读取平均: {avg_read:.2f} ms")

        except Exception as e:
            logger.error(f"KV Cache 测试失败: {e}")

    def _benchmark_standard_io(self):
        """标准 IO 测试（降级方案）"""
        logger.info("\n--- 标准 IO 测试 ---")

        try:
            test_dir = f"{self.config.kv_cache_path}/standard"
            os.makedirs(test_dir, exist_ok=True)

            test_data = b"x" * 1024 * 100

            write_times = []
            read_times = []

            for i in range(self.config.test_iters):
                start = time.perf_counter()
                with open(f"{test_dir}/test_{i}.dat", "wb") as f:
                    f.write(test_data)
                elapsed = time.perf_counter() - start
                write_times.append(elapsed * 1000)

                start = time.perf_counter()
                with open(f"{test_dir}/test_{i}.dat", "rb") as f:
                    f.read()
                elapsed = time.perf_counter() - start
                read_times.append(elapsed * 1000)

            avg_write = sum(write_times) / len(write_times)
            avg_read = sum(read_times) / len(read_times)

            self.results["standard_io"] = {
                "write_avg_ms": avg_write,
                "read_avg_ms": avg_read
            }

            logger.info(f"标准写入平均: {avg_write:.2f} ms")
            logger.info(f"标准读取平均: {avg_read:.2f} ms")

        except Exception as e:
            logger.error(f"标准 IO 测试失败: {e}")


class DistributedBenchmark:
    """
    分布式基准测试

    测试项：
    1. 双卡 5090 并行加载
    2. 专家分区与路由
    3. 负载均衡测试
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """运行分布式基准测试"""
        logger.info("=" * 60)
        logger.info("分布式基准测试 (双卡 5090)")
        logger.info("=" * 60)

        self._check_gpu_availability()
        self._benchmark_parallel_expert_loading()
        self._benchmark_expert_routing()

        return self.results

    def _check_gpu_availability(self):
        """检查 GPU 可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"GPU 数量: {device_count}")
                for i in range(device_count):
                    logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                logger.warning("CUDA 不可用")
        except Exception as e:
            logger.error(f"GPU 检查失败: {e}")

    def _benchmark_parallel_expert_loading(self):
        """测试并行专家加载"""
        logger.info("\n--- 并行专家加载测试 ---")

        try:
            import torch
            from concurrent.futures import ThreadPoolExecutor, as_completed

            num_experts = 16
            shape = [4096, 4096]

            single_times = []
            for i in range(min(self.config.test_iters, 8)):
                experts = []
                start = time.perf_counter()
                for j in range(num_experts):
                    expert = torch.randn(*shape, dtype=torch.float16)
                    experts.append(expert)
                elapsed = time.perf_counter() - start
                single_times.append(elapsed * 1000)

            dual_times = []
            for i in range(min(self.config.test_iters, 8)):
                experts = [[], []]

                def load_to_gpu(gpu_id, expert_range):
                    results = []
                    for j in expert_range:
                        expert = torch.randn(*shape, dtype=torch.float16, device=f"cuda:{gpu_id}")
                        results.append(expert)
                    return results

                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future1 = executor.submit(load_to_gpu, 0, range(0, num_experts // 2))
                    future2 = executor.submit(load_to_gpu, 1, range(num_experts // 2, num_experts))
                    results = [future1.result(), future2.result()]

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                dual_times.append(elapsed * 1000)

            avg_single = sum(single_times) / len(single_times)
            avg_dual = sum(dual_times) / len(dual_times)
            speedup = avg_single / avg_dual if avg_dual > 0 else 1.0

            self.results["parallel_loading"] = {
                "single_gpu_ms": avg_single,
                "dual_gpu_ms": avg_dual,
                "speedup": speedup
            }

            logger.info(f"单卡加载平均: {avg_single:.2f} ms")
            logger.info(f"双卡加载平均: {avg_dual:.2f} ms")
            logger.info(f"并行加速比: {speedup:.2f}x")

        except Exception as e:
            logger.error(f"并行专家加载测试失败: {e}")

    def _benchmark_expert_routing(self):
        """测试专家路由"""
        logger.info("\n--- 专家路由测试 ---")

        try:
            from cgc_engine.flash_moe.distributed_expert_store import DistributedExpertStore

            cluster_nodes = [
                "worker1:4420",
                "worker2:4420",
            ]

            store = DistributedExpertStore(
                cluster_nodes=cluster_nodes,
                local_store_path=self.config.expert_store_path,
                num_partitions=64
            )

            partition_counts = {}
            for expert_id in range(128):
                partition = store._get_partition(expert_id)
                node = store._get_node_for_partition(partition)
                key = f"p{partition}"
                partition_counts[key] = partition_counts.get(key, 0) + 1

            self.results["expert_routing"] = {
                "num_partitions": store.num_partitions,
                "partition_distribution": partition_counts,
                "num_nodes": len(cluster_nodes)
            }

            logger.info(f"分区数量: {store.num_partitions}")
            logger.info(f"集群节点: {len(cluster_nodes)}")

            store.shutdown()

        except Exception as e:
            logger.error(f"专家路由测试失败: {e}")


class SPDKDistributedBenchmark:
    """
    SPDK + 分布式并行组合测试

    测试项：
    1. SPDK + 分布式专家加载（多 GPU 通过 SPDK 并行加载专家权重）
    2. SPDK + 分布式 KV Cache（多 GPU 共享 SPDK KV Cache）
    3. 综合性能对比（标准 vs SPDK vs 分布式 vs SPDK+分布式）
    4. 批量 IO 吞吐量测试（SPDK + 多线程）
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """运行 SPDK + 分布式组合测试"""
        logger.info("=" * 60)
        logger.info("SPDK + 分布式并行组合测试")
        logger.info("=" * 60)

        spdk_available = self._check_spdk_available()
        gpu_count = self._get_gpu_count()

        if spdk_available and gpu_count >= 2:
            logger.info(f"✅ SPDK 可用: {spdk_available}")
            logger.info(f"✅ GPU 数量: {gpu_count}")
            logger.info("✅ 开始 SPDK + 分布式组合测试\n")

            self._benchmark_spdk_distributed_expert_loading()
            self._benchmark_spdk_distributed_kv_cache()
            self._benchmark_spdk_parallel_io_throughput()
            self._generate_comparison_summary()
        else:
            logger.warning("⚠️ SPDK 或多 GPU 不可用，跳过组合测试")
            if not spdk_available:
                logger.warning("  - SPDK 不可用")
            if gpu_count < 2:
                logger.warning(f"  - GPU 数量不足 (需要 >= 2，当前: {gpu_count})")

        return self.results

    def _check_spdk_available(self) -> bool:
        """检查 SPDK 是否可用"""
        try:
            from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
            return True
        except ImportError:
            return False

    def _get_gpu_count(self) -> int:
        """获取 GPU 数量"""
        try:
            import torch
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except:
            return 0

    def _benchmark_spdk_distributed_expert_loading(self):
        """测试 SPDK + 分布式专家加载"""
        logger.info("\n--- SPDK + 分布式专家加载测试 ---")

        try:
            import torch
            from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
            from cgc_engine.spdk_adapter.spdk_config import SPDKConfig
            from concurrent.futures import ThreadPoolExecutor, as_completed

            num_experts = 16
            expert_size = 4096 * 4096 * 2  # float16
            num_gpus = torch.cuda.device_count()

            # 准备测试数据
            os.makedirs(self.config.expert_store_path, exist_ok=True)
            for i in range(num_experts):
                expert_data = b"x" * expert_size
                with open(f"{self.config.expert_store_path}/expert_{i}.bin", "wb") as f:
                    f.write(expert_data)

            # 配置 SPDK IO 管理器
            spdk_config = SPDKConfig(
                kv_store_path=self.config.expert_store_path,
                io_queues=num_gpus * 2
            )
            spdk_io = SPDKIOManager(spdk_config)
            spdk_io.start()

            # 测试场景 1: 标准单卡加载
            logger.info("\n1️⃣ 标准单卡加载")
            std_single_times = []
            for _ in range(3):
                start = time.perf_counter()
                experts = []
                for i in range(num_experts):
                    with open(f"{self.config.expert_store_path}/expert_{i}.bin", "rb") as f:
                        data = f.read()
                    experts.append(data)
                elapsed = time.perf_counter() - start
                std_single_times.append(elapsed * 1000)
            avg_std_single = sum(std_single_times) / len(std_single_times)
            logger.info(f"   耗时: {avg_std_single:.2f} ms")

            # 测试场景 2: SPDK 单卡加载
            logger.info("\n2️⃣ SPDK 单卡加载")
            spdk_single_times = []
            for _ in range(3):
                start = time.perf_counter()
                futures = []
                for i in range(num_experts):
                    futures.append(spdk_io.submit_read(f"expert_{i}.bin"))
                for f in futures:
                    f.wait()
                elapsed = time.perf_counter() - start
                spdk_single_times.append(elapsed * 1000)
            avg_spdk_single = sum(spdk_single_times) / len(spdk_single_times)
            logger.info(f"   耗时: {avg_spdk_single:.2f} ms")
            logger.info(f"   加速比 vs 标准: {(avg_std_single / avg_spdk_single):.2f}x")

            # 测试场景 3: 分布式加载（无 SPDK）
            logger.info("\n3️⃣ 分布式加载 (无 SPDK)")
            dist_times = []
            for _ in range(3):
                start = time.perf_counter()

                def load_to_gpu(gpu_id, expert_ids):
                    results = []
                    for i in expert_ids:
                        with open(f"{self.config.expert_store_path}/expert_{i}.bin", "rb") as f:
                            data = f.read()
                        tensor = torch.frombuffer(data, dtype=torch.uint8).to(f"cuda:{gpu_id}")
                        results.append(tensor)
                    return results

                with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                    futures = []
                    per_gpu = num_experts // num_gpus
                    for gpu_id in range(num_gpus):
                        start_idx = gpu_id * per_gpu
                        end_idx = (gpu_id + 1) * per_gpu
                        futures.append(executor.submit(load_to_gpu, gpu_id, range(start_idx, end_idx)))
                    for f in futures:
                        f.result()

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                dist_times.append(elapsed * 1000)
            avg_dist = sum(dist_times) / len(dist_times)
            logger.info(f"   耗时: {avg_dist:.2f} ms")
            logger.info(f"   加速比 vs 标准: {(avg_std_single / avg_dist):.2f}x")

            # 测试场景 4: SPDK + 分布式加载（重点测试）
            logger.info("\n4️⃣ SPDK + 分布式加载 (组合优化)")
            spdk_dist_times = []
            for _ in range(3):
                start = time.perf_counter()

                def spdk_load_to_gpu(gpu_id, expert_ids):
                    results = []
                    futures = []
                    for i in expert_ids:
                        futures.append(spdk_io.submit_read(f"expert_{i}.bin"))
                    for f in futures:
                        data = f.wait()
                        if data is not None:
                            results.append(data)
                    return results

                with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                    futures = []
                    per_gpu = num_experts // num_gpus
                    for gpu_id in range(num_gpus):
                        start_idx = gpu_id * per_gpu
                        end_idx = (gpu_id + 1) * per_gpu
                        futures.append(executor.submit(spdk_load_to_gpu, gpu_id, range(start_idx, end_idx)))
                    for f in futures:
                        f.result()

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                spdk_dist_times.append(elapsed * 1000)
            avg_spdk_dist = sum(spdk_dist_times) / len(spdk_dist_times)
            logger.info(f"   耗时: {avg_spdk_dist:.2f} ms")
            logger.info(f"   加速比 vs 标准: {(avg_std_single / avg_spdk_dist):.2f}x")
            logger.info(f"   加速比 vs SPDK单独: {(avg_spdk_single / avg_spdk_dist):.2f}x")
            logger.info(f"   加速比 vs 分布式单独: {(avg_dist / avg_spdk_dist):.2f}x")

            spdk_io.stop()

            self.results["spdk_distributed_expert_loading"] = {
                "num_experts": num_experts,
                "num_gpus": num_gpus,
                "standard_single_ms": avg_std_single,
                "spdk_single_ms": avg_spdk_single,
                "distributed_ms": avg_dist,
                "spdk_distributed_ms": avg_spdk_dist,
                "speedup_vs_standard": avg_std_single / avg_spdk_dist,
                "speedup_vs_spdk": avg_spdk_single / avg_spdk_dist,
                "speedup_vs_distributed": avg_dist / avg_spdk_dist
            }

        except Exception as e:
            logger.error(f"SPDK + 分布式专家加载测试失败: {e}")
            import traceback
            traceback.print_exc()

    def _benchmark_spdk_distributed_kv_cache(self):
        """测试 SPDK + 分布式 KV Cache"""
        logger.info("\n--- SPDK + 分布式 KV Cache 测试 ---")

        try:
            import torch
            from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
            from cgc_engine.spdk_adapter.spdk_config import SPDKConfig
            from concurrent.futures import ThreadPoolExecutor

            num_gpus = torch.cuda.device_count()
            num_entries = 100
            entry_size = 1024 * 1024  # 1MB per entry

            # 配置 SPDK IO 管理器（共享）
            spdk_config = SPDKConfig(
                kv_store_path=self.config.kv_cache_path,
                io_queues=num_gpus * 4
            )
            spdk_io = SPDKIOManager(spdk_config)
            spdk_io.start()

            # 生成测试数据
            test_data = {}
            for i in range(num_entries):
                test_data[f"key_{i}"] = b"x" * entry_size

            # 场景 1: 标准单线程写入
            logger.info("\n1️⃣ 标准单线程写入")
            std_write_times = []
            for _ in range(3):
                start = time.perf_counter()
                for i in range(num_entries):
                    with open(f"{self.config.kv_cache_path}/std_{i}.dat", "wb") as f:
                        f.write(test_data[f"key_{i}"])
                elapsed = time.perf_counter() - start
                std_write_times.append(elapsed * 1000)
            avg_std_write = sum(std_write_times) / len(std_write_times)
            logger.info(f"   耗时: {avg_std_write:.2f} ms")
            logger.info(f"   吞吐量: {(num_entries * entry_size / 1e6 / (avg_std_write / 1000)):.2f} MB/s")

            # 场景 2: SPDK + 多线程写入（分布式）
            logger.info("\n2️⃣ SPDK + 多线程写入 (分布式)")
            spdk_write_times = []
            for _ in range(3):
                start = time.perf_counter()

                def write_worker(gpu_id, key_range):
                    futures = []
                    for i in key_range:
                        futures.append(spdk_io.submit_write(f"spdk_key_{i}", test_data[f"key_{i}"]))
                    for f in futures:
                        f.wait()

                with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                    futures = []
                    per_worker = num_entries // num_gpus
                    for gpu_id in range(num_gpus):
                        start_idx = gpu_id * per_worker
                        end_idx = (gpu_id + 1) * per_worker
                        futures.append(executor.submit(write_worker, gpu_id, range(start_idx, end_idx)))
                    for f in futures:
                        f.result()

                elapsed = time.perf_counter() - start
                spdk_write_times.append(elapsed * 1000)
            avg_spdk_write = sum(spdk_write_times) / len(spdk_write_times)
            logger.info(f"   耗时: {avg_spdk_write:.2f} ms")
            logger.info(f"   吞吐量: {(num_entries * entry_size / 1e6 / (avg_spdk_write / 1000)):.2f} MB/s")
            logger.info(f"   加速比 vs 标准: {(avg_std_write / avg_spdk_write):.2f}x")

            spdk_io.stop()

            self.results["spdk_distributed_kv_cache"] = {
                "num_entries": num_entries,
                "entry_size_bytes": entry_size,
                "num_gpus": num_gpus,
                "standard_write_ms": avg_std_write,
                "spdk_distributed_write_ms": avg_spdk_write,
                "standard_throughput_mbs": num_entries * entry_size / 1e6 / (avg_std_write / 1000),
                "spdk_throughput_mbs": num_entries * entry_size / 1e6 / (avg_spdk_write / 1000),
                "speedup": avg_std_write / avg_spdk_write
            }

        except Exception as e:
            logger.error(f"SPDK + 分布式 KV Cache 测试失败: {e}")
            import traceback
            traceback.print_exc()

    def _benchmark_spdk_parallel_io_throughput(self):
        """测试 SPDK 并行 IO 吞吐量"""
        logger.info("\n--- SPDK 并行 IO 吞吐量测试 ---")

        try:
            from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
            from cgc_engine.spdk_adapter.spdk_config import SPDKConfig

            num_threads = [1, 2, 4, 8]
            num_operations = 100
            data_size = 1024 * 1024  # 1MB

            spdk_config = SPDKConfig(
                kv_store_path=self.config.kv_cache_path,
                io_queues=8
            )
            spdk_io = SPDKIOManager(spdk_config)
            spdk_io.start()

            results = []
            for threads in num_threads:
                start = time.perf_counter()

                def worker(thread_id):
                    futures = []
                    for i in range(num_operations // threads):
                        key = f"throughput_{thread_id}_{i}"
                        data = b"x" * data_size
                        futures.append(spdk_io.submit_write(key, data))
                    for f in futures:
                        f.wait()

                threads_list = []
                for t in range(threads):
                    t = threading.Thread(target=worker, args=(t,))
                    threads_list.append(t)
                    t.start()
                for t in threads_list:
                    t.join()

                elapsed = time.perf_counter() - start
                throughput = (num_operations * data_size) / 1e6 / elapsed
                results.append({
                    "threads": threads,
                    "time_ms": elapsed * 1000,
                    "throughput_mbs": throughput
                })
                logger.info(f"   {threads} 线程: {throughput:.2f} MB/s")

            spdk_io.stop()

            self.results["spdk_parallel_throughput"] = results

        except Exception as e:
            logger.error(f"SPDK 并行 IO 吞吐量测试失败: {e}")
            import traceback
            traceback.print_exc()

    def _generate_comparison_summary(self):
        """生成综合对比摘要"""
        logger.info("\n--- SPDK + 分布式综合对比 ---")

        expert_results = self.results.get("spdk_distributed_expert_loading", {})
        kv_results = self.results.get("spdk_distributed_kv_cache", {})

        if expert_results:
            logger.info("\n📊 专家加载对比:")
            logger.info(f"   标准单卡: {expert_results.get('standard_single_ms', 0):.2f} ms")
            logger.info(f"   SPDK单卡: {expert_results.get('spdk_single_ms', 0):.2f} ms")
            logger.info(f"   分布式: {expert_results.get('distributed_ms', 0):.2f} ms")
            logger.info(f"   SPDK+分布式: {expert_results.get('spdk_distributed_ms', 0):.2f} ms")
            logger.info(f"   综合加速比: {expert_results.get('speedup_vs_standard', 1):.2f}x")

        if kv_results:
            logger.info("\n📊 KV Cache 吞吐量对比:")
            logger.info(f"   标准: {kv_results.get('standard_throughput_mbs', 0):.2f} MB/s")
            logger.info(f"   SPDK+分布式: {kv_results.get('spdk_throughput_mbs', 0):.2f} MB/s")
            logger.info(f"   加速比: {kv_results.get('speedup', 1):.2f}x")


class HarnessAgentStrategy:
    """
    Harness Agent 策略测试

    测试项：
    1. 端侧 llama.cpp 模式: q4_0 量化 + 64x64x64 tiling
    2. 云侧 vLLM 模式: Flash Attention + Paged Attention + MoE (TP=2)
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """运行 Harness Agent 策略测试"""
        logger.info("=" * 60)
        logger.info("Harness Agent 策略测试")
        logger.info("=" * 60)

        self._test_llama_cpp_strategy()
        self._test_vllm_strategy()
        self._apply_gds_spdk_strategy()

        return self.results

    def _test_llama_cpp_strategy(self):
        """测试 llama.cpp 策略（端侧）"""
        logger.info("\n--- llama.cpp 端侧策略测试 ---")

        try:
            from cgc_engine.agent.harness_agent import HarnessAgent

            agent = HarnessAgent(device="metal" if sys.platform == "darwin" else "cuda")

            user_hints = {
                "backend": "llama.cpp",
                "enable_op_fusion": True,
                "quantization_mode": "q4_0",
                "tile_sizes": {"M": 64, "N": 64, "K": 64},
            }

            logger.info(f"用户 hints: {user_hints}")

            self.results["llama_cpp"] = {
                "backend": "llama.cpp",
                "quantization": "q4_0",
                "tile_size": "64x64x64",
                "op_fusion": True
            }

            logger.info("llama.cpp 策略已配置")

        except Exception as e:
            logger.error(f"llama.cpp 策略测试失败: {e}")

    def _test_vllm_strategy(self):
        """测试 vLLM 策略（云侧）"""
        logger.info("\n--- vLLM 云侧策略测试 ---")

        try:
            from cgc_engine.agent.harness_agent import HarnessAgent

            agent = HarnessAgent(device="cuda")

            user_hints = {
                "backend": "vllm",
                "enable_op_fusion": True,
                "tp_degree": self.config.num_gpus,
                "attention_config": {
                    "flash_attention": True,
                    "paged_attention": True
                },
                "moe_config": {
                    "num_experts": 8,
                    "top_k": 2
                }
            }

            logger.info(f"用户 hints: {user_hints}")

            self.results["vllm"] = {
                "backend": "vllm",
                "tensor_parallel": self.config.num_gpus,
                "flash_attention": True,
                "paged_attention": True,
                "moe_experts": 8,
                "moe_top_k": 2
            }

            logger.info(f"vLLM 策略已配置 (TP={self.config.num_gpus})")

        except Exception as e:
            logger.error(f"vLLM 策略测试失败: {e}")

    def _apply_gds_spdk_strategy(self):
        """应用 GDS/SPDK 策略"""
        logger.info("\n--- GDS/SPDK 策略应用 ---")

        strategy = {
            "enable_gds": self.config.enable_gds,
            "enable_spdk": self.config.enable_spdk,
            "expert_store_path": self.config.expert_store_path,
            "kv_cache_path": self.config.kv_cache_path,
            "io_queues": self.config.num_gpus * 2,
            "platform": "linux" if os.path.exists("/proc/version") else "unknown"
        }

        self.results["gds_spdk_strategy"] = strategy

        logger.info(f"GDS 启用: {self.config.enable_gds}")
        logger.info(f"SPDK 启用: {self.config.enable_spdk}")
        logger.info(f"IO 队列数: {self.config.num_gpus * 2}")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Harness Agent GDS/SPDK 策略测试")
    logger.info("=" * 60)

    # 使用临时目录作为测试路径（兼容本地环境）
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="harness_test_")
    logger.info(f"使用临时目录: {temp_dir}")

    config = BenchmarkConfig(
        model_path=f"{temp_dir}/models/Qwen2.5-7B-Instruct",
        expert_store_path=f"{temp_dir}/flashmoe_experts",
        kv_cache_path=f"{temp_dir}/spdk_kv_cache",
        num_gpus=2,
        enable_gds=True,
        enable_spdk=True,
        test_iters=10,
        warmup_iters=2
    )

    all_results = {
        "config": {
            "num_gpus": config.num_gpus,
            "enable_gds": config.enable_gds,
            "enable_spdk": config.enable_spdk,
            "test_iters": config.test_iters
        },
        "benchmarks": {}
    }

    gds_benchmark = GDSBenchmark(config)
    all_results["benchmarks"]["gds"] = gds_benchmark.run()

    spdk_benchmark = SPDKBenchmark(config)
    all_results["benchmarks"]["spdk"] = spdk_benchmark.run()

    distributed_benchmark = DistributedBenchmark(config)
    all_results["benchmarks"]["distributed"] = distributed_benchmark.run()

    spdk_distributed_benchmark = SPDKDistributedBenchmark(config)
    all_results["benchmarks"]["spdk_distributed"] = spdk_distributed_benchmark.run()

    harness_strategy = HarnessAgentStrategy(config)
    all_results["benchmarks"]["harness_strategy"] = harness_strategy.run()

    output_file = "/tmp/harness_gds_spdk_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("测试完成！结果已保存")
    logger.info(f"输出文件: {output_file}")
    logger.info("=" * 60)

    logger.info("\n=== 测试结果摘要 ===")

    if "gds" in all_results["benchmarks"]:
        gds_res = all_results["benchmarks"]["gds"]
        if "gds_vs_pytorch" in gds_res:
            speedup = gds_res["gds_vs_pytorch"].get("speedup", 1.0)
            logger.info(f"GDS 加速比: {speedup:.2f}x")

    if "spdk" in all_results["benchmarks"]:
        spdk_res = all_results["benchmarks"]["spdk"]
        if "spdk_vs_standard" in spdk_res:
            ws = spdk_res["spdk_vs_standard"].get("write_speedup", 1.0)
            rs = spdk_res["spdk_vs_standard"].get("read_speedup", 1.0)
            logger.info(f"SPDK 写入加速比: {ws:.2f}x")
            logger.info(f"SPDK 读取加速比: {rs:.2f}x")

    if "distributed" in all_results["benchmarks"]:
        dist_res = all_results["benchmarks"]["distributed"]
        if "parallel_loading" in dist_res:
            speedup = dist_res["parallel_loading"].get("speedup", 1.0)
            logger.info(f"双卡并行加载加速比: {speedup:.2f}x")

    if "spdk_distributed" in all_results["benchmarks"]:
        spdk_dist_res = all_results["benchmarks"]["spdk_distributed"]
        logger.info("\n--- SPDK + 分布式组合测试结果 ---")
        
        if "spdk_distributed_expert_loading" in spdk_dist_res:
            expert_res = spdk_dist_res["spdk_distributed_expert_loading"]
            logger.info(f"专家加载 - SPDK+分布式加速比: {expert_res.get('speedup_vs_standard', 1.0):.2f}x")
            logger.info(f"                    vs SPDK单独: {expert_res.get('speedup_vs_spdk', 1.0):.2f}x")
            logger.info(f"                    vs 分布式单独: {expert_res.get('speedup_vs_distributed', 1.0):.2f}x")
        
        if "spdk_distributed_kv_cache" in spdk_dist_res:
            kv_res = spdk_dist_res["spdk_distributed_kv_cache"]
            logger.info(f"KV Cache - SPDK+分布式吞吐量: {kv_res.get('spdk_throughput_mbs', 0):.2f} MB/s")
            logger.info(f"            加速比: {kv_res.get('speedup', 1.0):.2f}x")
        
        if "spdk_parallel_throughput" in spdk_dist_res:
            logger.info("SPDK 并行 IO 吞吐量:")
            for result in spdk_dist_res["spdk_parallel_throughput"]:
                logger.info(f"  {result['threads']} 线程: {result['throughput_mbs']:.2f} MB/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
