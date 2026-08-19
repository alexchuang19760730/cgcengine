# Copyright (c) 2026 SandAI. All Rights Reserved.
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
Harness Agent 模块测试模板

==========================================================================
                          使用说明 (Harness Agent 必读)
==========================================================================

本模板为 Harness Agent 提供统一的模块测试框架。Agent 需要：

1. 【自动检测】检测硬件环境 (CPU/GPU/Metal/统一内存)
2. 【自动选择】根据硬件自动选择最优后端 (llama.cpp/vLLM/MegaTrain/MLX)
3. 【收集指标】自动收集以下指标：
   - 内存峰值 (peak_memory_gb)
   - 总耗时 / 平均耗时 (total_time_ms, avg_time_ms)
   - H2D / D2H 流量 (h2d_bytes, d2h_bytes)
   - 数据拷贝开销 (copy_count, copy_overhead_ms)
   - 算子计算量 (gflops, total_ops)
   - 调度延迟 (scheduling_delay_ms)
   - 后端 / 模式 / 优化策略 (backend, mode, strategy)
4. 【生成报告】生成标准化 JSON 报告

==========================================================================
                          快速开始
==========================================================================

# 方式1: 直接使用装饰器
@module_test(module_name="flashmoe", device="auto")
def my_test():
    # 你的测试代码
    pass

# 方式2: 手动调用
tester = ModuleTestTemplate(module_name="my_module")
tester.run_test(your_test_function, *args)
report = tester.generate_report()

# 方式3: Harness Agent 自动发现并运行
runner = HarnessTestRunner()
runner.discover_and_run_tests()
"""

import sys
import os
import json
import uuid
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# 数据存储和可视化模块
try:
    from cgc_engine.utils.test_result_storage import TestResultStorage, TestRecord
    from cgc_engine.utils.test_result_visualizer import TestResultVisualizer
    from cgc_engine.utils.knowledge_storage import KnowledgeStorage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    print("⚠️ 存储模块不可用，将使用基础功能")

# ============================================================================
#                           核心数据类 (Harness Agent 填充)
# ============================================================================

@dataclass
class HardwareInfo:
    """
    硬件信息 - Harness Agent 负责填充

    检测内容:
    - platform: 操作系统 (linux/darwin/win)
    - device_type: 设备类型 (cpu/cuda/metal)
    - device_count: GPU 数量
    - device_names: GPU 设备名称列表
    - total_memory_gb: 总内存/显存
    - unified_memory: 是否支持统一内存
    - cuda_version: CUDA 版本 (如有)
    - compute_capability: GPU 计算能力 (如有)
    """
    platform: str = ""
    device_type: str = ""
    device_count: int = 0
    device_names: List[str] = field(default_factory=list)
    total_memory_gb: float = 0.0
    unified_memory: bool = False
    cuda_version: Optional[str] = None
    compute_capability: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """
    性能指标 - Harness Agent 负责填充

    必须收集的指标:
    - 时间指标: total_time_ms, avg_time_ms, min_time_ms, max_time_ms
    - 内存指标: peak_memory_gb, avg_memory_gb
    - IO 指标: h2d_bytes, d2h_bytes, copy_count
    - 计算指标: gflops, total_ops
    - 调度指标: scheduling_delay_ms, overhead_ratio
    """
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    peak_memory_gb: float = 0.0
    avg_memory_gb: float = 0.0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    copy_count: int = 0
    gflops: float = 0.0
    total_ops: int = 0
    scheduling_delay_ms: float = 0.0
    overhead_ratio: float = 0.0
    iterations: int = 0
    warmup_iterations: int = 0


@dataclass
class TestResult:
    """
    测试结果 - Harness Agent 负责填充

    必须包含:
    - test_id: 测试唯一标识
    - module_name: 模块名称
    - device: 运行设备
    - backend: 使用后端
    - success: 是否成功
    - metrics: PerformanceMetrics 实例
    - hardware_info: HardwareInfo 实例
    - timestamp: 测试时间
    - error_message: 错误信息 (如有)
    """
    test_id: str
    module_name: str
    device: str
    backend: str
    success: bool
    metrics: PerformanceMetrics
    hardware_info: HardwareInfo
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None


@dataclass
class TestReport:
    """
    测试报告 - Harness Agent 负责生成

    结构:
    - report_id: 报告唯一标识
    - module_name: 模块名称
    - test_results: TestResult 列表
    - summary: 汇总信息 (由 Agent 自动计算)
    - generated_at: 生成时间
    """
    report_id: str
    module_name: str
    test_results: List[TestResult]
    summary: Dict[str, Any]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
#                           测试模板类 (Harness Agent 使用)
# ============================================================================

class ModuleTestTemplate:
    """
    模块测试模板

    =========================================================================
                             Agent 使用指南
    =========================================================================

    1. 初始化时指定:
       - module_name: 要测试的模块名称 (如 "flashmoe", "gds", "spdk")
       - device: 设备类型 ("auto" 表示自动检测)
       - backend: 后端类型 ("auto" 表示自动选择)

    2. 调用 run_test() 时传入:
       - test_fn: 测试函数
       - *args, **kwargs: 测试函数参数

    3. 调用 generate_report() 生成报告

    4. 调用 print_report_summary() 打印摘要

    =========================================================================
    """

    def __init__(
        self,
        module_name: str,
        device: str = "auto",
        backend: str = "auto",
        warmup_iters: int = None,
        test_iters: int = None,
        output_dir: str = "./test_reports"
    ):
        """
        初始化测试模板

        Args:
            module_name: 模块名称 (必填)
            device: 设备类型 ["auto", "cuda", "cpu", "metal"]
            backend: 后端类型 ["auto", "llama.cpp", "vllm", "megatrain", "mlx"]
            warmup_iters: 预热迭代次数 (Agent 根据实际情况设置)
            test_iters: 测试迭代次数 (Agent 根据实际情况设置)
            output_dir: 报告输出目录
        """
        self.module_name = module_name
        self.device = self._detect_device(device)
        self.backend = self._select_backend(backend)
        self.warmup_iters = warmup_iters
        self.test_iters = test_iters
        self.output_dir = output_dir
        self.metrics = PerformanceMetrics()
        self.hardware_info = self._detect_hardware()
        
        # 初始化数据存储和可视化
        self._init_storage()
        self._init_visualizer()
        
        print(f"✅ 测试模板初始化完成: {module_name}@{device}")

    def _init_storage(self):
        """初始化数据存储"""
        if STORAGE_AVAILABLE:
            self.storage = TestResultStorage()
            self.knowledge_storage = KnowledgeStorage()
            print(f"📦 数据存储已初始化")
        else:
            self.storage = None
            self.knowledge_storage = None

    def _init_visualizer(self):
        """初始化可视化工具"""
        if STORAGE_AVAILABLE:
            self.visualizer = TestResultVisualizer()
            print(f"📊 可视化工具已初始化")
        else:
            self.visualizer = None

    # =========================================================================
    #                           Agent 必须实现的方法
    # =========================================================================

    def _detect_device(self, device: str) -> str:
        """
        检测设备 - Agent 负责实现

        实现逻辑:
        1. 如果 device 不是 "auto"，直接返回
        2. 如果是 "auto":
           - 检测 torch.cuda.is_available() -> "cuda"
           - 检测 torch.backends.mps.is_available() -> "metal"
           - 否则 -> "cpu"
        """
        import torch
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "metal"
        return "cpu"

    def _select_backend(self, backend: str) -> str:
        """
        选择后端 - Agent 负责实现

        实现逻辑:
        1. 如果 backend 不是 "auto"，直接返回
        2. 如果是 "auto"，根据 device 选择:
           - "cuda" -> "vllm"
           - "metal" -> "mlx"
           - "cpu" -> "llama.cpp"
        """
        if backend != "auto":
            return backend
        if self.device == "cuda":
            return "vllm"
        if self.device == "metal":
            return "mlx"
        return "llama.cpp"

    def _detect_hardware(self) -> HardwareInfo:
        """
        检测硬件 - Agent 负责实现

        检测内容:
        1. platform: 通过 sys.platform 检测
        2. device_type/device_count/device_names: 通过 torch.cuda 检测
        3. total_memory_gb: 通过 torch.cuda.get_device_properties 检测
        4. unified_memory: 通过 torch.backends.mps.is_available() 检测
        5. cuda_version: 通过 torch.version.cuda 检测
        6. compute_capability: 通过 torch.cuda.get_device_capability 检测
        """
        import torch
        import psutil

        hardware = HardwareInfo()
        hardware.platform = sys.platform

        if torch.cuda.is_available():
            hardware.device_type = "cuda"
            hardware.device_count = torch.cuda.device_count()
            hardware.device_names = [
                torch.cuda.get_device_name(i)
                for i in range(hardware.device_count)
            ]
            hardware.total_memory_gb = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(hardware.device_count)
            ) / (1024**3)
            hardware.cuda_version = torch.version.cuda
            hardware.compute_capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
        elif torch.backends.mps.is_available():
            hardware.device_type = "metal"
            hardware.device_count = 1
            hardware.device_names = ["Apple Metal"]
            hardware.unified_memory = True
        else:
            hardware.device_type = "cpu"
            hardware.device_count = 1
            hardware.device_names = ["CPU"]
            hardware.total_memory_gb = psutil.virtual_memory().total / (1024**3)

        return hardware

    def _record_memory_usage(self) -> float:
        """
        记录内存使用 - Agent 负责实现

        返回:
            当前内存使用 (GB)
        """
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0

    def _record_io_transfer(self, src_device: str, dst_device: str, bytes_transferred: int):
        """
        记录 IO 传输 - Agent 负责实现

        Args:
            src_device: 源设备
            dst_device: 目标设备
            bytes_transferred: 传输字节数
        """
        # TODO: Agent 实现 IO 传输记录逻辑
        # 更新 self.metrics.h2d_bytes 和 d2h_bytes
        pass

    def _calculate_gflops(self, compute_time_ms: float, ops: int):
        """
        计算 GFLOPs - Agent 负责实现

        Args:
            compute_time_ms: 计算时间 (毫秒)
            ops: 操作数
        """
        # TODO: Agent 实现 GFLOPs 计算逻辑
        # 更新 self.metrics.gflops
        pass

    # =========================================================================
    #                           Agent 调用的方法
    # =========================================================================

    def run_test(
        self,
        test_fn: Callable,
        *args,
        warmup_fn: Optional[Callable] = None,
        **kwargs
    ) -> TestResult:
        """
        运行测试 - Agent 调用此方法执行测试

        Args:
            test_fn: 测试函数
            warmup_fn: 预热函数 (默认为 test_fn)
            args: 测试函数位置参数
            kwargs: 测试函数关键字参数

        Returns:
            TestResult: 包含所有指标的测试结果

        Agent 实现要点:
        1. 调用 _detect_hardware() 获取硬件信息
        2. 执行预热 (warmup_iters 次)
        3. 执行测试 (test_iters 次)
        4. 收集所有指标
        5. 返回 TestResult
        """
        import time
        import uuid
        import torch

        test_id = str(uuid.uuid4())[:8]
        times = []

        try:
            # 设置默认值
            warmup_iterations = self.warmup_iters if self.warmup_iters is not None else 2
            test_iterations = self.test_iters if self.test_iters is not None else 3
            
            # 预热
            warmup_fn = warmup_fn or test_fn
            for _ in range(warmup_iterations):
                warmup_fn(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # 测试迭代
            for _ in range(test_iterations):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                start = time.perf_counter()
                test_fn(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)

                # 记录内存
                current_mem = self._record_memory_usage()
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                self.metrics.peak_memory_gb = max(self.metrics.peak_memory_gb, peak_mem)

            # 填充指标
            self.metrics.total_time_ms = sum(times)
            self.metrics.avg_time_ms = sum(times) / len(times)
            self.metrics.min_time_ms = min(times)
            self.metrics.max_time_ms = max(times)
            self.metrics.iterations = test_iterations
            self.metrics.warmup_iterations = warmup_iterations

            return TestResult(
                test_id=test_id,
                module_name=self.module_name,
                device=self.device,
                backend=self.backend,
                success=True,
                metrics=self.metrics,
                hardware_info=self.hardware_info
            )

        except Exception as e:
            return TestResult(
                test_id=test_id,
                module_name=self.module_name,
                device=self.device,
                backend=self.backend,
                success=False,
                metrics=self.metrics,
                hardware_info=self.hardware_info,
                error_message=str(e)
            )

    def generate_report(self, results: Optional[List[TestResult]] = None) -> TestReport:
        """
        生成报告 - Agent 调用此方法生成报告

        Args:
            results: TestResult 列表 (默认为当前结果)

        Returns:
            TestReport: 标准化测试报告

        Agent 实现要点:
        1. 计算所有 TestResult 的汇总指标
        2. 生成 summary 字典
        3. 保存到 output_dir
        4. 返回 TestReport
        """
        import json

        if results is None:
            results = [TestResult(
                test_id="N/A",
                module_name=self.module_name,
                device=self.device,
                backend=self.backend,
                success=False,
                metrics=self.metrics,
                hardware_info=self.hardware_info,
                error_message="No test results"
            )]

        # 计算汇总
        successful = [r for r in results if r.success]

        summary = {
            "total_tests": len(results),
            "successful_tests": len(successful),
            "failed_tests": len(results) - len(successful),
            "success_rate": (len(successful) / len(results)) * 100 if results else 0,
            "aggregated_metrics": {
                "total_time_ms": sum(r.metrics.total_time_ms for r in successful),
                "avg_time_ms": sum(r.metrics.avg_time_ms for r in successful) / len(successful) if successful else 0,
                "peak_memory_gb": max(r.metrics.peak_memory_gb for r in successful) if successful else 0,
                "h2d_bytes": sum(r.metrics.h2d_bytes for r in successful),
                "d2h_bytes": sum(r.metrics.d2h_bytes for r in successful),
                "gflops": sum(r.metrics.gflops for r in successful) / len(successful) if successful else 0,
            },
            "hardware": {
                "device_type": results[0].hardware_info.device_type if results else "unknown",
                "device_count": results[0].hardware_info.device_count if results else 0,
                "total_memory_gb": results[0].hardware_info.total_memory_gb if results else 0,
            },
            "backend": results[0].backend if results else "unknown",
        }

        report = TestReport(
            report_id=str(hash(datetime.now())),
            module_name=self.module_name,
            test_results=results,
            summary=summary
        )

        # 保存报告
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f"{self.module_name}_report.json")

        report_dict = {
            "report_id": report.report_id,
            "module_name": report.module_name,
            "generated_at": report.generated_at,
            "summary": report.summary,
            "test_results": [
                {
                    "test_id": r.test_id,
                    "success": r.success,
                    "device": r.device,
                    "backend": r.backend,
                    "metrics": {
                        "total_time_ms": r.metrics.total_time_ms,
                        "avg_time_ms": r.metrics.avg_time_ms,
                        "peak_memory_gb": r.metrics.peak_memory_gb,
                        "h2d_bytes": r.metrics.h2d_bytes,
                        "d2h_bytes": r.metrics.d2h_bytes,
                        "gflops": r.metrics.gflops,
                    },
                    "error_message": r.error_message,
                }
                for r in results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)

        return report

    def print_report_summary(self, report: Optional[TestReport] = None):
        """
        打印报告摘要 - Agent 调用此方法打印结果

        Args:
            report: TestReport 实例
        """
        if report is None:
            report = self.generate_report()

        print("=" * 70)
        print(f"测试报告: {report.module_name}")
        print("=" * 70)

        summary = report.summary
        print(f"\n[测试概览]")
        print(f"  总测试数: {summary['total_tests']}")
        print(f"  通过: {summary['successful_tests']}")
        print(f"  失败: {summary['failed_tests']}")
        print(f"  成功率: {summary['success_rate']:.1f}%")

        metrics = summary['aggregated_metrics']
        print(f"\n[性能指标]")
        print(f"  总耗时: {metrics['total_time_ms']:.2f} ms")
        print(f"  平均耗时: {metrics['avg_time_ms']:.2f} ms")
        print(f"  峰值内存: {metrics['peak_memory_gb']:.2f} GB")
        print(f"  H2D 流量: {metrics['h2d_bytes'] / (1024**2):.2f} MB")
        print(f"  D2H 流量: {metrics['d2h_bytes'] / (1024**2):.2f} MB")

        hw = summary.get('hardware', {})
        print(f"\n[硬件信息]")
        print(f"  设备类型: {hw.get('device_type', 'N/A')}")
        print(f"  设备数量: {hw.get('device_count', 0)}")
        print(f"  总内存: {hw.get('total_memory_gb', 0):.1f} GB")

        print(f"\n[后端配置]")
        print(f"  后端: {summary.get('backend', 'N/A')}")
        print(f"  设备: {self.device}")

        print("\n" + "=" * 70)

    def save_test_record(self, result: TestResult):
        """
        保存测试记录到数据库
        
        Args:
            result: TestResult 实例
        """
        if not STORAGE_AVAILABLE or self.storage is None:
            print("⚠️ 存储模块不可用，跳过保存")
            return
        
        record = TestRecord(
            test_id=result.test_id,
            module_name=result.module_name,
            device=result.device,
            backend=result.backend,
            success=result.success,
            timestamp=result.timestamp,
            total_time_ms=result.metrics.total_time_ms,
            avg_time_ms=result.metrics.avg_time_ms,
            min_time_ms=result.metrics.min_time_ms,
            max_time_ms=result.metrics.max_time_ms,
            peak_memory_gb=result.metrics.peak_memory_gb,
            avg_memory_gb=result.metrics.avg_memory_gb,
            h2d_bytes=result.metrics.h2d_bytes,
            d2h_bytes=result.metrics.d2h_bytes,
            copy_count=result.metrics.copy_count,
            gflops=result.metrics.gflops,
            total_ops=result.metrics.total_ops,
            scheduling_delay_ms=result.metrics.scheduling_delay_ms,
            overhead_ratio=result.metrics.overhead_ratio,
            platform=result.hardware_info.platform,
            device_type=result.hardware_info.device_type,
            device_count=result.hardware_info.device_count,
            total_memory_gb_sys=result.hardware_info.total_memory_gb,
            unified_memory=result.hardware_info.unified_memory,
            error_message=result.error_message
        )
        
        self.storage.save_record(record)
        print(f"📥 测试记录已保存: {result.test_id}")

    def generate_visual_report(self, html_file: str = None):
        """
        生成可视化 HTML 报告
        
        Args:
            html_file: 输出文件名 (可选)
        """
        if not STORAGE_AVAILABLE or self.visualizer is None:
            print("⚠️ 可视化模块不可用")
            return
        
        if html_file is None:
            html_file = f"{self.module_name}_report.html"
        
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, html_file)
        
        self.visualizer.generate_html_report(
            module_name=self.module_name,
            output_file=output_path
        )
        
        print(f"📄 可视化报告已生成: {output_path}")
        return output_path

    def export_results(self, format: str = "json"):
        """
        导出测试结果
        
        Args:
            format: 导出格式 ["json", "csv"]
        
        Returns:
            导出文件路径
        """
        if not STORAGE_AVAILABLE or self.storage is None:
            print("⚠️ 存储模块不可用")
            return None
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if format == "json":
            filepath = os.path.join(self.output_dir, f"{self.module_name}_results.json")
            self.storage.export_to_json(filepath, module_name=self.module_name)
        elif format == "csv":
            filepath = os.path.join(self.output_dir, f"{self.module_name}_results.csv")
            self.storage.export_to_csv(filepath, module_name=self.module_name)
        else:
            print(f"❌ 不支持的格式: {format}")
            return None
        
        print(f"📤 结果已导出到: {filepath}")
        return filepath

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        if not STORAGE_AVAILABLE or self.storage is None:
            return {"error": "存储模块不可用"}
        
        return self.storage.get_statistics(module_name=self.module_name)


# ============================================================================
#                           装饰器 (可选)
# ============================================================================

def module_test(module_name: str, device: str = "auto", backend: str = "auto"):
    """
    模块测试装饰器

    用法:
        @module_test(module_name="flashmoe")
        def my_test():
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            tester = ModuleTestTemplate(module_name=module_name, device=device, backend=backend)
            result = tester.run_test(func, *args, **kwargs)
            report = tester.generate_report([result])
            tester.print_report_summary(report)
            return result
        return wrapper
    return decorator


# ============================================================================
#                           使用示例 (供 Agent 参考)
# ============================================================================

"""
# 示例 1: 测试 GDS 专家加载器
def test_gds_expert_loader():
    tester = ModuleTestTemplate(module_name="gds_expert_loader")
    from cgc_engine.flash_moe.gds_expert_loader import GDSExpertLoader

    def test_fn():
        loader = GDSExpertLoader()
        return loader.load_expert(0, [4096, 4096])

    result = tester.run_test(test_fn)
    report = tester.generate_report([result])
    tester.print_report_summary(report)

# 示例 2: 测试 SPDK KV Cache
def test_spdk_kv_cache():
    tester = ModuleTestTemplate(module_name="spdk_kv_cache")
    from cgc_engine.pd.spdk_kv_cache import SPDKKVCache

    def test_fn():
        kv_cache = SPDKKVCache()
        k = torch.randn(1, 32, 128, 64)
        v = torch.randn(1, 32, 128, 64)
        return kv_cache.set_kv("test", k, v)

    result = tester.run_test(test_fn)
    report = tester.generate_report([result])
    tester.print_report_summary(report)

# 示例 3: Agent 自动发现测试
def run_all_module_tests():
    # Agent 自动发现所有模块测试
    modules = [
        "gds_expert_loader",
        "spdk_kv_cache",
        "distributed_expert_store",
        "unified_io_controller",
        "inference_service",
    ]

    for module in modules:
        tester = ModuleTestTemplate(module_name=module)
        # Agent 根据模块类型加载对应的测试函数
        test_fn = load_test_function(module)
        result = tester.run_test(test_fn)
        report = tester.generate_report([result])
"""
