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
Harness Agent 测试运行器

==========================================================================
                          使用说明 (Harness Agent 必读)
==========================================================================

本运行器为 Harness Agent 提供自动测试能力。Agent 需要：

1. 【自动发现】发现所有需要测试的模块
   - 扫描 cgc_engine/ 目录下的所有模块
   - 识别包含测试函数的模块
   - 优先测试 GDS/SPDK/KV Cache 等关键模块

2. 【自动运行】按顺序运行所有测试
   - GDS Expert Loader 测试
   - SPDK KV Cache 测试
   - Distributed Expert Store 测试
   - Unified IO Controller 测试
   - Inference Service 测试
   - 各后端特定测试 (vLLM/llama.cpp/MLX)

3. 【收集指标】收集所有测试的指标
   - 内存峰值
   - 耗时/H2D/D2H/GFLOPs
   - 调度延迟
   - 后端/模式/策略

4. 【生成报告】生成综合测试报告
   - 汇总所有模块测试结果
   - 输出标准化 JSON 报告
   - 打印摘要信息

==========================================================================
                          快速开始
==========================================================================

# 方式1: Agent 自动发现并运行
runner = HarnessTestRunner()
runner.discover_and_run_tests()
report = runner.generate_combined_report()

# 方式2: 运行特定模块测试
runner = HarnessTestRunner()
runner.run_module_tests("gds_expert_loader")
runner.run_module_tests("spdk_kv_cache")

# 方式3: 运行所有后端测试
runner = HarnessTestRunner()
runner.run_backend_tests()
"""

import sys
import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================================
#                           Agent 必须实现的部分
# ============================================================================

# TODO: Agent 实现模块发现逻辑
# 扫描 cgc_engine/ 目录，识别需要测试的模块
MODULE_DISCOVERY_PATHS = [
    "cgc_engine/flash_moe/",
    "cgc_engine/pd/",
    "cgc_engine/spdk_adapter/",
    "cgc_engine/io_unified/",
    "cgc_engine/inference/",
]

# TODO: Agent 实现模块测试函数映射
# 将模块名称映射到对应的测试函数
MODULE_TEST_FUNCTIONS = {
    "gds_expert_loader": None,         # Agent 填充: cgc_engine.flash_moe.gds_expert_loader
    "spdk_kv_cache": None,             # Agent 填充: cgc_engine.pd.spdk_kv_cache
    "distributed_expert_store": None,   # Agent 填充: cgc_engine.flash_moe.distributed_expert_store
    "unified_io_controller": None,      # Agent 填充: cgc_engine.io_unified.unified_io_controller
    "inference_service": None,          # Agent 填充: cgc_engine.inference.inference_service
}

# TODO: Agent 实现后端测试函数
BACKEND_TEST_FUNCTIONS = {
    "vllm": None,       # Agent 填充: vLLM 推理测试
    "llama_cpp": None,  # Agent 填充: llama.cpp 推理测试
    "mlx": None,        # Agent 填充: MLX 推理测试
    "megatrain": None,  # Agent 填充: MegaTrain 训练测试
}


# ============================================================================
#                           数据类 (Harness Agent 填充)
# ============================================================================

@dataclass
class TestResult:
    """
    测试结果 - Agent 负责填充

    必须包含:
    - test_id: 测试唯一标识
    - module_name: 模块名称
    - device: 运行设备
    - backend: 使用后端
    - success: 是否成功
    - metrics: 性能指标字典
    - error_message: 错误信息 (如有)
    """
    test_id: str
    module_name: str
    device: str
    backend: str
    success: bool
    metrics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None


# ============================================================================
#                           测试运行器 (Harness Agent 使用)
# ============================================================================

class HarnessTestRunner:
    """
    Harness Agent 测试运行器

    =========================================================================
                             Agent 使用指南
    =========================================================================

    1. 初始化
       runner = HarnessTestRunner()

    2. 自动发现并运行测试
       runner.discover_and_run_tests()

    3. 运行特定模块测试
       runner.run_module_tests("gds_expert_loader")
       runner.run_module_tests("spdk_kv_cache")

    4. 运行后端测试
       runner.run_backend_tests()

    5. 生成综合报告
       report = runner.generate_combined_report()
       runner.print_combined_report(report)

    =========================================================================
    """

    def __init__(self, output_dir: str = "./test_reports"):
        """
        初始化测试运行器

        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        self.test_results: Dict[str, List[TestResult]] = {}

    # =========================================================================
    #                           Agent 必须实现的方法
    # =========================================================================

    def discover_modules(self) -> List[str]:
        """
        发现需要测试的模块 - Agent 负责实现

        实现逻辑:
        1. 扫描 MODULE_DISCOVERY_PATHS 目录
        2. 识别包含测试函数的 Python 文件
        3. 返回模块名称列表

        Returns:
            模块名称列表
        """
        # TODO: Agent 实现模块发现
        # 示例:
        # modules = []
        # for path in MODULE_DISCOVERY_PATHS:
        #     for root, dirs, files in os.walk(path):
        #         for file in files:
        #             if file.endswith("_test.py") or "test_" in file:
        #                 modules.append(file)
        # return modules
        return list(MODULE_TEST_FUNCTIONS.keys())

    def load_module_test_function(self, module_name: str) -> Optional[Callable]:
        """
        加载模块测试函数 - Agent 负责实现

        实现逻辑:
        1. 根据 module_name 查找对应的测试函数
        2. 动态导入模块
        3. 返回测试函数

        Args:
            module_name: 模块名称

        Returns:
            测试函数 (Callable) 或 None
        """
        # TODO: Agent 实现测试函数加载
        # 示例:
        # module_map = {
        #     "gds_expert_loader": "cgc_engine.flash_moe.gds_expert_loader.GDSExpertLoader",
        #     "spdk_kv_cache": "cgc_engine.pd.spdk_kv_cache.SPDKKVCache",
        # }
        # if module_name in module_map:
        #     module_path, class_name = module_map[module_name].rsplit(".", 1)
        #     module = importlib.import_module(module_path)
        #     return getattr(module, class_name)
        return None

    def run_single_test(self, module_name: str) -> TestResult:
        """
        运行单个模块测试 - Agent 负责实现

        实现逻辑:
        1. 加载测试函数
        2. 创建测试实例
        3. 运行测试
        4. 收集指标
        5. 返回 TestResult

        Args:
            module_name: 模块名称

        Returns:
            TestResult
        """
        import uuid
        import torch

        test_id = str(uuid.uuid4())[:8]
        device = "auto"
        backend = "auto"

        # 自动检测设备和后端
        if torch.cuda.is_available():
            device = "cuda"
            backend = "vllm"
        elif torch.backends.mps.is_available():
            device = "metal"
            backend = "mlx"
        else:
            device = "cpu"
            backend = "llama.cpp"

        try:
            # 加载测试函数
            test_fn = self.load_module_test_function(module_name)
            if test_fn is None:
                # 如果没有特定测试函数，执行通用测试
                test_fn = self._generic_test_function

            # TODO: Agent 运行测试并收集指标
            # 收集的指标应包括:
            # - total_time_ms, avg_time_ms
            # - peak_memory_gb
            # - h2d_bytes, d2h_bytes
            # - gflops

            return TestResult(
                test_id=test_id,
                module_name=module_name,
                device=device,
                backend=backend,
                success=True,
                metrics={}
            )

        except Exception as e:
            return TestResult(
                test_id=test_id,
                module_name=module_name,
                device=device,
                backend=backend,
                success=False,
                metrics={},
                error_message=str(e)
            )

    # =========================================================================
    #                           Agent 调用的方法
    # =========================================================================

    def run_module_tests(self, module_name: str) -> List[TestResult]:
        """
        运行指定模块的测试 - Agent 调用

        Args:
            module_name: 模块名称

        Returns:
            TestResult 列表
        """
        import logging
        logging.info(f"[Test] 运行模块测试: {module_name}")

        result = self.run_single_test(module_name)
        self.test_results.setdefault(module_name, []).append(result)

        return [result]

    def discover_and_run_tests(self) -> Dict[str, List[TestResult]]:
        """
        自动发现并运行所有测试 - Agent 调用

        Returns:
            模块名称 -> TestResult 列表 的映射
        """
        import logging
        logging.info("=" * 70)
        logging.info("[Runner] 开始自动发现并运行测试")
        logging.info("=" * 70)

        # 发现模块
        modules = self.discover_modules()
        logging.info(f"[Runner] 发现 {len(modules)} 个模块: {modules}")

        # 运行测试
        for module in modules:
            self.run_module_tests(module)

        logging.info(f"[Runner] 所有测试完成，共 {len(self.test_results)} 个模块")
        return self.test_results

    def run_backend_tests(self) -> Dict[str, List[TestResult]]:
        """
        运行所有后端测试 - Agent 调用

        Returns:
            后端名称 -> TestResult 列表 的映射
        """
        import logging
        import torch

        logging.info("=" * 70)
        logging.info("[Runner] 运行后端测试")
        logging.info("=" * 70)

        backends_to_test = []

        # 根据硬件自动选择要测试的后端
        if torch.cuda.is_available():
            backends_to_test.extend(["vllm"])
        if torch.backends.mps.is_available():
            backends_to_test.extend(["mlx"])
        backends_to_test.extend(["llama_cpp"])

        for backend in backends_to_test:
            logging.info(f"[Runner] 测试后端: {backend}")
            # TODO: Agent 实现后端测试
            # result = self.run_backend_test(backend)
            # self.test_results.setdefault(f"backend_{backend}", []).append(result)

        return self.test_results

    def generate_combined_report(self) -> Dict[str, Any]:
        """
        生成综合测试报告 - Agent 调用

        Returns:
            综合报告字典
        """
        import json
        import logging

        logging.info("[Report] 生成综合测试报告")

        # 汇总所有测试结果
        combined_report = {
            "report_id": str(hash(datetime.now())),
            "generated_at": datetime.now().isoformat(),
            "modules": {},
            "summary": {
                "total_modules": len(self.test_results),
                "total_tests": sum(len(results) for results in self.test_results.values()),
                "successful_tests": sum(
                    sum(1 for r in results if r.success)
                    for results in self.test_results.values()
                ),
                "failed_tests": sum(
                    sum(1 for r in results if not r.success)
                    for results in self.test_results.values()
                ),
            },
            "aggregated_metrics": {
                # TODO: Agent 计算聚合指标
                "total_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "peak_memory_gb": 0.0,
                "h2d_bytes": 0,
                "d2h_bytes": 0,
                "gflops": 0.0,
            },
            "hardware": {
                # TODO: Agent 填充硬件信息
            },
            "backends": {
                # TODO: Agent 填充后端信息
            }
        }

        # 计算成功率
        total = combined_report["summary"]["total_tests"]
        successful = combined_report["summary"]["successful_tests"]
        combined_report["summary"]["success_rate"] = (
            (successful / total) * 100 if total > 0 else 0
        )

        # 填充模块详情
        for module_name, results in self.test_results.items():
            module_summary = {
                "tests": [],
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
            }

            for result in results:
                module_summary["tests"].append({
                    "test_id": result.test_id,
                    "success": result.success,
                    "device": result.device,
                    "backend": result.backend,
                    "metrics": result.metrics,
                    "error_message": result.error_message,
                })

            combined_report["modules"][module_name] = module_summary

        # 保存报告
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, "harness_combined_report.json")

        with open(filepath, 'w') as f:
            json.dump(combined_report, f, indent=2)

        logging.info(f"[Report] 报告已保存: {filepath}")

        return combined_report

    def print_combined_report(self, report: Optional[Dict[str, Any]] = None):
        """
        打印综合报告摘要 - Agent 调用

        Args:
            report: 报告字典 (可选)
        """
        if report is None:
            report = self.generate_combined_report()

        print("=" * 80)
        print("Harness Agent 综合测试报告")
        print("=" * 80)

        # 测试概览
        summary = report["summary"]
        print(f"\n[测试概览]")
        print(f"  模块数: {summary['total_modules']}")
        print(f"  测试数: {summary['total_tests']}")
        print(f"  通过: {summary['successful_tests']}")
        print(f"  失败: {summary['failed_tests']}")
        print(f"  成功率: {summary['success_rate']:.1f}%")

        # 聚合指标
        metrics = report.get("aggregated_metrics", {})
        print(f"\n[聚合性能指标]")
        print(f"  总耗时: {metrics.get('total_time_ms', 0):.2f} ms")
        print(f"  平均耗时: {metrics.get('avg_time_ms', 0):.2f} ms")
        print(f"  峰值内存: {metrics.get('peak_memory_gb', 0):.2f} GB")
        print(f"  H2D 流量: {metrics.get('h2d_bytes', 0) / (1024**2):.2f} MB")
        print(f"  D2H 流量: {metrics.get('d2h_bytes', 0) / (1024**2):.2f} MB")

        # 模块详情
        print(f"\n[模块详情]")
        for module_name, module_data in report.get("modules", {}).items():
            status = "✅" if module_data["failed"] == 0 else "⚠️"
            print(f"  {status} {module_name}:")
            print(f"      通过: {module_data['successful']}, 失败: {module_data['failed']}")

        print("\n" + "=" * 80)

    # =========================================================================
    #                           辅助方法
    # =========================================================================

    def _generic_test_function(self):
        """
        通用测试函数 - 当没有特定测试函数时使用

        TODO: Agent 实现通用测试逻辑
        """
        pass


# ============================================================================
#                           辅助方法 (Harness Agent 使用)
# ============================================================================

def run_harness_module_tests() -> Dict[str, Any]:
    """
    Harness Agent 自动调用接口

    用法:
        from test.harness_test_runner import run_harness_module_tests
        report = run_harness_module_tests()

    Returns:
        综合测试报告
    """
    runner = HarnessTestRunner()
    runner.discover_and_run_tests()
    report = runner.generate_combined_report()
    runner.print_combined_report(report)
    return report


# ============================================================================
#                           使用示例 (供 Agent 参考)
# ============================================================================

"""
# 示例 1: 完整测试流程
def main():
    # 1. 创建运行器
    runner = HarnessTestRunner()

    # 2. 自动发现并运行测试
    runner.discover_and_run_tests()

    # 3. 或者运行特定模块测试
    runner.run_module_tests("gds_expert_loader")
    runner.run_module_tests("spdk_kv_cache")
    runner.run_module_tests("distributed_expert_store")

    # 4. 运行后端测试
    runner.run_backend_tests()

    # 5. 生成综合报告
    report = runner.generate_combined_report()
    runner.print_combined_report(report)

    return report

# 示例 2: 单模块测试
def test_single_module(module_name: str):
    runner = HarnessTestRunner()
    results = runner.run_module_tests(module_name)
    report = runner.generate_combined_report()
    runner.print_combined_report(report)
    return results

# 示例 3: 自定义测试流程
def custom_test_flow():
    runner = HarnessTestRunner()

    # 只测试关心的模块
    modules = ["gds_expert_loader", "spdk_kv_cache"]
    for module in modules:
        runner.run_module_tests(module)

    report = runner.generate_combined_report()
    return report
"""
