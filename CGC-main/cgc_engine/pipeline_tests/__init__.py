# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Harness Agent 测试模块

==========================================================================
                          使用说明 (Harness Agent 必读)
==========================================================================

本模块为 Harness Agent 提供统一的测试框架。

核心组件:
1. ModuleTestTemplate - 模块测试模板类
2. HarnessTestRunner - 测试运行器
3. TestResult - 测试结果数据类
4. TestReport - 测试报告数据类
5. PerformanceMetrics - 性能指标数据类
6. HardwareInfo - 硬件信息数据类

Agent 工作流程:
1. 使用 ModuleTestTemplate 创建测试实例
2. 调用 run_test() 运行测试
3. 调用 generate_report() 生成报告
4. 调用 print_report_summary() 打印摘要

或者使用 HarnessTestRunner:
1. 创建运行器实例
2. 调用 discover_and_run_tests()
3. 调用 generate_combined_report()
4. 调用 print_combined_report()

==========================================================================
"""

from .harness_module_test_template import (
    ModuleTestTemplate,
    TestResult,
    TestReport,
    PerformanceMetrics,
    HardwareInfo,
    module_test,
)

from .harness_test_runner import (
    HarnessTestRunner,
    run_harness_module_tests,
)

__all__ = [
    # 模板类
    "ModuleTestTemplate",
    "HarnessTestRunner",
    # 数据类
    "TestResult",
    "TestReport",
    "PerformanceMetrics",
    "HardwareInfo",
    # 辅助函数
    "module_test",
    "run_harness_module_tests",
]
