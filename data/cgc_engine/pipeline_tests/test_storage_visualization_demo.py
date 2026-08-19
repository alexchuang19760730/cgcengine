#!/usr/bin/env python3
"""
Harness Agent 数据存储与可视化演示

展示功能:
1. 测试结果持久化存储 (SQLite)
2. 知识存储系统
3. HTML 可视化报告生成
4. 数据导出功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.harness_module_test_template import ModuleTestTemplate
from cgc_engine.utils.knowledge_storage import KnowledgeStorage


def test_function():
    """示例测试函数"""
    import time
    # 模拟计算
    time.sleep(0.1)
    return {"result": "success"}


def main():
    print("=" * 80)
    print("🚀 Harness Agent 数据存储与可视化演示")
    print("=" * 80)
    
    # 1. 创建测试模板
    print("\n📋 1. 创建测试模板")
    tester = ModuleTestTemplate(module_name="flashmoe_demo", device="auto")
    
    # 2. 运行测试
    print("\n🔧 2. 运行测试")
    result = tester.run_test(test_function)
    print(f"   测试结果: {'成功' if result.success else '失败'}")
    print(f"   平均耗时: {result.metrics.avg_time_ms:.2f} ms")
    
    # 3. 保存测试记录
    print("\n💾 3. 保存测试记录")
    tester.save_test_record(result)
    
    # 4. 生成可视化报告
    print("\n📊 4. 生成可视化报告")
    report_path = tester.generate_visual_report()
    
    # 5. 导出结果
    print("\n📤 5. 导出结果")
    json_path = tester.export_results(format="json")
    csv_path = tester.export_results(format="csv")
    
    # 6. 获取统计信息
    print("\n📈 6. 获取统计信息")
    stats = tester.get_statistics()
    print(f"   总测试数: {stats.get('total_records', 0)}")
    print(f"   成功率: {stats.get('success_rate', 0):.1f}%")
    print(f"   平均耗时: {stats.get('avg_total_time_ms', 0):.2f} ms")
    
    # 7. 初始化知识存储
    print("\n🧠 7. 初始化知识存储")
    knowledge_storage = KnowledgeStorage()
    knowledge_storage.load_default_knowledge()
    
    # 8. 获取知识
    print("\n🔍 8. 查询知识")
    backend = knowledge_storage.get_backend_knowledge("vllm")
    if backend:
        print(f"   后端: {backend.name}")
        print(f"   支持操作: {backend.supported_ops}")
        print(f"   优化能力: {backend.optimization_capabilities}")
    
    hardware = knowledge_storage.get_hardware_knowledge("nvidia-rtx-5090")
    if hardware:
        print(f"   硬件: {hardware.vendor} {hardware.model}")
        print(f"   显存: {hardware.memory_gb} GB")
        print(f"   支持后端: {hardware.supported_backends}")
    
    # 9. 导出知识
    print("\n📥 9. 导出知识")
    knowledge_storage.export_knowledge("knowledge_export.json")
    
    print("\n" + "=" * 80)
    print("✅ 演示完成!")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - 可视化报告: {report_path}")
    print(f"  - JSON 导出: {json_path}")
    print(f"  - CSV 导出: {csv_path}")
    print(f"  - 知识导出: knowledge_export.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
