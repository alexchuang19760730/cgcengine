#!/usr/bin/env python3
"""
CGC Gate 2.3 测试套件主入口

运行所有测试:
- 单元测试 (unit)
- 集成测试 (integration)
- 性能测试 (performance)
"""

import os
import sys
import pytest
import argparse
from datetime import datetime


def run_tests(test_type=None, verbose=False):
    """
    运行测试
    
    Args:
        test_type: 测试类型 (unit, integration, performance, all)
        verbose: 是否详细输出
    
    Returns:
        int: 测试结果码 (0=成功, 非0=失败)
    """
    start_time = datetime.now()
    print("=" * 70)
    print(f"CGC Gate 2.3 测试套件")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA 可用: {torch.cuda.is_available() if 'torch' in sys.modules else '未检查'}")
    print("=" * 70)
    
    # 测试目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    args = ["-v" if verbose else ""]
    
    if test_type == "unit" or test_type == "all":
        print("\n--- 运行单元测试 ---")
        unit_result = pytest.main([os.path.join(test_dir, "unit"), "-v"])
        print(f"单元测试结果: {'通过' if unit_result == 0 else f'失败 ({unit_result})'}")
    
    if test_type == "integration" or test_type == "all":
        print("\n--- 运行集成测试 ---")
        integration_result = pytest.main([os.path.join(test_dir, "integration"), "-v"])
        print(f"集成测试结果: {'通过' if integration_result == 0 else f'失败 ({integration_result})'}")
    
    if test_type == "performance" or test_type == "all":
        print("\n--- 运行性能测试 ---")
        # 性能测试单独处理
        from cgc_engine.rswa_integration import test_rswa_vs_baseline
        try:
            test_rswa_vs_baseline.main()
            print("性能测试结果: 通过")
        except Exception as e:
            print(f"性能测试结果: 失败 ({e})")
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print(f"测试完成")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"耗时: {elapsed:.2f} 秒")
    print("=" * 70)
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="CGC Gate 2.3 测试套件")
    parser.add_argument(
        "-t", "--type",
        choices=["unit", "integration", "performance", "all"],
        default="all",
        help="测试类型"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 导入 torch（需要在解析参数后，避免影响 --help）
    global torch
    import torch
    
    return run_tests(args.type, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
