#!/usr/bin/env python3
"""
CGC 核心整合测试（简化版）
只测试我们的核心整合模块
"""

import sys
import torch
import os

# 直接添加核心模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MagiCompiler-main/magi_compiler/cgc"))

# 直接导入我们的核心模块（跳过项目其他部分）
from cgc_opcodes import CGC_OP_CODES


def print_divider(title: str):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_opcode_integrity():
    """测试 opcode 完整性"""
    print_divider("Opcode 完整性测试")

    print(f"\nvLLM/通用计算 (0x00-0xBF):")
    vllm_ops = []
    for attr in dir(CGC_OP_CODES):
        val = getattr(CGC_OP_CODES, attr)
        if isinstance(val, int) and 0x00 <= val < 0xC0:
            vllm_ops.append((attr, val))
    print(f"  ✓ 共有 {len(vllm_ops)} 个 vLLM/通用 opcode")
    for name, val in sorted(vllm_ops, key=lambda x: x[1]):
        print(f"    0x{val:02X} : {name}")

    print(f"\nllama.cpp 专用 (0xC0-0xDF):")
    llama_ops = []
    for attr in dir(CGC_OP_CODES):
        val = getattr(CGC_OP_CODES, attr)
        if isinstance(val, int) and 0xC0 <= val <= 0xDF:
            llama_ops.append((attr, val))
    print(f"  ✓ 共有 {len(llama_ops)} 个 llama.cpp opcode")
    for name, val in sorted(llama_ops, key=lambda x: x[1]):
        print(f"    0x{val:02X} : {name}")

    print(f"\n量化/反量化 (0xA0-0xAF):")
    quant_ops = []
    for attr in dir(CGC_OP_CODES):
        val = getattr(CGC_OP_CODES, attr)
        if isinstance(val, int) and 0xA0 <= val <= 0xAF:
            quant_ops.append((attr, val))
    print(f"  ✓ 共有 {len(quant_ops)} 个量化 opcode")
    for name, val in sorted(quant_ops, key=lambda x: x[1]):
        print(f"    0x{val:02X} : {name}")

    return len(vllm_ops) > 0 and len(llama_ops) > 0 and len(quant_ops) > 0


def test_unified_command_import():
    """测试统一命令模块导入"""
    print_divider("统一执行层测试")

    try:
        from cgc_unified_executor import (
            UnifiedCommand,
            execute_unified,
            UnifiedCommandToOpcode,
            BackendStrategy,
        )
        print(f"\n✓ cgc_unified_executor 导入成功！")

        print(f"\n统一 Command 列表:")
        cmd_list = []
        for attr in dir(UnifiedCommand):
            if not attr.startswith('_'):
                cmd_list.append(attr)
        print(f"  ✓ 共有 {len(cmd_list)} 个统一 Command:")
        for cmd in cmd_list:
            print(f"    • {cmd}")

        return True

    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        return False


def test_simd_executor_import():
    """测试 SIMD 执行器导入"""
    print_divider("SIMD 执行器测试")

    try:
        from cgc_simd_executor import (
            CGCExecutor,
            CGCKernelRegistry,
        )
        print(f"\n✓ cgc_simd_executor 导入成功！")
        print(f"  - CGCExecutor: OK")
        print(f"  - CGCKernelRegistry: OK")
        return True
    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        return False


def test_backend_import():
    """测试后端模块导入"""
    print_divider("调度层测试")

    try:
        from cgc_backend import CGCBackend, EnhancedPDClient
        print(f"\n✓ cgc_backend 导入成功！")
        print(f"  - CGCBackend: OK")
        print(f"  - EnhancedPDClient: OK")
        return True
    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_mapping():
    """测试统一 Command 到 opcode 的映射"""
    print_divider("Command → Opcode 映射测试")

    try:
        from cgc_unified_executor import UnifiedCommand, UnifiedCommandToOpcode
        from cgc_opcodes import CGC_OP_CODES

        print(f"\n测试 vLLM 映射:")
        vllm_mapped = 0
        for attr in dir(UnifiedCommand):
            if not attr.startswith('_'):
                cmd_val = getattr(UnifiedCommand, attr)
                op_val = UnifiedCommandToOpcode.vllm(cmd_val)
                if op_val is not None:
                    vllm_mapped += 1
        print(f"  ✓ {vllm_mapped} 个统一 Command 有 vLLM 映射")

        print(f"\n测试 llama.cpp 映射:")
        llama_mapped = 0
        for attr in dir(UnifiedCommand):
            if not attr.startswith('_'):
                cmd_val = getattr(UnifiedCommand, attr)
                op_val = UnifiedCommandToOpcode.llama_cpp(cmd_val, "q4_k")
                if op_val is not None:
                    llama_mapped += 1
        print(f"  ✓ {llama_mapped} 个统一 Command 有 llama.cpp 映射")

        return True
    except Exception as e:
        print(f"\n✗ 映射测试失败: {e}")
        return False


def main():
    """主函数"""
    print_divider("CGC 核心整合测试启动")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")

    all_passed = True

    # 测试 1: Opcode 完整性
    if test_opcode_integrity():
        print(f"\n✓ Opcode 完整性测试通过!")
    else:
        print(f"\n✗ Opcode 完整性测试失败!")
        all_passed = False

    # 测试 2: 统一执行层
    if test_unified_command_import():
        print(f"\n✓ 统一执行层导入测试通过!")
    else:
        print(f"\n✗ 统一执行层导入测试失败!")
        all_passed = False

    # 测试 3: SIMD 执行器
    if test_simd_executor_import():
        print(f"\n✓ SIMD 执行器导入测试通过!")
    else:
        print(f"\n✗ SIMD 执行器导入测试失败!")
        all_passed = False

    # 测试 4: 调度层
    if test_backend_import():
        print(f"\n✓ 调度层导入测试通过!")
    else:
        print(f"\n✗ 调度层导入测试失败!")
        all_passed = False

    # 测试 5: Command → Opcode 映射
    if test_integration_mapping():
        print(f"\n✓ 映射测试通过!")
    else:
        print(f"\n✗ 映射测试失败!")
        all_passed = False

    # 总结
    print_divider("测试总结")
    if all_passed:
        print("\n🎉 所有核心整合测试通过！整合成功！")
    else:
        print("\n✗ 部分测试失败！")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
