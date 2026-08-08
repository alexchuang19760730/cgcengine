#!/usr/bin/env python3
"""
CGC 完整整合测试脚本
测试所有 llama.cpp/vLLM 的计算、量化、反量化功能
"""

import sys
import torch
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MagiCompiler-main"))

from magi_compiler.cgc.cgc_simd_executor import CGCExecutor
from magi_compiler.cgc.cgc_unified_executor import (
    UnifiedCommand,
    execute_unified,
)
from magi_compiler.cgc.cgc_commands import CGCCommand


def print_divider(title: str):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_general_computations():
    """测试通用计算"""
    print_divider("通用计算测试")

    executor = CGCExecutor()

    # 创建测试数据
    batch_size = 2
    seq_len = 16
    hidden_dim = 128

    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
    weight = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32)
    rms_weight = torch.randn(hidden_dim, dtype=torch.float32)

    print(f"\n测试数据:")
    print(f"  x.shape: {x.shape}")
    print(f"  weight.shape: {weight.shape}")

    # =====================================
    # 测试 1: Linear/GEMM
    # =====================================
    print(f"\n1. 测试 Linear/GEMM:")
    try:
        # vLLM 域（大显存）
        result_vllm = execute_unified(
            executor,
            UnifiedCommand.LINEAR_GEMM,
            inputs=[x, weight],
            hints={
                "device": "cpu",
                "backend": "vllm",
                "memory_available_gb": 24.0
            }
        )
        print(f"  ✓ vLLM 域执行成功! Output shape: {result_vllm[0].shape}")

        # llama.cpp 域（小显存）
        result_llama = execute_unified(
            executor,
            UnifiedCommand.LINEAR_GEMM,
            inputs=[x, weight],
            hints={
                "device": "cpu",
                "backend": "llama_cpp",
                "quant_type": "q4_k"
            }
        )
        print(f"  ✓ llama.cpp 域执行成功! Output shape: {result_llama[0].shape}")
    except Exception as e:
        print(f"  ✗ Linear/GEMM 测试失败: {e}")

    # =====================================
    # 测试 2: RMSNorm
    # =====================================
    print(f"\n2. 测试 RMSNorm:")
    try:
        result = execute_unified(
            executor,
            UnifiedCommand.RMS_NORM,
            inputs=[x, rms_weight],
            hints={"device": "cpu", "backend": "vllm"}
        )
        print(f"  ✓ RMSNorm 执行成功! Output shape: {result[0].shape}")
    except Exception as e:
        print(f"  ✗ RMSNorm 测试失败: {e}")

    # =====================================
    # 测试 3: SiLU
    # =====================================
    print(f"\n3. 测试 SiLU:")
    try:
        result = execute_unified(
            executor,
            UnifiedCommand.SILU,
            inputs=[x],
            hints={"device": "cpu", "backend": "vllm"}
        )
        print(f"  ✓ SiLU 执行成功! Output shape: {result[0].shape}")
    except Exception as e:
        print(f"  ✗ SiLU 测试失败: {e}")

    return True


def test_quantization():
    """测试量化/反量化"""
    print_divider("量化/反量化测试")

    executor = CGCExecutor()

    # 创建测试数据
    hidden_dim = 128
    x = torch.randn(2, 16, hidden_dim, dtype=torch.float32)

    print(f"\n测试数据:")
    print(f"  x.shape: {x.shape}")

    # =====================================
    # 测试 1: W8A16 量化
    # =====================================
    print(f"\n1. 测试 W8A16 量化:")
    try:
        quantized = execute_unified(
            executor,
            UnifiedCommand.QUANTIZE_W8A16,
            inputs=[x],
            params={"scale": 0.1},
            hints={"device": "cpu", "backend": "vllm"}
        )
        print(f"  ✓ W8A16 量化成功!")

        # 反量化
        dequantized = execute_unified(
            executor,
            UnifiedCommand.DEQUANTIZE,
            inputs=[quantized[0]],
            params={"scale": 0.1},
            hints={"device": "cpu", "backend": "vllm"}
        )
        print(f"  ✓ 反量化成功!")
    except Exception as e:
        print(f"  ✗ W8A16 测试失败: {e}")

    # =====================================
    # 测试 2: GGUF 量化/反量化
    # =====================================
    print(f"\n2. 测试 GGUF 量化/反量化:")
    try:
        # GGUF 量化
        quantized = execute_unified(
            executor,
            UnifiedCommand.GGUF_QUANTIZE,
            inputs=[x],
            hints={"device": "cpu", "backend": "llama_cpp"}
        )
        print(f"  ✓ GGUF 量化成功!")

        # GGUF 反量化
        dequantized = execute_unified(
            executor,
            UnifiedCommand.GGUF_DEQUANTIZE,
            inputs=[quantized[0]],
            hints={"device": "cpu", "backend": "llama_cpp"}
        )
        print(f"  ✓ GGUF 反量化成功!")
    except Exception as e:
        print(f"  ✗ GGUF 测试失败: {e}")

    return True


def test_llama_cpp_special():
    """测试 llama.cpp 专用功能"""
    print_divider("llama.cpp 专用功能测试")

    executor = CGCExecutor()

    # 创建测试数据
    x = torch.randn(2, 16, 128, dtype=torch.float32)

    print(f"\n测试数据:")
    print(f"  x.shape: {x.shape}")

    # =====================================
    # 测试 MoE 路由
    # =====================================
    print(f"\n1. 测试 MoE 路由:")
    try:
        routing = execute_unified(
            executor,
            UnifiedCommand.MOE_ROUTING,
            inputs=[x],
            hints={"device": "cpu", "backend": "llama_cpp"}
        )
        print(f"  ✓ MoE 路由执行成功!")
    except Exception as e:
        print(f"  ✗ MoE 路由测试失败: {e}")

    # =====================================
    # 测试 MoE 专家前向
    # =====================================
    print(f"\n2. 测试 MoE 专家前向:")
    try:
        expert_weight = torch.randn(128, 128, dtype=torch.float32)
        output = execute_unified(
            executor,
            UnifiedCommand.MOE_EXPERT_FWD,
            inputs=[x, expert_weight],
            hints={"device": "cpu", "backend": "llama_cpp"}
        )
        print(f"  ✓ MoE 专家前向执行成功!")
    except Exception as e:
        print(f"  ✗ MoE 专家前向测试失败: {e}")

    return True


def test_attention():
    """测试 Attention 计算"""
    print_divider("Attention 计算测试")

    executor = CGCExecutor()

    # 创建测试数据
    batch_size = 2
    seq_len = 16
    hidden_dim = 128
    num_heads = 8
    head_dim = hidden_dim // num_heads

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    print(f"\n测试数据:")
    print(f"  q.shape: {q.shape}")
    print(f"  k.shape: {k.shape}")
    print(f"  v.shape: {v.shape}")

    # =====================================
    # 测试 SDPA Attention
    # =====================================
    print(f"\n1. 测试 SDPA Attention:")
    try:
        result = execute_unified(
            executor,
            UnifiedCommand.ATTENTION_SDPA,
            inputs=[q, k, v],
            hints={"device": "cpu", "backend": "vllm"}
        )
        print(f"  ✓ SDPA Attention 执行成功!")
    except Exception as e:
        print(f"  ✗ SDPA Attention 测试失败: {e}")

    # =====================================
    # 测试 KDA Attention
    # =====================================
    print(f"\n2. 测试 KDA Attention:")
    try:
        result = execute_unified(
            executor,
            UnifiedCommand.ATTENTION_KDA,
            inputs=[q, k, v],
            hints={"device": "cpu", "backend": "vllm"}
        )
        print(f"  ✓ KDA Attention 执行成功!")
    except Exception as e:
        print(f"  ✗ KDA Attention 测试失败: {e}")

    return True


def main():
    """主函数"""
    print_divider("CGC 完整整合测试启动")
    print(f"PyTorch version: {torch.__version__}")

    all_passed = True

    # 测试 1: 通用计算
    try:
        if test_general_computations():
            print(f"\n✓ 通用计算测试通过!")
        else:
            print(f"\n✗ 通用计算测试失败!")
            all_passed = False
    except Exception as e:
        print(f"\n✗ 通用计算测试异常: {e}")
        all_passed = False

    # 测试 2: 量化/反量化
    try:
        if test_quantization():
            print(f"\n✓ 量化/反量化测试通过!")
        else:
            print(f"\n✗ 量化/反量化测试失败!")
            all_passed = False
    except Exception as e:
        print(f"\n✗ 量化/反量化测试异常: {e}")
        all_passed = False

    # 测试 3: llama.cpp 专用
    try:
        if test_llama_cpp_special():
            print(f"\n✓ llama.cpp 专用功能测试通过!")
        else:
            print(f"\n✗ llama.cpp 专用功能测试失败!")
            all_passed = False
    except Exception as e:
        print(f"\n✗ llama.cpp 专用功能测试异常: {e}")
        all_passed = False

    # 测试 4: Attention
    try:
        if test_attention():
            print(f"\n✓ Attention 计算测试通过!")
        else:
            print(f"\n✗ Attention 计算测试失败!")
            all_passed = False
    except Exception as e:
        print(f"\n✗ Attention 计算测试异常: {e}")
        all_passed = False

    # 总结
    print_divider("测试总结")
    if all_passed:
        print("\n🎉 所有测试通过！整合成功！")
    else:
        print("\n✗ 部分测试失败！")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
