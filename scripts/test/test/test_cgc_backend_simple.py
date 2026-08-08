#!/usr/bin/env python3
"""
CGC Backend + llama.cpp Integration Test (Simplified)
測試重構後的 CGC Backend 和 llama.cpp 整合
"""

import sys
import os
from pathlib import Path

print("="*80)
print("CGC Backend + llama.cpp Integration Test")
print("="*80)


# 測試 1: 從 cgc 目錄導入所有模塊
print("\n[1/5] Testing module imports from cgc directory...")
print("-"*80)
try:
    # 先切換到 cgc 目錄
    magi_compiler_dir = Path(__file__).parent / "MagiCompiler-main"
    cgc_dir = magi_compiler_dir / "magi_compiler" / "cgc"

    os.chdir(cgc_dir)
    print(f"✓ Changed to: {cgc_dir}")

    sys.path.insert(0, str(cgc_dir))

    import cgc_opcodes
    print("✓ cgc_opcodes imported successfully")

    import cgc_commands
    print("✓ cgc_commands imported successfully")

    import cgc_simd_executor
    print("✓ cgc_simd_executor imported successfully")

    print(f"\n  LLAMA_CPP_AVAILABLE: {cgc_simd_executor.LLAMA_CPP_AVAILABLE}")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# 測試 2: 檢查 llama.cpp opcode
print("\n[2/5] Testing llama.cpp opcodes...")
print("-"*80)
try:
    llama_ops = cgc_opcodes.list_llama_cpp_opcodes()
    print(f"✓ Found {len(llama_ops)} llama.cpp opcodes:")
    for op in llama_ops[:5]:
        print(f"  - 0x{op.value:02X}: {op.name}")
    if len(llama_ops) > 5:
        print(f"  ... and {len(llama_ops) - 5} more")

    # 檢查輔助函數
    for opcode in [0xC0, 0xC3, 0xCA, 0xD2]:
        is_llama = cgc_opcodes.is_llama_cpp_opcode(opcode)
        category = cgc_opcodes.get_category(opcode)
        name = cgc_opcodes.get_opcode_name(opcode)
        print(f"  0x{opcode:02X}: {name:<30} (llama: {is_llama}, category: {category})")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()


# 測試 3: 檢查 llama.cpp commands
print("\n[3/5] Testing llama.cpp commands...")
print("-"*80)
try:
    llama_cmds = cgc_commands.get_commands_by_category("llama_cpp")
    print(f"✓ Found {len(llama_cmds)} llama.cpp commands:")

    for cmd in llama_cmds[:5]:
        print(f"  - {cmd.name} (0x{cmd.opcode:02X})")
    if len(llama_cmds) > 5:
        print(f"  ... and {len(llama_cmds) - 5} more")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()


# 測試 4: 檢查 cgc_simd_executor 的 llama.cpp 支持
print("\n[4/5] Testing cgc_simd_executor llama.cpp support...")
print("-"*80)
try:
    print(f"✓ LLAMA_CPP_AVAILABLE: {cgc_simd_executor.LLAMA_CPP_AVAILABLE}")
    print(f"✓ LLAMA_CPP_OPCODE_START: 0x{cgc_simd_executor.LLAMA_CPP_OPCODE_START:02X}")
    print(f"✓ LLAMA_CPP_OPCODE_END: 0x{cgc_simd_executor.LLAMA_CPP_OPCODE_END:02X}")

    # 測試 is_llama_cpp_opcode
    test_opcodes = [0xC0, 0xC3, 0xCA, 0xD2, 0x10, 0x20]
    for opcode in test_opcodes:
        result = cgc_simd_executor.is_llama_cpp_opcode(opcode)
        print(f"  is_llama_cpp_opcode(0x{opcode:02X}): {result}")

    # 列出所有註冊的 kernel
    all_kernels = cgc_simd_executor.list_available_kernels()
    llama_kernels = {k: v for k, v in all_kernels.items() if 0xC0 <= k <= 0xD5}
    print(f"\n✓ Total kernels registered: {len(all_kernels)}")
    print(f"✓ llama.cpp kernels registered: {len(llama_kernels)}")

    if llama_kernels:
        sample = list(llama_kernels.items())[:3]
        print(f"  Sample llama.cpp kernels: {sample}")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()


# 測試 5: 測試 llama.cpp fallback 執行
print("\n[5/5] Testing llama.cpp fallback execution...")
print("-"*80)
try:
    import torch

    # 測試簡單的張量
    x = torch.randn(1, 32, 1024)
    w = torch.randn(1024, 2048)

    print(f"✓ Test tensors created:")
    print(f"  x: {x.shape}, {x.dtype}, {x.device}")
    print(f"  w: {w.shape}, {w.dtype}, {w.device}")

    # 測試 Q4_K_MATMUL 回退
    try:
        result = cgc_simd_executor._llama_q4_k_matmul(
            inputs=[x, w],
            params={}
        )
        print(f"✓ _llama_q4_k_matmul fallback worked: {result[0].shape}")
    except Exception as e:
        print(f"⚠️ _llama_q4_k_matmul test: {e}")

    # 測試 RMSNorm
    try:
        weight = torch.ones(1024)
        result = cgc_simd_executor._llama_rmsnorm_gguf(
            inputs=[x, weight],
            params={"eps": 1e-6}
        )
        print(f"✓ _llama_rmsnorm_gguf fallback worked: {result[0].shape}")
    except Exception as e:
        print(f"⚠️ _llama_rmsnorm_gguf test: {e}")

    # 測試 SiLU
    try:
        result = cgc_simd_executor._llama_silu_gguf(
            inputs=[x],
            params={}
        )
        print(f"✓ _llama_silu_gguf fallback worked: {result[0].shape}")
    except Exception as e:
        print(f"⚠️ _llama_silu_gguf test: {e}")

    # 測試 MoE 路由
    try:
        result = cgc_simd_executor._llama_moe_routing(
            inputs=[x],
            params={}
        )
        print(f"✓ _llama_moe_routing fallback worked: {result[0].shape}")
    except Exception as e:
        print(f"⚠️ _llama_moe_routing test: {e}")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()


# 總結
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
summary = cgc_commands.get_command_summary()
print(f"\nTotal commands: {summary['total_commands']}")
print("\nCommands by category:")
for category, cmds in summary['categories'].items():
    if category == "llama_cpp":
        print(f"  ✓ {category:<30}: {len(cmds)} commands (NEW!)")
    else:
        print(f"  {category:<30}: {len(cmds)} commands")

print("\n" + "="*80)
print("✓ ALL TESTS COMPLETED!")
print("="*80)
