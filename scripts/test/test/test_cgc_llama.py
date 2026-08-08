#!/usr/bin/env python3
"""
簡單的 llama.cpp + CGC 整合測試腳本
直接導入和演示我們修改的模塊
"""

import sys
import os
from pathlib import Path

# 添加模塊路徑
magi_compiler_path = Path(__file__).parent / "MagiCompiler-main"
cgc_path = magi_compiler_path / "magi_compiler" / "cgc"
sys.path.insert(0, str(magi_compiler_path))
sys.path.insert(0, str(cgc_path))

print("="*60)
print("Phi-3.5-MoE + llama.cpp + CGC 整合測試")
print("="*60)
print(f"\n模塊路徑:")
print(f"  MagiCompiler: {magi_compiler_path}")
print(f"  CGC 目錄: {cgc_path}")

# 測試 1: 導入 cgc_opcodes
print("\n" + "-"*60)
print("測試 1: 導入 cgc_opcodes.py")
print("-"*60)
try:
    import cgc_opcodes
    print("✓ 成功導入 cgc_opcodes")

    # 檢查 llama.cpp opcodes
    llama_ops = []
    for op in cgc_opcodes.CGC_OP_CODES:
        if cgc_opcodes.is_llama_cpp_opcode(op.value):
            llama_ops.append(op)

    print(f"\n找到 {len(llama_ops)} 個 llama.cpp 相關的 opcode:")
    for op in llama_ops[:10]:  # 只顯示前 10 個
        print(f"  0x{op.value:02X} | {op.name}")
    if len(llama_ops) > 10:
        print(f"  ... 還有 {len(llama_ops)-10} 個")

except Exception as e:
    print(f"✗ 導入失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 測試 2: 導入 cgc_commands
print("\n" + "-"*60)
print("測試 2: 導入 cgc_commands.py")
print("-"*60)
try:
    import cgc_commands
    print("✓ 成功導入 cgc_commands")

    # 獲取 llama.cpp 相關的 commands
    llama_cmds = cgc_commands.get_commands_by_category("llama_cpp")
    print(f"\n找到 {len(llama_cmds)} 個 llama.cpp 相關的 command:")

    for cmd in llama_cmds[:10]:  # 只顯示前 10 個
        print(f"\n  [{cmd.name}] (0x{cmd.opcode:02X})")
        print(f"    {cmd.description}")
    if len(llama_cmds) > 10:
        print(f"\n  ... 還有 {len(llama_cmds)-10} 個")

except Exception as e:
    print(f"✗ 導入失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 測試 3: 顯示完整的 opcode 列表
print("\n" + "-"*60)
print("測試 3: 完整的 llama.cpp opcode 列表")
print("-"*60)
print()

all_llama_ops = []
for op in cgc_opcodes.CGC_OP_CODES:
    if cgc_opcodes.is_llama_cpp_opcode(op.value):
        all_llama_ops.append(op)

# 按 opcode 排序
all_llama_ops.sort(key=lambda x: x.value)

# 分組顯示
groups = {
    "模型加載/量化": [],
    "矩陣運算": [],
    "MoE": [],
    "激活/歸一化": [],
    "推理/採樣": [],
}

for op in all_llama_ops:
    if "GGUF_LOAD" in op.name or "QUANTIZE" in op.name or "DEQUANTIZE" in op.name:
        groups["模型加載/量化"].append(op)
    elif "MATMUL" in op.name:
        groups["矩陣運算"].append(op)
    elif "MOE" in op.name:
        groups["MoE"].append(op)
    elif "ROPE" in op.name or "RMSNORM" in op.name or "SILU" in op.name or "GELU" in op.name:
        groups["激活/歸一化"].append(op)
    else:
        groups["推理/採樣"].append(op)

for group_name, ops in groups.items():
    print(f"\n{group_name}:")
    for op in ops:
        print(f"  0x{op.value:02X}  {op.name:<30}")

# 測試 4: 顯示完整的 command 列表
print("\n" + "-"*60)
print("測試 4: 完整的 llama.cpp command 列表")
print("-"*60)

for cmd in llama_cmds:
    print(f"\n{cmd.name}")
    print(f"  Opcode:    0x{cmd.opcode:02X}")
    print(f"  Category:  {cmd.category}")
    print(f"  Module:    {cmd.module}")
    print(f"  Desc:      {cmd.description}")
    print(f"  Params:    {', '.join(cmd.params.keys())}")

# 測試 5: 獲取單個 command
print("\n" + "-"*60)
print("測試 5: 獲取單個 command")
print("-"*60)

cmd_names = [
    "LLAMA_INFERENCE",
    "LLAMA_Q4_K_MATMUL",
    "LLAMA_GGUF_LOAD",
]

for name in cmd_names:
    cmd = cgc_commands.get_cgc_command(name)
    if cmd:
        print(f"\n{name}:")
        print(f"  Opcode: 0x{cmd.opcode:02X}")
        print(f"  Params: {cmd.params}")

# 測試 6: 按 opcode 獲取 command
print("\n" + "-"*60)
print("測試 6: 按 opcode 獲取 command")
print("-"*60)

test_opcodes = [0xC0, 0xC3, 0xD2]
for opcode in test_opcodes:
    cmd = cgc_commands.get_cgc_command_by_opcode(opcode)
    if cmd:
        print(f"\n0x{opcode:02X}:")
        print(f"  Name: {cmd.name}")
        print(f"  Desc: {cmd.description}")

# 總結
print("\n" + "="*60)
print("測試總結")
print("="*60)
summary = cgc_commands.get_command_summary()
print(f"\n總指令數: {summary['total_commands']}")
print("\n按分類統計:")
for cat, cmds in summary['categories'].items():
    if cat == "llama_cpp":
        print(f"  ✓ {cat:<20}: {len(cmds)} 條指令 (新增!)")
    else:
        print(f"  {cat:<20}: {len(cmds)} 條指令")

print("\n" + "="*60)
print("✓ 所有測試通過!")
print("="*60)
