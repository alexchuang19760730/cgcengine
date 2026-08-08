#!/usr/bin/env python3
"""
最簡單的測試腳本，只測試核心功能
"""

import sys
from pathlib import Path

# 正確的路徑設置
current_dir = Path(__file__).parent
magi_compiler_dir = current_dir / "MagiCompiler-main"
cgc_dir = magi_compiler_dir / "magi_compiler" / "cgc"

sys.path.insert(0, str(magi_compiler_dir))
sys.path.insert(0, str(cgc_dir))

print("="*60)
print("Phi-3.5-MoE + llama.cpp + CGC 整合測試 (簡化版)")
print("="*60)

# 導入模塊
print("\n[1] 導入 cgc_opcodes...")
try:
    import cgc_opcodes
    print("✓ 成功導入 cgc_opcodes")

    # 檢查 llama.cpp opcodes
    llama_ops = []
    for op in cgc_opcodes.CGC_OP_CODES:
        if cgc_opcodes.is_llama_cpp_opcode(op.value):
            llama_ops.append(op)

    print(f"  找到 {len(llama_ops)} 個 llama.cpp opcode")

except Exception as e:
    print(f"✗ 失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2] 導入 cgc_commands...")
try:
    import cgc_commands
    print("✓ 成功導入 cgc_commands")

    llama_cmds = cgc_commands.get_commands_by_category("llama_cpp")
    print(f"  找到 {len(llama_cmds)} 個 llama.cpp command")

except Exception as e:
    print(f"✗ 失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("  整合成功！")
print("="*60)

# 顯示一些信息
print("\n新增的 llama.cpp 相關功能:")
print("-"*60)
print(f"  Opcodes: {len(llama_ops)} 條 (0xC0 ~ 0xD5)")
print(f"  Commands: {len(llama_cmds)} 條")
print("\n主要功能類型:")

# 按功能分組
quant_ops = [op for op in llama_ops if "QUANTIZE" in op.name or "MATMUL" in op.name]
moe_ops = [op for op in llama_ops if "MOE" in op.name]
infer_ops = [op for op in llama_ops if "INFERENCE" in op.name or "SAMPLING" in op.name]

print(f"  - 量化運算: {len(quant_ops)} 條")
print(f"  - MoE 專用: {len(moe_ops)} 條")
print(f"  - 推理/採樣: {len(infer_ops)} 條")

print("\n" + "="*60)
print("  測試完成！")
print("="*60)
