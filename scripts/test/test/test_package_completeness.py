
#!/usr/bin/env python3
"""
测试 MagiCompiler 包完整性！
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent / "MagiCompiler-main"
sys.path.insert(0, str(PROJECT_ROOT))

import magi_compiler as mc

print("=" * 80)
print(" MagiCompiler 包完整性检查")
print("=" * 80)
print()

print("1. 包版本:", getattr(mc, "__version__", "unknown"))
print()

print("2. __all__ 导出的成员数:", len(getattr(mc, "__all__", [])))
if getattr(mc, "__all__", []):
    print("   导出的成员:")
    for name in sorted(getattr(mc, "__all__", [])):
        print(f"     - {name}")
print()

print("3. CGC 核心模块检查:")
print(f"   - CGCExecutor: {'✓ 已导入' if hasattr(mc, 'CGCExecutor') else '✗ 未导入'}")
print(f"   - CGCBackend: {'✓ 已导入' if hasattr(mc, 'CGCBackend') else '✗ 未导入'}")
print(f"   - CGCCommand: {'✓ 已导入' if hasattr(mc, 'CGCCommand') else '✗ 未导入'}")
print(f"   - CGC_OP_CODES: {'✓ 已导入' if hasattr(mc, 'CGC_OP_CODES') else '✗ 未导入'}")
print()

print("4. PD 模块检查:")
print(f"   - PDClient: {'✓ 已导入' if hasattr(mc, 'PDClient') else '✗ 未导入'}")
print()

print("5. Model Parsers 检查:")
print(f"   - GGUFParser: {'✓ 已导入' if hasattr(mc, 'GGUFParser') else '✗ 未导入'}")
print()

print("6. MagiCompiler 核心 API 检查:")
print(f"   - magi_compile: {'✓ 已导入' if hasattr(mc, 'magi_compile') else '✗ 未导入'}")
print(f"   - magi_register_custom_op: {'✓ 已导入' if hasattr(mc, 'magi_register_custom_op') else '✗ 未导入'}")
print()

print("=" * 80)
print(" ✓ MagiCompiler 包检查完成！")
print("=" * 80)
