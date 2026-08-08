#!/usr/bin/env python3
"""测试 cgc_cpp pybind11 模块加载"""

import sys
import os
from pathlib import Path

_build_dir = Path(__file__).resolve().parents[2] / "cgc_engine" / "cgc" / "cgc_cpp" / "build"
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

print('Testing cgc_cpp pybind11 module...')
print(f'Python: {sys.version}')
print(f'Executable: {sys.executable}')

try:
    import cgc_cpp
    print('✅ cgc_cpp module loaded successfully!')
    print(f'Module: {cgc_cpp}')
    print(f'Doc: {cgc_cpp.__doc__}')

    if hasattr(cgc_cpp, 'init'):
        print('Calling cgc_cpp.init()...')
        cgc_cpp.init()
        print('✅ init() called successfully!')

    if hasattr(cgc_cpp, 'has_opcode'):
        print('\nSupported opcodes:')
        for opcode in [0x10, 0x11, 0x12, 0x13, 0x20, 0x30, 0x80, 0x81, 0x82]:
            result = cgc_cpp.has_opcode(opcode)
            print(f'  0x{opcode:02x}: {result}')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
