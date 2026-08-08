
#!/usr/bin/env python3
"""测试直接导入双分层模块"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent / "MagiCompiler-main"
sys.path.insert(0, str(PROJECT_ROOT))

# 直接导入
print("尝试直接导入...")
import magi_compiler as mc

print("\nmagi_compiler.__all__:", [x for x in dir(mc) if not x.startswith('_')])
print("\n检查是否有 DualLayerConfig:")
print("  - DualLayerConfig:", hasattr(mc, "DualLayerConfig"))
print("  - DualLayerManager:", hasattr(mc, "DualLayerManager"))
print("\n\n现在尝试直接从模块导入:")

try:
    from magi_compiler.dual_layer_manager import DualLayerConfig, DualLayerManager
    print("✅ 直接导入成功！")
except Exception as e:
    print("❌ 直接导入失败:", e)
    import traceback
    traceback.print_exc()
print("\n检查 dual_layer_manager 模块是否存在:")
import os
print(os.path.exists("/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/magi_compiler/dual_layer_manager.py"))
