
#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent / "MagiCompiler-main"
sys.path.insert(0, str(PROJECT_ROOT))

print("Testing GGUFParser import...")
print()

try:
    from magi_compiler.model_parsers import GGUFParser
    print("✓ GGUFParser imported from magi_compiler.model_parsers")
except Exception as e:
    print(f"✗ Failed to import GGUFParser from magi_compiler.model_parsers: {e}")

print()
print()
print("Testing from magi_compiler import GGUFParser...")
try:
    import magi_compiler as mc
    print("✓ magi_compiler imported successfully!")
    if hasattr(mc, "GGUFParser"):
        print("✓ GGUFParser is in magi_compiler namespace!")
    else:
        print("✗ GGUFParser NOT in magi_compiler namespace!")
        print("  magi_compiler namespace members:", [x for x in dir(mc) if not x.startswith("__")])
except Exception as e:
    print(f"✗ Failed: {e}")
