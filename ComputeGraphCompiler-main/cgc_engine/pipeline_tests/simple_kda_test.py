#!/usr/bin/env python3
"""简单的 KDA backend 测试"""
import os
import sys
import torch
from pathlib import Path

print("="*80)
print("简单的 KDA Backend 导入和基本测试")
print("="*80)

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ["VLLM_USE_CGC_KDA"] = "1"
print(f"✅ Setting VLLM_USE_CGC_KDA=1")

print("\n🚀 Importing cgc_kda_backend...")
try:
    import Backend.Vllm.vllm_backend.cgc_kda_backend
    print("✅ 成功导入 cgc_kda_backend!")

    print("\n🚀 Importing vllm...")
    from vllm import LLM, SamplingParams

    print("✅ 成功导入 vllm!")

    print("\n" + "="*80)
    print("✅ 环境测试通过！")
    print("="*80)

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
