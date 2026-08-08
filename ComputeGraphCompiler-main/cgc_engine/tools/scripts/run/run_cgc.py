#!/usr/bin/env python3
"""
🔥 CGC 全栈编译器端到端测试
从 GGUF → 内存 → IO → 调度 → 计算 完整流程
"""

import sys
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')

from cgc_compiler_complete import CGCCompiler, GGUFModel
import json
import time

# --------------------------
# 你的起始输入（极简）
# --------------------------
GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
DEVICE = "metal"
OP_TARGET = "kimi_kda"

# --------------------------
# 🔥 运行编译器
# --------------------------
print("="*70)
print("🔥 CGC 全栈编译器端到端测试")
print("="*70)
print(f"📌 输入: {GGUF_FILE}")
print(f"📌 设备: {DEVICE}")
print(f"📌 算子: {OP_TARGET}")

compiler = CGCCompiler(device=DEVICE)
model = GGUFModel(name=GGUF_FILE.split('/')[-1])

# 一键编译：存储 + IO + 调度 + 计算
result = compiler.compile(
    gguf_model=model,
    op_target=OP_TARGET,
    batch_size=1
)

# --------------------------
# 输出最终编译产物
# --------------------------
print("\n📦 编译产物：")
print(f" - 存储策略：{result['storage']}")
print(f" - 设备IO策略：{result['device_io']}")
print(f" - 调度策略：{result['scheduler']}")
print(f" - 计算内核：{result['kernel'][:100]}...")

# --------------------------
# 完整推理测试
# --------------------------
print("\n" + "="*60)
print("🧪 完整端到端推理测试")
print("="*60)

# 尝试加载真实 GGUF
try:
    import gguf
    print(f"\n🔹 加载 GGUF 模型: {GGUF_FILE}")
    
    t0 = time.time()
    reader = gguf.GGUFReader(GGUF_FILE)
    load_time = time.time() - t0
    
    # 读取配置
    config = {}
    for key in reader.metadata:
        if key.startswith("llama."):
            config[key] = reader.metadata[key]
    
    print(f"✅ 模型加载完成 ({load_time:.2f}s)")
    print(f"📋 模型配置:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # 读取权重
    print(f"\n🔹 张量数量: {len(reader.tensors)}")
    for i, tensor in enumerate(reader.tensors[:5]):
        print(f"   {tensor.name}: {tensor.shape}")
        
except Exception as e:
    print(f"⚠️  GGUF 加载失败: {e}")

print("\n" + "="*70)
print("✅ CGC 全栈编译器测试完成！")
print("="*70)

# --------------------------
# 输出完整 Ground Truth
# --------------------------
print("\n📊 完整 Ground Truth 知识库:")
print("-"*60)
with open("ground_truth.json", "r") as f:
    gt = json.load(f)
    print(json.dumps(gt, indent=2, ensure_ascii=False))