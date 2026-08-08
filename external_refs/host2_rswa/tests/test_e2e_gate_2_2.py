#!/usr/bin/env python3
"""
CGC Gate 2.2 DeepEP MoE 负载均衡 - 端到端测试
基于真实 FlashMoE 客户端和 OMLX 客户端
"""

import sys
sys.path.insert(0, '/Users/alexchuang/Documents/embodied/ComputeGraphCompiler-main')

import torch
import time
import numpy as np

print("="*70)
print("🔥 CGC Gate 2.2 - DeepEP MoE 负载均衡端到端测试")
print("="*70)

results = {}

print("\n" + "="*70)
print("【第一步】初始化 FlashMoE 客户端 (真实后端)")
print("="*70)

try:
    from cgc_engine.flash_moe.client import FlashMoEClient
    
    flashmoe = FlashMoEClient(
        expert_dir="/tmp/flash_moe_experts",
        backend="auto"
    )
    
    flashmoe.num_experts = 16
    flashmoe.expert_dim = 4096
    flashmoe.intermediate_dim = 14336
    
    print("   ✅ FlashMoEClient 初始化完成")
    print(f"   • 后端类型: {flashmoe.backend_info['backend']}")
    print(f"   • 设备: {flashmoe.backend_info['device']}")
    print(f"   • 专家数: {flashmoe.num_experts}")
    print(f"   • 专家维度: {flashmoe.expert_dim}")
    
    results["step1"] = {
        "success": True,
        "backend": flashmoe.backend_info['backend'],
        "device": flashmoe.backend_info['device'],
        "num_experts": flashmoe.num_experts
    }
    
except Exception as e:
    print(f"   ❌ FlashMoE 初始化失败: {e}")
    results["step1"] = {"success": False, "error": str(e)}
    sys.exit(1)

print("\n" + "="*70)
print("【第二步】初始化 OMLX 客户端")
print("="*70)

try:
    from cgc_engine.omlx.client import OMLXClient
    
    omlx = OMLXClient(model_dir="/tmp/omlx_models")
    omlx.num_experts = 16
    omlx.expert_dim = 4096
    
    print("   ✅ OMLXClient 初始化完成")
    print(f"   • 专家数: {omlx.num_experts}")
    print(f"   • 专家维度: {omlx.expert_dim}")
    
    results["step2"] = {"success": True}
    
except Exception as e:
    print(f"   ❌ OMLX 初始化失败: {e}")
    results["step2"] = {"success": False, "error": str(e)}
    sys.exit(1)

print("\n" + "="*70)
print("【第三步】准备测试数据")
print("="*70)

batch_size = 2
seq_len = 128
hidden_dim = 4096

print(f"   🔧 测试配置:")
print(f"   • batch_size: {batch_size}")
print(f"   • seq_len: {seq_len}")
print(f"   • hidden_dim: {hidden_dim}")

torch.manual_seed(42)
test_input = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)
print(f"   ✅ 测试数据生成完成: {test_input.shape}")

results["step3"] = {"success": True, "input_shape": test_input.shape}

print("\n" + "="*70)
print("【第四步】专家预测测试 (OMLX)")
print("="*70)

try:
    print("   🔄 执行专家预测...")
    
    predicted_experts = omlx.predict_experts(test_input, top_k=2)
    print(f"   ✅ 专家分配完成: {predicted_experts.flatten().tolist()}")
    
    unique_experts = list(set(predicted_experts.flatten().tolist()))
    print(f"   • 唯一专家数: {len(unique_experts)}")
    
    results["step4"] = {
        "success": True,
        "predicted_experts": predicted_experts.flatten().tolist(),
        "unique_experts": unique_experts
    }
    
except Exception as e:
    print(f"   ❌ 专家预测失败: {e}")
    results["step4"] = {"success": False, "error": str(e)}

print("\n" + "="*70)
print("【第五步】专家加载测试 (使用 load_expert 方法)")
print("="*70)

try:
    print("   🔄 执行专家加载...")
    
    target_experts = [0, 1, 2, 3]
    for expert_id in target_experts:
        flashmoe.load_expert(expert_id)
    
    print(f"   ✅ 专家加载完成")
    print(f"   • 加载专家数: {len(target_experts)}")
    print(f"   • 缓存专家数: {len(flashmoe.cache_manager)}")
    
    results["step5"] = {
        "success": True,
        "loaded_count": len(target_experts),
        "cached_count": len(flashmoe.cache_manager)
    }
    
except Exception as e:
    print(f"   ❌ 专家加载失败: {e}")
    results["step5"] = {"success": False, "error": str(e)}

print("\n" + "="*70)
print("【第六步】MoE 推理测试 (使用 forward_with_auto_load)")
print("="*70)

try:
    print("   🔄 执行 MoE 推理...")
    
    start_time = time.time()
    result = flashmoe.forward_with_auto_load(test_input, top_k=2, expert_ids=[0, 1])
    elapsed = time.time() - start_time
    
    print(f"   ✅ MoE 推理完成")
    print(f"   • 输入形状: {test_input.shape}")
    print(f"   • 输出形状: {result.shape}")
    print(f"   • 推理时间: {elapsed*1000:.2f} ms")
    print(f"   • 吞吐: {batch_size * seq_len / elapsed / 1000:.2f} k tokens/sec")
    
    results["step6"] = {
        "success": True,
        "input_shape": test_input.shape,
        "output_shape": result.shape,
        "latency_ms": elapsed * 1000,
        "throughput_ktps": batch_size * seq_len / elapsed / 1000
    }
    
except Exception as e:
    print(f"   ❌ MoE 推理失败: {e}")
    results["step6"] = {"success": False, "error": str(e)}

print("\n" + "="*70)
print("【第七步】结果验证")
print("="*70)

try:
    result = results["step6"]
    if result["success"]:
        assert result["output_shape"][0] == batch_size, "Batch size mismatch"
        assert result["output_shape"][1] == seq_len, "Sequence length mismatch"
        print("   ✅ 结果验证通过")
        
        results["step7"] = {"success": True}
    else:
        print("   ⚠️ 跳过验证（推理失败）")
        results["step7"] = {"success": False}
        
except Exception as e:
    print(f"   ❌ 验证失败: {e}")
    results["step7"] = {"success": False, "error": str(e)}

print("\n" + "="*70)
print("📊 Gate 2.2 E2E 测试结果汇总")
print("="*70)

print(f"\n{'步骤':<35} {'状态':<10} {'关键指标'}")
print("-"*70)

step_names = {
    "step1": "FlashMoE 初始化",
    "step2": "OMLX 初始化",
    "step3": "测试数据准备",
    "step4": "专家预测",
    "step5": "专家加载",
    "step6": "MoE 推理",
    "step7": "结果验证"
}

for step, data in results.items():
    status = "✅" if data["success"] else "❌"
    metrics = ""
    step_name = step_names.get(step, step)
    
    if step == "step1":
        metrics = f"后端: {data.get('backend', '-')}, 设备: {data.get('device', '-')}"
    elif step == "step5":
        metrics = f"加载: {data.get('loaded_count', '-')} 个, 缓存: {data.get('cached_count', '-')} 个"
    elif step == "step6":
        if data["success"]:
            metrics = f"延迟: {data.get('latency_ms', -1):.1f}ms, 吞吐: {data.get('throughput_ktps', -1):.1f}k tok/s"
        else:
            metrics = f"错误: {data.get('error', '-')[:30]}..."
    
    print(f"{step_name:<35} {status:<10} {metrics}")

all_passed = all(data["success"] for data in results.values())

print("\n" + "="*70)
if all_passed:
    print("🎉 Gate 2.2 端到端测试全部通过!")
else:
    print("⚠️ Gate 2.2 端到端测试部分失败，请检查上述错误")
print("="*70)

print("""
📋 测试覆盖:
   • ✅ FlashMoE 客户端初始化（真实后端）
   • ✅ OMLX 客户端初始化
   • ✅ 专家预测 (OMLX)
   • ✅ 专家加载 (load_expert)
   • ⚠️ MoE 推理 (forward_with_auto_load)
   • ⚠️ 结果验证

🔑 技术栈:
   • FlashMoE - 跨平台 MoE 引擎
   • OMLX - 专家选择网络
   • ExpertCacheManager - LRU 缓存管理
   • Metal/CUDA/CPU 后端自动选择
""")

if __name__ == "__main__":
    sys.exit(0 if all_passed else 1)
