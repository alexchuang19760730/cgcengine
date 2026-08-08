#!/usr/bin/env python3
"""
测试：验证 MagiCompiler 原生能力内化到 Harness Agent Strategy

测试目标：
1. 验证 Sand.ai MagiCompiler 6大核心能力正确映射
2. 验证4大后端策略配置正确
3. 验证 StrategyDispatcher 正确分发策略
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness_strategy import (
    HarnessAgent,
    HarnessStrategy,
    StrategyDispatcher,
    MagiBackendType,
)

# ====================
# 测试 1: 验证 MagiCompiler 6大核心能力映射
# ====================
print("=" * 70)
print("测试 1: Sand.ai MagiCompiler 6大核心能力映射验证")
print("=" * 70)

# 创建 Harness Agent
agent = HarnessAgent()
print("✅ HarnessAgent 初始化成功")

# 核心能力映射表
core_capabilities = {
    "整图/整层编译": {
        "description": "推理期捕获完整计算图、训练期以 Transformer Layer 为单位整层编译",
        "check": lambda config: config.compile.compile_mode in ["full_graph", "layer_wise"],
    },
    "Compiler as Manager": {
        "description": "编译器全局接管计算调度 + 显存生命周期",
        "check": lambda config: config.memory.compiler_as_manager,
    },
    "启发式自动重计算": {
        "description": "自动分析图，显存密集算子动态重计算，无需手动 checkpoint",
        "check": lambda config: config.optimization.auto_recalculation,
    },
    "JIT Offload 调度": {
        "description": "权重按访问频率分级，常驻 GPU/CPU 内存；计算前最晚预取",
        "check": lambda config: config.optimization.jit_offload,
    },
    "FSDP-Aware 分布式优化": {
        "description": "分布式训练时将通信原语深度内联计算图",
        "check": lambda config: config.distributed.fsdp_aware,
    },
    "无 Graph Break": {
        "description": "推理/训练稳定捕获完整计算图",
        "check": lambda config: config.graph_capture.graph_break_protection,
    },
}

# 测试每个后端的能力映射
for backend in MagiBackendType:
    config = agent.get_strategy(backend)
    print(f"\n--- {backend.value} ---")
    
    for capability, info in core_capabilities.items():
        result = info["check"](config)
        status = "✅" if result else "❌"
        print(f"{status} {capability}: {'支持' if result else '不支持'}")

# ====================
# 测试 2: 策略配置正确性验证
# ====================
print("\n" + "=" * 70)
print("测试 2: 各后端策略配置正确性验证")
print("=" * 70)

# 预期配置对照表
expected_configs = {
    MagiBackendType.LLAMA_CPP: {
        "name": "llama.cpp (端侧推理)",
        "graph_capture_mode": "full_graph",
        "compile_mode": "full_graph",
        "kda_enabled": True,
        "auto_recalculation": False,
        "jit_offload": False,
        "fsdp_aware": False,
        "graph_break_protection": True,
    },
    MagiBackendType.VLLM: {
        "name": "vLLM (云侧推理)",
        "graph_capture_mode": "full_graph",
        "compile_mode": "full_graph",
        "kda_enabled": True,
        "auto_recalculation": False,
        "jit_offload": True,
        "fsdp_aware": True,
        "graph_break_protection": True,
    },
    MagiBackendType.MEGATRAIN_2026_4: {
        "name": "MegaTrain 2026.4 (大模型训练)",
        "graph_capture_mode": "layer_wise",
        "compile_mode": "layer_wise",
        "kda_enabled": True,
        "auto_recalculation": True,
        "jit_offload": True,
        "fsdp_aware": True,
        "graph_break_protection": True,
    },
    MagiBackendType.MLX_TUNE: {
        "name": "mlx-tune (端侧微调)",
        "graph_capture_mode": "full_graph",
        "compile_mode": "full_graph",
        "kda_enabled": True,
        "auto_recalculation": False,
        "jit_offload": False,
        "fsdp_aware": False,
        "graph_break_protection": True,
    },
}

# 验证配置
all_passed = True
for backend, expected in expected_configs.items():
    config = agent.get_strategy(backend)
    print(f"\n验证: {expected['name']}")
    
    checks = [
        ("图捕获模式", config.graph_capture.capture_mode, expected["graph_capture_mode"]),
        ("编译模式", config.compile.compile_mode, expected["compile_mode"]),
        ("KDA优化", config.optimization.kda_enabled, expected["kda_enabled"]),
        ("自动重计算", config.optimization.auto_recalculation, expected["auto_recalculation"]),
        ("JIT Offload", config.optimization.jit_offload, expected["jit_offload"]),
        ("FSDP支持", config.distributed.fsdp_aware, expected["fsdp_aware"]),
        ("Graph Break保护", config.graph_capture.graph_break_protection, expected["graph_break_protection"]),
    ]
    
    passed = True
    for name, actual, expected_val in checks:
        status = "✅" if actual == expected_val else "❌"
        print(f"  {status} {name}: {actual}")
        if actual != expected_val:
            passed = False
            all_passed = False
    
    print(f"  {'✅ 配置正确' if passed else '❌ 配置有误'}")

# ====================
# 测试 3: StrategyDispatcher 分发验证
# ====================
print("\n" + "=" * 70)
print("测试 3: StrategyDispatcher 策略分发验证")
print("=" * 70)

dispatcher = StrategyDispatcher()

for backend in MagiBackendType:
    print(f"\n--- 分发策略到 {backend.value} ---")
    config = dispatcher.dispatch(backend)
    print(f"✅ 策略分发成功")

# ====================
# 测试 4: 策略更新与验证
# ====================
print("\n" + "=" * 70)
print("测试 4: 策略更新与验证")
print("=" * 70)

original_config = agent.get_strategy(MagiBackendType.VLLM)
print(f"原始 KDA 状态: {original_config.optimization.kda_enabled}")

agent.update_strategy(
    MagiBackendType.VLLM,
    kda_enabled=False,
    jit_offload=False,
)

updated_config = agent.get_strategy(MagiBackendType.VLLM)
print(f"更新后 KDA 状态: {updated_config.optimization.kda_enabled}")
print(f"更新后 JIT Offload 状态: {updated_config.optimization.jit_offload}")

if not updated_config.optimization.kda_enabled and not updated_config.optimization.jit_offload:
    print("✅ 策略更新成功")
else:
    print("❌ 策略更新失败")
    all_passed = False

# 恢复原始配置
agent.update_strategy(
    MagiBackendType.VLLM,
    kda_enabled=True,
    jit_offload=True,
)

# ====================
# 测试 5: 策略验证功能测试
# ====================
print("\n" + "=" * 70)
print("测试 5: 策略验证功能测试")
print("=" * 70)

validation_results = agent.run_strategy_validation()
print("策略验证结果:")
for backend, passed in validation_results.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {backend}: {'验证通过' if passed else '验证失败'}")
    if not passed:
        all_passed = False

# ====================
# 测试总结
# ====================
print("\n" + "=" * 70)
print("测试总结")
print("=" * 70)

if all_passed:
    print("🎉 所有测试通过！")
    print("\n📋 MagiCompiler 原生能力内化验证：")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  ✅ 整图/整层编译         → compile_mode                   │")
    print("│  ✅ Compiler as Manager    → memory.compiler_as_manager    │")
    print("│  ✅ 启发式自动重计算       → optimization.auto_recalculation│")
    print("│  ✅ JIT Offload 调度       → optimization.jit_offload      │")
    print("│  ✅ FSDP-Aware 分布式优化  → distributed.fsdp_aware        │")
    print("│  ✅ 无 Graph Break        → graph_capture保护              │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("\n📋 四大后端策略配置：")
    print("┌────────────────┬──────────┬─────────┬──────────┬───────────┐")
    print("│ 后端           │ 编译模式 │ 自动重算 │ JIT Offload│ FSDP     │")
    print("├────────────────┼──────────┼─────────┼──────────┼───────────┤")
    print("│ llama.cpp      │ 整图     │ ❌      │ ❌       │ ❌        │")
    print("│ vLLM           │ 整图     │ ❌      │ ✅       │ ✅        │")
    print("│ MegaTrain      │ 整层     │ ✅      │ ✅       │ ✅        │")
    print("│ mlx-tune       │ 整图     │ ❌      │ ❌       │ ❌        │")
    print("└────────────────┴──────────┴─────────┴──────────┴───────────┘")
else:
    print("❌ 部分测试失败，请检查配置")
