#!/usr/bin/env python3
"""
Harness Agent 六大策略测试
展示知识库中的六大策略如何工作
"""

import sys

from cgc_engine.utils.knowledge_storage import KnowledgeStorage


def test_six_strategies():
    """测试六大策略"""
    print("=" * 100)
    print("🎯 Harness Agent 六大策略测试")
    print("=" * 100)
    
    # 初始化知识库
    knowledge = KnowledgeStorage()
    
    # 获取所有策略
    all_strategies = knowledge.get_optimization_strategies()
    
    print(f"\n📊 知识库中共有 {len(all_strategies)} 个优化策略\n")
    
    # 按类型分组
    heuristic_strategies = [s for s in all_strategies if s.strategy_type == 'heuristic']
    reference_strategies = [s for s in all_strategies if s.strategy_type == 'reference']
    
    print("=" * 100)
    print("📋 启发式策略 (Heuristic)")
    print("=" * 100)
    for strategy in heuristic_strategies:
        print(f"\n🔹 {strategy.name} ({strategy.strategy_id})")
        print(f"   优先级：{strategy.priority}")
        print(f"   条件：{len(strategy.conditions)} 个")
        print(f"   动作：{len(strategy.actions)} 个")
        
        # 显示条件
        print("   触发条件:")
        for key, value in strategy.conditions.items():
            print(f"     - {key}: {value}")
        
        # 显示动作
        print("   执行动作:")
        for i, action in enumerate(strategy.actions, 1):
            action_type = action.get('action')
            print(f"     {i}. {action_type}")
    
    print("\n" + "=" * 100)
    print("📚 参考策略 (Reference)")
    print("=" * 100)
    for strategy in reference_strategies:
        print(f"\n🔹 {strategy.name} ({strategy.strategy_id})")
        print(f"   优先级：{strategy.priority}")
        print(f"   来源：{strategy.metadata.get('backend', 'unknown')}")
        print(f"   动作：{len(strategy.actions)} 个")
        
        # 显示动作
        print("   执行动作:")
        for i, action in enumerate(strategy.actions, 1):
            action_type = action.get('action')
            print(f"     {i}. {action_type}")
    
    # 测试策略匹配
    print("\n" + "=" * 100)
    print("🧪 策略匹配测试")
    print("=" * 100)
    
    test_contexts = [
        {
            "name": "Metal 后端 + Flash Attention",
            "context": {
                "device_type": "metal",
                "num_devices": 1,
                "has_flash_attention": True,
                "has_moe": False,
                "has_tensor_parallel": False,
                "has_vlm": False,
                "optimization_space_available": True,
                "enable_llama_cpp_reference": True,
                "enable_vllm_reference": False,
                "backend": "metal"
            }
        },
        {
            "name": "CUDA 后端 + MoE + vLLM 参考",
            "context": {
                "device_type": "cuda",
                "num_devices": 2,
                "has_flash_attention": True,
                "has_moe": True,
                "has_tensor_parallel": True,
                "has_vlm": False,
                "optimization_space_available": True,
                "enable_llama_cpp_reference": False,
                "enable_vllm_reference": True,
                "backend": "cuda"
            }
        },
        {
            "name": "CPU 后端 + llama.cpp 参考",
            "context": {
                "device_type": "cpu",
                "num_devices": 1,
                "has_flash_attention": False,
                "has_moe": False,
                "has_tensor_parallel": False,
                "has_vlm": False,
                "optimization_space_available": True,
                "enable_llama_cpp_reference": True,
                "enable_vllm_reference": False,
                "backend": "cpu"
            }
        }
    ]
    
    for test_case in test_contexts:
        print(f"\n📋 测试场景：{test_case['name']}")
        print("-" * 50)
        
        matched = knowledge.match_optimization_strategies(test_case['context'])
        
        print(f"✅ 匹配到 {len(matched)} 个策略:")
        for strategy in matched:
            print(f"   - {strategy.name} (优先级：{strategy.priority})")
    
    print("\n" + "=" * 100)
    print("✅ 测试完成")
    print("=" * 100)


if __name__ == "__main__":
    test_six_strategies()
