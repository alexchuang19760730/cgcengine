#!/usr/bin/env python3
"""
Harness Agent 知识库策略调度测试
验证 Harness Agent 如何使用知识库进行策略调度
"""

import sys

import torch
import torch.nn as nn
from cgc_engine.agent.harness_agent import HarnessAgent, HarnessCompileStrategy


class SimpleTransformer(nn.Module):
    """简单的 Transformer 模型用于测试"""
    def __init__(self, vocab_size=1000, dim=512, heads=8, layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
            for _ in range(layers)
        ])
        self.lm_head = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def test_harness_with_knowledge():
    """测试 Harness Agent 使用知识库"""
    print("=" * 100)
    print("🤖 Harness Agent 知识库策略调度测试")
    print("=" * 100)
    
    # 1. 创建模型
    print("\n📦 创建测试模型...")
    model = SimpleTransformer(vocab_size=1000, dim=512, heads=8, layers=4)
    input_shape = (32, 512)  # (batch_size, seq_len)
    
    # 2. 创建 Harness Agent（启用知识库）
    print("\n🧠 初始化 Harness Agent（启用知识库）...")
    agent = HarnessAgent(
        device="metal",  # 使用 Metal 后端
        enable_llama_cpp_reference=True,
        enable_vllm_reference=True,
        enable_heuristic=True,
        use_knowledge_base=True  # 启用知识库
    )
    
    # 3. 执行策略决策
    print("\n⚡ 执行策略决策...")
    strategy = agent.decide(model, input_shape)
    
    # 4. 打印策略结果
    print("\n📊 策略决策结果")
    print("-" * 50)
    print(f"  后端: {strategy.backend}")
    print(f"  启用算子融合: {strategy.enable_op_fusion}")
    print(f"  融合区域数量: {len(strategy.fusion_regions)}")
    print(f"  张量并行度: {strategy.tp_degree}")
    print(f"  流水线并行度: {strategy.pp_degree}")
    print(f"  调度配置: {strategy.schedules}")
    print(f"  注意力配置: {strategy.attention_config}")
    print(f"  MoE 配置: {strategy.moe_config}")
    print(f"  量化模式: {strategy.quantization_mode}")
    
    if strategy.op_hints:
        print(f"  操作提示: {[h.value for h in strategy.op_hints]}")
    
    # 5. 打印知识库信息
    print("\n📚 知识库信息")
    print("-" * 50)
    if agent.knowledge:
        platform_info = agent.knowledge.detect_current_platform()
        print(f"  检测平台: {platform_info['backend']} - {platform_info['hardware']}")
        
        # 获取存储的策略数量
        try:
            import sqlite3
            conn = sqlite3.connect(agent.knowledge.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM edge_cloud_strategies')
            strategy_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM graph_patterns')
            pattern_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM optimization_codes')
            code_count = cursor.fetchone()[0]
            conn.close()
            
            print(f"  端云策略数: {strategy_count}")
            print(f"  图模式数: {pattern_count}")
            print(f"  优化代码数: {code_count}")
        except Exception as e:
            print(f"  ⚠️ 无法查询知识库统计: {e}")
    else:
        print("  ⚠️ 知识库未初始化")
    
    print("\n" + "=" * 100)
    print("✅ 测试完成")
    print("=" * 100)
    
    return strategy


def test_knowledge_base_dispatcher():
    """测试知识库作为 Dispatcher"""
    print("\n\n" + "=" * 100)
    print("🎯 知识库 Dispatcher 测试")
    print("=" * 100)
    
    from cgc_engine.utils.knowledge_storage import KnowledgeStorage
    
    # 初始化知识库
    knowledge = KnowledgeStorage()
    
    # 模拟不同的请求场景
    test_scenarios = [
        {
            "name": "低延迟端侧推理",
            "request": {
                "max_requests_per_second": 5,
                "min_latency_requirement_ms": 50,
                "model_size_gb": 7,
                "edge_hardware_available": True
            }
        },
        {
            "name": "高吞吐量云端推理",
            "request": {
                "min_requests_per_second": 100,
                "model_size_gb": 30,
                "cloud_hardware_available": True
            }
        },
        {
            "name": "云端 Prefill + 端侧 Decode",
            "request": {
                "min_requests_per_second": 50,
                "prefill_seq_len": 1024,
                "decode_tokens": 128,
                "cloud_hardware_available": True,
                "edge_hardware_available": True
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📋 场景：{scenario['name']}")
        print("-" * 50)
        
        # 查找匹配的策略
        strategy = knowledge.find_matching_strategy(scenario['request'])
        
        if strategy:
            print(f"  ✅ 匹配策略：{strategy.name}")
            print(f"  描述：{strategy.description}")
            print(f"  优先级：{strategy.priority}")
            print(f"  动作数：{len(strategy.actions)}")
            
            for i, action in enumerate(strategy.actions, 1):
                print(f"    {i}. {action.get('action')}: {action}")
        else:
            print(f"  ⚠️ 未找到匹配策略")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    # 测试 1: Harness Agent 使用知识库
    strategy = test_harness_with_knowledge()
    
    # 测试 2: 知识库 Dispatcher
    test_knowledge_base_dispatcher()
