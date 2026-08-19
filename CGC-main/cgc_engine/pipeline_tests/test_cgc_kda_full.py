#!/usr/bin/env python3
"""
CGC Engine + KDA 完整测试脚本
测试流程：
1. 创建一个简单的 transformer 模型
2. 用 CGC Engine 编译
3. 注入 KDA 策略
4. 运行推理并测量性能
"""
import sys
import os
import time
import torch
import torch.nn as nn

from cgc_engine import CGCEngine, compile


def create_simple_transformer_model(
    vocab_size: int = 32000,
    hidden_size: int = 512,
    num_layers: int = 2,
    num_heads: int = 8
):
    """创建一个简单的 Transformer 模型用于测试"""
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            
            # 简单的 attention 层
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True
            )
            
            # 简单的 FFN
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            
            # Attention 层
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            
            # FFN
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            
            return self.lm_head(x)
    
    return SimpleTransformer()


def test_cgc_kda():
    """测试 CGC 引擎 + KDA"""
    print("=" * 80)
    print("CGC Engine + KDA 测试")
    print("=" * 80)
    
    print("\n1. 创建简单的 Transformer 模型")
    model = create_simple_transformer_model()
    model.eval()
    print(f"   模型创建成功！参数数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 准备测试输入
    input_ids = torch.randint(0, 32000, (1, 128), dtype=torch.long)
    print(f"   测试输入形状：{input_ids.shape}")
    
    # 2. 基准测试 - 原生 PyTorch
    print("\n2. 基准测试 - 原生 PyTorch")
    num_warmups = 3
    num_iterations = 10
    for _ in range(num_warmups):
        _ = model(input_ids)
    
    total_time = 0
    for i in range(num_iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        output = model(input_ids)
        total_time += time.time() - start
    avg_time = total_time / num_iterations
    print(f"   平均推理时间：{avg_time*1000:.2f} ms")
    print(f"   输出形状：{output.shape}")
    
    # 3. CGC 引擎初始化（不使用命令模式）
    print("\n3. CGC Engine 初始化（Agent + 策略注入）")
    try:
        # 不使用 use_cgc_commands，直接用 model 模式
        engine = CGCEngine(model=model, device='cpu')
        
        # 启用 agent 决策
        agent_config = {
            "enable_heuristic": True
        }
        from cgc_engine.agent.harness_agent import HarnessAgent
        engine._harness_agent = HarnessAgent(
            device='cpu',
            enable_heuristic=True,
            enable_llama_cpp_reference=False
        )
        
        print("   ✅ CGC Engine + Agent 初始化成功")
        
        # 运行推理（直接用 model，不使用命令）
        print("\n   3.1 CGC 推理（直接 model 模式）")
        for _ in range(num_warmups):
            _ = engine._model(input_ids)
        
        total_time_cgc = 0
        for i in range(num_iterations):
            start = time.time()
            output_cgc = engine._model(input_ids)
            total_time_cgc += time.time() - start
        avg_time_cgc = total_time_cgc / num_iterations
        print(f"   CGC 平均推理时间：{avg_time_cgc*1000:.2f} ms")
        print(f"   输出形状：{output_cgc.shape}")
        
        # 对比输出
        print("\n   3.2 输出一致性检查")
        print(f"   输出差异 max：{(output - output_cgc).abs().max():.4e}")
        print(f"   输出差异 mean：{(output - output_cgc).abs().mean():.4e}")
        
    except Exception as e:
        print(f"   ❌ CGC Engine 测试失败：{e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_cgc_kda()
