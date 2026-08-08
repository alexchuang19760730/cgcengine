#!/usr/bin/env python3
# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
CGC Engine - MagiCompiler + Harness Agent 完整测试

完整流程：
1. MagiCompiler 捕获整图
2. Harness Agent 决策策略
3. 注入策略到编译器
4. 编译生成模型
5. 执行推理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

from cgc_engine.magicompiler_integration import MagiCompiler
from cgc_engine.agent.harness_agent import HarnessAgent
from cgc_engine.agent.strategy_executor import StrategyExecutor
from cgc_engine.hardware.hardware_constraints import HardwareConstraints


def create_simple_transformer_model():
    """创建简单 Transformer 模型用于测试"""
    d_model = 128
    nhead = 2
    dim_feedforward = 256
    dropout = 0.1

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=2)


def cgc_engine_main(model, device="metal"):
    """
    CGC 主引擎（正确流程）

    流程：
    1. MagiCompiler 捕获整图
    2. Harness Agent 决策策略
    3. 注入策略到编译器
    4. 编译生成模型
    """
    print("\n" + "=" * 60)
    print("🚀 CGC Engine 启动 (MagiCompiler + Agent)")
    print("=" * 60 + "\n")

    mgc = MagiCompiler(model)
    graph = mgc.capture_full_graph()
    print(f"✅ 计算图捕获完成，设备: {device}")

    hw = HardwareConstraints(device=device)
    print(f"✅ 硬件信息加载完成: {hw}")

    agent = HarnessAgent(device=device)
    strategy = agent.decide(graph, hw)
    print("✅ Harness Agent 策略决策完成")

    executor = StrategyExecutor()
    executor.apply_to_magicompiler(strategy, mgc)
    print("✅ 策略注入 MagiCompiler 完成")

    compiled_model = mgc.compile()
    print("✅ 模型编译完成！")

    return compiled_model


if __name__ == "__main__":
    model = create_simple_transformer_model()

    compiled_model = cgc_engine_main(model, device="metal")

    batch_size = 1
    seq_len = 64
    d_model = 128
    x = torch.randn(batch_size, seq_len, d_model)

    print("\n▶️  开始执行编译后模型")
    output = compiled_model(x)
    print(f"\n✅ 执行完成，输出 shape: {output.shape}")
