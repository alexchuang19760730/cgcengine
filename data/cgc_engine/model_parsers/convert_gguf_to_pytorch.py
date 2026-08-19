# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
GGUF → PyTorch 模型转换脚本
"""

import os
import torch
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/cgc_engine')

from model_parsers.gguf_parser import GGUFParser
from model_parsers.parsed_model_adapter import parsed_model_to_pytorch


def convert_gguf_to_pytorch(gguf_path: str, output_path: str = None):
    """
    将 GGUF 模型转换为 PyTorch 模型
    
    Args:
        gguf_path: GGUF 模型路径
        output_path: 输出 PyTorch 模型路径（可选）
    """
    logger.info(f"⏳ 开始转换 GGUF → PyTorch")
    logger.info(f"   输入: {gguf_path}")
    
    # 1. 解析 GGUF
    logger.info("▶️ Step 1: 解析 GGUF 模型结构...")
    parser = GGUFParser(gguf_path)
    parsed_model = parser.parse_model()
    logger.info(f"   ✅ 模型类型: {parsed_model.model_type}")
    logger.info(f"   ✅ 词汇量: {parsed_model.vocab_size}")
    logger.info(f"   ✅ 隐藏层维度: {parsed_model.hidden_dim}")
    logger.info(f"   ✅ 层数: {parsed_model.num_layers}")
    logger.info(f"   ✅ 头数: {parsed_model.num_heads}")
    logger.info(f"   ✅ 头维度: {parsed_model.head_dim}")
    
    # 2. 加载权重
    logger.info("▶️ Step 2: 加载 GGUF 权重...")
    weights = parser.load_weights()
    logger.info(f"   ✅ 加载了 {len(weights)} 个权重张量")
    
    # 3. 转换为 PyTorch
    logger.info("▶️ Step 3: 转换为 PyTorch Module...")
    torch_model = parsed_model_to_pytorch(parsed_model, weights)
    logger.info(f"   ✅ 转换完成")
    
    # 4. 保存模型
    if output_path is None:
        output_path = gguf_path.replace(".gguf", ".pth")
    
    logger.info(f"▶️ Step 4: 保存 PyTorch 模型...")
    torch.save(torch_model.state_dict(), output_path)
    logger.info(f"   ✅ 保存到: {output_path}")
    
    # 5. 验证加载
    logger.info("▶️ Step 5: 验证模型...")
    test_input = torch.randint(0, parsed_model.vocab_size, (1, 128))
    torch_model.eval()
    with torch.no_grad():
        output = torch_model(test_input)
    logger.info(f"   ✅ 验证通过，输出形状: {output.shape}")
    
    logger.info("🎉 GGUF → PyTorch 转换完成！")
    return torch_model, output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GGUF → PyTorch 模型转换器")
    parser.add_argument("gguf_path", help="输入 GGUF 模型路径")
    parser.add_argument("-o", "--output", help="输出 PyTorch 模型路径")
    
    args = parser.parse_args()
    
    convert_gguf_to_pytorch(args.gguf_path, args.output)