# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
CGC Engine + KDA Benchmark 测试脚本
对比 GGUF → PyTorch + CGC KDA 与原生 llama.cpp Metal 性能
"""

import os
import sys
import time
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')

# 使用绝对导入
from cgc_engine.model_parsers import GGUFParser, parsed_model_to_pytorch
from cgc_engine import run_cgc_with_kda


def benchmark_cgc_kda(model, device="metal", seq_lens=[128, 512, 1024, 2048]):
    """
    测试 CGC Engine + KDA 性能
    """
    results = {}
    
    for seq_len in seq_lens:
        logger.info(f"▶️ 测试序列长度: {seq_len}")
        
        # Prefill 测试
        logger.info("  - Prefill 测试...")
        start_time = time.time()
        x = torch.randint(0, model.config.vocab_size, (1, seq_len))
        with torch.no_grad():
            out = model(x)
        prefill_time = time.time() - start_time
        prefill_tps = seq_len / prefill_time
        
        # Decode 测试（模拟）
        logger.info("  - Decode 测试...")
        start_time = time.time()
        for _ in range(128):  # 生成 128 tokens
            with torch.no_grad():
                out = model(out.argmax(-1)[:, -1:])
        decode_time = time.time() - start_time
        decode_tps = 128 / decode_time
        
        results[seq_len] = {
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "prefill_time": prefill_time,
            "decode_time": decode_time
        }
        
        logger.info(f"  ✅ Prefill: {prefill_tps:.2f} tokens/s")
        logger.info(f"  ✅ Decode: {decode_tps:.2f} tokens/s")
    
    return results


def run_full_benchmark():
    """
    运行完整的 benchmark 测试
    """
    gguf_path = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
    output_path = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m_pytorch.pth"
    
    logger.info("=" * 60)
    logger.info("🔥 CGC Engine + KDA Benchmark 测试")
    logger.info("=" * 60)
    
    # 1. 解析 GGUF
    logger.info("\n📥 Step 1: 解析 GGUF 模型")
    parser = GGUFParser(gguf_path)
    parsed_model = parser.parse_model()
    logger.info(f"   ✅ 模型类型: {parsed_model.model_type}")
    logger.info(f"   ✅ 层数: {parsed_model.num_layers}, 头数: {parsed_model.num_heads}")
    
    # 2. 加载权重
    logger.info("\n📥 Step 2: 加载权重")
    weights = parser.load_weights()
    logger.info(f"   ✅ 加载 {len(weights)} 个权重张量")
    
    # 3. 转换为 PyTorch
    logger.info("\n🔄 Step 3: GGUF → PyTorch 转换")
    torch_model = parsed_model_to_pytorch(parsed_model, weights)
    logger.info(f"   ✅ 转换完成")
    
    # 4. 保存 PyTorch 模型
    logger.info("\n💾 Step 4: 保存 PyTorch 模型")
    torch.save(torch_model.state_dict(), output_path)
    logger.info(f"   ✅ 保存到: {output_path}")
    
    # 5. 使用 CGC Engine + KDA 运行
    logger.info("\n🚀 Step 5: 运行 CGC Engine + KDA")
    try:
        out = run_cgc_with_kda(torch_model, device="metal")
        logger.info(f"   ✅ CGC + KDA 执行成功，输出形状: {out.shape}")
    except Exception as e:
        logger.error(f"   ❌ CGC + KDA 执行失败: {e}")
        return
    
    # 6. Benchmark 测试
    logger.info("\n📊 Step 6: Benchmark 测试")
    results = benchmark_cgc_kda(torch_model, device="metal")
    
    # 7. 输出对比结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 Benchmark 结果对比")
    logger.info("=" * 60)
    
    print("\n" + "=" * 70)
    print(f"| {'上下文长度':<12} | {'CGC KDA Prefill':<18} | {'CGC KDA Decode':<18} |")
    print("=" * 70)
    
    for seq_len in [128, 512, 1024, 2048]:
        if seq_len in results:
            print(f"| {seq_len:<12} | {results[seq_len]['prefill_tps']:<18.2f} | {results[seq_len]['decode_tps']:<18.2f} |")
    
    print("=" * 70)
    
    # 对比原生 llama.cpp Metal 结果
    print("\n📈 与原生 llama.cpp Metal 对比（参考值）:")
    print("┌────────────────┬─────────────────┬─────────────────┐")
    print("│   上下文长度   │  llama.cpp Prefill │  llama.cpp Decode │")
    print("├────────────────┼─────────────────┼─────────────────┤")
    print("│      128      │     176.29      │     20.84       │")
    print("│      512      │     176.38      │     20.85       │")
    print("│     1024      │     171.24      │     20.64       │")
    print("│     2048      │     155.78      │     18.74       │")
    print("└────────────────┴─────────────────┴─────────────────┘")
    
    logger.info("\n🎉 Benchmark 测试完成！")


if __name__ == "__main__":
    run_full_benchmark()