#!/usr/bin/env python3
"""
CGC vs ds4.c Shader 對比測試
=============================

使用實際下載的 ds4.c 原始 Shader 與 CGC 生成的 Shader 進行對比

Author: CGC Engine Team
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from cgc_engine.cgc.metal_shader_generator import MetalShaderGenerator, ShaderType, GraphAnalysisResult
from cgc_engine.cgc.ds4_shader_comparator import DS4ShaderComparator


def main():
    print("=" * 70)
    print("CGC vs ds4.c Shader 對比測試")
    print("=" * 70)

    comparator = DS4ShaderComparator()

    print("\n【1. 生成 CGC Shaders】")
    generator = MetalShaderGenerator(output_dir="./generated_shaders")

    graph_analysis = GraphAnalysisResult(
        has_moe=True,
        has_moe_router=True,
        num_experts=8,
        num_active_experts=2,
        has_gqa=True,
        has_flash_attention=True,
        has_rope=True,
        has_rms_norm=True,
        has_quantization=True,
        quantization_bits=2,
        num_layers=28,
        hidden_dim=4096,
        num_heads=32,
        head_dim=128,
        seq_len=4096,
        batch_size=1,
        detected_ops=["matmul", "softmax", "rms_norm", "moe_router", "moe_expert"],
    )

    config = {
        "top_k": 2,
        "num_experts": 8,
        "quantization": "2bit",
    }

    shaders = generator.generate(graph_analysis, config)
    print(f"生成 {len(shaders)} 個 CGC Shaders")

    shaders_str = {k.value: v for k, v in shaders.items()}

    print("\n【2. 使用實際 ds4.c 文件進行對比】")
    comparison = comparator.compare_with_actual_ds4(shaders_str)

    print("\n【3. 對比摘要】")
    has_cgc_count = sum(1 for c in comparison.values() if c["has_cgc"])
    has_ds4_count = sum(1 for c in comparison.values() if c["has_ds4"])
    print(f"ds4.c 有文件: {has_ds4_count} 個")
    print(f"CGC 有代碼:   {has_cgc_count} 個")

    total_ds4 = sum(c["ds4_lines"] for c in comparison.values())
    total_cgc = sum(c["cgc_lines"] for c in comparison.values())
    print(f"\nds4.c 總行數: {total_ds4}")
    print(f"CGC 總行數:   {total_cgc}")
    print(f"壓縮比:       {total_cgc/total_ds4*100:.1f}% (CGC 更簡潔)")

    print("\n【4. 各 Shader 詳細對比】")
    print("-" * 70)
    for name, info in sorted(comparison.items(), key=lambda x: -x[1]["ds4_lines"]):
        ratio = info["cgc_lines"] / info["ds4_lines"] * 100 if info["ds4_lines"] > 0 else 0
        print(f"  {name:<25} ds4:{info['ds4_lines']:>5} 行 -> cgc:{info['cgc_lines']:>4} 行 ({ratio:>5.1f}%)")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("對比完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()