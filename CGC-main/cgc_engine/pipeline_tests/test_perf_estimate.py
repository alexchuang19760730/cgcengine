#!/usr/bin/env python3
"""CGC Shader vs ds4.c 性能預估"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from cgc_engine.cgc.metal_shader_generator import MetalShaderGenerator, GraphAnalysisResult
from cgc_engine.cgc.ds4_shader_comparator import DS4ShaderComparator

print("=" * 70)
print("CGC Shader vs ds4.c 性能預估")
print("=" * 70)

comparator = DS4ShaderComparator()
generator = MetalShaderGenerator(output_dir="./generated_shaders")

graph_analysis = GraphAnalysisResult(
    has_moe=True, has_moe_router=True, num_experts=8, num_active_experts=2,
    has_gqa=True, has_flash_attention=True, has_rope=True, has_rms_norm=True,
    has_quantization=True, quantization_bits=2, num_layers=28, hidden_dim=4096,
    num_heads=32, head_dim=128, seq_len=4096, batch_size=1,
)

shaders = generator.generate(graph_analysis, {"top_k": 2, "num_experts": 8})
shaders_str = {k.value: v for k, v in shaders.items()}

result = comparator.compare(shaders_str)
comparator.print_report(result)