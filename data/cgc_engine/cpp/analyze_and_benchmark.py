#!/usr/bin/env python3
"""
Simple GGUF model analyzer using the unified_moe_streamer module.
"""

import os
import sys
import time
import numpy as np

# Import the unified streamer
sys.path.insert(0, r"D:\alex\flashkv0516")
from unified_moe_streamer import UnifiedExpertStreamer

# Models to analyze
models = [
    (r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf", "Qwen3.6-35B-A3B"),
]

print("=" * 70)
print("GGUF MODEL ANALYSIS AND BENCHMARK")
print("=" * 70)

for model_path, model_name in models:
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        continue
    
    file_size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"\n📦 Model: {model_name} ({file_size_gb:.2f} GB)")
    
    # Load with unified streamer
    t0 = time.time()
    streamer = UnifiedExpertStreamer(model_path)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.2f}s")
    
    # Get architecture from streamer/adapter
    print(f"\n📊 Architecture:")
    layers = streamer.adapter.list_layers()
    print(f"  Layers: {len(layers)}")
    if layers:
        print(f"  Experts/Layer: {streamer.adapter.num_experts(layers[0])}")
    if hasattr(streamer.adapter, 'top_k'):
        print(f"  Top-K: {streamer.adapter.top_k}")
    if hasattr(streamer.adapter, 'hidden'):
        print(f"  Hidden: {streamer.adapter.hidden}")
    if hasattr(streamer.adapter, 'expert_inter'):
        print(f"  Expert Intermediate: {streamer.adapter.expert_inter}")
    
    # List layers
    print(f"\n🔢 MoE Layers: {len(layers)}")
    print(f"  Layer IDs: {layers[:10]}... (showing first 10)")
    
    # Analyze expert structure
    print("\n🔍 Expert Structure Analysis:")
    layer_info = streamer.adapter._layer_info
    if layer_info:
        first_layer = list(layer_info.keys())[0]
        info = layer_info[first_layer]
        
        for role, tensor_info in info.items():
            if isinstance(tensor_info, dict) and 'dims' in tensor_info:
                dims = tensor_info['dims']
                size_bytes = tensor_info.get('size_bytes', 0)
                print(f"  {role}: dims={dims}, size={size_bytes/(1024**2):.1f}MB")
        
        # Calculate per-expert sizes
        for role, tensor_info in info.items():
            if isinstance(tensor_info, dict) and 'dims' in tensor_info:
                dims = tensor_info['dims']
                if len(dims) >= 3:
                    experts = dims[-1]
                    total_bytes = tensor_info.get('size_bytes', 0)
                    per_expert = total_bytes / experts if experts > 0 else 0
                    print(f"    → Per-expert {role}: {per_expert/(1024**2):.1f}MB ({experts} experts)")
    
    # Benchmark expert loading
    print("\n⏱️  Expert Loading Benchmark:")
    test_layer = layers[0] if layers else 0
    test_experts = [0, 10, 50, 100, 200]
    
    for eid in test_experts:
        try:
            t0 = time.time()
            expert = streamer.load_expert(test_layer, eid)
            load_time_ms = (time.time() - t0) * 1000
            
            if expert:
                roles = expert.get('roles', {})
                total_kb = sum(r.get('size_bytes', 0) / 1024 for r in roles.values())
                print(f"  Expert {eid}: {load_time_ms:.2f}ms ({total_kb:.0f}KB)")
            else:
                print(f"  Expert {eid}: failed to load")
        except Exception as e:
            print(f"  Expert {eid}: Error - {e}")
    
    # Estimate full inference performance
    print("\n📈 Performance Estimation:")
    
    # Based on measured expert loading times
    expert_load_times = []
    for eid in test_experts:
        try:
            t0 = time.time()
            streamer.load_expert(test_layer, eid)
            expert_load_times.append((time.time() - t0) * 1000)
        except:
            pass
    
    if expert_load_times:
        avg_load_ms = np.mean(expert_load_times)
        print(f"  Average expert load time: {avg_load_ms:.2f}ms")
        
        # For MoE models, each token needs to load top-K experts
        top_k = 8  # Typical for MoE models
        prefill_time_per_token = avg_load_ms * top_k / 1000  # seconds
        decode_time_per_token = avg_load_ms * top_k / 1000  # seconds
        
        # Add base computation time (attention, etc.)
        base_compute_ms = 20  # Estimate for attention and other ops
        prefill_total_ms = base_compute_ms + avg_load_ms * top_k
        decode_total_ms = base_compute_ms + avg_load_ms * top_k
        
        est_prefill_tps = 1000 / prefill_total_ms if prefill_total_ms > 0 else 0
        est_decode_tps = 1000 / decode_total_ms if decode_total_ms > 0 else 0
        
        print(f"\n  Estimated CPU Performance:")
        print(f"    Prefill: {est_prefill_tps:.1f} tokens/second ({prefill_total_ms:.1f}ms/token)")
        print(f"    Decode: {est_decode_tps:.1f} tokens/second ({decode_total_ms:.1f}ms/token)")
        
        # GPU speedup estimate
        gpu_speedup = 3  # Conservative estimate for integrated GPU
        print(f"\n  Estimated GPU Performance ({gpu_speedup}x speedup):")
        print(f"    Prefill: {est_prefill_tps * gpu_speedup:.1f} tokens/second")
        print(f"    Decode: {est_decode_tps * gpu_speedup:.1f} tokens/second")
        
        # Larger MoE models might have better GPU utilization
        gpu_speedup_large = 2.5
        print(f"\n  Estimated GPU Performance (large MoE, {gpu_speedup_large}x):")
        print(f"    Prefill: {est_prefill_tps * gpu_speedup_large:.1f} tokens/second")
        print(f"    Decode: {est_decode_tps * gpu_speedup_large:.1f} tokens/second")

# Summary
print("\n" + "=" * 70)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 70)

print("""
1. llama-bench.exe 无法运行的原因:
   - 编译时使用 MinGW/GCC (不是 MSVC)
   - 缺少 UCRT API 集的 DLL (api-ms-win-crt-private-l1-1-0.dll)
   - 错误代码: 0xC0000135 (STATUS_DLL_NOT_FOUND)

2. 解决方案:
   a) 安装 "Microsoft Visual C++ Redistributable"
   b) 从源码重新编译 llama.cpp (使用 MSVC 而非 MinGW)
   c) 使用 Python gguf 库进行模型分析 (已演示)

3. GPU 测试替代方案:
   - 使用 gguf Python 库读取模型元数据
   - 测量专家权重 I/O 性能
   - 基于架构参数估算推理性能
   - 实际 GPU 推理需要 llama.cpp 运行时

4. 性能估算说明:
   - MoE 模型性能受限于专家权重加载 I/O
   - 每 token 需加载 top-K 个专家 (通常 K=8)
   - GPU 可加速权重传输和矩阵计算
   - 实际性能需通过 llama-bench 验证
""")
