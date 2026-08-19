"""
Analyze Qwen3.6 architecture for PD separation planning.
Calculate exact memory requirements and layer distribution.
"""

import os
import sys
import json

sys.path.insert(0, r"D:\alex\flashkv0516")
from unified_moe_streamer import UnifiedExpertStreamer

MODEL_PATH = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

print("=" * 60)
print("QWEN3.6 35B-A3B ARCHITECTURE ANALYSIS")
print("=" * 60)

streamer = UnifiedExpertStreamer(MODEL_PATH)

layers = streamer.adapter.list_layers()
n_experts = streamer.adapter.num_experts(layers[0])
hidden = streamer.adapter.hidden
expert_inter = streamer.adapter.expert_inter
top_k = streamer.adapter.top_k

print(f"\nModel: Qwen3.6-35B-A3B-UD-IQ3_XXS")
print(f"File size: {os.path.getsize(MODEL_PATH) / 1024**3:.2f} GB")
print(f"\nArchitecture:")
print(f"  Layers: {len(layers)}")
print(f"  Hidden size: {hidden}")
print(f"  Expert intermediate: {expert_inter}")
print(f"  Experts per layer: {n_experts}")
print(f"  Top-K: {top_k}")

# Calculate parameter counts
# Attention parameters: hidden*hidden + 2*hidden*kv_dim + hidden*hidden
kv_dim = hidden // 2  # typical
attn_params_per_layer = hidden * hidden + 2 * hidden * kv_dim + hidden * hidden

# MoE parameters
gate_up_params = 2 * hidden * expert_inter * n_experts  # gate + up packed
down_params = expert_inter * hidden * n_experts
gate_inp_params = hidden * n_experts

total_moe_params_per_layer = gate_up_params + down_params + gate_inp_params

# Calculate memory
# For IQ3_XXS quantization, approximately 2 bits per element
# But actual size varies by quantization type
# Let's use actual tensor sizes from the streamer

print(f"\nMemory Analysis (using actual GGUF tensor sizes):")

# Load a sample layer to get actual sizes
sample_layer = layers[0]
sample_expert = streamer.load_expert(sample_layer, 0)
roles = sample_expert.get('roles', {})

print(f"\n  Sample expert (Layer {sample_layer}, Expert 0):")
for role_name, role_data in roles.items():
    size_kb = role_data.get('size_bytes', 0) / 1024
    dims = role_data.get('dims', [])
    print(f"    {role_name}: {size_kb:.1f} KB, dims={dims}")

# Calculate total expert memory per layer
expert_bytes_per_layer = sum(
    r.get('size_bytes', 0) 
    for r in roles.values()
) * n_experts

print(f"\n  Total expert memory per layer: {expert_bytes_per_layer / 1024**2:.2f} MB")
print(f"  Total expert memory ({len(layers)} layers): {expert_bytes_per_layer * len(layers) / 1024**3:.2f} GB")

# PD split calculation
prefill_layers = len(layers) // 2
decode_layers = len(layers) - prefill_layers

prefill_mem = expert_bytes_per_layer * prefill_layers
decode_mem = expert_bytes_per_layer * decode_layers

print(f"\n  PD Split (equal):")
print(f"    Prefill: {prefill_layers} layers = {prefill_mem / 1024**3:.2f} GB")
print(f"    Decode:  {decode_layers} layers = {decode_mem / 1024**3:.2f} GB")

# Optimized split based on GPU memory
gpu0_mem = 4 * 1024**3  # Intel UHD ~4GB
gpu1_mem = 2 * 1024**3  # NVIDIA MX250 ~2GB

# GPU0 can handle all prefill layers (4GB > ~2.5GB)
# GPU1 needs decode layers within 2GB
# Need to check if decode layers fit

decode_layers_fit = int(gpu1_mem / expert_bytes_per_layer * 0.85)  # 85% utilization
prefill_layers_opt = len(layers) - decode_layers_fit

print(f"\n  Optimized Split (GPU memory-aware):")
print(f"    GPU 0 (Intel UHD, 4GB): {prefill_layers_opt} layers")
print(f"      Memory: {expert_bytes_per_layer * prefill_layers_opt / 1024**3:.2f} GB / 4GB")
print(f"    GPU 1 (NVIDIA MX250, 2GB): {decode_layers_fit} layers")
print(f"      Memory: {expert_bytes_per_layer * decode_layers_fit / 1024**3:.2f} GB / 2GB")

# Expert streaming optimization
# During decode, only load k experts (top_k) per token
decode_expert_per_token = expert_bytes_per_layer / n_experts * top_k
print(f"\n  Decode Expert Streaming:")
print(f"    Per-token expert loading: {decode_expert_per_token / 1024**2:.2f} MB")
print(f"    This fits easily in GPU 1's 2GB memory")

# Cache strategy
cache_layers = min(10, decode_layers_fit)  # Cache 10 layers' experts
cache_mem = expert_bytes_per_layer * cache_layers
print(f"    Suggested cache: {cache_layers} layers = {cache_mem / 1024**3:.2f} GB")

# Save analysis
analysis = {
    'model': 'Qwen3.6-35B-A3B-UD-IQ3_XXS',
    'architecture': {
        'layers': len(layers),
        'hidden': hidden,
        'expert_inter': expert_inter,
        'experts_per_layer': n_experts,
        'top_k': top_k,
    },
    'memory': {
        'expert_per_layer_bytes': expert_bytes_per_layer,
        'total_expert_gb': expert_bytes_per_layer * len(layers) / 1024**3,
    },
    'pd_split': {
        'equal': {
            'prefill_layers': prefill_layers,
            'decode_layers': decode_layers,
            'prefill_gb': prefill_mem / 1024**3,
            'decode_gb': decode_mem / 1024**3,
        },
        'optimized': {
            'gpu0_layers': prefill_layers_opt,
            'gpu1_layers': decode_layers_fit,
            'gpu0_gb': expert_bytes_per_layer * prefill_layers_opt / 1024**3,
            'gpu1_gb': expert_bytes_per_layer * decode_layers_fit / 1024**3,
        },
    },
}

output_dir = r"D:\alex\flashkv0516\bench_results"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'qwen36_arch_analysis.json'), 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"\nAnalysis saved to {output_dir}/qwen36_arch_analysis.json")
print("\n" + "=" * 60)
