#!/usr/bin/env python3
"""Test ExpertStreamerLite with the updated gguf parser."""
import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")

from llama_monkey_patch import ExpertStreamerLite

# Test with qwen3.6 A3B model
model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

print("Testing ExpertStreamerLite with official gguf library:")
print("=" * 60)

streamer = ExpertStreamerLite(model_path)

print(f"\nModel parameters:")
print(f"  architecture: {streamer.architecture}")
print(f"  hidden: {streamer.hidden}")
print(f"  intermediate: {streamer.inter}")
print(f"  num_experts: {streamer.num_experts}")
print(f"  num_layers: {streamer.num_layers}")
print(f"  has_experts: {streamer.has_experts()}")

print(f"\nOffset map stats:")
print(f"  per-expert entries: {len(streamer._offsets)}")
print(f"  per-layer entries: {len(streamer._layer_offsets)}")

# Check per-layer offsets
if streamer._layer_offsets:
    print(f"\nPer-layer offset details (first 3 layers):")
    for layer_id in sorted(streamer._layer_offsets.keys())[:3]:
        layer = streamer._layer_offsets[layer_id]
        print(f"  Layer {layer_id}:")
        for role, t in layer.items():
            print(f"    {role}: dims={t['dims']} offset={t['offset']}")

# Test get_expert_weight for per-layer mode
print(f"\nTesting get_expert_weight (per-layer mode):")
for layer_id in [0, 1, 2]:
    for expert_id in [0, 1, 2]:
        for role in ["gate", "up", "down"]:
            try:
                weight = streamer.get_expert_weight(layer_id, expert_id, role)
                print(f"  Layer {layer_id}, Expert {expert_id}, {role}: shape={weight.shape if hasattr(weight, 'shape') else 'N/A'} dtype={weight.dtype if hasattr(weight, 'dtype') else 'N/A'}")
            except Exception as e:
                print(f"  Layer {layer_id}, Expert {expert_id}, {role}: ERROR - {e}")
                break  # Stop on first error

# Test caching
print(f"\nCache stats:")
print(f"  hits: {streamer._hits}")
print(f"  misses: {streamer._misses}")
