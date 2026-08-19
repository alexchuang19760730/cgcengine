#!/usr/bin/env python3
"""Test ExpertStreamerLite with get_expert_weight method."""
import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")

from llama_monkey_patch import ExpertStreamerLite

# Test with qwen3.6 A3B model
model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

print("Testing get_expert_weight for per-layer mode:")
print("=" * 60)

streamer = ExpertStreamerLite(model_path)

print(f"\nModel parameters:")
print(f"  architecture: {streamer.architecture}")
print(f"  hidden: {streamer.hidden}")
print(f"  intermediate: {streamer.inter}")
print(f"  num_experts: {streamer.num_experts}")
print(f"  num_layers: {streamer.num_layers}")

print(f"\nTesting get_expert_weight:")
for layer_id in [0, 1]:
    for expert_id in [0, 1, 127, 255]:
        for role in ["gate", "up", "down"]:
            try:
                weight = streamer.get_expert_weight(layer_id, expert_id, role)
                if weight is not None:
                    print(f"  Layer {layer_id}, Expert {expert_id}, {role}: shape={weight.shape} dtype={weight.dtype}")
                else:
                    print(f"  Layer {layer_id}, Expert {expert_id}, {role}: None (invalid)")
            except Exception as e:
                print(f"  Layer {layer_id}, Expert {expert_id}, {role}: ERROR - {e}")

print(f"\nTesting load_expert:")
for layer_id in [0]:
    for expert_id in [0, 1, 255]:
        try:
            result = streamer.load_expert(expert_id, layer_id)
            if result:
                print(f"  Layer {layer_id}, Expert {expert_id}:")
                for role, data in result.get("roles", {}).items():
                    print(f"    {role}: dims={data.get('dims')} size={data.get('data_size')} type={data.get('ggml_type')}")
            else:
                print(f"  Layer {layer_id}, Expert {expert_id}: None")
        except Exception as e:
            print(f"  Layer {layer_id}, Expert {expert_id}: ERROR - {e}")

# Verify data integrity
print(f"\nVerifying data integrity (Layer 0, Expert 0, gate):")
weight = streamer.get_expert_weight(0, 0, "gate")
if weight is not None:
    print(f"  shape: {weight.shape}")
    print(f"  dtype: {weight.dtype}")
    print(f"  first 10 bytes: {weight.flatten()[:10]}")
    print(f"  min: {weight.min()}, max: {weight.max()}")

# Test caching
print(f"\nCache stats:")
stats = streamer.cache_stats()
for key, val in stats.items():
    print(f"  {key}: {val}")
