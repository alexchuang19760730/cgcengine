#!/usr/bin/env python3
"""Test ExpertStreamerLite with Gemma 4 model (per-expert convention)."""
import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")

from llama_monkey_patch import ExpertStreamerLite, parse_gguf_header

# Test with Gemma 4 model
model_path = r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf"

print("Testing ExpertStreamerLite with Gemma 4 model:")
print("=" * 60)

# First just parse the header
print("\n--- Step 1: Parse GGUF header ---")
result = parse_gguf_header(model_path)

print(f"\nHeader info:")
print(f"  version: {result['version']}")
print(f"  n_tensors: {result['n_tensors']}")
print(f"  n_kv: {result['n_kv']}")
print(f"  data_start: {result['data_start']} (0x{result['data_start']:X})")

kv = result['kv']
print(f"\nKV metadata:")
for key in sorted(kv.keys())[:20]:
    val = kv[key]
    val_str = str(val)
    if len(val_str) > 80:
        val_str = val_str[:80] + "..."
    print(f"  {key}: {val_str}")

print(f"\nArchitecture: {kv.get('general.architecture', 'N/A')}")

# Check for gemma4-specific parameters
arch = kv.get('general.architecture', '')
print(f"\nGemma4-specific parameters:")
for prefix in [arch, 'gemma4', 'gemma']:
    for key in ['hidden_size', 'embedding_length', 'intermediate_size', 'feed_forward_length',
                'moe_intermediate_size', 'expert_count', 'num_experts', 'block_count', 'num_layers']:
        full_key = f"{prefix}.{key}"
        if full_key in kv:
            print(f"  {full_key}: {kv[full_key]}")

# Check expert tensors
print(f"\nExpert tensor analysis:")
expert_tensors = [t for t in result['tensors'] if 'expert' in t['name'].lower()]
print(f"  Total expert tensors: {len(expert_tensors)}")

# Show first 10 expert tensors
for t in expert_tensors[:10]:
    print(f"    '{t['name']}' dims={t['dims']} type={t['type']}")

# Check for per-expert convention (blk.X.expert.Y.role)
per_expert_tensors = [t for t in expert_tensors if 'expert' in t['name'] and 'exps' not in t['name']]
print(f"\n  Per-expert convention tensors (like blk.X.expert.Y.role): {len(per_expert_tensors)}")

# Check for per-layer convention (blk.X.ffn_{role}_exps)
per_layer_tensors = [t for t in expert_tensors if 'exps' in t['name']]
print(f"  Per-layer convention tensors (like blk.X.ffn_{{role}}_exps): {len(per_layer_tensors)}")

if per_expert_tensors:
    print(f"\n  First per-expert tensor: {per_expert_tensors[0]['name']}")
if per_layer_tensors:
    print(f"\n  First per-layer tensor: {per_layer_tensors[0]['name']}")

# Now test full ExpertStreamerLite
print("\n--- Step 2: Create ExpertStreamerLite ---")
try:
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
    
    if streamer._offsets:
        # Show first few per-expert entries
        print(f"\n  First per-expert entries:")
        for i, ((layer_id, expert_id, role), info) in enumerate(sorted(streamer._offsets.items())[:5]):
            print(f"    Layer {layer_id}, Expert {expert_id}, {role}: dims={info['dims']} offset={info['offset']}")
    
    if streamer._layer_offsets:
        # Show per-layer entries
        print(f"\n  Per-layer entries (first 3 layers):")
        for layer_id in sorted(streamer._layer_offsets.keys())[:3]:
            layer = streamer._layer_offsets[layer_id]
            print(f"    Layer {layer_id}:")
            for role, info in layer.items():
                print(f"      {role}: dims={info['dims']} offset={info['offset']}")
    
    print(f"\n✅ Gemma 4 model test completed!")
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()
