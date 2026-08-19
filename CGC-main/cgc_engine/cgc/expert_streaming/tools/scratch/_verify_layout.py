import sys
import os
import json

sys.path.insert(0, r"D:\alex\flashkv0516\CGC-main\cgc_engine\cpp\expert_streaming")

from gguf import GGUFReader

GGML_QUANT_SIZES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
    14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2),
}

def ggml_bpe(ggml_type):
    if ggml_type in GGML_QUANT_SIZES:
        bs, bb = GGML_QUANT_SIZES[ggml_type]
        return bb / bs
    return 0.0

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

print("=" * 60)
print("Gemma 4 Expert Streaming Layout Verification")
print("=" * 60)

reader = GGUFReader(gguf_path)

print("\n--- Model Metadata ---")
meta = {
    'architecture': reader.fields['general.architecture'].contents(),
    'expert_count': reader.fields['gemma4.expert_count'].contents(),
    'active_experts': reader.fields['gemma4.expert_used_count'].contents(),
    'block_count': reader.fields['gemma4.block_count'].contents(),
    'hidden_size': reader.fields['gemma4.embedding_length'].contents(),
    'expert_ffn_length': reader.fields['gemma4.expert_feed_forward_length'].contents(),
    'file_type': reader.fields['general.file_type'].contents(),
}
for k, v in meta.items():
    print(f"  {k}: {v}")

print("\n--- Expert Tensor Layout Analysis ---")

expert_tensors = []
for i, t in enumerate(reader.tensors):
    if hasattr(t, 'name') and '_exps' in t.name.lower():
        expert_tensors.append({
            'index': i,
            'name': t.name,
            'shape': list(t.shape),
            'type': t.tensor_type,
            'offset': t.data_offset,
            'n_elements': t.n_elements,
            'n_bytes': t.n_bytes,
        })

print(f"Total expert tensors: {len(expert_tensors)}")

if expert_tensors:
    sorted_tensors = sorted(expert_tensors, key=lambda x: x['offset'])
    
    print("\nFirst 10 expert tensors (sorted by offset):")
    for t in sorted_tensors[:10]:
        print(f"  [{t['index']}] {t['name']}")
        print(f"    shape={t['shape']}, type={t['type']}, offset={t['offset']}")
        print(f"    size={t['n_bytes']} bytes ({t['n_bytes']/1024/1024:.1f} MB)")
    
    print("\n--- Per-Layer Expert Analysis ---")
    
    layers = {}
    for t in sorted_tensors:
        name = t['name']
        layer_idx = int(name.split('.')[1])
        
        if layer_idx not in layers:
            layers[layer_idx] = []
        layers[layer_idx].append(t)
    
    for layer_idx in sorted(layers.keys())[:5]:
        layer_tensors = layers[layer_idx]
        print(f"\n  Layer {layer_idx}:")
        for t in layer_tensors:
            print(f"    {t['name']}: offset={t['offset']}, size={t['n_bytes']/1024/1024:.1f} MB")
        
        ffn_down = [t for t in layer_tensors if 'ffn_down_exps.weight' in t['name']]
        ffn_gate_up = [t for t in layer_tensors if 'ffn_gate_up_exps.weight' in t['name']]
        
        if ffn_down:
            t = ffn_down[0]
            expert_dims = t['shape'][-1]
            print(f"    Expert count (last dim): {expert_dims}")
            
            stride_elems = 1
            for d in t['shape'][:-1]:
                stride_elems *= d
            bpe = ggml_bpe(t['type'])
            expert_stride = int(stride_elems * bpe)
            print(f"    Per-expert stride: {expert_stride} bytes ({expert_stride/1024/1024:.1f} MB)")
        
        if len(layers) > 1 and layer_idx == 0:
            next_layer = layers[1]
            first_of_next = sorted(next_layer, key=lambda x: x['offset'])[0]
            stride = first_of_next['offset'] - sorted(layer_tensors, key=lambda x: x['offset'])[0]['offset']
            print(f"    Layer stride (to next layer): {stride} bytes ({stride/1024/1024:.1f} MB)")
    
    print("\n--- C Implementation Reference ---")
    
    first_layer = layers[0]
    first_tensor = sorted(first_layer, key=lambda x: x['offset'])[0]
    
    layout_info = {
        'layout_type': 'PER_LAYER',
        'expert_count': meta['expert_count'],
        'active_experts': meta['active_experts'],
        'block_count': meta['block_count'],
        'hidden_size': meta['hidden_size'],
        'expert_ffn_length': meta['expert_ffn_length'],
        'stream_offset': first_tensor['offset'],
        'layers': []
    }
    
    for layer_idx in sorted(layers.keys()):
        layer_tensors = layers[layer_idx]
        ffn_down_scale = None
        ffn_down_weight = None
        ffn_gate_up_weight = None
        
        for t in layer_tensors:
            if 'ffn_down_exps.scale' in t['name']:
                ffn_down_scale = t
            elif 'ffn_down_exps.weight' in t['name']:
                ffn_down_weight = t
            elif 'ffn_gate_up_exps.weight' in t['name']:
                ffn_gate_up_weight = t
        
        layer_info = {
            'layer_index': layer_idx,
            'ffn_down_scale_offset': ffn_down_scale['offset'] if ffn_down_scale else 0,
            'ffn_down_weight_offset': ffn_down_weight['offset'] if ffn_down_weight else 0,
            'ffn_down_weight_size': ffn_down_weight['n_bytes'] if ffn_down_weight else 0,
            'ffn_gate_up_weight_offset': ffn_gate_up_weight['offset'] if ffn_gate_up_weight else 0,
            'ffn_gate_up_weight_size': ffn_gate_up_weight['n_bytes'] if ffn_gate_up_weight else 0,
        }
        
        if ffn_down_weight:
            t = ffn_down_weight
            stride_elems = 1
            for d in t['shape'][:-1]:
                stride_elems *= d
            bpe = ggml_bpe(t['type'])
            layer_info['expert_stride'] = int(stride_elems * bpe)
            layer_info['expert_dims'] = t['shape'][:-1]
            layer_info['expert_count_last_dim'] = t['shape'][-1]
        
        layout_info['layers'].append(layer_info)
    
    output_path = os.path.join(os.path.dirname(gguf_path), 'gemma4_layout_info.json')
    with open(output_path, 'w') as f:
        json.dump(layout_info, f, indent=2)
    
    print(f"\n  Layout info saved to: {output_path}")
    
    print("\n--- Summary for C Implementation ---")
    print(f"  1. Layout: PER_LAYER (all {meta['expert_count']} experts packed per tensor)")
    print(f"  2. Per layer: 3 expert tensors (ffn_down.scale, ffn_down.weight, ffn_gate_up.weight)")
    print(f"  3. Expert count in last dim of weight tensors")
    print(f"  4. Per-expert slice = tensor_size / expert_count")
    print(f"  5. IQ4_XS type (23): block=256, bytes=136 → {136/256:.4f} bytes/elem")
    
    if len(layout_info['layers']) >= 2:
        l0 = layout_info['layers'][0]
        l1 = layout_info['layers'][1]
        print(f"\n  Layer 0 -> Layer 1 stride: {l1['ffn_down_weight_offset'] - l0['ffn_down_weight_offset']} bytes")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
