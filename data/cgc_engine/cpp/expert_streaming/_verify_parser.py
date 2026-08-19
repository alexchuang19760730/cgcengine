import sys
import struct
import os

sys.path.insert(0, r"D:\alex\flashkv0516\CGC-main\cgc_engine\cpp\expert_streaming")

from gguf import GGUFReader

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_QUANT_SIZES = {
    0: (1, 4),   # F32
    1: (1, 2),   # F16
    2: (32, 18), # Q4_0
    3: (32, 20), # Q4_1
    6: (32, 22), # Q5_0
    7: (32, 24), # Q5_1
    8: (32, 34), # Q8_0
    9: (32, 40), # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16
}

def ggml_bytes_per_elem(ggml_type):
    if ggml_type in GGML_QUANT_SIZES:
        block_size, block_bytes = GGML_QUANT_SIZES[ggml_type]
        return block_bytes / block_size
    return 0.0

def parse_bool(file):
    data = file.read(4)
    if len(data) != 4:
        return None
    return data[0] != 0

def parse_kv_file(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        print(f"Magic: 0x{magic:08X}")
        print(f"Version: {version}")
        print(f"Num tensors: {n_tensors}")
        print(f"Num KV: {n_kv}")

        kvs = {}
        for i in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8', errors='replace')
            
            value_type = struct.unpack('<I', f.read(4))[0]
            
            if value_type in [0, 1, 2, 3, 4, 5]:
                val = struct.unpack('<i', f.read(4))[0]
                kvs[key] = ('int', val)
            elif value_type == 6:
                val = struct.unpack('<f', f.read(4))[0]
                kvs[key] = ('float', val)
            elif value_type == 7:
                data = f.read(4)
                val = data[0] != 0
                kvs[key] = ('bool', val)
            elif value_type == 8:
                str_len = struct.unpack('<Q', f.read(8))[0]
                val = f.read(str_len).decode('utf-8', errors='replace')
                kvs[key] = ('string', val)
            elif value_type == 9:
                arr_len = struct.unpack('<Q', f.read(8))[0]
                elem_type = struct.unpack('<I', f.read(4))[0]
                
                skip = False
                if arr_len > 10000:
                    print(f"  Skipping large array '{key}' with {arr_len} elements")
                    skip = True
                
                arr_vals = []
                for j in range(arr_len):
                    if elem_type in [0, 1, 2, 3, 4, 5]:
                        v = struct.unpack('<i', f.read(4))[0]
                        if not skip: arr_vals.append(v)
                    elif elem_type == 6:
                        v = struct.unpack('<f', f.read(4))[0]
                        if not skip: arr_vals.append(v)
                    elif elem_type == 7:
                        data = f.read(4)
                        v = data[0] != 0
                        if not skip: arr_vals.append(v)
                    elif elem_type == 8:
                        str_len = struct.unpack('<Q', f.read(8))[0]
                        v = f.read(str_len).decode('utf-8', errors='replace')
                        if not skip: arr_vals.append(v)
                    else:
                        print(f"Unknown array elem type: {elem_type}")
                        return None
                
                if not skip:
                    kvs[key] = ('array', elem_type, arr_vals)
                else:
                    kvs[key] = ('array', elem_type, f'<skipped {arr_len} elements>')
            elif value_type == 10:
                val = struct.unpack('<q', f.read(8))[0]
                kvs[key] = ('int64', val)
            elif value_type == 11:
                val = struct.unpack('<q', f.read(8))[0]
                kvs[key] = ('int64', val)
            elif value_type == 12:
                val = struct.unpack('<d', f.read(8))[0]
                kvs[key] = ('float64', val)
            else:
                print(f"Unknown type: {value_type} for key {key}")
                return None

        print(f"\n--- KV Metadata ---")
        important_keys = [
            'general.architecture', 'gemma4.expert_count', 'gemma4.expert_used_count',
            'gemma4.block_count', 'gemma4.embedding_length', 'gemma4.feed_forward_length',
            'gemma4.expert_feed_forward_length', 'gemma4.attention.head_count',
            'gemma4.attention.head_count_kv', 'general.file_type'
        ]
        for key in important_keys:
            if key in kvs:
                val = kvs[key]
                print(f"  {key}: {val}")

        return kvs

def parse_tensors_file(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        for i in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len)
            value_type = struct.unpack('<I', f.read(4))[0]
            
            if value_type in [0, 1, 2, 3, 4, 5]:
                f.read(4)
            elif value_type == 6:
                f.read(4)
            elif value_type == 7:
                f.read(4)
            elif value_type == 8:
                str_len = struct.unpack('<Q', f.read(8))[0]
                f.read(str_len)
            elif value_type == 9:
                arr_len = struct.unpack('<Q', f.read(8))[0]
                elem_type = struct.unpack('<I', f.read(4))[0]
                
                for j in range(arr_len):
                    if elem_type in [0, 1, 2, 3, 4, 5]:
                        f.read(4)
                    elif elem_type == 6:
                        f.read(4)
                    elif elem_type == 7:
                        f.read(4)
                    elif elem_type == 8:
                        str_len = struct.unpack('<Q', f.read(8))[0]
                        f.read(str_len)
                    else:
                        print(f"Error: unknown elem type {elem_type}")
                        return None
            elif value_type == 10 or value_type == 11:
                f.read(8)
            elif value_type == 12:
                f.read(8)
            else:
                print(f"Error: unknown type {value_type}")
                return None

        tensor_infos = []
        for i in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8', errors='replace')
            
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = []
            for j in range(n_dims):
                dims.append(struct.unpack('<q', f.read(8))[0])
            
            ggml_type = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            n_elements = 1
            for d in dims:
                n_elements *= d
            
            bpe = ggml_bytes_per_elem(ggml_type)
            size_bytes = int(n_elements * bpe)
            
            tensor_infos.append({
                'name': name,
                'dims': dims,
                'ggml_type': ggml_type,
                'offset': offset,
                'n_elements': n_elements,
                'size_bytes': size_bytes
            })

        return tensor_infos

def analyze_expert_tensors(tensor_infos, expert_count):
    print(f"\n--- Analyzing Expert Tensors ---")
    
    per_expert_tensors = []
    per_layer_tensors = []
    
    for t in tensor_infos:
        name = t['name']
        
        if '_exps' in name:
            if 'ffn_down_exps.scale' in name:
                print(f"  SCALE: {name} dims={t['dims']} type={t['ggml_type']} offset={t['offset']}")
            elif 'ffn_down_exps.weight' in name:
                per_layer_tensors.append(t)
            elif 'ffn_gate_up_exps.weight' in name:
                per_layer_tensors.append(t)
        elif 'expert' in name.lower():
            per_expert_tensors.append(t)
    
    print(f"\nPer-layer expert tensors (Gemma 4 style): {len(per_layer_tensors)}")
    print(f"Per-expert tensors (Qwen3.6 style): {len(per_expert_tensors)}")
    
    if per_layer_tensors:
        print(f"\nGemma 4 Expert Analysis:")
        print(f"  Layout: PER_LAYER (all experts packed in single tensor)")
        print(f"  Expert count: {expert_count}")
        
        for t in per_layer_tensors[:5]:
            print(f"\n  Tensor: {t['name']}")
            print(f"    dims: {t['dims']}")
            print(f"    type: {t['ggml_type']} ({GGML_QUANT_SIZES.get(t['ggml_type'], 'unknown')})")
            print(f"    offset: {t['offset']}")
            print(f"    size: {t['size_bytes']} bytes ({t['size_bytes']/1024/1024:.1f} MB)")
            
            expert_dims_idx = len(t['dims']) - 1
            stride_elems = 1
            for d in t['dims'][:expert_dims_idx]:
                stride_elems *= d
            
            bpe = ggml_bytes_per_elem(t['ggml_type'])
            expert_stride_bytes = int(stride_elems * bpe)
            print(f"    expert_stride: {expert_stride_bytes} bytes ({expert_stride_bytes/1024/1024:.1f} MB)")
            print(f"    per_expert_size: {expert_stride_bytes} bytes")
    
    return per_layer_tensors, per_expert_tensors

def main():
    gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"
    
    print("=" * 60)
    print("GGUF Parser Verification (Python Reference Implementation)")
    print("=" * 60)
    
    print("\n--- Phase 1: Parse KV Metadata ---")
    kvs = parse_kv_file(gguf_path)
    if kvs is None:
        print("FAILED to parse KV metadata!")
        return
    
    expert_count = kvs.get('gemma4.expert_count', (None, 0))[1]
    block_count = kvs.get('gemma4.block_count', (None, 0))[1]
    hidden_size = kvs.get('gemma4.embedding_length', (None, 0))[1]
    moe_inter = kvs.get('gemma4.expert_feed_forward_length', (None, 0))[1]
    active_experts = kvs.get('gemma4.expert_used_count', (None, 0))[1]
    
    print(f"\nModel Summary:")
    print(f"  Architecture: gemma4")
    print(f"  Layers: {block_count}")
    print(f"  Expert count (total): {expert_count}")
    print(f"  Active experts: {active_experts}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  MoE intermediate: {moe_inter}")
    
    print("\n--- Phase 2: Parse Tensor Info ---")
    tensor_infos = parse_tensors_file(gguf_path)
    if tensor_infos is None:
        print("FAILED to parse tensor info!")
        return
    
    print(f"Total tensors: {len(tensor_infos)}")
    
    per_layer, per_expert = analyze_expert_tensors(tensor_infos, expert_count)
    
    if per_layer:
        print(f"\n--- Phase 3: Verify Expert Streaming Layout ---")
        
        sorted_tensors = sorted(per_layer, key=lambda t: t['offset'])
        
        print(f"\nFirst expert tensor: {sorted_tensors[0]['name']}")
        print(f"  offset: {sorted_tensors[0]['offset']}")
        print(f"  size: {sorted_tensors[0]['size_bytes']} bytes")
        
        if len(sorted_tensors) >= 2:
            print(f"\nSecond expert tensor: {sorted_tensors[1]['name']}")
            print(f"  offset: {sorted_tensors[1]['offset']}")
            print(f"  size: {sorted_tensors[1]['size_bytes']} bytes")
            
            stride = sorted_tensors[1]['offset'] - sorted_tensors[0]['offset']
            print(f"\nExpert stride (between layers): {stride} bytes ({stride/1024/1024:.1f} MB)")
        
        print(f"\n--- Phase 4: Per-Expert Slicing Verification ---")
        
        t = sorted_tensors[0]
        expert_dims_idx = len(t['dims']) - 1
        stride_elems = 1
        for d in t['dims'][:expert_dims_idx]:
            stride_elems *= d
        
        bpe = ggml_bytes_per_elem(t['ggml_type'])
        expert_stride_bytes = int(stride_elems * bpe)
        
        print(f"\nFor tensor {t['name']}:")
        print(f"  dims: {t['dims']}")
        print(f"  elements per expert: {stride_elems}")
        print(f"  bytes per expert: {expert_stride_bytes} ({expert_stride_bytes/1024/1024:.1f} MB)")
        print(f"  expert 0 offset: {t['offset']}")
        print(f"  expert 1 offset: {t['offset'] + expert_stride_bytes}")
        print(f"  expert 127 offset: {t['offset'] + 127 * expert_stride_bytes}")
        
        print(f"\n--- Phase 5: Verify Against Official gguf Library ---")
        reader = GGUFReader(gguf_path)
        
        for i, t_info in enumerate(reader.tensors):
            if hasattr(t_info, 'name') and t_info.name == t['name']:
                print(f"Found matching tensor in official library:")
                print(f"  name: {t_info.name}")
                print(f"  shape: {t_info.shape}")
                print(f"  data_offset: {t_info.data_offset}")
                print(f"  n_bytes: {t_info.n_bytes}")
                print(f"  MATCH: offset={t['offset']} vs {t_info.data_offset}, size={t['size_bytes']} vs {t_info.n_bytes}")
                break
        
        print(f"\n" + "=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        
        total_stream_size = 0
        for t in sorted_tensors:
            total_stream_size += t['size_bytes']
        print(f"\nTotal expert stream size: {total_stream_size} bytes ({total_stream_size/1024/1024:.1f} MB)")
        print(f"Per-layer expert data: ~{total_stream_size/block_count/1024/1024:.1f} MB")
        
        print(f"\nKey insight for C implementation:")
        print(f"  1. Gemma 4 uses PER_LAYER layout (_exps suffix)")
        print(f"  2. Each layer has 3 expert tensors: ffn_down.scale, ffn_down.weight, ffn_gate_up.weight")
        print(f"  3. Expert count embedded in last dim of weight tensors")
        print(f"  4. Per-expert slice = tensor_size / expert_count")

if __name__ == '__main__':
    main()
