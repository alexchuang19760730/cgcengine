#!/usr/bin/env python3
"""Complete verification of qwen35moe GGUF parsing and expert streaming."""

import sys
sys.path.insert(0, "D:/alex/flashkv0516/app/edge_engine")
from llama_monkey_patch import parse_gguf_header, ExpertStreamerLite

model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

header = parse_gguf_header(model_path)
print(f"=== GGUF Header Parsing ===")
print(f"Version: {header['version']}")
print(f"KV items: {len(header['kv'])}")
print(f"Tensor items: {len(header['tensors'])}")

if header['tensors'] == 0:
    print("\nTensor count is 0! Let's fix the parser...")
    import struct
    with open(model_path, "rb") as f:
        f.seek(24)
        
        for _ in range(header['n_kv']):
            klen_raw = f.read(8)
            if len(klen_raw) < 8:
                break
            klen = struct.unpack("<Q", klen_raw)[0]
            
            if klen > 65536:
                try: f.seek(klen)
                except: _ = f.read(klen)
                dtype = struct.unpack("<I", f.read(4))[0]
                if dtype in (4, 5, 6, 10):
                    f.seek(4, 1)
                elif dtype in (7, 11, 12):
                    f.seek(8, 1)
                elif dtype == 8:
                    slen = struct.unpack("<Q", f.read(8))[0]
                    try: f.seek(slen, 1)
                    except: pass
                elif dtype == 9:
                    elem_type = struct.unpack("<I", f.read(4))[0]
                    array_len = struct.unpack("<Q", f.read(8))[0]
                    if elem_type == 8:
                        for _ in range(array_len):
                            sl = struct.unpack("<Q", f.read(8))[0]
                            try: f.seek(sl, 1)
                            except: pass
                    else:
                        sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                        total = sizes.get(elem_type, 0) * array_len
                        try: f.seek(total, 1)
                        except: pass
                continue
            
            key = f.read(klen).decode("utf-8", errors="replace")
            dtype = struct.unpack("<I", f.read(4))[0]
            
            if dtype in (4, 5):
                f.read(4)
            elif dtype == 6:
                f.read(4)
            elif dtype == 7:
                f.read(8)
            elif dtype == 8:
                slen = struct.unpack("<Q", f.read(8))[0]
                if slen <= 1048576:
                    f.read(slen)
                else:
                    try: f.seek(slen, 1)
                    except: pass
            elif dtype == 9:
                elem_type = struct.unpack("<I", f.read(4))[0]
                array_len = struct.unpack("<Q", f.read(8))[0]
                if elem_type == 8:
                    for _ in range(array_len):
                        sl = struct.unpack("<Q", f.read(8))[0]
                        try: f.seek(sl, 1)
                        except: pass
                else:
                    sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                    total = sizes.get(elem_type, 0) * array_len
                    try: f.seek(total, 1)
                    except: pass
            elif dtype == 10:
                f.read(4)
            elif dtype == 11:
                f.read(8)
            elif dtype == 12:
                f.read(8)
        
        pos_after_kv = f.tell()
        print(f"Pos after all KVs: {pos_after_kv}")
        
        # Now parse tensors
        tensor_count = 0
        expert_tensor_count = 0
        first_tensor = None
        
        for idx in range(header['n_tensors']):
            nlen_raw = f.read(8)
            if len(nlen_raw) < 8:
                print(f"Stopped at tensor {idx} (EOF)")
                break
            nlen = struct.unpack("<Q", nlen_raw)[0]
            if nlen > 65536:
                print(f"Stopped at tensor {idx}: nlen={nlen} too large")
                break
            
            name = f.read(nlen).decode("utf-8", errors="replace")
            
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            
            if first_tensor is None:
                first_tensor = (name, dims, ggml_type, offset)
            
            tensor_count += 1
            if 'expert' in name.lower():
                expert_tensor_count += 1
        
        print(f"\nManual parse result:")
        print(f"  Total tensors: {tensor_count}")
        print(f"  Expert tensors: {expert_tensor_count}")
        if first_tensor:
            print(f"  First tensor: {first_tensor[0]} dims={first_tensor[1]} type={first_tensor[2]} offset={first_tensor[3]}")

# Now test ExpertStreamerLite
print(f"\n=== ExpertStreamerLite Test ===")
streamer = ExpertStreamerLite(model_path)
stats = streamer.cache_stats()
print(f"Architecture: {stats['architecture']}")
print(f"Hidden: {stats['hidden']}")
print(f"Inter: {stats['inter']}")
print(f"Num experts: {stats['num_experts']}")
print(f"Has experts: {stats['has_experts']}")
print(f"Num layers: {stats['num_layers']}")

if stats['has_experts']:
    eids = streamer.list_experts()
    print(f"Expert IDs: {eids[:10]}{'...' if len(eids) > 10 else ''}")
    
    # Test expert loading
    if eids:
        print(f"\nLoading expert {eids[0]}...")
        result = streamer.load_expert(eids[0])
        if result:
            print(f"  Success! Roles: {list(result.get('roles', {}).keys())}")
        else:
            print(f"  Failed (likely because tensor parsing returned 0 tensors)")

print("\n[DONE]")