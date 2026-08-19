import struct
import sys

def inspect_gguf(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]

        print(f"GGUF v{version}, {tensor_count} tensors, {kv_count} KV entries")

        for i in range(kv_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8', errors='replace')
            dtype = struct.unpack('<I', f.read(4))[0]

            if dtype == 0: value = struct.unpack('<B', f.read(1))[0]
            elif dtype == 1: value = struct.unpack('<b', f.read(1))[0]
            elif dtype == 2: value = struct.unpack('<H', f.read(2))[0]
            elif dtype == 3: value = struct.unpack('<h', f.read(2))[0]
            elif dtype == 4: value = struct.unpack('<I', f.read(4))[0]
            elif dtype == 5: value = struct.unpack('<i', f.read(4))[0]
            elif dtype == 6: value = struct.unpack('<f', f.read(4))[0]
            elif dtype == 7:
                val = struct.unpack('<B', f.read(1))[0]
                f.read(3)
                value = val
            elif dtype == 8:
                str_len = struct.unpack('<Q', f.read(8))[0]
                value = f.read(str_len).decode('utf-8', errors='replace')
            elif dtype == 9:
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_count = struct.unpack('<Q', f.read(8))[0]
                if arr_type == 8:
                    strs = []
                    for _ in range(min(arr_count, 5)):
                        sl = struct.unpack('<Q', f.read(8))[0]
                        strs.append(f.read(sl).decode('utf-8', errors='replace'))
                    value = f"string_array[{arr_count}]: {strs[:3]}..."
                elif arr_type == 0:
                    vals = struct.unpack(f'<{arr_count}B', f.read(arr_count))
                    value = f"uint8_array[{arr_count}]: {list(vals[:5])}..."
                elif arr_type == 4:
                    vals = struct.unpack(f'<{arr_count}I', f.read(arr_count*4))
                    value = f"uint32_array[{arr_count}]: {list(vals[:5])}..."
                elif arr_type == 10:
                    vals = struct.unpack(f'<{arr_count}Q', f.read(arr_count*8))
                    value = f"uint64_array[{arr_count}]: {list(vals[:5])}..."
                else:
                    f.seek(4 + 8 * arr_count, 1)
                    value = f"array[{arr_type}x{arr_count}]"
            elif dtype == 10: value = struct.unpack('<Q', f.read(8))[0]
            elif dtype == 11: value = struct.unpack('<q', f.read(8))[0]
            else:
                print(f"  Unknown dtype {dtype} for key {key}, skipping")
                break

            if key == 'image-text-to-text':
                print(f"  >>> Found image-text-to-text (dtype={dtype})")
                break

            print(f"  {key}: {value}")

        print("\n--- Tensor names (expert tensors only) ---")
        expert_tensors = []
        for i in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8', errors='replace')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', f.read(n_dims * 8))
            ggml_type = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]

            if 'expert' in name.lower():
                expert_tensors.append((name, dims, ggml_type, offset))
                if len(expert_tensors) <= 20:
                    print(f"  {name}: dims={dims}, type={ggml_type}, offset={offset}")

        print(f"\nTotal expert tensors: {len(expert_tensors)}")

        if len(expert_tensors) >= 2:
            offs = [t[3] for t in expert_tensors[:10]]
            if all(offs[i+1] > offs[i] for i in range(len(offs)-1)):
                strides = [offs[i+1] - offs[i] for i in range(len(offs)-1)]
                print(f"Expert strides: {strides[:5]}")

if __name__ == "__main__":
    path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"
    inspect_gguf(path)
