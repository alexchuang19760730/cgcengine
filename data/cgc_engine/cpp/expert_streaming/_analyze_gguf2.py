import sys
sys.path.insert(0, r"D:\alex\flashkv0516\CGC-main\cgc_engine\cpp\expert_streaming")
from gguf import GGUFReader

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

reader = GGUFReader(gguf_path)

print("=== Gemma 4 Expert Tensors Analysis ===\n")

expert_info = []
for i, t in enumerate(reader.tensors):
    if hasattr(t, 'name') and '_exps' in t.name.lower():
        expert_info.append((i, t.name, t.shape, t.tensor_type, t.data_offset, t.n_bytes))

print(f"Total expert tensors (_exps): {len(expert_info)}\n")

for idx, name, shape, ttype, offset, nbytes in expert_info[:20]:
    print(f"  [{idx}] {name}:")
    print(f"    shape={shape}, type={ttype}, offset={offset}, size={nbytes}")

print("\n=== Layer 0 Expert Tensors ===")
layer0 = [(i, n, s, t, o, b) for i, n, s, t, o, b in expert_info if n.startswith('blk.0.')]
for idx, name, shape, ttype, offset, nbytes in layer0:
    print(f"  {name}: shape={shape}, type={ttype}")
    print(f"    offset={offset}, size={nbytes} bytes")

print("\n=== Layer 1 Expert Tensors ===")
layer1 = [(i, n, s, t, o, b) for i, n, s, t, o, b in expert_info if n.startswith('blk.1.')]
for idx, name, shape, ttype, offset, nbytes in layer1:
    print(f"  {name}: shape={shape}, type={ttype}")
    print(f"    offset={offset}, size={nbytes} bytes")

print("\n=== Offset Analysis ===")
sorted_by_offset = sorted(expert_info, key=lambda x: x[4])
print(f"Sorted by offset (first 20):")
for idx, name, shape, ttype, offset, nbytes in sorted_by_offset[:20]:
    print(f"  {offset}: {name} ({nbytes} bytes)")

if len(sorted_by_offset) >= 2:
    print(f"\nStrides between consecutive tensors:")
    for i in range(min(10, len(sorted_by_offset)-1)):
        diff = sorted_by_offset[i+1][4] - sorted_by_offset[i][4]
        print(f"  {sorted_by_offset[i][1]} -> {sorted_by_offset[i+1][1]}: {diff} bytes")
