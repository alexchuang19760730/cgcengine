import sys
sys.path.insert(0, r"D:\alex\flashkv0516\CGC-main\cgc_engine\cpp\expert_streaming")
from gguf import GGUFReader

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

reader = GGUFReader(gguf_path)

print("Inspecting tensor structure...")
if reader.tensors:
    first = reader.tensors[0]
    print(f"First tensor type: {type(first)}")
    print(f"First tensor dir: {[x for x in dir(first) if not x.startswith('_')]}")
    if hasattr(first, 'name'):
        print(f"First tensor name: {first.name}")
    if hasattr(first, 'info'):
        print(f"First tensor info: {first.info}")
    if hasattr(first, 'parts'):
        print(f"First tensor parts: {first.parts}")

print("\n--- All tensor names ---")
tensor_names = []
for i, t in enumerate(reader.tensors):
    if isinstance(t, str):
        name = t
    elif hasattr(t, 'name'):
        name = t.name
    elif hasattr(t, 'info') and hasattr(t.info, 'name'):
        name = t.info.name
    elif hasattr(t, 'parts'):
        name = t.parts[0] if t.parts else str(t)
    else:
        name = str(t)[:100]
    
    tensor_names.append(name)
    
    if i < 20 or 'expert' in name.lower():
        print(f"  [{i}] {name}")

print(f"\nTotal tensors: {len(tensor_names)}")

expert_names = [n for n in tensor_names if 'expert' in n.lower()]
print(f"Expert tensors: {len(expert_names)}")
for n in expert_names[:10]:
    print(f"  {n}")

blktensors = [n for n in tensor_names if 'blk.' in n and 'expert' not in n]
print(f"\nNon-expert blk tensors: {len(blktensors)}")
for n in blktensors[:10]:
    print(f"  {n}")
