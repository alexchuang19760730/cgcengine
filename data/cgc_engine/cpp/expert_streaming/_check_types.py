import sys
sys.path.insert(0, r"D:\alex\flashkv0516\CGC-main\cgc_engine\cpp\expert_streaming")
from gguf import GGUFReader

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

reader = GGUFReader(gguf_path)

print("=== GGML Type Info ===")
from gguf import GGMLQuantizationType
for name in dir(GGMLQuantizationType):
    if not name.startswith('_'):
        val = getattr(GGMLQuantizationType, name)
        print(f"  {name}: {val}")

print("\n=== Checking bool handling ===")
import gguf
print(f"gguf module: {gguf.__file__}")

print("\n=== Checking IQ4_XS type ===")
try:
    t = GGMLQuantizationType.IQ4_XS
    print(f"IQ4_XS value: {t}")
except:
    try:
        t = GGMLQuantizationType(7)
        print(f"Type 7: {t}")
    except Exception as e:
        print(f"Error: {e}")

print("\n=== Gemma4 architecture details ===")
print(f"gemma4.expert_count: {reader.fields['gemma4.expert_count'].contents()}")
print(f"gemma4.expert_feed_forward_length: {reader.fields['gemma4.expert_feed_forward_length'].contents()}")
print(f"gemma4.expert_used_count: {reader.fields['gemma4.expert_used_count'].contents()}")
print(f"gemma4.embedding_length: {reader.fields['gemma4.embedding_length'].contents()}")
print(f"gemma4.feed_forward_length: {reader.fields['gemma4.feed_forward_length'].contents()}")
print(f"gemma4.block_count: {reader.fields['gemma4.block_count'].contents()}")

print("\n=== IQ4_XS block size ===")
import gguf
if hasattr(gguf, 'GGML_QUANT_SIZES'):
    print(f"GGML_QUANT_SIZES: {gguf.GGML_QUANT_SIZES}")
elif hasattr(gguf, 'GGML_BLOCK_SIZES'):
    print(f"GGML_BLOCK_SIZES: {gguf.GGML_BLOCK_SIZES}")

print("\n=== Find IQ4_XS info ===")
from gguf import GGMLQuantizationType as T
print(f"Type 7 exists: {hasattr(T, 'IQ4_XS')}")

try:
    for attr in ['IQ4_XS', 'IQ3_M', 'IQ2_XXS', 'IQ2_XS']:
        if hasattr(T, attr):
            val = getattr(T, attr)
            print(f"  {attr}: {val}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Full GGML type map ===")
for i in range(50):
    try:
        name = GGMLQuantizationType(i)
        print(f"  {i}: {name}")
    except:
        pass
