import sys
sys.path.insert(0, r"D:\alex\flashkv0516\CGC-main\cgc_engine\cpp\expert_streaming")

from gguf import GGUFReader

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

print(f"Loading: {gguf_path}")
reader = GGUFReader(gguf_path)

print(f"\nArchitecture: {reader.fields['general.architecture'].contents()}")

print("\n--- Key KV Metadata ---")
keys_of_interest = [
    'general.architecture', 'general.type', 'general.name',
    'general.base_model.name', 'general.size_label',
    'gemma4.block_count', 'gemma4.attention.head_count',
    'gemma4.attention.head_count_kv', 'gemma4.context_length',
    'gemma4.embedding_length', 'gemma4.feed_forward_length',
    'gemma4.expert_count', 'gemma4.expert_stride',
    'gemma4.attention.head_count', 'gemma4.attention.layer_norm_epsilon',
    'general.alignment', 'general.file_type',
]

for key in keys_of_interest:
    if key in reader.fields:
        val = reader.fields[key].contents()
        print(f"  {key}: {val}")

print("\n--- All KV keys ---")
for key in sorted(reader.fields.keys()):
    field = reader.fields[key]
    print(f"  {key}: type={field.types}")

print("\n--- Expert Tensor Sample ---")
expert_tensors = []
for name in reader.names:
    if 'expert' in name.lower():
        info = reader.tensor_info[name]
        expert_tensors.append((name, info))
        if len(expert_tensors) <= 10:
            print(f"  {name}: dims={info.shape}, type={info.ggml_type}, offset={info.offset}")

print(f"\nTotal expert tensors: {len(expert_tensors)}")

if len(expert_tensors) >= 2:
    offsets = [t[1].offset for t in expert_tensors[:10]]
    print(f"First 10 offsets: {offsets}")
    strides = [offsets[i+1] - offsets[i] for i in range(len(offsets)-1)]
    print(f"Strides: {strides[:5]}")
    
    first_tensor = expert_tensors[0][1]
    print(f"\nFirst expert tensor details:")
    print(f"  name: {expert_tensors[0][0]}")
    print(f"  shape: {first_tensor.shape}")
    print(f"  ggml_type: {first_tensor.ggml_type}")
    print(f"  offset: {first_tensor.offset}")

print("\n--- Gemma4 Expert Info ---")
print(f"  expert_count: {reader.fields['gemma4.expert_count'].contents()}")
print(f"  expert_feed_forward_length: {reader.fields['gemma4.expert_feed_forward_length'].contents()}")
print(f"  expert_used_count: {reader.fields['gemma4.expert_used_count'].contents()}")
print(f"  embedding_length: {reader.fields['gemma4.embedding_length'].contents()}")
print(f"  feed_forward_length: {reader.fields['gemma4.feed_forward_length'].contents()}")
print(f"  block_count: {reader.fields['gemma4.block_count'].contents()}")
