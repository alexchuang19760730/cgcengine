import sys
sys.path.insert(0, r"D:\alex\flashkv0516\CGC-main\cgc_engine\cpp\expert_streaming")
from cgc_expert_streamer_ctypes import CGCExpertStreamer

cgc = CGCExpertStreamer(auto_build=False)
gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"
print("Loading GGUF layout...")
layout = cgc.load_layout_from_gguf(gguf_path)
print(f"offset={layout.stream_offset}")
print(f"size={layout.stream_size}")
print(f"experts={layout.experts_per_layer}")
print(f"stride={layout.expert_stride}")
print(f"explicit_offsets={layout.has_explicit_offsets}")
if layout.has_explicit_offsets:
    for i in range(min(layout.experts_per_layer, 8)):
        print(f"  expert[{i}] offset={layout.expert_offsets[i]}")
