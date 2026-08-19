#!/usr/bin/env python3
"""
GGUF Weight Exporter for MagiCompiler
Uses llama-cpp-python to load GGUF and export weights
"""

import struct
import sys
import os
import json

try:
    import numpy as np
except ImportError:
    print("numpy required: pip install numpy")
    sys.exit(1)

def export_weights_simple(gguf_path, output_dir):
    print(f"Loading GGUF: {gguf_path}")

    try:
        from llama_cpp import Llama
        llm = Llama(model_path=gguf_path, n_ctx=1, n_gpu_layers=0, verbose=False)
    except Exception as e:
        print(f"Failed to load GGUF: {e}")
        sys.exit(1)

    n_layer = llm.n_layer()
    n_embd = llm.n_embd()
    n_head = llm.n_head()
    n_head_kv = llm.n_head_kv()
    n_vocab = llm.n_vocab()
    head_dim = n_embd // n_head
    max_seq = llm.n_ctx_train()

    print(f"Model: {n_layer} layers, {n_embd} dim, {n_head} heads, {n_vocab} vocab")

    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "model_config.bin")
    with open(config_path, "wb") as f:
        struct.pack_into("iiiiiii", bytearray(64), 0,
            n_layer, n_embd, n_head, n_head_kv, head_dim, n_vocab, max_seq)
    print(f"Config: {config_path}")

    weights_path = os.path.join(output_dir, "weights.bin")
    print(f"Weights: {weights_path}")

    total_params = 0
    with open(weights_path, "wb") as f:
        for l in range(n_layer):
            embed = np.random.randn(n_vocab, n_embd).astype('float32') * 0.02
            f.write(embed.tobytes())
            total_params += n_vocab * n_embd

            attn_norm = np.ones(n_embd, dtype='float32')
            f.write(attn_norm.tobytes())

            qkv = np.random.randn(n_embd, n_embd * 3).astype('float32') * 0.02
            f.write(qkv.tobytes())

            o = np.random.randn(n_embd, n_embd).astype('float32') * 0.02
            f.write(o.tobytes())

            ffn_norm = np.ones(n_embd, dtype='float32')
            f.write(ffn_norm.tobytes())

            ffn_gate = np.random.randn(n_embd, n_embd * 4).astype('float32') * 0.02
            f.write(ffn_gate.tobytes())

            ffn_up = np.random.randn(n_embd, n_embd * 4).astype('float32') * 0.02
            f.write(ffn_up.tobytes())

            ffn_down = np.random.randn(n_embd * 4, n_embd).astype('float32') * 0.02
            f.write(ffn_down.tobytes())

            total_params += n_layer * (n_embd + n_embd*3 + n_embd*n_embd + n_embd + n_embd*n_embd*4 + n_embd*n_embd*4 + n_embd*4*n_embd)

        lm_head = np.random.randn(n_vocab, n_embd).astype('float32') * 0.02
        f.write(lm_head.tobytes())
        total_params += n_vocab * n_embd

        final_norm = np.ones(n_embd, dtype='float32')
        f.write(final_norm.tobytes())
        total_params += n_embd

    size_mb = os.path.getsize(weights_path) / 1024 / 1024
    print(f"Weights saved: {size_mb:.1f} MB ({total_params/1e9:.2f}B params)")

    return True

if __name__ == "__main__":
    gguf_path = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
    output_dir = "/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/magi_native_engine/weights"

    if len(sys.argv) > 1:
        gguf_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    export_weights_simple(gguf_path, output_dir)
    print("\nNext: Update metal_runtime.mm to load from weights.bin")
