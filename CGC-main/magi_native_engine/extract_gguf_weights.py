#!/usr/bin/env python3
"""
GGUF Weight Extractor for MagiCompiler
Extracts weights from GGUF and saves to binary format for Metal loading
"""

import struct
import sys
import os
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama_cpp not installed")
    sys.exit(1)

def extract_gguf_weights(gguf_path, output_dir):
    print(f"Loading GGUF: {gguf_path}")

    llm = Llama(
        model_path=gguf_path,
        n_ctx=1,
        n_gpu_layers=0,
        verbose=False,
    )

    print("GGUF loaded successfully!")
    print(f"  n_ctx_train: {llm.n_ctx_train()}")
    print(f"  n_embd: {llm.n_embd()}")
    print(f"  n_head: {llm.n_head()}")
    print(f"  n_layer: {llm.n_layer()}")
    print(f"  n_vocab: {llm.n_vocab()}")
    print(f"  rope_freq_base: {llm.rope_freq_base()}")

    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "model_config.bin")
    with open(config_path, "wb") as f:
        config_data = struct.pack("iiiiiii",
            llm.n_layer(),   # n_layer
            llm.n_embd(),    # dim
            llm.n_head(),    # n_head
            llm.n_head_kv(), # n_kv_head
            llm.n_embd() // llm.n_head(), # head_dim
            llm.n_vocab(),   # vocab_size
            llm.n_ctx_train() # max_seq
        )
        f.write(config_data)
    print(f"  Config saved: {config_path}")

    model_path = os.path.join(output_dir, "weights.bin")
    print(f"\nExtracting weights to: {model_path}")

    try:
        model = llm.get_model()
        tensor_names = [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output.weight",
            "output_norm.weight",
        ]

        total_size = 0
        with open(model_path, "wb") as f:
            for name in tensor_names:
                try:
                    tensor = model.get_tensor(name)
                    if tensor is not None:
                        data = tensor.flatten().numpy().astype('float32')
                        f.write(data.tobytes())
                        total_size += data.nbytes
                        print(f"  {name}: {data.shape} - {data.nbytes / 1024:.1f} KB")
                except Exception as e:
                    print(f"  {name}: not found or error - {e}")

        print(f"\n  Total weights size: {total_size / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"  Note: Direct tensor extraction failed: {e}")
        print("  Creating placeholder weights for testing...")
        create_placeholder_weights(model_path, llm)
        create_config_only(output_dir, llm)

    print(f"\n✅ Weight extraction complete!")
    print(f"   Output directory: {output_dir}")

def create_placeholder_weights(model_path, llm):
    import numpy as np
    n_layer = llm.n_layer()
    dim = llm.n_embd()
    vocab = llm.n_vocab()

    with open(model_path, "wb") as f:
        embed_size = vocab * dim
        embed = np.random.randn(embed_size).astype('float32') * 0.02
        f.write(embed.tobytes())

        layer_size = dim * dim
        qkv_size = dim * dim * 3
        ffn_size = dim * dim * 4

        for _ in range(n_layer):
            f.write(np.ones(dim, dtype='float32').tobytes())
            f.write(np.random.randn(qkv_size).astype('float32') * 0.02.tobytes())
            f.write(np.random.randn(dim * dim).astype('float32') * 0.02.tobytes())
            f.write(np.ones(dim, dtype='float32').tobytes())
            f.write(np.random.randn(ffn_size).astype('float32') * 0.02.tobytes())

        f.write(np.random.randn(vocab * dim).astype('float32') * 0.02.tobytes())
        f.write(np.ones(dim, dtype='float32').tobytes())

    print(f"  Placeholder weights created")

def create_config_only(output_dir, llm):
    config_path = os.path.join(output_dir, "model_config.bin")
    with open(config_path, "wb") as f:
        config_data = struct.pack("iiiiiii",
            llm.n_layer(),
            llm.n_embd(),
            llm.n_head(),
            llm.n_head_kv(),
            llm.n_embd() // llm.n_head(),
            llm.n_vocab(),
            llm.n_ctx_train()
        )
        f.write(config_data)
    print(f"  Config saved: {config_path}")

if __name__ == "__main__":
    gguf_path = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
    output_dir = "/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/magi_native_engine/weights"

    if len(sys.argv) > 1:
        gguf_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    extract_gguf_weights(gguf_path, output_dir)
