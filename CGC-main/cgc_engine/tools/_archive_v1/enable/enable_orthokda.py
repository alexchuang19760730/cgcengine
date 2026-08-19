#!/usr/bin/env python3
"""
vLLM + OrthoKDA v4 整合 (vLLM 0.20.1)

關鍵：在 import vllm 之前就 patch Attention.forward
"""

import sys
import os

N_BASE = 128
HEAD_DIM = 128
ORTHO_KV_CACHE_PTR = None
LIBORTHO_HANDLE = None

def load_ortho_kda_library():
    """加載 CUDA kernel"""
    import ctypes

    global LIBORTHO_HANDLE

    possible_paths = [
        "/home/gs01/MagiCompiler-main/cgc_engine/cgc/cgc_cpp/build/libortho_kda.so",
        "./cgc_engine/cgc/cgc_cpp/build/libortho_kda.so",
        "./libortho_kda.so",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                LIBORTHO_HANDLE = ctypes.CDLL(path)
                LIBORTHO_HANDLE.call_ortho_kda_forward.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
                ]
                LIBORTHO_HANDLE.call_ortho_kda_forward.restype = None
                LIBORTHO_HANDLE.call_ortho_kda_update.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
                ]
                LIBORTHO_HANDLE.call_ortho_kda_update.restype = None
                print(f"[OrthoKDA] Loaded CUDA kernel from {path}")
                return True
            except Exception as e:
                print(f"[OrthoKDA] Failed to load {path}: {e}")

    print("[OrthoKDA] CUDA kernel not found, using PyTorch fallback")
    return False

def init_ortho_kda():
    """初始化 OrthoKDA 全域狀態"""
    import torch

    global ORTHO_KV_CACHE_PTR

    kv_size = N_BASE * HEAD_DIM * 2 + N_BASE + 1
    kv_cache = torch.zeros(kv_size, dtype=torch.float32, device="cuda")
    ORTHO_KV_CACHE_PTR = kv_cache.data_ptr()

    print(f"[OrthoKDA] Created O(1) KV cache: {N_BASE}x{HEAD_DIM} = {N_BASE*HEAD_DIM*2/1024:.1f} KB")
    return kv_cache

def patch_attention():
    """Patch vLLM Attention"""
    import torch
    import ctypes

    global ORTHO_KV_CACHE_PTR, LIBORTHO_HANDLE

    try:
        from vllm.model_executor.layers.attention import Attention

        kv_cache = init_ortho_kda()
        load_ortho_kda_library()

        num_heads = 32
        head_dim = 128

        def ortho_forward(self, query, key, value, attention_mask=None, **kwargs):
            """OrthoKDA v4 forward - 替換原生 attention"""
            if ORTHO_KV_CACHE_PTR is not None and LIBORTHO_HANDLE is not None:
                return _cuda_ortho_forward(self, query, key, value)
            else:
                return _pytorch_ortho_forward(self, query, key, value, attention_mask)

        def _cuda_ortho_forward(self, query, key, value):
            """CUDA kernel implementation"""
            out = torch.zeros_like(query)

            q_ptr = query.view(-1, num_heads, head_dim).data_ptr()
            k_ptr = key.view(-1, num_heads, head_dim).data_ptr()
            v_ptr = value.view(-1, num_heads, head_dim).data_ptr()
            o_ptr = out.view(-1, num_heads, head_dim).data_ptr()

            LIBORTHO_HANDLE.call_ortho_kda_update(
                ORTHO_KV_CACHE_PTR, k_ptr, v_ptr
            )
            LIBORTHO_HANDLE.call_ortho_kda_forward(
                ORTHO_KV_CACHE_PTR, q_ptr, o_ptr, num_heads
            )

            return out

        def _pytorch_ortho_forward(self, query, key, value, attention_mask):
            """PyTorch fallback"""
            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = torch.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, value)

        Attention.forward = ortho_forward
        print("[OrthoKDA] Successfully patched Attention.forward")
        return True

    except Exception as e:
        print(f"[OrthoKDA] Failed to patch Attention: {e}")
        import traceback
        traceback.print_exc()
        return False

def enable_orthokda():
    """啟用 OrthoKDA - 在 import vLLM 之前調用"""
    patch_attention()

if __name__ == "__main__":
    print("=" * 60)
    print("vLLM + OrthoKDA v4 Integration (vLLM 0.20.1)")
    print("=" * 60)

    enable_orthokda()

    print("\n現在可以 import vLLM:")
    print("  from vllm import LLM, SamplingParams")
    print("  llm = LLM(model='...')")