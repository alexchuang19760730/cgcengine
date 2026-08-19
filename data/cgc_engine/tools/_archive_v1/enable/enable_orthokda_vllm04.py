#!/usr/bin/env python3
"""
vLLM 0.4.3 + OrthoKDA v4 - Patch Attention layer

問題：vLLM 0.4.3 的 Attention.forward 調用 self.impl.forward()
我们需要替換這個實現來使用 OrthoKDA
"""

import sys
import os

N_BASE = 128
HEAD_DIM = 128
ORTHO_KV_CACHE_PTR = None
LIBORTHO_HANDLE = None
CUDA_KERNEL_CALLED = False

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
    print("[OrthoKDA] CUDA kernel not found")
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

def patch_vllm_attention():
    """Patch vLLM Attention layer"""
    import torch
    import ctypes

    global ORTHO_KV_CACHE_PTR, LIBORTHO_HANDLE, CUDA_KERNEL_CALLED

    print("[OrthoKDA] Initializing OrthoKDA...")
    kv_cache = init_ortho_kda()
    load_ortho_kda_library()

    num_heads = 32
    head_dim = 128

    try:
        from vllm.attention.layer import Attention

        original_forward = Attention.forward

        def ortho_forward(self, query, key, value, kv_cache, attn_metadata):
            """OrthoKDA v4 forward - 替換原生 attention"""
            global CUDA_KERNEL_CALLED
            CUDA_KERNEL_CALLED = True

            print(f"[OrthoKDA] Attention.forward called - Query: {query.shape}")

            if ORTHO_KV_CACHE_PTR is not None and LIBORTHO_HANDLE is not None:
                try:
                    out = torch.zeros_like(query)

                    q_ptr = query.flatten().data_ptr()
                    k_ptr = key.flatten().data_ptr() if key is not None else 0
                    v_ptr = value.flatten().data_ptr() if value is not None else 0
                    o_ptr = out.flatten().data_ptr()

                    if key is not None and value is not None:
                        LIBORTHO_HANDLE.call_ortho_kda_update(
                            ORTHO_KV_CACHE_PTR, k_ptr, v_ptr
                        )
                        print("[OrthoKDA] KV update done")

                    LIBORTHO_HANDLE.call_ortho_kda_forward(
                        ORTHO_KV_CACHE_PTR, q_ptr, o_ptr, num_heads
                    )
                    print("[OrthoKDA] Attention forward done")
                    return out
                except Exception as e:
                    print(f"[OrthoKDA] CUDA kernel call failed: {e}")

            print("[OrthoKDA] Falling back to original forward")
            return original_forward(self, query, key, value, kv_cache, attn_metadata)

        Attention.forward = ortho_forward
        print("[OrthoKDA] Successfully patched Attention.forward")
        return True

    except Exception as e:
        print(f"[OrthoKDA] Failed to patch Attention: {e}")
        import traceback
        traceback.print_exc()
        return False

def enable_orthokda():
    """啟用 OrthoKDA - 在 import vLLM 之後調用"""
    patch_vllm_attention()

if __name__ == "__main__":
    print("=" * 60)
    print("vLLM 0.4.3 + OrthoKDA v4 (Attention Layer Patch)")
    print("=" * 60)
    enable_orthokda()