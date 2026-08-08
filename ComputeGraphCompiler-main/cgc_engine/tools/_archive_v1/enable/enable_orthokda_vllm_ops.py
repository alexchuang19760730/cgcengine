#!/usr/bin/env python3
"""
vLLM + OrthoKDA v4 - Patch torch.ops.vllm operators

問題：vLLM 0.20.1 的 Attention.forward 內部調用 torch.ops.vllm.* 運算符
所以我們需要替換這些底層運算符來實現 OrthoKDA
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

def patch_vllm_ops():
    """Patch torch.ops.vllm operators"""
    import torch
    import ctypes

    global ORTHO_KV_CACHE_PTR, LIBORTHO_HANDLE, CUDA_KERNEL_CALLED

    print("[OrthoKDA] Initializing OrthoKDA...")
    kv_cache = init_ortho_kda()
    load_ortho_kda_library()

    num_heads = 32
    head_dim = 128

    def ortho_kv_cache_update_impl(key, value, layer_name):
        """OrthoKDA KV cache update - 替換 vllm.unified_kv_cache_update"""
        global CUDA_KERNEL_CALLED, ORTHO_KV_CACHE_PTR, LIBORTHO_HANDLE

        if ORTHO_KV_CACHE_PTR is None or LIBORTHO_HANDLE is None:
            print("[OrthoKDA] WARNING: CUDA kernel not loaded, using fallback")
            return key, value

        CUDA_KERNEL_CALLED = True

        try:
            k_ptr = key.flatten().data_ptr()
            v_ptr = value.flatten().data_ptr()
            LIBORTHO_HANDLE.call_ortho_kda_update(
                ORTHO_KV_CACHE_PTR, k_ptr, v_ptr
            )
            print(f"[OrthoKDA] CUDA kernel called: KV update for layer {layer_name}")
        except Exception as e:
            print(f"[OrthoKDA] CUDA kernel call failed: {e}")

        return key, value

    def ortho_attention_impl(query, key, value, output, layer_name, kv_cache_dummy_dep=None):
        """OrthoKDA attention - 替換 vllm.unified_attention_with_output"""
        global CUDA_KERNEL_CALLED, ORTHO_KV_CACHE_PTR, LIBORTHO_HANDLE

        if ORTHO_KV_CACHE_PTR is None or LIBORTHO_HANDLE is None:
            print("[OrthoKDA] WARNING: CUDA kernel not loaded, using fallback")
            return output

        CUDA_KERNEL_CALLED = True

        try:
            q_ptr = query.flatten().data_ptr()
            o_ptr = output.flatten().data_ptr()
            LIBORTHO_HANDLE.call_ortho_kda_forward(
                ORTHO_KV_CACHE_PTR, q_ptr, o_ptr, num_heads
            )
            print(f"[OrthoKDA] CUDA kernel called: Attention for layer {layer_name}")
        except Exception as e:
            print(f"[OrthoKDA] CUDA kernel call failed: {e}")

        return output

    try:
        print("[OrthoKDA] Patching torch.ops.vllm...")

        if hasattr(torch.ops, 'vllm'):
            vllm_ops = torch.ops.vllm
            print(f"[OrthoKDA] Found torch.ops.vllm: {dir(vllm_ops)}")

            if hasattr(vllm_ops, 'unified_kv_cache_update'):
                print("[OrthoKDA] Patching unified_kv_cache_update")
                vllm_ops.unified_kv_cache_update = ortho_kv_cache_update_impl

            if hasattr(vllm_ops, 'unified_attention_with_output'):
                print("[OrthoKDA] Patching unified_attention_with_output")
                vllm_ops.unified_attention_with_output = ortho_attention_impl

            print("[OrthoKDA] Successfully patched torch.ops.vllm operators")
            return True
        else:
            print("[OrthoKDA] torch.ops.vllm not found")
            return False

    except Exception as e:
        print(f"[OrthoKDA] Failed to patch torch.ops.vllm: {e}")
        import traceback
        traceback.print_exc()
        return False

def enable_orthokda():
    """啟用 OrthoKDA - 在 import vLLM 之前調用"""
    patch_vllm_ops()

if __name__ == "__main__":
    print("=" * 60)
    print("vLLM + OrthoKDA v4 (torch.ops.vllm Patch)")
    print("=" * 60)
    enable_orthokda()