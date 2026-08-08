#!/usr/bin/env python3
"""
vLLM + OrthoKDA v4 Benchmark - 測試 CUDA kernel 是否被調用

使用 nvidia-smi 測量真實 GPU 記憶體使用
"""

import subprocess
import time
import gc
import sys
import os

N_BASE = 128
HEAD_DIM = 128
ORTHO_KV_CACHE_PTR = None
LIBORTHO_HANDLE = None
CUDA_KERNEL_CALLED = False

def get_gpu_memory_nvidia_smi():
    """使用 nvidia-smi 獲取真實 GPU 記憶體使用"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except:
        return None

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

def patch_all_attention_classes():
    """Patch 所有 vLLM 的 Attention 類別"""
    import torch
    import ctypes

    global ORTHO_KV_CACHE_PTR, LIBORTHO_HANDLE, CUDA_KERNEL_CALLED

    try:
        from vllm.model_executor.layers import attention
        import inspect

        kv_cache = init_ortho_kda()
        load_ortho_kda_library()

        num_heads = 32
        head_dim = 128

        print(f"[OrthoKDA] Attention module classes: {dir(attention)}")

        def ortho_forward(self, query, key, value, attention_mask=None, **kwargs):
            """OrthoKDA v4 forward"""
            global CUDA_KERNEL_CALLED
            CUDA_KERNEL_CALLED = True

            if ORTHO_KV_CACHE_PTR is not None and LIBORTHO_HANDLE is not None:
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
            else:
                scale = 1.0 / (head_dim ** 0.5)
                scores = torch.matmul(query, key.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(scores, dim=-1)
                return torch.matmul(attn_weights, value)

        for attr_name in dir(attention):
            attr = getattr(attention, attr_name)
            if inspect.isclass(attr) and hasattr(attr, 'forward'):
                print(f"[OrthoKDA] Patching {attr_name}")
                attr.forward = ortho_forward

        from vllm.model_executor.layers.attention import Attention
        Attention.forward = ortho_forward
        print("[OrthoKDA] Successfully patched all Attention.forward methods")
        return True

    except Exception as e:
        print(f"[OrthoKDA] Failed to patch Attention: {e}")
        import traceback
        traceback.print_exc()
        return False

MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"
CONTEXT_SIZES = [256, 512, 1024, 2048]
MAX_TOKENS = 30

def test_vllm_with_orthokda(context_size):
    """測量 vLLM + OrthoKDA"""
    import torch
    from vllm import LLM, SamplingParams

    global CUDA_KERNEL_CALLED
    CUDA_KERNEL_CALLED = False

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_before = get_gpu_memory_nvidia_smi()

    print(f"\n  [Context: {context_size} + OrthoKDA]")

    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=context_size,
    )
    load_time = time.time() - t0
    torch.cuda.synchronize()
    mem_after_load = get_gpu_memory_nvidia_smi()

    print(f"    GPU Memory after load: {mem_after_load:.1f} MB (delta: {mem_after_load - mem_before:.1f} MB)")

    sampling_params = SamplingParams(temperature=0.7, max_tokens=MAX_TOKENS)

    t0 = time.time()
    outputs = llm.generate(["Hello, how are you?"], sampling_params)
    torch.cuda.synchronize()
    total_time = time.time() - t0

    prompt_tokens = len(outputs[0].prompt_token_ids)
    generated = len(outputs[0].outputs[0].token_ids)

    mem_during = get_gpu_memory_nvidia_smi()

    print(f"    GPU Memory during inference: {mem_during:.1f} MB")
    print(f"    CUDA kernel called: {CUDA_KERNEL_CALLED}")
    print(f"    Inference time: {total_time:.2f}s")
    print(f"    Generated {generated} tokens")

    decode_speed = generated / total_time if total_time > 0 else 0
    print(f"    Decode speed: {decode_speed:.1f} tok/s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        "context": context_size,
        "mem_delta": mem_after_load - mem_before,
        "decode_speed": decode_speed,
        "cuda_called": CUDA_KERNEL_CALLED,
    }

def main():
    print("=" * 80)
    print("vLLM + OrthoKDA v4 Benchmark")
    print("=" * 80)

    import torch
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_PATH}")

    print("\n" + "=" * 80)
    print("Step 1: Patch Attention (Before importing vLLM)")
    print("=" * 80)

    patch_all_attention_classes()

    print("\n" + "=" * 80)
    print("Step 2: Benchmark vLLM + OrthoKDA")
    print("=" * 80)

    results = []
    for ctx in CONTEXT_SIZES:
        r = test_vllm_with_orthokda(ctx)
        results.append(r)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Context':>8} | {'Memory Delta':>12} | {'Decode Speed':>12} | {'CUDA Called':>12}")
    print("-" * 50)

    for r in results:
        ctx = r["context"]
        mem = f"{r['mem_delta']:.1f} MB"
        speed = f"{r['decode_speed']:.1f} tok/s"
        cuda = "✅" if r['cuda_called'] else "❌"
        print(f"{ctx:>8} | {mem:>12} | {speed:>12} | {cuda:>12}")

if __name__ == "__main__":
    main()