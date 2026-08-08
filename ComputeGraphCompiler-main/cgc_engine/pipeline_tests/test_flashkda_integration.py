#!/usr/bin/env python3
"""
FlashKDA + CGC SIMD Engine 集成測試
"""

import sys
import os
sys.path.insert(0, '/home/gs01')

print("=" * 80)
print("FlashKDA + CGC SIMD Engine 集成測試")
print("=" * 80)

# 測試 FlashKDA
print("\n【1】測試 FlashKDA")
print("-" * 80)

try:
    import flash_kda
    print(f"✅ FlashKDA 導入成功")
    print(f"   可用函數: {[x for x in dir(flash_kda) if not x.startswith('_')]}")
except Exception as e:
    print(f"❌ FlashKDA 導入失敗: {e}")

# 測試 CGC SIMD Engine
print("\n【2】測試 CGC SIMD Engine")
print("-" * 80)

try:
    from cgc_engine.cgc.cgc_simd_executor import CGCKernelRegistry, KernelType
    
    registry = CGCKernelRegistry()
    
    # 檢查 KDA kernel 是否註冊
    if 0x11 in registry._kernels:
        kda_kernel = registry._kernels[0x11]
        print(f"✅ KDA Kernel 已註冊: {kda_kernel.name}")
        print(f"   Supports FlashKDA: {kda_kernel.supports_flashkda}")
    else:
        print("❌ KDA Kernel 未註冊")
        
    # 列出所有已註冊的 attention kernels
    print("\n已註冊的 Attention Kernels:")
    for opcode, spec in registry._kernels.items():
        if spec.kernel_type == KernelType.ATTENTION:
            print(f"  0x{opcode:02X}: {spec.name} (FlashKDA: {spec.supports_flashkda})")
            
except Exception as e:
    print(f"❌ CGC SIMD Engine 測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 測試 KDA 推理
print("\n【3】測試 KDA 推理")
print("-" * 80)

try:
    import torch
    
    # 創建測試張量
    batch_size = 1
    n_heads = 28
    seq_len = 64
    head_dim = 128
    
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    print(f"  輸入 shape: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # 獲取 KDA kernel
    kda_kernel = registry._kernels[0x11].cuda_kernel
    
    # 執行 KDA
    import time
    start = time.time()
    output = kda_kernel(q, k, v, scale=1.0)
    elapsed = time.time() - start
    
    print(f"  輸出 shape: {output.shape}")
    print(f"  執行時間: {elapsed*1000:.2f} ms")
    print(f"  輸出均值: {output.mean().item():.4f}")
    print(f"  ✅ KDA 推理成功!")
    
except Exception as e:
    print(f"❌ KDA 推理失敗: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("測試完成!")
print("=" * 80)