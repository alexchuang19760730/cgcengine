#!/usr/bin/env python3
"""
CGC Kimi KDA - C++ SIMD 性能对比测试

比较：
1. PyTorch CPU
2. PyTorch MPS (Metal)
3. C++ NEON SIMD

运行：
python3 test_kda_cpp_perf.py
"""

import sys
import os
from pathlib import Path

_build_dir = Path(__file__).resolve().parents[2] / "cgc_engine" / "cgc" / "cgc_cpp" / "build"
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

import numpy as np
import time
import torch
import torch.nn.functional as F

print("""
╔════════════════════════════════════════════════════════════════╗
║     CGC Kimi KDA - C++ SIMD 性能对比                        ║
╚════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 1. 加载 C++ KDA
# =============================================================================
print("🔍 加载 C++ KDA 模块...")
try:
    import kda_cpp
    kda_cpp_available = True
    print("✅ C++ KDA 已加载 (NEON SIMD)")
except ImportError as e:
    kda_cpp_available = False
    print(f"❌ C++ KDA 加载失败: {e}")

# =============================================================================
# 2. PyTorch 检测
# =============================================================================
print("\n🔍 检测 PyTorch 设备...")
mps_available = torch.backends.mps.is_available()
device_mps = torch.device("mps" if mps_available else "cpu")
device_cpu = torch.device("cpu")
print(f"   MPS (Metal): {'✅ 可用' if mps_available else '❌ 不可用'}")
print(f"   CPU: ✅ 可用")

# =============================================================================
# 3. Kimi KDA PyTorch 实现
# =============================================================================
def kimi_kda_torch(Q, K, V, beta=0.1):
    """PyTorch 实现"""
    B, H, L, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    S = torch.zeros(B, H, D, D, device=Q.device)
    O = torch.zeros_like(Q)

    for l in range(L):
        k_l = K[:, :, l, :]
        v_l = V[:, :, l, :]
        q_l = Q[:, :, l, :]

        kkt = torch.einsum('bhd,bhe->bhde', k_l, k_l)
        kv = torch.einsum('bhd,bhe->bhde', k_l, v_l)
        S = S * (1.0 - beta * kkt) + beta * kv
        o_l = torch.einsum('bhd,bhde->bhe', q_l, S) * scale
        O[:, :, l, :] = o_l

    return O

# =============================================================================
# 4. C++ KDA 封装
# =============================================================================
def kda_cpp_forward(Q, K, V, beta=0.1):
    """C++ KDA 封装"""
    if not kda_cpp_available:
        return None

    B, H, L, D = Q.shape

    kda = kda_cpp.KDA()
    kda.init(B, H, D)

    Q_arr = np.ascontiguousarray(Q.numpy().astype(np.float32))
    K_arr = np.ascontiguousarray(K.numpy().astype(np.float32))
    V_arr = np.ascontiguousarray(V.numpy().astype(np.float32))

    O = kda.forward(Q_arr, K_arr, V_arr, beta)

    return torch.from_numpy(np.array(O)).reshape(B, H, L, D)

# =============================================================================
# 5. 性能测试
# =============================================================================
def benchmark():
    print("\n" + "="*70)
    print("📊 KDA 性能对比测试")
    print("="*70)

    configs = [
        (1, 28, 128, 128, "Qwen2.5-7B 配置"),
        (1, 28, 256, 128, "长序列"),
        (1, 28, 512, 128, "更长序列"),
    ]

    for B, H, L, D, desc in configs:
        print(f"\n🔹 {desc}: B={B}, H={H}, L={L}, D={D}")

        torch.manual_seed(42)
        Q = torch.randn(B, H, L, D) * 0.1
        K = torch.randn(B, H, L, D) * 0.1
        V = torch.randn(B, H, L, D) * 0.1

        iterations = 20

        # PyTorch CPU
        Q_cpu = Q.to(device_cpu)
        K_cpu = K.to(device_cpu)
        V_cpu = V.to(device_cpu)

        for _ in range(3):
            _ = kimi_kda_torch(Q_cpu, K_cpu, V_cpu)

        torch.mps.synchronize() if mps_available else None
        t0 = time.time()
        for _ in range(iterations):
            O_cpu = kimi_kda_torch(Q_cpu, K_cpu, V_cpu)
        torch.mps.synchronize() if mps_available else None
        cpu_time = (time.time() - t0) / iterations
        cpu_speed = L / cpu_time

        print(f"   PyTorch CPU:      {cpu_time:.6f}s ({cpu_speed:>10.2f} tok/s)")

        # PyTorch MPS
        if mps_available:
            Q_mps = Q.to(device_mps)
            K_mps = K.to(device_mps)
            V_mps = V.to(device_mps)

            for _ in range(3):
                _ = kimi_kda_torch(Q_mps, K_mps, V_mps)

            torch.mps.synchronize()
            t0 = time.time()
            for _ in range(iterations):
                O_mps = kimi_kda_torch(Q_mps, K_mps, V_mps)
            torch.mps.synchronize()
            mps_time = (time.time() - t0) / iterations
            mps_speed = L / mps_time

            print(f"   PyTorch MPS:     {mps_time:.6f}s ({mps_speed:>10.2f} tok/s)")

        # C++ NEON SIMD
        if kda_cpp_available:
            for _ in range(3):
                _ = kda_cpp_forward(Q, K, V)

            t0 = time.time()
            for _ in range(iterations):
                O_cpp = kda_cpp_forward(Q, K, V)
            cpp_time = (time.time() - t0) / iterations
            cpp_speed = L / cpp_time

            print(f"   C++ NEON SIMD:   {cpp_time:.6f}s ({cpp_speed:>10.2f} tok/s)")

            if mps_available:
                speedup = mps_speed / cpp_speed if cpp_speed > 0 else 0
                print(f"   MPS vs C++:      {speedup:.2f}x")

# =============================================================================
# 6. 数学正确性验证
# =============================================================================
def verify_correctness():
    print("\n" + "="*70)
    print("🔬 KDA 数学正确性验证")
    print("="*70)

    B, H, L, D = 1, 4, 32, 32
    beta = 0.1

    torch.manual_seed(42)
    Q = torch.randn(B, H, L, D) * 0.1
    K = torch.randn(B, H, L, D) * 0.1
    V = torch.randn(B, H, L, D) * 0.1

    # PyTorch CPU
    O_torch = kimi_kda_torch(Q, K, V, beta=beta)

    # C++ SIMD
    if kda_cpp_available:
        O_cpp = kda_cpp_forward(Q, K, V, beta=beta)

        if O_cpp is not None:
            mse = F.mse_loss(O_torch, O_cpp).item()
            mae = F.l1_loss(O_torch, O_cpp).item()

            print(f"\nPyTorch vs C++ NEON:")
            print(f"   MSE: {mse:.6e}")
            print(f"   MAE: {mae:.6e}")
            print(f"   ✅ 一致性: {'PASS' if mae < 0.01 else 'FAIL'}")

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    verify_correctness()
    benchmark()

    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)
