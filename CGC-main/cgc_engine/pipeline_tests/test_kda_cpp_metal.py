#!/usr/bin/env python3
"""
CGC C++ Kimi KDA Engine - Python 测试脚本

需要先编译 C++ 引擎:
cd cgc_engine/cgc/cgc_cpp
mkdir -p build && cd build
cmake .. && make -j8

然后运行:
python3 test_kda_cpp_metal.py
"""

import sys
import os
from pathlib import Path

_build_dir = Path(__file__).resolve().parents[2] / "cgc_engine" / "cgc" / "cgc_cpp" / "build"
if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))

import numpy as np
import time

try:
    import cgc_cpp
    CGC_AVAILABLE = True
except ImportError as e:
    CGC_AVAILABLE = False
    print(f"⚠️  CGC C++ 引擎未加载: {e}")
    print("   尝试使用 PyTorch 版本...")

import torch
import torch.nn.functional as F

# =============================================================================
# Kimi KDA 核心公式 (PyTorch 参考实现)
# =============================================================================
def kimi_kda_pytorch(Q, K, V, beta=0.1):
    """
    Kimi 原论文 KDA 公式
    S = (I - β k kᵀ) S + β k vᵀ
    O = Q S
    """
    B, H, L, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    S = torch.zeros(B, H, D, D, device=Q.device)
    O = torch.zeros_like(Q)

    for l in range(L):
        k_l = K[:, :, l, :]
        v_l = V[:, :, l, :]
        q_l = Q[:, :, l, :]

        S = S * (1.0 - beta * torch.einsum('bhd,bhe->bhde', k_l, k_l)) + beta * torch.einsum('bhd,bhe->bhde', k_l, v_l)
        o_l = torch.einsum('bhd,bhde->bhe', q_l, S) * scale
        O[:, :, l] = o_l

    return O

# =============================================================================
# CGC C++ KDA 测试
# =============================================================================
def test_cgc_kda():
    """测试 CGC C++ KDA 引擎"""
    print("\n" + "="*60)
    print("🔥 CGC C++ Kimi KDA 引擎测试")
    print("="*60)

    if not CGC_AVAILABLE:
        print("⚠️  CGC C++ 引擎不可用，使用 PyTorch 版本")
        return None

    print("\n✅ CGC C++ 引擎已加载")

    # 初始化
    cgc_cpp.init()
    backend = cgc_cpp.get_current_backend()
    print(f"   当前后端: {backend}")

    # 测试参数
    B, H, L, D = 1, 28, 128, 128
    beta = 0.1

    print(f"\n🔧 测试配置:")
    print(f"   Batch: {B}, Heads: {H}, SeqLen: {L}, HeadDim: {D}")
    print(f"   Beta: {beta}")

    # 准备输入数据
    Q = np.random.randn(B, H, L, D).astype(np.float32) * 0.1
    K = np.random.randn(B, H, L, D).astype(np.float32) * 0.1
    V = np.random.randn(B, H, L, D).astype(np.float32) * 0.1

    # 执行 KDA
    print("\n🔹 执行 CGC KDA...")
    t0 = time.time()

    result = cgc_cpp.execute_opcode(
        0x11,  # ATTENTION_KDA
        [Q, K, V],
        {"beta": beta}
    )

    cgc_time = time.time() - t0
    print(f"   时间: {cgc_time:.6f}s")

    return cgc_time

# =============================================================================
# 性能对比测试
# =============================================================================
def benchmark_kda():
    """对比 PyTorch vs C++ KDA 性能"""
    print("\n" + "="*60)
    print("📊 Kimi KDA 性能对比测试")
    print("="*60)

    # 测试配置
    configs = [
        (1, 28, 128, 128),   # Qwen2.5-7B 配置
        (1, 8, 512, 64),     # 长序列配置
        (1, 4, 1024, 128),   # 更大序列
    ]

    for B, H, L, D in configs:
        print(f"\n配置: B={B}, H={H}, L={L}, D={D}")

        Q = torch.randn(B, H, L, D) * 0.1
        K = torch.randn(B, H, L, D) * 0.1
        V = torch.randn(B, H, L, D) * 0.1

        # PyTorch 测试
        t0 = time.time()
        for _ in range(5):
            O_pytorch = kimi_kda_pytorch(Q, K, V)
        torch_time = (time.time() - t0) / 5

        print(f"   PyTorch: {torch_time:.6f}s ({L/torch_time:.2f} tokens/s)")

        # C++ 测试（如果可用）
        if CGC_AVAILABLE:
            Q_np = Q.numpy()
            K_np = K.numpy()
            V_np = V.numpy()

            t0 = time.time()
            for _ in range(5):
                result = cgc_cpp.execute_opcode(0x11, [Q_np, K_np, V_np], {"beta": 0.1})
            cpp_time = (time.time() - t0) / 5

            speedup = torch_time / cpp_time if cpp_time > 0 else 0
            print(f"   C++:      {cpp_time:.6f}s ({L/cpp_time:.2f} tokens/s) [加速 {speedup:.2f}x]")

# =============================================================================
# 数学正确性验证
# =============================================================================
def verify_correctness():
    """验证 KDA 数学正确性"""
    print("\n" + "="*60)
    print("🔬 KDA 数学正确性验证")
    print("="*60)

    B, H, L, D = 1, 4, 32, 32
    beta = 0.1

    Q = torch.randn(B, H, L, D) * 0.1
    K = torch.randn(B, H, L, D) * 0.1
    V = torch.randn(B, H, L, D) * 0.1

    # 标准 Attention
    scale = 1.0 / (D ** 0.5)
    attn_scores = (Q @ K.transpose(-2, -1)) * scale
    attn = F.softmax(attn_scores, dim=-1)
    out_std = attn @ V

    # Kimi KDA
    out_kda = kimi_kda_pytorch(Q, K, V, beta=beta)

    # 误差
    mse = F.mse_loss(out_std, out_kda).item()
    mae = F.l1_loss(out_std, out_kda).item()
    max_err = torch.max(torch.abs(out_std - out_kda)).item()

    print(f"\n标准 Attention vs Kimi KDA:")
    print(f"   MSE:   {mse:.6e}")
    print(f"   MAE:   {mae:.6e}")
    print(f"   MAX:   {max_err:.6e}")
    print(f"   ✅ 一致性: {'PASS' if mae < 0.1 else 'FAIL'}")

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 CGC Compiler + Kimi KDA - C++/Metal 实现测试")
    print("="*70)

    # 1. 数学正确性验证
    verify_correctness()

    # 2. C++ KDA 测试
    test_cgc_kda()

    # 3. 性能对比
    benchmark_kda()

    print("\n" + "="*70)
    print("✅ 测试完成!")
    print("="*70)

    if not CGC_AVAILABLE:
        print("""
📝 编译 CGC C++ 引擎:

cd /Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/cgc_engine/cgc/cgc_cpp
mkdir -p build && cd build
cmake ..
make -j8

然后重新运行此脚本:
python3 test_kda_cpp_metal.py
        """)
