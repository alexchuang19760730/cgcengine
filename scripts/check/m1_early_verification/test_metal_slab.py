#!/usr/bin/env python3
"""
M1 早期驗證測試 3：Metal 整層 Slab 可行性

目的：驗證 Metal 是否能分配 280MB+ 連續 buffer 並做 batch matmul，
這是 prefill 整層 slab 路徑的硬體基礎。

使用方式：
    /Users/alexchuang/Documents/edge0/.venv/bin/python test_metal_slab.py

測試環境：
    - MLX 0.30.4 + Metal
    - MacBook Air M4 16GB

測試結果（2026-09-13）：
    ✅ 280MB 連續 buffer 分配成功（299ms）
    ✅ Batch matmul [256,1024,512]×[256,512,256] 成功（70ms）
    ✅ 整層 MoE 三個 proj 共 768MB 分配成功
    ✅ FFN 計算（8 tokens × 單 expert）成功（68.6ms）
    ✅ Prefill 整層計算（256 tokens）成功（23.5ms）
"""

import time
import sys

try:
    import mlx.core as mx
except ImportError:
    print("❌ MLX 未安裝，請使用 Edge0 的 .venv：")
    print("   /Users/alexchuang/Documents/edge0/.venv/bin/python test_metal_slab.py")
    sys.exit(1)


def silu(x):
    """手動實現 silu：x * sigmoid(x)"""
    return x * mx.sigmoid(x)


def test_1_allocate_280mb_buffer():
    """測試 1：分配 280MB 連續 Metal buffer（256 experts × [1024,512] FP16）"""
    print("=== 測試 1：分配 280MB Metal buffer（256 experts × [1024,512] FP16）===")
    t0 = time.time()
    slab = mx.random.normal((256, 1024, 512), dtype=mx.float16)
    mx.eval(slab)
    t1 = time.time()
    print(f"✅ 分配成功，耗時: {(t1-t0)*1000:.1f} ms")
    print(f"   Shape: {slab.shape}, dtype: {slab.dtype}")
    print(f"   Size: {slab.nbytes / 1024 / 1024:.1f} MB")
    print()
    return slab


def test_2_batch_matmul():
    """測試 2：Batch matmul [256,1024,512] × [256,512,256]"""
    print("=== 測試 2：Batch matmul [256,1024,512] × [256,512,256] ===")
    a = mx.random.normal((256, 1024, 512), dtype=mx.float16)
    b = mx.random.normal((256, 512, 256), dtype=mx.float16)
    mx.eval(a, b)
    t0 = time.time()
    c = mx.matmul(a, b)
    mx.eval(c)
    t1 = time.time()
    print(f"✅ Batch matmul 成功，耗時: {(t1-t0)*1000:.1f} ms")
    print(f"   Output shape: {c.shape}, size: {c.nbytes / 1024 / 1024:.1f} MB")
    print()


def test_3_full_layer_moe():
    """測試 3：整層 MoE 三個 proj（gate/up/down），總計 ~768MB"""
    print("=== 測試 3：整層 MoE 三個 proj（gate/up/down），總計 ~768MB ===")
    t0 = time.time()
    gate = mx.random.normal((256, 1024, 512), dtype=mx.float16)
    up = mx.random.normal((256, 1024, 512), dtype=mx.float16)
    down = mx.random.normal((256, 512, 1024), dtype=mx.float16)
    mx.eval(gate, up, down)
    t1 = time.time()
    total_mb = (gate.nbytes + up.nbytes + down.nbytes) / 1024 / 1024
    print(f"✅ 分配 3 個 proj 成功，耗時: {(t1-t0)*1000:.1f} ms")
    print(f"   總大小: {total_mb:.1f} MB")
    print()
    return gate, up, down


def test_4_ffn_compute(gate, up, down):
    """測試 4：FFN 計算（8 tokens × 單 expert）"""
    print("=== 測試 4：FFN 計算（8 tokens × 單 expert）===")
    x = mx.random.normal((8, 1024), dtype=mx.float16)
    mx.eval(x)
    t0 = time.time()
    gate_out = mx.matmul(x, gate[0])
    up_out = mx.matmul(x, up[0])
    activated = gate_out * silu(up_out)
    down_out = mx.matmul(activated, down[0])
    mx.eval(down_out)
    t1 = time.time()
    print(f"✅ FFN 計算成功，耗時: {(t1-t0)*1000:.1f} ms")
    print(f"   Output shape: {down_out.shape}")
    print()


def test_5_prefill_full_layer(gate, up, down):
    """測試 5：Prefill 整層計算（256 tokens，batch expert）"""
    print("=== 測試 5：Prefill 整層計算（256 tokens，batch expert）===")
    x_prefill = mx.random.normal((256, 1024), dtype=mx.float16)
    mx.eval(x_prefill)
    t0 = time.time()
    # batch matmul: [256, 1, 1024] × [256, 1024, 512] -> [256, 1, 512]
    x_expanded = x_prefill[:, None, :]  # [256, 1, 1024]
    gate_all = mx.matmul(x_expanded, gate)
    up_all = mx.matmul(x_expanded, up)
    activated_all = gate_all * silu(up_all)
    down_all = mx.matmul(activated_all, down)  # [256, 1, 1024]
    mx.eval(down_all)
    t1 = time.time()
    print(f"✅ Prefill 整層計算成功，耗時: {(t1-t0)*1000:.1f} ms")
    print(f"   Output shape: {down_all.shape}")
    print(f"   （實際 prefill 會用 top-k routing，不會算全部 256 experts）")
    print()


def main():
    print("=" * 60)
    print("M1 早期驗證測試 3：Metal 整層 Slab 可行性")
    print("=" * 60)
    print(f"MLX version: {mx.__version__}")
    print(f"Metal available: {mx.metal.is_available()}")
    print()

    # 測試 1
    slab = test_1_allocate_280mb_buffer()

    # 測試 2
    test_2_batch_matmul()

    # 測試 3
    gate, up, down = test_3_full_layer_moe()

    # 測試 4
    test_4_ffn_compute(gate, up, down)

    # 測試 5
    test_5_prefill_full_layer(gate, up, down)

    # 總結
    print("=" * 60)
    print("=== 總結 ===")
    print("=" * 60)
    print("✅ Metal 可以分配 280MB+ 的連續 buffer")
    print("✅ 可以同時分配多個大 buffer（總計 ~768MB）")
    print("✅ Batch matmul 和 FFN 計算正常")
    print("✅ Prefill 整層 slab 計算可行")
    print()
    print("⚠️  實際完整尺寸（256 × [2048,512] × FP16 × 3 proj）= ~1.5GB")
    print("⚠️  在 16GB 機器上，需要與 expert pool（6GB）、KV cache、模型權重共存")
    print("⚠️  建議：prefill 用整層 slab，decode 用 compacted gather，動態切換")


if __name__ == "__main__":
    main()
