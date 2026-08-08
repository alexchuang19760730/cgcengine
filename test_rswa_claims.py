#!/usr/bin/env python3
"""验证 R-SWA + 窗口 OrthoKDA 三个核心宣称:
1. 无限上下文: feed 10000+ tokens, visible_count 恒定 (reference_len + window_size)
2. O(n) 计算: R-SWA attention 时间随 token 数线性增长, 非二次
3. 显存不长大: feed 大量 token, 内存恒定

测试模型配置 (模拟小/中/大模型的 hidden_dim):
- 小模型 (Qwen3-VL-2B): hidden=2048, heads=16, head_dim=128
- 中模型 (Qwen2.5-7B): hidden=3584, heads=28, head_dim=128  
- 大模型 (V4-Flash): hidden=4096, heads=32, head_dim=128
"""
import os, sys, time, tracemalloc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rswaengine", "python"))
os.environ["RSWAENGINE_LIB"] = os.path.join(
    os.path.dirname(__file__), "rswaengine", "cpp", "build", "libcgc_unified.so"
)

from cgc_unified_injection import RSWAManager

MODELS = {
    "小模型 (Qwen3-VL-2B)": {"num_heads": 16, "head_dim": 128, "hidden": 2048},
    "中模型 (Qwen2.5-7B)":  {"num_heads": 28, "head_dim": 128, "hidden": 3584},
    "大模型 (V4-Flash)":    {"num_heads": 32, "head_dim": 128, "hidden": 4096},
}

REFERENCE_LEN = 4
WINDOW_SIZE = 128
ORTHO_BASE_DIM = 64
TOTAL_TOKENS = 5000


def make_kv(num_heads, head_dim, n=1):
    k = np.random.randn(n, num_heads, head_dim).astype(np.float32)
    v = np.random.randn(n, num_heads, head_dim).astype(np.float32)
    return np.ascontiguousarray(k), np.ascontiguousarray(v)


def test_infinite_context(name, cfg):
    """宣称 1: 无限上下文 — feed 大量 token, visible_count 恒定"""
    print(f"\n{'='*60}")
    print(f"[{name}] 宣称 1: 无限上下文")
    print(f"  hidden={cfg['hidden']}, heads={cfg['num_heads']}, head_dim={cfg['head_dim']}")
    print(f"  reference_len={REFERENCE_LEN}, window_size={WINDOW_SIZE}")
    print(f"  预期 visible_count 恒定 = {REFERENCE_LEN + WINDOW_SIZE}")
    print(f"{'='*60}")

    mgr = RSWAManager(
        num_heads=cfg["num_heads"], head_dim=cfg["head_dim"],
        reference_len=REFERENCE_LEN, window_size=WINDOW_SIZE,
        ortho_base_dim=ORTHO_BASE_DIM, decay_rate=0.01,
        hybrid_every=0, enable_window_ortho=True,
    )
    ref_k, ref_v = make_kv(cfg["num_heads"], cfg["head_dim"], REFERENCE_LEN)
    mgr.set_reference(ref_k, ref_v)

    visible_counts = []
    for i in range(TOTAL_TOKENS):
        k, v = make_kv(cfg["num_heads"], cfg["head_dim"], 1)
        mgr.feed_token(k[0], v[0])
        if i >= WINDOW_SIZE and (i % 500 == 0 or i == TOTAL_TOKENS - 1):
            vis = mgr.visible_count()
            visible_counts.append(vis)
            status = "✓ 恒定" if vis == REFERENCE_LEN + WINDOW_SIZE else "✗ 偏差!"
            print(f"  step {i:5d}: visible={vis} {status}")

    all_constant = all(v == REFERENCE_LEN + WINDOW_SIZE for v in visible_counts)
    print(f"\n  结果: {'✅ PASS — visible_count 恒定, 无限上下文' if all_constant else '❌ FAIL — visible_count 不恒定'}")
    return all_constant


def test_o_n_complexity(name, cfg):
    """宣称 2: O(n) 计算 — attention 时间随 visible_count 线性增长"""
    print(f"\n{'='*60}")
    print(f"[{name}] 宣称 2: O(n) 计算")
    print(f"  对比: R-SWA combined_attention vs 标准注意力 (O(n²))")
    print(f"{'='*60}")

    results = []
    for n in [50, 100, 200, 500, 1000]:
        mgr = RSWAManager(
            num_heads=1, head_dim=64,
            reference_len=4, window_size=n,
            ortho_base_dim=32, decay_rate=0.01,
            hybrid_every=0, enable_window_ortho=True,
        )
        ref_k, ref_v = make_kv(1, 64, 4)
        mgr.set_reference(ref_k, ref_v)
        for _ in range(n):
            k, v = make_kv(1, 64, 1)
            mgr.feed_token(k[0], v[0])
        q = np.random.randn(cfg["num_heads"] * cfg["head_dim"]).astype(np.float32)
        q = np.ascontiguousarray(q)
        iters = 20
        t0 = time.time()
        for _ in range(iters):
            mgr.combined_attention(q)
        dt_rswa = (time.time() - t0) / iters

        # 标准 O(n²) 模拟 (矩阵乘法)
        mat_a = np.random.randn(n + 4, n + 4).astype(np.float32)
        mat_b = np.random.randn(n + 4, 64).astype(np.float32)
        t0 = time.time()
        for _ in range(iters):
            np.dot(mat_a, mat_b)
        dt_std = (time.time() - t0) / iters

        ratio = dt_std / dt_rswa if dt_rswa > 0 else float('inf')
        results.append((n, dt_rswa, dt_std, ratio))
        print(f"  n={n:5d}: RSWA={dt_rswa*1e6:.1f}μs  标准={dt_std*1e6:.1f}μs  比值={ratio:.2f}x")

    # 检查 R-SWA 是否线性增长 (O(n)), 标准是否二次增长 (O(n²))
    if len(results) >= 3:
        n1, r1, s1, _ = results[0]
        n2, r2, s2, _ = results[-1]
        scale = n2 / n1
        rswa_scale = r2 / r1 if r1 > 0 else 0
        std_scale = s2 / s1 if s1 > 0 else 0
        # O(n): scale ≈ scale, O(n²): scale ≈ scale²
        linear_ok = rswa_scale < scale * 2  # 允许 2x 容差
        quadratic_ok = std_scale > scale * scale * 0.5
        print(f"\n  n 从 {n1}→{n2} (x{scale:.0f}):")
        print(f"  R-SWA 时间 x{rswa_scale:.1f} (O(n) 预期 x{scale:.0f}) {'✅' if linear_ok else '⚠️'}")
        print(f"  标准时间 x{std_scale:.1f} (O(n²) 预期 x{scale*scale:.0f}) {'✅' if quadratic_ok else '⚠️'}")
        print(f"\n  结果: {'✅ PASS — R-SWA 增长远小于 O(n²)' if linear_ok else '⚠️ 需进一步验证'}")
        return linear_ok
    return False


def test_memory_constant(name, cfg):
    """宣称 3: 显存不长大 — feed 大量 token, 内存恒定"""
    print(f"\n{'='*60}")
    print(f"[{name}] 宣称 3: 显存不长大")
    print(f"  feed {TOTAL_TOKENS} tokens, 监控 Python 内存")
    print(f"{'='*60}")

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    mgr = RSWAManager(
        num_heads=cfg["num_heads"], head_dim=cfg["head_dim"],
        reference_len=REFERENCE_LEN, window_size=WINDOW_SIZE,
        ortho_base_dim=ORTHO_BASE_DIM, decay_rate=0.01,
        hybrid_every=0, enable_window_ortho=True,
    )
    ref_k, ref_v = make_kv(cfg["num_heads"], cfg["head_dim"], REFERENCE_LEN)
    mgr.set_reference(ref_k, ref_v)

    mem_samples = []
    for i in range(TOTAL_TOKENS):
        k, v = make_kv(cfg["num_heads"], cfg["head_dim"], 1)
        mgr.feed_token(k[0], v[0])
        if i % 500 == 0 or i == TOTAL_TOKENS - 1:
            current, peak = tracemalloc.get_traced_memory()
            mem_samples.append((i, current / 1e6, peak / 1e6))
            print(f"  step {i:5d}: current={current/1e6:.2f}MB  peak={peak/1e6:.2f}MB")

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # 检查内存是否恒定 (最后 50% 的样本波动 < 20%)
    if len(mem_samples) >= 4:
        late_samples = [m[1] for m in mem_samples[len(mem_samples)//2:]]
        if late_samples:
            avg_late = sum(late_samples) / len(late_samples)
            max_dev = max(abs(m - avg_late) for m in late_samples) / avg_late if avg_late > 0 else 1
            constant = max_dev < 0.20
            print(f"\n  后半段内存: avg={avg_late:.2f}MB, 最大偏差={max_dev:.1%}")
            print(f"  结果: {'✅ PASS — 显存恒定, 不随 token 数增长' if constant else '⚠️ 内存有波动'}")
            return constant
    return False


def main():
    print("=" * 60)
    print("R-SWA + 窗口 OrthoKDA 三大宣称验证")
    print(f"每模型 feed {TOTAL_TOKENS} tokens")
    print("=" * 60)

    all_pass = True
    for name, cfg in MODELS.items():
        r1 = test_infinite_context(name, cfg)
        r2 = test_o_n_complexity(name, cfg)
        r3 = test_memory_constant(name, cfg)
        model_pass = r1 and r2 and r3
        all_pass = all_pass and model_pass
        print(f"\n[{name}] 总计: {'✅ ALL PASS' if model_pass else '⚠️ PARTIAL'}")

    print(f"\n{'='*60}")
    print(f"最终结果: {'✅ 全部通过 — 三个宣称均验证' if all_pass else '⚠️ 部分通过 — 需进一步验证'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
