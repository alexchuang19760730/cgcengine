"""叠加测试: 投机 decode + R-SWA attention (MLX, Mac M4).

测试方案:
1. baseline: 标准 attention, 无投机 (26.8 tok/s)
2. R-SWA only: R-SWA attention, 无投机 (~35 tok/s 预估)
3. 投机 decode only: 标准 attention, 0.5B draft (53.2 tok/s)
4. 投机 decode + R-SWA: R-SWA attention + 0.5B draft (? tok/s)

用简化模型模拟 (不加载完整 LLM, 用 attention 模块模拟):
- 模拟 43 层 transformer (每层 = attention + MLP)
- attention 部分: 标准 vs R-SWA
- MLP 部分: 固定开销
- 投机 decode: 模拟 N=16 chain (draft + verify)
"""
import time
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except Exception:
    HAS_MLX = False

if HAS_MLX:
    from rswa_mlx import RSWAOrthoAttentionMLX


def simulate_transformer_layer(attn_fn, mlp_fn, q, k_cache, v_cache, num_heads, head_dim):
    """模拟一层 transformer: attention + MLP."""
    # attention
    attn_out = attn_fn(q, k_cache, v_cache)
    # MLP (固定开销, 模拟 4x hidden)
    mlp_out = mlp_fn(attn_out)
    return mlp_out


def make_standard_attn(num_heads, head_dim):
    """标准全量 attention."""
    scale = head_dim ** -0.5
    def attn(q, k_cache, v_cache):
        # q: [num_heads, head_dim]
        # k_cache: [seq_len, num_heads, head_dim]
        # v_cache: [num_heads, head_dim]
        q_exp = mx.expand_dims(q, 1)  # [num_heads, 1, head_dim]
        k = k_cache.transpose(1, 0, 2)  # [num_heads, seq_len, head_dim]
        v = v_cache.transpose(1, 0, 2)  # [num_heads, seq_len, head_dim]
        scores = mx.matmul(q_exp, k.transpose(0, 2, 1)) * scale
        weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(weights, v)[:, 0, :]
        return out
    return attn


def make_rswa_attn(num_heads, head_dim, ref_len, win_size, ortho_dim):
    """R-SWA attention (恒定范围)."""
    rswa = RSWAOrthoAttentionMLX(num_heads, head_dim, ref_len, win_size, ortho_dim)
    rswa.set_reference(mx.random.normal((ref_len, num_heads, head_dim)),
                       mx.random.normal((ref_len, num_heads, head_dim)))
    # 预填充窗口
    for _ in range(win_size):
        rswa.feed_token(mx.random.normal((num_heads, head_dim)),
                        mx.random.normal((num_heads, head_dim)))
    mx.eval(rswa.win_K)

    def attn(q, k_cache, v_cache):
        # R-SWA: feed + forward (attention 范围恒定)
        k_new = mx.random.normal((num_heads, head_dim))
        v_new = mx.random.normal((num_heads, head_dim))
        rswa.feed_token(k_new, v_new)
        return rswa.forward(q)
    return attn


def make_mlp(num_heads, head_dim):
    """模拟 MLP (4x hidden, 固定开销)."""
    hidden = num_heads * head_dim
    W1 = mx.random.normal((hidden, hidden * 4)) * 0.02
    W2 = mx.random.normal((hidden * 4, hidden)) * 0.02
    def mlp(x):
        x_flat = x.reshape(-1)
        h = mx.matmul(x_flat, W1)
        h = mx.maximum(h, 0)  # ReLU
        out = mx.matmul(h, W2)
        return out.reshape(num_heads, head_dim)
    return mlp


def bench_decode(attn_fn, mlp_fn, num_layers, num_heads, head_dim,
                 num_steps, spec_decode=False, spec_N=16, spec_accept=0.65):
    """模拟 decode, 返回 tok/s.

    spec_decode=False: 每 step 1 token
    spec_decode=True: 每 step 平均 (1 + spec_N * spec_accept) tokens
    """
    q = mx.random.normal((num_heads, head_dim))
    k_cache = mx.random.normal((128, num_heads, head_dim))  # 初始 128 tokens
    v_cache = mx.random.normal((128, num_heads, head_dim))

    # warmup
    for _ in range(5):
        for _ in range(num_layers):
            simulate_transformer_layer(attn_fn, mlp_fn, q, k_cache, v_cache, num_heads, head_dim)
    mx.eval(q)

    t0 = time.time()
    total_tokens = 0
    for step in range(num_steps):
        if spec_decode:
            # 投机 decode: draft N tokens, verify
            # 模拟: 1 次 target forward + N 次 draft forward
            accepted = int(spec_N * spec_accept)
            total_tokens += 1 + accepted
            # target forward (N+1 tokens verify, 但我们模拟 1 次)
            for _ in range(num_layers):
                simulate_transformer_layer(attn_fn, mlp_fn, q, k_cache, v_cache, num_heads, head_dim)
            # draft forward (N 次, 简化为 1 次, 开销小)
            # draft overhead ~ 0.3x target (0.5B vs 2B)
            for _ in range(num_layers):
                mlp_fn(q)  # 简化 draft
        else:
            # 标准 decode: 每 step 1 token
            total_tokens += 1
            for _ in range(num_layers):
                simulate_transformer_layer(attn_fn, mlp_fn, q, k_cache, v_cache, num_heads, head_dim)

    mx.eval(q)
    dt = time.time() - t0
    return total_tokens / dt if dt > 0 else 0


def main():
    if not HAS_MLX:
        print("MLX not available")
        return

    print("=" * 70)
    print(" 叠加测试: 投机 decode + R-SWA attention (MLX, Mac M4)")
    print("=" * 70)

    NH, HD = 16, 128
    NUM_LAYERS = 8  # 简化: 8 层 (不是 43, 加快测试)
    REF, WIN, ORTHO = 4, 128, 64
    STEPS = 100
    SPEC_N = 16
    SPEC_ACCEPT = 0.65  # 0.5B draft accept rate

    mlp = make_mlp(NH, HD)
    std_attn = make_standard_attn(NH, HD)
    rswa_attn = make_rswa_attn(NH, HD, REF, WIN, ORTHO)

    configs = [
        ("1. baseline (标准, 无投机)", std_attn, False, 1.0),
        ("2. R-SWA only (无投机)", rswa_attn, False, 1.0),
        ("3. 投机 decode only (标准 attn)", std_attn, True, SPEC_ACCEPT),
        ("4. 投机 decode + R-SWA", rswa_attn, True, SPEC_ACCEPT),
    ]

    results = []
    for label, attn, spec, accept in configs:
        tps = bench_decode(attn, mlp, NUM_LAYERS, NH, HD, STEPS,
                          spec_decode=spec, spec_N=SPEC_N, spec_accept=accept)
        results.append((label, tps))
        print(f"  {label}: {tps:.1f} tok/s")

    print(f"\n{'='*70}")
    print(f" 对比 (相对 baseline)")
    print(f"{'='*70}")
    base_tps = results[0][1]
    for label, tps in results:
        speedup = tps / base_tps if base_tps > 0 else 0
        print(f"  {label}: {tps:.1f} tok/s ({speedup:.2f}x)")

    # 叠加效果
    rswa_only = results[1][1] / base_tps
    spec_only = results[2][1] / base_tps
    both = results[3][1] / base_tps
    print(f"\n  R-SWA only: {rswa_only:.2f}x")
    print(f"  投机 only:  {spec_only:.2f}x")
    print(f"  叠加:       {both:.2f}x")
    print(f"  预期叠加:   {rswa_only * spec_only:.2f}x (独立相乘)")
    print(f"  实际/预期:  {both / (rswa_only * spec_only):.2f}x")


if __name__ == "__main__":
    main()
