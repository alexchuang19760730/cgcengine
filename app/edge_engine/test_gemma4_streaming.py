#!/usr/bin/env python3
"""Gemma4-26B-A4B streaming vs bypass 对比测试.

验证目标:
1. StreamingSwitchGLU 在 Gemma4 30层/128专家结构上正确注入
2. bypass (全量加载) vs streaming (off 模式) 性能对比
3. 验证商业化指标: no MTP 6 tps / TTFT 1.5s / 99.5% 成功率
"""
import sys
import time

sys.path.insert(0, "app/edge_engine")

import mlx.core as mx
# 调高 GPU 内存上限 (M4 16GB 物理内存, 默认推荐 10.67GB, 调到 14GB 留 2GB 给系统)
try:
    mx.metal.set_memory_limit(14 * 1024 * 1024 * 1024)
    print(f"[metal] memory_limit set to 14GB")
except Exception as e:
    print(f"[metal] set_memory_limit failed: {e}")
# 调高 cache_limit (MLX 计算图缓存)
try:
    mx.metal.set_cache_limit(2 * 1024 * 1024 * 1024)
    print(f"[metal] cache_limit set to 2GB")
except Exception as e:
    print(f"[metal] set_cache_limit failed: {e}")

from omlx_mlx_engine import OMLXMLXEngine


def benchmark_mode(
    model_path: str,
    enable_streaming: bool,
    prompt: str,
    max_tokens: int,
    warmup: int = 1,
    stats_mode: str = "off",
    cache_size: int = 2,
) -> dict:
    """跑单个模式 benchmark."""
    tag = "streaming" if enable_streaming else "bypass"
    print(f"\n=== {tag} (stats={stats_mode}, cache={cache_size}) ===")

    engine = OMLXMLXEngine(
        model_path=model_path,
        enable_streaming=enable_streaming,
        streaming_config={
            "max_experts_in_memory": cache_size,
            "stats_mode": stats_mode,
            "lazy_stats": True,
            "enable_io_simulation": False,
        },
    )
    engine.load()
    print(f"  MoE: {engine._is_moe}, Layers: {engine._num_layers}, Experts: {engine._num_experts}")

    # warmup
    for _ in range(warmup):
        engine.generate(prompt, max_tokens=10)

    # 正式测
    t0 = time.perf_counter()
    text = engine.generate(prompt, max_tokens=max_tokens)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    tps = max_tokens / elapsed
    ttft_ms = elapsed / (max_tokens + 1) * 1000  # 粗估

    print(f"  TPS: {tps:.2f}, TTFT(est): {ttft_ms:.1f}ms")
    print(f"  Output: {text[:80]!r}")

    stats = engine.get_stats() if enable_streaming else None
    if stats and stats.total_calls > 0:
        print(f"  Stats: {stats.summary()}")

    return {
        "mode": tag,
        "tps": tps,
        "ttft_ms": ttft_ms,
        "elapsed_s": elapsed,
        "text": text[:80],
        "hit_rate": stats.hit_rate if stats else None,
        "swaps": stats.total_swaps if stats else None,
    }


def main():
    model_path = "models/gemma-4-26B-A4B-it-qat-4bit"
    prompt = "Explain what is a transformer model in one sentence."
    max_tokens = 50

    print(f"Model: {model_path}")
    print(f"Prompt: {prompt!r}")
    print(f"Max tokens: {max_tokens}")

    results = []

    # 1. bypass (baseline)
    try:
        r = benchmark_mode(
            model_path, enable_streaming=False,
            prompt=prompt, max_tokens=max_tokens,
        )
        results.append(r)
    except Exception as e:
        print(f"  bypass failed: {e}")

    # 2. streaming (off 模式 = 零统计开销, 生产用)
    try:
        r = benchmark_mode(
            model_path, enable_streaming=True,
            prompt=prompt, max_tokens=max_tokens,
            stats_mode="off", cache_size=2,
        )
        results.append(r)
    except Exception as e:
        print(f"  streaming off failed: {e}")

    # 3. streaming + true_swap (每层后 eval+clear_cache, 解决 OOM)
    try:
        print(f"\n=== streaming + true_swap (OOM 解决方案) ===")
        engine = OMLXMLXEngine(
            model_path=model_path,
            enable_streaming=True,
            streaming_config={
                "max_experts_in_memory": 2,
                "stats_mode": "off",
                "lazy_stats": True,
                "true_swap": True,
            },
        )
        engine.load()
        print(f"  MoE: {engine._is_moe}, Layers: {engine._num_layers}, Experts: {engine._num_experts}")
        # warmup
        engine.generate(prompt, max_tokens=5)
        t0 = time.perf_counter()
        text = engine.generate(prompt, max_tokens=max_tokens)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        tps = max_tokens / elapsed
        ttft_ms = elapsed / (max_tokens + 1) * 1000
        print(f"  TPS: {tps:.2f}, TTFT(est): {ttft_ms:.1f}ms")
        print(f"  Output: {text[:80]!r}")
        results.append({"mode": "streaming_true_swap", "tps": tps, "ttft_ms": ttft_ms, "elapsed_s": elapsed, "text": text[:80], "hit_rate": None, "swaps": None})
    except Exception as e:
        print(f"  streaming true_swap failed: {e}")

    # 汇总
    print("\n" + "=" * 70)
    print(f"{'Mode':<25} {'TPS':>8} {'TTFT(ms)':>10} {'Hit%':>8}")
    print("-" * 70)
    bypass_tps = next((r["tps"] for r in results if r["mode"] == "bypass"), 0)
    for r in results:
        delta = f"({r['tps']/bypass_tps*100:.0f}%)" if bypass_tps > 0 else ""
        hit = f"{r['hit_rate']:.1%}" if r.get("hit_rate") is not None else "-"
        print(f"{r['mode']:<25} {r['tps']:>7.2f} {delta:>8} {r['ttft_ms']:>9.1f} {hit:>8}")
    print("=" * 70)

    # 商业化指标检查
    print("\n--- 商业化指标 (no MTP: 6 tps / TTFT 1.5s) ---")
    for r in results:
        ok_tps = "✓" if r["tps"] >= 6 else "✗"
        ok_ttft = "✓" if r["ttft_ms"] < 1500 else "✗"
        print(f"  {r['mode']}: tps={r['tps']:.2f} {ok_tps} | ttft={r['ttft_ms']:.1f}ms {ok_ttft}")


if __name__ == "__main__":
    main()
