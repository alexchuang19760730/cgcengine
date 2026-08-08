#!/usr/bin/env python3
"""oMLX+FlashMoE streaming vs bypass 对比测试.

验证目标:
1. streaming 路径正确 work (hit/miss/swap 统计合理)
2. streaming 与 bypass 性能相近 (无 I/O 模拟时)
3. I/O 模拟下 streaming 的性能退化曲线
4. 不同 cache size 下的命中率变化
"""
import json
import time
from typing import Any, Dict, List

from app.edge_engine.omlx_mlx_engine import OMLXMLXEngine


def run_benchmark(
    model_path: str,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    max_tokens: int = 50,
    warmup: int = 2,
) -> Dict[str, Any]:
    """跑 bypass vs streaming 对比."""
    results = {}

    # 1. Bypass (全量加载, 无 streaming)
    print("\n=== Bypass Mode (full load) ===")
    engine = OMLXMLXEngine(
        model_path=model_path,
        enable_streaming=False,
    )
    engine.load()
    t0 = time.perf_counter()
    for _ in range(warmup):
        engine.generate(prompt, max_tokens=10)
    t1 = time.perf_counter()

    t0 = time.perf_counter()
    text = engine.generate(prompt, max_tokens=max_tokens)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    tps = max_tokens / elapsed
    results["bypass"] = {
        "elapsed_s": elapsed,
        "tps": tps,
        "ttft_ms": elapsed / (max_tokens + 1) * 1000,
        "text": text[:80],
    }
    print(f"  TPS: {tps:.1f}, TTFT: {results['bypass']['ttft_ms']:.1f}ms")
    del engine

    # 2. Streaming (不同 cache size)
    for cache_size in [2, 4, 8]:
        print(f"\n=== Streaming Mode (cache_size={cache_size}) ===")
        engine = OMLXMLXEngine(
            model_path=model_path,
            enable_streaming=True,
            streaming_config={
                "max_experts_in_memory": cache_size,
                "enable_io_simulation": False,
            },
        )
        engine.load()
        for _ in range(warmup):
            engine.generate(prompt, max_tokens=10)

        # 重置 stats
        if engine.manager:
            from app.edge_engine.omlx_mlx_engine import StreamingStats
            engine.manager.stats = StreamingStats()

        t0 = time.perf_counter()
        text = engine.generate(prompt, max_tokens=max_tokens)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        tps = max_tokens / elapsed
        stats = engine.get_stats()

        results[f"streaming_cache{cache_size}"] = {
            "elapsed_s": elapsed,
            "tps": tps,
            "ttft_ms": elapsed / (max_tokens + 1) * 1000,
            "hit_rate": stats.hit_rate,
            "hits": stats.total_expert_hits,
            "misses": stats.total_expert_misses,
            "swaps": stats.total_swaps,
            "text": text[:80],
        }
        print(
            f"  TPS: {tps:.1f}, hit_rate: {stats.hit_rate:.2%}, "
            f"swaps: {stats.total_swaps}"
        )
        del engine

    # 3. Streaming with I/O simulation
    for io_ms in [0.5, 1.0, 5.0]:
        print(f"\n=== Streaming + I/O Sim (io_ms={io_ms}) ===")
        engine = OMLXMLXEngine(
            model_path=model_path,
            enable_streaming=True,
            streaming_config={
                "max_experts_in_memory": 2,
                "swap_time_per_expert_ms": io_ms,
                "enable_io_simulation": True,
            },
        )
        engine.load()

        from app.edge_engine.omlx_mlx_engine import StreamingStats
        engine.manager.stats = StreamingStats()

        t0 = time.perf_counter()
        text = engine.generate(prompt, max_tokens=max_tokens)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        tps = max_tokens / elapsed
        stats = engine.get_stats()

        results[f"streaming_io{io_ms}ms"] = {
            "elapsed_s": elapsed,
            "tps": tps,
            "ttft_ms": elapsed / (max_tokens + 1) * 1000,
            "hit_rate": stats.hit_rate,
            "swaps": stats.total_swaps,
            "swap_time_ms": stats.total_swap_time_ms,
            "text": text[:80],
        }
        print(
            f"  TPS: {tps:.1f}, hit_rate: {stats.hit_rate:.2%}, "
            f"swap_time: {stats.total_swap_time_ms:.1f}ms"
        )
        del engine

    return results


def print_summary(results: Dict[str, Any]) -> None:
    """打印对比汇总表."""
    print("\n" + "=" * 80)
    print(f"{'Mode':<25} {'TPS':>8} {'TTFT(ms)':>10} {'Hit%':>8} {'Swaps':>8}")
    print("-" * 80)
    bypass_tps = results.get("bypass", {}).get("tps", 0)
    for mode, r in results.items():
        tps = r.get("tps", 0)
        ttft = r.get("ttft_ms", 0)
        hit = r.get("hit_rate", 0)
        swaps = r.get("swaps", 0)
        delta = f"({tps/bypass_tps*100:.0f}%)" if bypass_tps > 0 else ""
        print(f"{mode:<25} {tps:>7.1f} {delta:>8} {ttft:>9.1f} {hit:>7.1%} {swaps:>8}")
    print("=" * 80)

    # 验证结论
    print("\n--- 验证结论 ---")
    streaming_base = results.get("streaming_cache2", {})
    if streaming_base and bypass_tps > 0:
        ratio = streaming_base["tps"] / bypass_tps
        if ratio > 0.7:
            print(f"✓ streaming 路径正确 work, 性能为 bypass 的 {ratio:.0%}")
        else:
            print(f"⚠ streaming 性能仅为 bypass 的 {ratio:.0%}, 需优化 Python 开销")

    io1 = results.get("streaming_io1.0ms", {})
    if io1:
        ratio_io = io1["tps"] / bypass_tps
        print(f"✓ I/O 模拟 1ms/expert: 性能为 bypass 的 {ratio_io:.0%}")


def main():
    model_path = "models/test_moe_small"
    prompt = "The quick brown fox jumps over the lazy dog. " * 3
    max_tokens = 50

    print(f"Model: {model_path}")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Max tokens: {max_tokens}")

    results = run_benchmark(model_path, prompt, max_tokens)
    print_summary(results)

    # 保存结果
    with open("streaming_vs_bypass_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to streaming_vs_bypass_results.json")


if __name__ == "__main__":
    main()
