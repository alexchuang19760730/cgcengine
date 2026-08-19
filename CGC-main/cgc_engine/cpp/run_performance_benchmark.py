#!/usr/bin/env python3
"""
MoE Expert Streaming Performance Benchmark

测试内容:
1. Expert loading latency (single expert)
2. Cache hit rate (repeated access)
3. PD separation prewarm time
4. Memory savings analysis
5. Multi-layer batch loading
"""

import os
import sys
import time
import sys
from collections import defaultdict

# Import unified streamer
sys.path.insert(0, os.path.dirname(__file__))
from unified_moe_streamer import (
    UnifiedExpertStreamer, ExpertLayout, detect_layout
)

def run_performance_benchmark():
    """运行性能基准测试."""
    print("=" * 80)
    print("MOE EXPERT STREAMING PERFORMANCE BENCHMARK")
    print("=" * 80)

    model_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return 1

    # Test 1: Initialization
    print("\n📊 Test 1: Initialization Performance")
    print("-" * 60)

    start = time.time()
    streamer = UnifiedExpertStreamer(model_path)
    init_time = time.time() - start

    print(f"  Initialize time: {init_time:.2f}s")
    print(f"  Layout: {streamer.layout.value}")

    # Test 2: Single Expert Loading
    print("\n📈 Test 2: Single Expert Loading Latency")
    print("-" * 60)

    test_configs = [
        (0, 0, "Layer 0, Expert 0"),
        (0, 64, "Layer 0, Expert 64 (middle)"),
        (0, 127, "Layer 0, Expert 127 (last)"),
        (15, 0, "Layer 15, Expert 0 (mid layer)"),
        (29, 127, "Layer 29, Expert 127 (last layer)"),
    ]

    latencies = []
    for layer, expert_id, desc in test_configs:
        # Clear cache to force miss
        streamer._cache.clear()
        streamer._hits = 0
        streamer._misses = 0

        t0 = time.perf_counter()
        expert = streamer.load_expert(layer, expert_id)
        t1 = time.perf_counter()

        if expert:
            roles = expert.get("roles", {})
            total_mb = sum(r.get("size_bytes", 0) / 1024**2 for r in roles.values())
            latency_ms = (t1 - t0) * 1000
            latencies.append(latency_ms)
            print(f"  {desc}: {latency_ms:.1f}ms ({total_mb:.2f} MB)")
        else:
            print(f"  {desc}: FAILED")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print(f"\n  Summary: avg={avg_latency:.1f}ms, min={min_latency:.1f}ms, max={max_latency:.1f}ms")

    # Test 3: Cache Performance
    print("\n💾 Test 3: Cache Hit Rate")
    print("-" * 60)

    streamer._cache.clear()
    streamer._hits = 0
    streamer._misses = 0

    # First access (miss)
    streamer.load_expert(0, 0)
    streamer.load_expert(0, 1)
    streamer.load_expert(5, 0)

    # Second access (should be hit)
    for _ in range(10):
        streamer.load_expert(0, 0)
        streamer.load_expert(0, 1)
        streamer.load_expert(5, 0)

    # Third access
    for _ in range(20):
        streamer.load_expert(0, 0)

    stats = streamer.cache_stats()
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.1f}%")
    print(f"  Cached: {stats['cached']} experts")

    # Test 4: Batch Layer Loading
    print("\n📦 Test 4: Batch Layer Loading")
    print("-" * 60)

    streamer._cache.clear()
    streamer._hits = 0
    streamer._misses = 0

    layers = streamer.adapter.list_layers()
    if hasattr(streamer.adapter, 'top_k'):
        top_k = streamer.adapter.top_k
    else:
        top_k = 8

    batch_times = []
    for layer in layers[:10]:  # Test first 10 layers
        t0 = time.perf_counter()
        experts = streamer.load_layer(layer, list(range(top_k)))
        t1 = time.perf_counter()
        batch_times.append((t1 - t0) * 1000)
        if layer < 5:
            print(f"  Layer {layer}: {len(experts)} experts in {(t1-t0)*1000:.1f}ms")

    if batch_times:
        avg_batch = sum(batch_times) / len(batch_times)
        print(f"\n  Average batch time (8 experts/layer): {avg_batch:.1f}ms")
        print(f"  Throughput: {top_k / (avg_batch / 1000):.0f} experts/second per layer")

    # Test 5: PD Separation
    print("\n⚡ Test 5: PD Separation Prewarm")
    print("-" * 60)

    streamer._cache.clear()
    streamer._hits = 0
    streamer._misses = 0

    # Prefill phase
    t0 = time.perf_counter()
    prefill_warmed = streamer.prewarm_prefill()
    t1 = time.perf_counter()
    prefill_time = t1 - t0

    print(f"  Prefill Prewarm:")
    print(f"    Experts loaded: {prefill_warmed}")
    print(f"    Time: {prefill_time*1000:.0f}ms ({prefill_time:.2f}s)")
    print(f"    Avg per expert: {prefill_time / prefill_warmed * 1000:.1f}ms")

    # Decode phase
    t0 = time.perf_counter()
    decode_warmed = streamer.prewarm_decode()
    t1 = time.perf_counter()
    decode_time = t1 - t0

    print(f"\n  Decode Prewarm:")
    print(f"    Experts loaded: {decode_warmed}")
    print(f"    Time: {decode_time*1000:.0f}ms ({decode_time:.2f}s)")
    print(f"    Avg per expert: {decode_time / decode_warmed * 1000:.1f}ms")

    # Test 6: Memory Analysis
    print("\n📊 Test 6: Memory Savings Analysis")
    print("-" * 60)

    mem = streamer.get_memory_estimate()
    print(f"  Layout: {mem['layout']}")
    print(f"  Full expert weights: {mem['full_model_mb']:.0f} MB ({mem['full_model_mb']/1024:.2f} GB)")
    print(f"  With streaming: {mem['streaming_mb']:.0f} MB ({mem['streaming_mb']/1024:.2f} GB)")
    print(f"  Savings: {mem['saving_percent']:.1f}%")

    # Test 7: Cache Eviction
    print("\n🔄 Test 7: Cache Eviction Test")
    print("-" * 60)

    streamer._cache.clear()
    streamer.max_cache_size = 10  # Small cache for testing

    # Load 15 experts (should trigger eviction)
    for e in range(15):
        streamer.load_expert(0, e)

    stats_after = streamer.cache_stats()
    print(f"  After loading 15 experts (max_cache=10):")
    print(f"  Cached: {stats_after['cached']} (expected: 10)")
    print(f"  Misses: {stats_after['misses']} (expected: 15)")

    # Test 8: Multi-Layer Expert (Per-Layer Layout)
    print("\n🎯 Test 8: Per-Layer Layout Specifics")
    print("-" * 60)

    if streamer.layout == ExpertLayout.PER_LAYER and hasattr(streamer.adapter, 'hidden'):
        print(f"  Gemma4 Architecture:")
        print(f"    Hidden size: {streamer.adapter.hidden}")
        print(f"    Expert intermediate: {streamer.adapter.expert_inter}")
        print(f"    Num experts: {streamer.adapter.num_experts_val}")
        print(f"    Top-K: {streamer.adapter.top_k}")
        print(f"    Total layers: {len(streamer.adapter._layer_list)}")

        # Calculate per-expert sizes
        expert = streamer.load_expert(0, 0)
        if expert:
            roles = expert.get("roles", {})
            down_mb = roles.get("down", {}).get("size_bytes", 0) / 1024**2
            gate_mb = roles.get("gate", {}).get("size_bytes", 0) / 1024**2
            up_mb = roles.get("up", {}).get("size_bytes", 0) / 1024**2
            total_mb = down_mb + gate_mb + up_mb

            print(f"\n  Per-Expert Breakdown:")
            print(f"    Gate: {gate_mb:.2f} MB [{roles.get('gate', {}).get('dims', [])}]")
            print(f"    Up: {up_mb:.2f} MB [{roles.get('up', {}).get('dims', [])}]")
            print(f"    Down: {down_mb:.2f} MB [{roles.get('down', {}).get('dims', [])}]")
            print(f"    Total: {total_mb:.2f} MB")

            # Calculate per-layer total
            per_layer_total = total_mb * streamer.adapter.num_experts_val
            per_layer_streaming = total_mb * streamer.adapter.top_k
            print(f"\n  Per-Layer Analysis:")
            print(f"    Full layer (all {streamer.adapter.num_experts_val} experts): {per_layer_total:.0f} MB")
            print(f"    Streaming ({streamer.adapter.top_k} experts): {per_layer_streaming:.1f} MB")
            print(f"    Per-layer savings: {(1 - per_layer_streaming/per_layer_total)*100:.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK COMPLETE")
    print("=" * 80)

    print(f"""
📈 Key Metrics:
  Single Expert Load: {avg_latency:.1f}ms (avg)
  Cache Hit Rate: {stats['hit_rate']:.1f}%
  Batch Loading: {avg_batch:.1f}ms/layer (8 experts)
  Prefill Prewarm: {prefill_time*1000:.0f}ms ({prefill_warmed} experts)
  Decode Prewarm: {decode_time*1000:.0f}ms ({decode_warmed} experts)
  Memory Savings: {mem['saving_percent']:.1f}%

💡 Recommendations:
  1. Cache size: 64-128 experts for optimal hit rate
  2. Prewarm strategy: Load top-K experts for active layers
  3. PD separation: Switch at {len(layers)//2} layers (mid-point)
  4. GPU transfer: Consider staging buffer for async upload
""")

    return 0


if __name__ == "__main__":
    sys.exit(run_performance_benchmark())
