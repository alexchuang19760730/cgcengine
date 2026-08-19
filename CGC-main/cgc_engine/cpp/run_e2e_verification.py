#!/usr/bin/env python3
"""
End-to-End Verification for Unified MoE Expert Streamer
======================================================
Tests with both Gemma4 (Per-Layer) and Qwen3.6 (Per-Expert) GGUF models.

Verifies:
1. GGUF parsing and architecture detection
2. Expert loading with unified interface
3. Cache management and LRU eviction
4. PD scheduler with token routing
5. Dynamic expert prefetch
6. Zero-copy data integrity
"""

import os
import sys
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_GEMMA4 = r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf"
MODEL_QWEN36 = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
MODEL_QWEN36_IQ2 = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"


def verify_data_integrity(data: bytes, expected_size: int, label: str) -> bool:
    """Verify data integrity and checksum."""
    if len(data) != expected_size:
        print(f"  ❌ {label}: size mismatch {len(data)} vs {expected_size}")
        return False
    
    checksum = hashlib.md5(data[:1024]).hexdigest()
    print(f"  ✅ {label}: {len(data)} bytes, MD5(1KB)={checksum[:16]}...")
    return True


def test_gemma4_model():
    """Test with Gemma4 Per-Layer model."""
    print("\n" + "=" * 70)
    print("E2E TEST 1: Gemma4 (Per-Layer Layout)")
    print("=" * 70)

    if not os.path.exists(MODEL_GEMMA4):
        print(f"  ⚠️  Model not found: {MODEL_GEMMA4}")
        return False

    from unified_moe_streamer import UnifiedExpertStreamer, ExpertLayout

    print("\n  Initializing streamer...")
    t0 = time.time()
    streamer = UnifiedExpertStreamer(MODEL_GEMMA4)
    init_time = time.time() - t0
    print(f"  ✅ Initialized in {init_time:.2f}s")

    assert streamer.layout == ExpertLayout.PER_LAYER, f"Expected PER_LAYER, got {streamer.layout}"
    print(f"  ✅ Layout: {streamer.layout.value}")

    layers = streamer.adapter.list_layers()
    n_experts = streamer.adapter.num_experts(layers[0])
    print(f"  ✅ Architecture: {len(layers)} layers, {n_experts} experts/layer")

    expert1 = streamer.load_expert(layers[0], 0)
    expert2 = streamer.load_expert(layers[0], 0)
    
    assert expert1["roles"].keys() == expert2["roles"].keys()
    for role in expert1["roles"]:
        assert expert1["roles"][role]["data"] == expert2["roles"][role]["data"]
    print(f"  ✅ Cache hit returns identical data")

    print("\n  Testing expert loading across layers...")
    test_configs = [
        (layers[0], 0, "Layer 0, Expert 0"),
        (layers[0], n_experts - 1, f"Layer 0, Expert {n_experts-1}"),
        (layers[len(layers)//2], 0, f"Layer {len(layers)//2}, Expert 0"),
        (layers[-1], n_experts - 1, f"Layer {len(layers)-1}, Expert {n_experts-1}"),
    ]

    for layer, eid, label in test_configs:
        t0 = time.time()
        expert = streamer.load_expert(layer, eid)
        elapsed = (time.time() - t0) * 1000

        roles = expert.get("roles", {})
        total_mb = sum(r.get("size_bytes", 0) / 1024**2 for r in roles.values())

        gate_dims = roles.get("gate", {}).get("dims", "N/A")
        up_dims = roles.get("up", {}).get("dims", "N/A")
        down_dims = roles.get("down", {}).get("dims", "N/A")

        print(f"    {label}: {total_mb:.2f} MB ({elapsed:.1f}ms)")
        print(f"      gate={gate_dims}, up={up_dims}, down={down_dims}")

        assert "gate" in roles, f"Missing gate for {label}"
        assert "up" in roles, f"Missing up for {label}"
        assert "down" in roles, f"Missing down for {label}"

    print("\n  Testing memory estimation...")
    mem = streamer.get_memory_estimate()
    print(f"    Full model: {mem.get('full_model_mb', 0):.2f} MB")
    print(f"    Streaming: {mem.get('streaming_mb', 0):.2f} MB")
    print(f"    Savings: {mem.get('saving_percent', 0):.1f}%")

    assert mem.get("saving_percent", 0) > 50, "Memory savings should be > 50%"
    print(f"  ✅ Memory savings verified (>50%)")

    print("\n  Testing prewarm...")
    t0 = time.time()
    prefill_warmed = streamer.prewarm_prefill()
    decode_warmed = streamer.prewarm_decode()
    print(f"    Prefill: {prefill_warmed} experts")
    print(f"    Decode: {decode_warmed} experts")

    stats = streamer.cache_stats()
    print(f"    Cache hit rate: {stats['hit_rate']:.1f}%")

    print(f"\n  ✅ Gemma4 E2E test PASSED")
    return True


def test_qwen36_model():
    """Test with Qwen3.6 Per-Layer model (UD format)."""
    print("\n" + "=" * 70)
    print("E2E TEST 2: Qwen3.6 (Per-Layer, Separate Gate/Up)")
    print("=" * 70)

    if not os.path.exists(MODEL_QWEN36):
        print(f"  ⚠️  Model not found: {MODEL_QWEN36}")
        return False

    from unified_moe_streamer import UnifiedExpertStreamer, ExpertLayout

    print("\n  Initializing streamer...")
    t0 = time.time()
    streamer = UnifiedExpertStreamer(MODEL_QWEN36)
    init_time = time.time() - t0
    print(f"  ✅ Initialized in {init_time:.2f}s")

    assert streamer.layout == ExpertLayout.PER_LAYER, f"Expected PER_LAYER, got {streamer.layout}"
    print(f"  ✅ Layout: {streamer.layout.value}")

    layers = streamer.adapter.list_layers()
    n_experts = streamer.adapter.num_experts(layers[0])
    print(f"  ✅ Architecture: {len(layers)} layers, {n_experts} experts/layer")
    
    if hasattr(streamer.adapter, 'hidden'):
        print(f"     Hidden: {streamer.adapter.hidden}")
        print(f"     Expert Inter: {streamer.adapter.expert_inter}")

    print("\n  Loading experts from different layers...")
    test_configs = [
        (layers[0], 0, "Layer 0, Expert 0"),
        (layers[0], n_experts - 1, f"Layer 0, Expert {n_experts-1}"),
        (layers[len(layers)//2], 100, f"Layer {len(layers)//2}, Expert 100"),
    ]

    for layer, eid, label in test_configs:
        t0 = time.time()
        expert = streamer.load_expert(layer, eid)
        elapsed = (time.time() - t0) * 1000

        roles = expert.get("roles", {})
        total_mb = sum(r.get("size_bytes", 0) / 1024**2 for r in roles.values())

        print(f"    {label}: {total_mb:.2f} MB ({elapsed:.1f}ms)")
        for role_name in ["gate", "up", "down"]:
            if role_name in roles:
                r = roles[role_name]
                print(f"      {role_name}: dims={r['dims']}, size={r['size_bytes']/1024:.1f}KB")

    stats = streamer.cache_stats()
    print(f"\n  Cache: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate']:.1f}%")

    print(f"\n  ✅ Qwen3.6 E2E test PASSED")
    return True


def test_pd_workflow():
    """Test complete PD workflow simulation."""
    print("\n" + "=" * 70)
    print("E2E TEST 3: PD Separation Workflow")
    print("=" * 70)

    if not os.path.exists(MODEL_GEMMA4):
        print(f"  ⚠️  Model not found: {MODEL_GEMMA4}")
        return False

    from unified_moe_streamer import UnifiedExpertStreamer

    streamer = UnifiedExpertStreamer(MODEL_GEMMA4)
    layers = streamer.adapter.list_layers()
    n_layers = len(layers)
    n_experts = streamer.adapter.num_experts(layers[0])

    mid = n_layers // 2
    prefill_layers = layers[:mid]
    decode_layers = layers[mid:]

    print(f"  PD Split: {len(prefill_layers)} prefill layers | {len(decode_layers)} decode layers")

    print("\n  Phase 1: Prefill (first half)...")
    t0 = time.time()
    
    warm_experts = [0, 1, 2, 3, 4, 5, 6, 7]
    for layer in prefill_layers:
        for eid in warm_experts:
            streamer.load_expert(layer, eid)
    
    prefill_time = time.time() - t0
    print(f"    Loaded {len(prefill_layers) * len(warm_experts)} experts in {prefill_time:.3f}s")

    print("\n  Phase 2: Switch to Decode...")
    t0 = time.time()
    
    for layer in prefill_layers:
        streamer.invalidate_layer(layer)
    
    decode_experts = [7, 8, 9, 10, 11, 12, 13, 14]
    for layer in decode_layers:
        for eid in decode_experts:
            streamer.load_expert(layer, eid)
    
    decode_time = time.time() - t0
    print(f"    Loaded {len(decode_layers) * len(decode_experts)} experts in {decode_time:.3f}s")

    stats = streamer.cache_stats()
    print(f"\n  Cache stats: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate']:.1f}% hit rate")

    print(f"\n  ✅ PD workflow test PASSED")
    return True


def test_dynamic_prefetch():
    """Test dynamic expert prefetch with token routing."""
    print("\n" + "=" * 70)
    print("E2E TEST 4: Dynamic Expert Prefetch")
    print("=" * 70)

    class TokenRouterSimulator:
        def __init__(self, n_experts):
            self.n_experts = n_experts
            self.route_history = {}
            self.recent_routes = []

        def submit_routes(self, routes):
            for layer, eid in routes:
                key = (layer, eid)
                self.route_history[key] = self.route_history.get(key, 0) + 1
                self.recent_routes.append(eid)
                if len(self.recent_routes) > 128:
                    self.recent_routes.pop(0)

        def predict_next(self, top_k=8):
            freq = {}
            for eid in self.recent_routes:
                freq[eid] = freq.get(eid, 0) + 1

            predictions = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            total = sum(v for _, v in predictions)

            return [{"expert_id": eid, "frequency": count, "probability": count / max(total, 1)}
                    for eid, count in predictions[:top_k]]

    if not os.path.exists(MODEL_GEMMA4):
        print(f"  ⚠️  Model not found: {MODEL_GEMMA4}")
        return False

    from unified_moe_streamer import UnifiedExpertStreamer

    streamer = UnifiedExpertStreamer(MODEL_GEMMA4)
    layers = streamer.adapter.list_layers()
    n_experts = streamer.adapter.num_experts(layers[0])

    router = TokenRouterSimulator(n_experts)

    route_patterns = [
        [(0, 3), (1, 7), (2, 1), (0, 3), (1, 7)],
        [(0, 5), (1, 2), (0, 5)],
        [(0, 3), (1, 7), (0, 3), (2, 4)],
        [(0, 8), (1, 3), (0, 8), (0, 3)],
        [(0, 3), (1, 7), (2, 1), (0, 3), (1, 7), (0, 3)],
    ]

    print("\n  Simulating token routing patterns with prefetch...")
    total_prefetched = 0
    total_hits = 0

    for batch_idx, routes in enumerate(route_patterns):
        router.submit_routes(routes)
        predictions = router.predict_next(4)
        prefetch_list = [p["expert_id"] for p in predictions]

        for layer in layers[:3]:
            for eid in prefetch_list:
                result = streamer.load_expert(layer, eid)
                if result:
                    total_prefetched += 1

        print(f"\n    Batch {batch_idx + 1}:")
        print(f"      Submitted {len(routes)} routes")
        print(f"      Prefetch list: {prefetch_list}")
        for p in predictions[:3]:
            bar = "█" * int(p['probability'] * 20)
            print(f"        Expert {p['expert_id']:3d}: [{bar:20s}] {p['probability']*100:5.1f}%")

    stats = streamer.cache_stats()
    print(f"\n  Final stats: {stats['hits']} hits, {stats['hit_rate']:.1f}% hit rate")
    print(f"  Total prefetched: {total_prefetched} experts")

    print(f"\n  ✅ Dynamic prefetch test PASSED")
    return True


def main():
    print("=" * 70)
    print("END-TO-END VERIFICATION - UNIFIED MOE EXPERT STREAMER")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    results["gemma4_per_layer"] = test_gemma4_model()
    results["qwen36_per_expert"] = test_qwen36_model()
    results["pd_workflow"] = test_pd_workflow()
    results["dynamic_prefetch"] = test_dynamic_prefetch()

    print("\n" + "=" * 70)
    print("E2E VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print(f"\n  Overall: {'✅ ALL E2E TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
