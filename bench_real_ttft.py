#!/usr/bin/env python3
"""
Real cloud TTFT benchmark for edge_first_proxy.
Measures TTFT for: L1 cache hit, spec HIT, spec MISS, and cold request.
"""
import asyncio
import aiohttp
import time
import json
import sys

PROXY_URL = "http://127.0.0.1:30001"
MODEL = "gemma-4-26b-a4b-it"

async def measure_request(session, prompt, label):
    """Send a streaming request and measure TTFT + total time."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 15,
        "stream": True,
    }
    
    t0 = time.perf_counter()
    first_byte_time = None
    spec_status = None
    predicted_token = None
    first_cloud_token = None
    total_tokens = 0
    
    try:
        async with session.post(f"{PROXY_URL}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                
                if first_byte_time is None:
                    first_byte_time = time.perf_counter()
                
                try:
                    chunk = json.loads(data_str)
                except:
                    continue
                
                # Check for speculation markers
                if "x-cgc-speculation" in chunk:
                    spec_status = chunk.get("x-cgc-speculation")
                    predicted_token = chunk.get("x-cgc-predicted")
                
                # Count content tokens
                choices = chunk.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        total_tokens += 1
                        if first_cloud_token is None and chunk.get("id", "").startswith("edge_") == False and spec_status is not None:
                            first_cloud_token = content
                
                # Check finish
                if choices and choices[0].get("finish_reason"):
                    break
    except Exception as e:
        return {"label": label, "error": str(e)}
    
    ttft_ms = (first_byte_time - t0) * 1000 if first_byte_time else -1
    total_ms = (time.perf_counter() - t0) * 1000
    
    return {
        "label": label,
        "ttft_ms": round(ttft_ms, 1),
        "total_ms": round(total_ms, 1),
        "spec_status": spec_status or "cache_hit",
        "predicted": predicted_token,
        "first_cloud_token": first_cloud_token,
        "total_tokens": total_tokens,
    }

async def main():
    results = []
    
    # Reset stats first
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(f"{PROXY_URL}/stats/reset", timeout=aiohttp.ClientTimeout(total=5))
            print("[INFO] Stats reset")
        except:
            pass
        
        # Phase 1: Cold request (no cache, spec MISS likely)
        print("\n=== Phase 1: Cold request (unique prompt) ===")
        r = await measure_request(session, "refactor this python module for better error handling", "cold_unique")
        results.append(r)
        print(f"  TTFT: {r.get('ttft_ms')}ms, spec: {r.get('spec_status')}, predicted: {r.get('predicted')}")
        
        # Phase 2: Same prompt again (L1 cache hit)
        print("\n=== Phase 2: L1 cache hit (same prompt) ===")
        r = await measure_request(session, "refactor this python module for better error handling", "l1_cache_hit")
        results.append(r)
        print(f"  TTFT: {r.get('ttft_ms')}ms, spec: {r.get('spec_status')}")
        
        # Phase 3: Another L1 cache hit
        print("\n=== Phase 3: L1 cache hit again ===")
        r = await measure_request(session, "refactor this python module for better error handling", "l1_cache_hit_2")
        results.append(r)
        print(f"  TTFT: {r.get('ttft_ms')}ms, spec: {r.get('spec_status')}")
        
        # Phase 4: Different prompts (spec MISS with parallel preflight)
        print("\n=== Phase 4: Spec MISS (parallel preflight, miss penalty ~0ms) ===")
        for prompt in [
            "optimize this database query for performance",
            "add unit tests for the authentication module",
            "convert this function to use async await",
        ]:
            r = await measure_request(session, prompt, f"spec_miss_{prompt[:20]}")
            results.append(r)
            print(f"  TTFT: {r.get('ttft_ms')}ms, spec: {r.get('spec_status')}, predicted: {r.get('predicted')}, cloud_token: {r.get('first_cloud_token')}")
        
        # Phase 5: Send "fix this code" to test spec HIT
        print("\n=== Phase 5: Spec HIT test (fix family) ===")
        for i in range(3):
            r = await measure_request(session, "fix this code", f"fix_hit_{i}")
            results.append(r)
            print(f"  TTFT: {r.get('ttft_ms')}ms, spec: {r.get('spec_status')}, predicted: {r.get('predicted')}")
        
        # Get final stats
        print("\n=== Final Stats ===")
        try:
            async with session.get(f"{PROXY_URL}/stats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                stats = await resp.json()
                print(f"  Total requests: {stats.get('total_requests')}")
                print(f"  Cache hit L1: {stats.get('cache_hit_l1')}")
                print(f"  Spec correct: {stats.get('speculation_correct')}")
                print(f"  Spec wrong: {stats.get('speculation_wrong')}")
                print(f"  TTFT min: {stats.get('ttft_ms_min')}ms")
                print(f"  TTFT avg: {stats.get('ttft_ms_avg')}ms")
                print(f"  TTFT max: {stats.get('ttft_ms_max')}ms")
        except Exception as e:
            print(f"  Error getting stats: {e}")
        
        try:
            async with session.get(f"{PROXY_URL}/acceptance", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                tracker = await resp.json()
                print(f"\n  Tracker state: {tracker.get('state')}")
                print(f"  Accept rate: {tracker.get('global_accept_rate')}")
                print(f"  Samples: {tracker.get('global_samples')}")
                print(f"  Hits: {tracker.get('total_hits')}, Misses: {tracker.get('total_misses')}")
                print(f"  Family counts: {tracker.get('family_sample_counts')}")
        except Exception as e:
            print(f"  Error getting tracker: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TTFT SUMMARY")
    print("=" * 70)
    print(f"{'Label':<35} {'TTFT(ms)':<12} {'Spec':<12} {'Predicted':<15}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<35} ERROR: {r['error']}")
        else:
            print(f"{r['label']:<35} {r['ttft_ms']:<12} {r['spec_status']:<12} {str(r.get('predicted', '')):<15}")
    
    # Calculate averages
    cache_hits = [r["ttft_ms"] for r in results if r.get("spec_status") == "cache_hit" and "ttft_ms" in r]
    spec_misses = [r["ttft_ms"] for r in results if r.get("spec_status") == "miss" and "ttft_ms" in r]
    spec_hits = [r["ttft_ms"] for r in results if r.get("spec_status") == "hit" and "ttft_ms" in r]
    
    print(f"\n  L1 cache hit avg TTFT: {sum(cache_hits)/len(cache_hits):.1f}ms ({len(cache_hits)} samples)" if cache_hits else "  L1 cache hit: none")
    print(f"  Spec MISS avg TTFT:    {sum(spec_misses)/len(spec_misses):.1f}ms ({len(spec_misses)} samples)" if spec_misses else "  Spec MISS: none")
    print(f"  Spec HIT avg TTFT:     {sum(spec_hits)/len(spec_hits):.1f}ms ({len(spec_hits)} samples)" if spec_hits else "  Spec HIT: none")

if __name__ == "__main__":
    asyncio.run(main())
