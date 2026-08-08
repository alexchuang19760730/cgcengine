#!/usr/bin/env python3
"""Quick cloud decode rate measurement (50 tokens, streaming)."""
import asyncio, aiohttp, time, json, statistics

CLOUD_URL = "http://localhost:30000"
prompts = [
    "def fibonacci(n):",
    "import numpy as np\n\narr = np.array([",
    "class Solution:\n    def twoSum(self, nums, target):",
    "def quicksort(arr):",
    "async def fetch_data(url):",
]

async def main():
    async with aiohttp.ClientSession() as session:
        # Warmup
        await session.post(f"{CLOUD_URL}/generate", json={
            "text": "warmup", "sampling_params": {"temperature": 0, "max_new_tokens": 1}, "stream": False
        }, timeout=aiohttp.ClientTimeout(total=10))
        
        all_ttfts = []
        all_decode_tps = []
        all_e2e_tps = []
        
        for prompt in prompts:
            t0 = time.time()
            first_token_ms = None
            token_count = 0
            
            async with session.post(f"{CLOUD_URL}/generate", json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
                "stream": True,
            }, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        text = chunk.get("text", "")
                        elapsed = (time.time() - t0) * 1000
                        if text and first_token_ms is None:
                            first_token_ms = elapsed
                        if text:
                            token_count += 1
                    except:
                        continue
            
            total_ms = (time.time() - t0) * 1000
            decode_ms = total_ms - (first_token_ms or 0)
            decode_tps = token_count / (decode_ms / 1000) if decode_ms > 0 else 0
            e2e_tps = token_count / (total_ms / 1000) if total_ms > 0 else 0
            
            all_ttfts.append(first_token_ms or 0)
            all_decode_tps.append(decode_tps)
            all_e2e_tps.append(e2e_tps)
            
            print(f"{prompt[:35]:35s} | TTFT={first_token_ms:.0f}ms | decode={decode_tps:.1f} tok/s | e2e={e2e_tps:.1f} tok/s | {token_count} tokens")
        
        print(f"\nMedian TTFT: {statistics.median(all_ttfts):.1f}ms")
        print(f"Median decode tok/s: {statistics.median(all_decode_tps):.1f}")
        print(f"Median e2e tok/s: {statistics.median(all_e2e_tps):.1f}")

asyncio.run(main())
