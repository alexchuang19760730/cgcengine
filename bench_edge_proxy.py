#!/usr/bin/env python3
"""edge_first_proxy 缓存命中率 + TTFT 基准测试。

测试场景:
  1. 重复 prompt (L1 cache hit)
  2. 相似前缀 prompt (L2 prefix cache hit)
  3. 相似尾部 prompt (L3 tail cache hit)
  4. 代码补全 prompt (pattern matching)
  5. 聊天 prompt (pattern matching)
  6. 混合并发请求 (Pipeline 效果)
"""
import asyncio
import aiohttp
import time
import json
import sys
import os

PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:30002")
STATS_URL = f"{PROXY_URL}/stats"
HEALTH_URL = f"{PROXY_URL}/health"

# 测试 prompts
TEST_PROMPTS = {
    # === 代码补全 (pattern matching) ===
    "py_def": [
        {"role": "user", "content": "def calculate_sum(a, b):"}
    ],
    "py_class": [
        {"role": "user", "content": "class UserProfile:"}
    ],
    "py_import": [
        {"role": "user", "content": "import json\nimport os\n"}
    ],
    "py_self": [
        {"role": "user", "content": "self.get_user_data()"}
    ],
    "js_const": [
        {"role": "user", "content": "const result = await fetch"}
    ],
    "js_func": [
        {"role": "user", "content": "function handleSubmit(event)"}
    ],
    # === 聊天 (pattern matching) ===
    "write_code": [
        {"role": "user", "content": "Write a Python function to reverse a string"}
    ],
    "fix_bug": [
        {"role": "user", "content": "Fix this bug: IndexError: list index out of range"}
    ],
    "explain": [
        {"role": "user", "content": "Explain how this code works: def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"}
    ],
    # === 重复 prompt (L1 cache test) ===
    "repeat_1": [
        {"role": "user", "content": "Write a function to check if a number is prime"}
    ],
    "repeat_2": [
        {"role": "user", "content": "Write a function to check if a number is prime"}
    ],
    "repeat_3": [
        {"role": "user", "content": "Write a function to check if a number is prime"}
    ],
    # === 相似前缀 (L2 prefix cache test) ===
    "prefix_a": [
        {"role": "user", "content": "You are a helpful assistant. Write a Python function to sort a list using quicksort. Include comments and type hints."}
    ],
    "prefix_b": [
        {"role": "user", "content": "You are a helpful assistant. Write a Python function to sort a list using mergesort. Include comments and type hints."}
    ],
    # === 相似尾部 (L3 tail cache test) ===
    "tail_a": [
        {"role": "user", "content": "Please review this code. The function needs optimization. def process(data): return [x*2 for x in data]"}
    ],
    "tail_b": [
        {"role": "user", "content": "Can you help me? The function needs optimization. def process(data): return [x*2 for x in data]"}
    ],
}


async def send_request(session, url, messages, max_tokens=20, stream=True):
    """发送单个请求，返回 TTFT 和首 token。"""
    payload = {
        "model": "gemma-4-26b-a4b-it",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    t0 = time.monotonic()
    first_token_time = None
    first_token_text = ""
    full_text = ""

    try:
        resp = await session.post(url, json=payload)
        try:
            if stream:
                async for line in resp.content:
                    line = line.decode().strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        choices = obj.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content and first_token_time is None:
                                first_token_time = time.monotonic() - t0
                                first_token_text = content
                            full_text += content
                    except json.JSONDecodeError:
                        continue
            else:
                data = await resp.json()
                choices = data.get("choices", [])
                if choices:
                    full_text = choices[0].get("message", {}).get("content", "")
                    first_token_time = time.monotonic() - t0
                    first_token_text = full_text[:10] if full_text else ""
        finally:
            resp.close()

        total_time = time.monotonic() - t0
        return {
            "ttft_ms": round(first_token_time * 1000, 1) if first_token_time else None,
            "total_ms": round(total_time * 1000, 1),
            "first_token": first_token_text[:30] if first_token_text else "",
            "full_text": full_text[:80] if full_text else "",
            "tokens": len(full_text.split()),
        }
    except Exception as e:
        return {"error": str(e), "ttft_ms": None, "total_ms": round((time.monotonic() - t0) * 1000, 1)}


async def get_stats(session):
    """获取 proxy 统计。"""
    try:
        async with session.get(STATS_URL) as resp:
            return await resp.json()
    except:
        return {}


async def reset_stats(session):
    """重置统计。"""
    try:
        async with session.post(f"{PROXY_URL}/stats/reset") as resp:
            return await resp.json()
    except:
        return {}


async def main():
    print("=" * 70)
    print("edge_first_proxy 缓存命中率 + TTFT 基准测试")
    print("=" * 70)

    # 检查 proxy 是否在线
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(HEALTH_URL, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                health = await resp.json()
                print(f"\nProxy health: {health.get('status', 'unknown')}")
                print(f"  cache_sizes: {health.get('cache_sizes', {})}")
                print(f"  speculation_min_confidence: {health.get('edge_speculation_min_confidence')}")
        except Exception as e:
            print(f"\n❌ Proxy 不可达: {e}")
            print(f"   确认 proxy 在 {PROXY_URL} 运行")
            return

    # 重置统计
    async with aiohttp.ClientSession() as session:
        await reset_stats(session)

    completions_url = f"{PROXY_URL}/v1/chat/completions"

    # === Phase 1: 逐个测试 (验证 cache 填充) ===
    print("\n" + "=" * 70)
    print("Phase 1: 逐个测试 (验证多级缓存填充)")
    print("=" * 70)

    results = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        for name, messages in TEST_PROMPTS.items():
            result = await send_request(session, completions_url, messages, max_tokens=20)
            results[name] = result
            ttft = result.get("ttft_ms")
            ttft_str = f"{ttft:>7.1f}ms" if isinstance(ttft, (int, float)) else "     N/A"
            ft = result.get("first_token", "")
            err = result.get("error", "")
            status = "✅" if not err else "❌"
            print(f"  {status} {name:12s}  TTFT={ttft_str}  first='{ft}'  {err}")

    # 检查统计
    async with aiohttp.ClientSession() as session:
        stats = await get_stats(session)
        print(f"\n  📊 统计 (Phase 1 后):")
        print(f"     total_requests: {stats.get('total_requests', 0)}")
        print(f"     speculated: {stats.get('speculated', 0)}")
        print(f"     cache_hit_l1: {stats.get('cache_hit_l1', 0)}")
        print(f"     cache_hit_l2: {stats.get('cache_hit_l2', 0)}")
        print(f"     cache_hit_l3: {stats.get('cache_hit_l3', 0)}")
        print(f"     cache_miss: {stats.get('cache_miss', 0)}")
        print(f"     cache_hit_rate: {stats.get('cache_hit_rate', 0)}")
        print(f"     speculation_rate: {stats.get('speculation_rate', 0)}")
        print(f"     ttft_ms_avg: {stats.get('ttft_ms_avg', 0)}")
        print(f"     ttft_ms_min: {stats.get('ttft_ms_min', 0)}")
        print(f"     ttft_ms_max: {stats.get('ttft_ms_max', 0)}")
        print(f"     family_counts: {stats.get('family_counts', {})}")

    # === Phase 2: 重复请求 (验证 L1 cache hit) ===
    print("\n" + "=" * 70)
    print("Phase 2: 重复请求 (验证 L1 cache hit)")
    print("=" * 70)

    repeat_messages = [
        {"role": "user", "content": "Write a Python function to check if a string is a palindrome"}
    ]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        for i in range(5):
            result = await send_request(session, completions_url, repeat_messages, max_tokens=15)
            ttft = result.get("ttft_ms")
            ttft_str = f"{ttft:>7.1f}ms" if isinstance(ttft, (int, float)) else "     N/A"
            ft = result.get("first_token", "")
            print(f"  run {i+1}: TTFT={ttft_str}  first='{ft}'")

    # === Phase 3: 并发请求 (Pipeline 效果) ===
    print("\n" + "=" * 70)
    print("Phase 3: 4 路并发 (Pipeline 效果)")
    print("=" * 70)

    concurrent_messages = [
        [{"role": "user", "content": f"Write a Python function to process item {i}"}]
        for i in range(4)
    ]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        t0 = time.monotonic()
        tasks = [send_request(session, completions_url, msgs, max_tokens=30) for msgs in concurrent_messages]
        concurrent_results = await asyncio.gather(*tasks)
        total_time = time.monotonic() - t0

        total_tokens = 0
        for i, r in enumerate(concurrent_results):
            ttft = r.get("ttft_ms")
            ttft_str = f"{ttft:>7.1f}ms" if isinstance(ttft, (int, float)) else "     N/A"
            tokens = r.get("tokens", 0)
            total_tokens += tokens
            err = r.get("error", "")
            print(f"  req {i+1}: TTFT={ttft_str}  tokens={tokens}  {err}")

        aggregate_tok_s = round(total_tokens / total_time, 1) if total_time > 0 else 0
        print(f"\n  📊 4路并发: {total_time:.1f}s, {total_tokens} tokens, {aggregate_tok_s} tok/s aggregate")

    # === 最终统计 ===
    print("\n" + "=" * 70)
    print("最终统计")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:
        final_stats = await get_stats(session)
        print(json.dumps(final_stats, indent=2, ensure_ascii=False))

    # === 检查 cache sizes ===
    async with aiohttp.ClientSession() as session:
        async with session.get(HEALTH_URL) as resp:
            final_health = await resp.json()
            print(f"\nCache sizes: {final_health.get('cache_sizes', {})}")


if __name__ == "__main__":
    asyncio.run(main())
