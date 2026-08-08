"""分析 sglang 流式 chunk 结构 + 测试 WebSocket Pipeline 方案."""
import requests
import time
import json
import asyncio
from collections import defaultdict

URL = "http://39.106.118.206:30001"
MODEL = "/data/models/gemma-4-26b-a4b-it"


def analyze_streaming():
    """分析 sglang 流式 chunk 结构."""
    print("=" * 60)
    print(" 1. 分析 sglang 流式 chunk 结构")
    print("=" * 60)

    r = requests.post(f"{URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 30"}],
        "max_tokens": 80, "stream": True
    }, stream=True, timeout=30)

    chunks = []
    token_count = 0
    chunk_times = []
    t0 = time.time()
    last_chunk_time = t0
    for line in r.iter_lines():
        if line:
            l = line.decode("utf-8")
            if l.startswith("data: ") and l != "data: [DONE]":
                now = time.time()
                chunk_times.append(now - last_chunk_time)
                last_chunk_time = now
                chunks.append(l)
                try:
                    data = json.loads(l[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        token_count += 1
                except:
                    pass
    dt = time.time() - t0
    print(f"  总 chunks: {len(chunks)}")
    print(f"  总 tokens: {token_count}")
    print(f"  时间: {dt:.3f}s")
    print(f"  tok/s: {token_count/dt:.1f}")
    print(f"  每 chunk 间隔: {sum(chunk_times)/len(chunk_times)*1000:.1f}ms (avg)")
    print(f"  首 chunk 延迟: {chunk_times[0]*1000:.1f}ms" if chunk_times else "  N/A")
    return token_count, dt


def test_batch_pipeline():
    """方案 D: HTTP 批量请求 (伪 Pipeline).
    
    发送多个小 max_tokens 请求, 模拟批量流式.
    """
    print()
    print("=" * 60)
    print(" 2. 批量流式 Pipeline (HTTP batch)")
    print("=" * 60)

    # 策略: 发 1 个大请求, 但用 stream=True
    # vs 发多个小请求 (每个 max_tokens=8)
    # 对比吞吐

    # A. 单个大流式请求
    print("\n  --- A. 单个流式请求 (max_tokens=100) ---")
    t0 = time.time()
    r = requests.post(f"{URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a story"}],
        "max_tokens": 100, "stream": True
    }, stream=True, timeout=60)
    tokens_a = 0
    for line in r.iter_lines():
        if line:
            l = line.decode("utf-8")
            if l.startswith("data: ") and l != "data: [DONE]":
                tokens_a += 1
    dt_a = time.time() - t0
    print(f"  {tokens_a} tokens, {dt_a:.2f}s, {tokens_a/dt_a:.1f} tok/s")

    # B. 多个小非流式请求 (batch=10 tokens each)
    print("\n  --- B. 多个小非流式请求 (batch=10, 10次) ---")
    t0 = time.time()
    total_tokens = 0
    messages = [{"role": "user", "content": "Write a story"}]
    for i in range(10):
        r = requests.post(f"{URL}/v1/chat/completions", json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 10, "stream": False
        }, timeout=30)
        data = r.json()
        completion = data.get("usage", {}).get("completion_tokens", 0)
        total_tokens += completion
        # 把生成的 token 加到 messages 里继续
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "continue"})
    dt_b = time.time() - t0
    print(f"  {total_tokens} tokens, {dt_b:.2f}s, {total_tokens/dt_b:.1f} tok/s")

    # C. 单个非流式请求 (baseline)
    print("\n  --- C. 单个非流式请求 (max_tokens=100, baseline) ---")
    t0 = time.time()
    r = requests.post(f"{URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a story"}],
        "max_tokens": 100, "stream": False
    }, timeout=60)
    dt_c = time.time() - t0
    completion = r.json().get("usage", {}).get("completion_tokens", 0)
    print(f"  {completion} tokens, {dt_c:.2f}s, {completion/dt_c:.1f} tok/s")

    return tokens_a/dt_a, total_tokens/dt_b, completion/dt_c


async def test_websocket_pipeline():
    """方案 A: WebSocket 真 Pipeline.
    
    用 asyncio 并发发送多个流式请求 (HTTP/2 多路复用模拟).
    """
    print()
    print("=" * 60)
    print(" 3. asyncio 并发流式 Pipeline")
    print("=" * 60)

    import aiohttp

    async def fetch_stream(session, prompt, max_tokens, result_queue):
        """异步获取流式响应."""
        async with session.post(f"{URL}/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True
        }) as resp:
            tokens = 0
            first_token_time = None
            t0 = time.time()
            async for line in resp.content:
                l = line.decode("utf-8").strip()
                if l.startswith("data: ") and l != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.time() - t0
                    tokens += 1
            dt = time.time() - t0
            await result_queue.put((tokens, dt, first_token_time))
            return tokens, dt, first_token_time

    # 策略: 并发 4 个流式请求 (模拟 Pipeline batch)
    print("\n  --- 并发 4 个流式请求 (HTTP keep-alive 复用) ---")
    
    connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.time()
        tasks = []
        for i in range(4):
            task = asyncio.create_task(fetch_stream(
                session, f"Write story part {i+1}", 50, asyncio.Queue()
            ))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_dt = time.time() - t0
        total_tokens = sum(r[0] for r in results)
        
        print(f"  4 个并发请求完成")
        print(f"  总 tokens: {total_tokens}")
        print(f"  总时间: {total_dt:.2f}s")
        print(f"  整体吞吐: {total_tokens/total_dt:.1f} tok/s")
        for i, (tokens, dt, ttft) in enumerate(results):
            print(f"    请求 {i+1}: {tokens} tokens, TTFT={ttft:.3f}s, {tokens/dt:.1f} tok/s")

    # 策略: Pipeline 连续请求 (前一个还没完就发下一个)
    print("\n  --- Pipeline 连续流式 (重叠请求) ---")
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=10)) as session:
        t0 = time.time()
        # 发 1 个长流式 + 同时发 2 个短流式 (重叠)
        task_long = asyncio.create_task(fetch_stream(
            session, "Write a detailed essay", 100, asyncio.Queue()
        ))
        await asyncio.sleep(0.05)  # 稍微延迟
        task_short1 = asyncio.create_task(fetch_stream(
            session, "Quick reply 1", 30, asyncio.Queue()
        ))
        task_short2 = asyncio.create_task(fetch_stream(
            session, "Quick reply 2", 30, asyncio.Queue()
        ))
        
        results = await asyncio.gather(task_long, task_short1, task_short2)
        total_dt = time.time() - t0
        total_tokens = sum(r[0] for r in results)
        print(f"  重叠请求完成")
        print(f"  总 tokens: {total_tokens}")
        print(f"  总时间: {total_dt:.2f}s")
        print(f"  整体吞吐: {total_tokens/total_dt:.1f} tok/s")


def main():
    print("=" * 60)
    print(" WebSocket 真 Pipeline 可行性测试")
    print(" Host1 Gemma 4 26B-A4B (cuda-graph + MTP + CGC)")
    print("=" * 60)

    # 1. 分析当前流式结构
    analyze_streaming()

    # 2. 批量 Pipeline 测试
    test_batch_pipeline()

    # 3. asyncio 并发测试 (模拟 WebSocket Pipeline)
    asyncio.run(test_websocket_pipeline())

    print()
    print("=" * 60)
    print(" 总结")
    print("=" * 60)


if __name__ == "__main__":
    main()
