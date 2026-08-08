"""TTFT 优化验证: radix cache + 连接池.

测试:
  1. 新 prompt TTFT (无缓存)
  2. 重复 prompt TTFT (radix cache 命中)
  3. 重复 prefix + 不同 suffix TTFT (radix cache 部分命中)
  4. 连接池 vs 每次新连接
  5. HTTP keep-alive 效果
"""
import time
import requests
import json

CLOUD_URL = "http://47.95.250.55:30001"
MODEL = "Qwen3-VL-2B-Instruct"

def measure_ttft(url, payload, session=None, label=""):
    """测 TTFT (time_starttransfer)."""
    s = session or requests
    t0 = time.time()
    try:
        resp = s.post(url, json=payload, stream=True, timeout=30)
        # 读第一个 chunk
        for chunk in resp.iter_content(chunk_size=1):
            if chunk:
                ttft = (time.time() - t0) * 1000
                # 消费完剩余
                for _ in resp.iter_content(chunk_size=1024):
                    pass
                total = (time.time() - t0) * 1000
                return ttft, total
    except Exception as e:
        return -1, -1
    return -1, -1


def test_radix_cache():
    """测试 1: radix cache 效果."""
    print("\n=== 1. Radix Cache 测试 ===")

    # 新 prompt (无缓存)
    payload_new = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 5, "stream": True,
    }
    # 重复 prompt (radix cache 应命中)
    payload_repeat = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 5, "stream": True,
    }

    # 第一次 (冷启动, 填充缓存)
    ttft1, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload_new, label="冷启动")
    print(f"  冷启动 (填充缓存): TTFT={ttft1:.0f}ms")

    # 第二次 (相同 prompt, radix cache 命中)
    ttft2, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload_repeat, label="重复")
    print(f"  重复 prompt (cache命中): TTFT={ttft2:.0f}ms")

    # 第三次 (再次重复, 确认稳定)
    ttft3, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload_repeat, label="重复2")
    print(f"  重复 prompt 2: TTFT={ttft3:.0f}ms")

    # 不同 prompt (无缓存)
    payload_diff = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
        "max_tokens": 5, "stream": True,
    }
    ttft4, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload_diff, label="新prompt")
    print(f"  新 prompt (无缓存): TTFT={ttft4:.0f}ms")

    saved = ttft4 - ttft2
    pct = saved / ttft4 * 100 if ttft4 > 0 else 0
    print(f"  Radix cache 节省: {saved:.0f}ms ({pct:.0f}%)")
    return ttft2, ttft4


def test_prefix_cache():
    """测试 2: 重复 prefix (系统 prompt) + 不同 suffix."""
    print("\n=== 2. Prefix Cache (重复系统 prompt) ===")

    system_msg = "You are a helpful coding assistant. Always provide concise answers with code examples."
    # 第一次 (填充 prefix 缓存)
    payload1 = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "Write a Python hello world"},
        ],
        "max_tokens": 5, "stream": True,
    }
    ttft1, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload1)
    print(f"  首次 (系统prompt+问题1): TTFT={ttft1:.0f}ms")

    # 第二次 (相同系统 prompt + 不同问题, prefix 应命中)
    payload2 = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "Write a Python fibonacci function"},
        ],
        "max_tokens": 5, "stream": True,
    }
    ttft2, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload2)
    print(f"  重复系统prompt+新问题: TTFT={ttft2:.0f}ms")

    saved = ttft1 - ttft2
    print(f"  Prefix cache 节省: {saved:.0f}ms")


def test_connection_pool():
    """测试 3: 连接池 vs 每次新连接."""
    print("\n=== 3. 连接池 vs 新连接 ===")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 3, "stream": True,
    }

    # 每次新连接 (不用 session)
    print("  每次新连接:")
    new_times = []
    for i in range(5):
        ttft, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload)
        new_times.append(ttft)
        print(f"    run{i+1}: TTFT={ttft:.0f}ms")

    # 连接池 (用 session, keep-alive)
    print("  连接池 (keep-alive):")
    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})
    pool_times = []
    for i in range(5):
        ttft, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload, session=session)
        pool_times.append(ttft)
        print(f"    run{i+1}: TTFT={ttft:.0f}ms")

    avg_new = sum(new_times[1:]) / len(new_times[1:])  # 去冷启动
    avg_pool = sum(pool_times[1:]) / len(pool_times[1:])
    saved = avg_new - avg_pool
    print(f"  平均(去冷启动): 新连接={avg_new:.0f}ms, 连接池={avg_pool:.0f}ms, 节省={saved:.0f}ms")

    return avg_pool


def test_combined():
    """测试 4: radix cache + 连接池叠加."""
    print("\n=== 4. 叠加: radix cache + 连接池 ===")

    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})

    system_msg = "You are a helpful assistant."
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "max_tokens": 3, "stream": True,
    }

    # 第一次 (填充缓存 + 建立连接)
    ttft1, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload, session=session)
    print(f"  首次 (冷启动): TTFT={ttft1:.0f}ms")

    # 第二次 (radix cache + 连接池)
    ttft2, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload, session=session)
    print(f"  叠加 (cache+pool): TTFT={ttft2:.0f}ms")

    # 多次测稳定性
    times = []
    for i in range(3):
        ttft, _ = measure_ttft(f"{CLOUD_URL}/v1/chat/completions", payload, session=session)
        times.append(ttft)
    avg = sum(times) / len(times)
    print(f"  叠加稳定: {avg:.0f}ms (runs: {[f'{t:.0f}' for t in times]})")

    return avg


def main():
    print("=" * 60)
    print("TTFT 优化验证 (2B 直连, Mac M4)")
    print("=" * 60)
    print(f"Cloud: {CLOUD_URL}")
    print(f"Model: {MODEL}")

    # 1. radix cache
    cache_ttft, no_cache_ttft = test_radix_cache()

    # 2. prefix cache
    test_prefix_cache()

    # 3. 连接池
    pool_ttft = test_connection_pool()

    # 4. 叠加
    combined_ttft = test_combined()

    # 总结
    print("\n" + "=" * 60)
    print("TTFT 优化总结")
    print("=" * 60)
    print(f"  基准 (无优化):          ~122ms")
    print(f"  + radix cache (重复):   {cache_ttft:.0f}ms (省 {122-cache_ttft:.0f}ms)")
    print(f"  + 连接池 (keep-alive):  {pool_ttft:.0f}ms (省 {122-pool_ttft:.0f}ms)")
    print(f"  + 叠加 (cache+pool):    {combined_ttft:.0f}ms (省 {122-combined_ttft:.0f}ms)")
    print(f"  + MTP 首包预测 (待训练): ~55ms (省 67ms)")
    print(f"  + MTP warm cache:       0ms (重复 prompt)")


if __name__ == "__main__":
    main()
