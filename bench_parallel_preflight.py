#!/usr/bin/env python3
"""Minimal self-contained test: parallel preflight (miss不痛) timing verification.

核心指标:
  - TTFT (hit): 预测正确时, 用户拿到首 token 的时间 = spec_time
  - TTFT (miss): 预测错误时, 用户拿到正确首 token 的时间 = cloud_token_arrival_time
  - Miss penalty = TTFT(miss) - cloud_TTFT (无投机基线)
    - 串行: spec_time + cloud_TTFT - cloud_TTFT = spec_time (有惩罚)
    - 并行: max(spec_time, cloud_TTFT) - cloud_TTFT = 0 (无惩罚)

用法:
  python3 bench_parallel_preflight.py
"""
import asyncio
import time

CLOUD_DELAY_SEC = 0.054  # 54ms cloud TTFT
SPEC_TIME_FAST = 0.002   # 2ms (pattern matching)
SPEC_TIME_SLOW = 0.020   # 20ms (local GGUF model)


async def mock_cloud_post(delay: float):
    """模拟云端: 等待 delay 秒后返回 token."""
    await asyncio.sleep(delay)
    return "The"


def mock_speculate(spec_time: float, hit: bool):
    """模拟投机: 等待 spec_time 秒后返回预测."""
    time.sleep(spec_time)  # 同步阻塞
    return "The" if hit else "Because"


# === 串行模式 (旧) ===

async def serial_mode(cloud_delay: float, spec_time: float, hit: bool):
    """串行: 先投机, 再发云端请求."""
    t0 = time.monotonic()

    # 1. Speculate (blocking)
    predicted = await asyncio.get_event_loop().run_in_executor(
        None, mock_speculate, spec_time, hit
    )

    # 2. 发送预测 token (如果有)
    ttft = (time.monotonic() - t0) * 1000  # 预测 token 到达时间

    # 3. Cloud request (AFTER speculation)
    cloud_token = await mock_cloud_post(cloud_delay)
    cloud_arrival = (time.monotonic() - t0) * 1000  # 云端 token 到达时间

    # 如果预测错误 (miss), 正确 token 的到达时间 = cloud_arrival
    # 如果预测正确 (hit), 正确 token 的到达时间 = ttft
    correct_token_ttft = ttft if hit else cloud_arrival

    return {
        "ttft": ttft,               # 首个 token (可能错误) 到达时间
        "correct_ttft": correct_token_ttft,  # 正确 token 到达时间
        "cloud_arrival": cloud_arrival,      # 云端 token 到达时间
        "predicted": predicted,
        "cloud_token": cloud_token,
        "hit": hit,
    }


# === 并行 preflight 模式 (新) ===

async def parallel_preflight(cloud_delay: float, spec_time: float, hit: bool):
    """并行: 云端请求和投机同时启动."""
    t0 = time.monotonic()

    # 1. 立即启动云端请求 (async task)
    loop = asyncio.get_event_loop()
    cloud_future = loop.create_task(mock_cloud_post(cloud_delay))

    # 2. 并行运行投机
    predicted = await loop.run_in_executor(
        None, mock_speculate, spec_time, hit
    )

    # 3. 发送预测 token (如果有)
    ttft = (time.monotonic() - t0) * 1000

    # 4. 等待云端响应 (已在 spec 期间并行计算)
    cloud_token = await cloud_future
    cloud_arrival = (time.monotonic() - t0) * 1000

    correct_token_ttft = ttft if hit else cloud_arrival

    return {
        "ttft": ttft,
        "correct_ttft": correct_token_ttft,
        "cloud_arrival": cloud_arrival,
        "predicted": predicted,
        "cloud_token": cloud_token,
        "hit": hit,
    }


async def run_comparison():
    cloud_ms = CLOUD_DELAY_SEC * 1000

    print(f"\n{'='*85}")
    print(f"Parallel Preflight Benchmark — miss不痛验证")
    print(f"  Cloud TTFT: {cloud_ms:.0f}ms")
    print(f"  Spec time (fast): {SPEC_TIME_FAST*1000:.0f}ms (pattern matching)")
    print(f"  Spec time (slow): {SPEC_TIME_SLOW*1000:.0f}ms (local GGUF model)")
    print(f"{'='*85}\n")

    scenarios = [
        ("HIT  fast spec",  CLOUD_DELAY_SEC, SPEC_TIME_FAST,  True),
        ("MISS fast spec",  CLOUD_DELAY_SEC, SPEC_TIME_FAST,  False),
        ("HIT  slow spec",  CLOUD_DELAY_SEC, SPEC_TIME_SLOW, True),
        ("MISS slow spec",  CLOUD_DELAY_SEC, SPEC_TIME_SLOW, False),
    ]

    print(f"{'Scenario':<20} {'Mode':<10} {'TTFT':>6} {'Correct':>8} {'Cloud':>7} {'Miss Penalty':>14}")
    print(f"{'':20} {'':10} {'(pred)':>6} {'TTFT':>8} {'Arrival':>7} {'vs baseline':>14}")
    print("-" * 75)

    results = {}
    for label, cd, st, hit in scenarios:
        # 串行
        r_s = await serial_mode(cd, st, hit)
        penalty_s = r_s["correct_ttft"] - cloud_ms if not hit else 0
        print(f"{label:<20} {'serial':<10} {r_s['ttft']:>5.0f}ms {r_s['correct_ttft']:>7.0f}ms {r_s['cloud_arrival']:>6.0f}ms {penalty_s:>13.0f}ms")

        # 并行
        r_p = await parallel_preflight(cd, st, hit)
        penalty_p = r_p["correct_ttft"] - cloud_ms if not hit else 0
        print(f"{label:<20} {'parallel':<10} {r_p['ttft']:>5.0f}ms {r_p['correct_ttft']:>7.0f}ms {r_p['cloud_arrival']:>6.0f}ms {penalty_p:>13.0f}ms")
        print()

        results[label] = {"serial": r_s, "parallel": r_p, "penalty_s": penalty_s, "penalty_p": penalty_p}

    # === 总结 ===
    print(f"{'='*85}")
    print("Summary:")
    print(f"{'='*85}")

    for spec_label, spec_key in [("fast (2ms)", "MISS fast spec"), ("slow (20ms)", "MISS slow spec")]:
        r = results[spec_key]
        serial_correct = r["serial"]["correct_ttft"]
        parallel_correct = r["parallel"]["correct_ttft"]
        serial_penalty = r["penalty_s"]
        parallel_penalty = r["penalty_p"]
        savings = serial_correct - parallel_correct

        print(f"\n  MISS with {spec_label} spec:")
        print(f"    Serial  correct-token TTFT: {serial_correct:.0f}ms  (miss penalty: {serial_penalty:.0f}ms)")
        print(f"    Parallel correct-token TTFT: {parallel_correct:.0f}ms  (miss penalty: {parallel_penalty:.0f}ms)")
        print(f"    Savings on miss: {savings:.0f}ms")

        if abs(parallel_penalty) < 5:
            print(f"    ✅ miss penalty ≈ 0ms — parallel preflight working")
        else:
            print(f"    ⚠️  miss penalty = {parallel_penalty:.0f}ms — should be ~0")

    print(f"\n  Key insight:")
    print(f"    Parallel preflight saves spec_time on every miss.")
    print(f"    Fast spec: saves {results['MISS fast spec']['penalty_s']:.0f}ms per miss")
    print(f"    Slow spec: saves {results['MISS slow spec']['penalty_s']:.0f}ms per miss")
    print(f"    With 85.7% hit rate, expected savings per request:")
    fast_save = results["MISS fast spec"]["penalty_s"] * 0.143
    slow_save = results["MISS slow spec"]["penalty_s"] * 0.143
    print(f"      Fast spec: {fast_save:.1f}ms avg")
    print(f"      Slow spec: {slow_save:.1f}ms avg")


if __name__ == "__main__":
    asyncio.run(run_comparison())
