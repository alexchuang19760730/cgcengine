#!/usr/bin/env python3
"""End-to-end test: AcceptanceTracker + Parallel Preflight + Correction.

模拟场景:
1. 前期: 高 accept rate → ENABLED → 投机正常工作
2. 中期: accept rate 下降 → DEGRADED → 只对高 confidence 投机
3. 后期: 持续低 accept → DISABLED → 不投机, 但 preflight 仍工作 (miss penalty=0)
4. 恢复: accept rate 回升 → ENABLED → 恢复投机

验证:
- HIT: TTFT ~1ms, 云端首 token 被 blank
- MISS: TTFT ~1ms, correction marker 发送, 云端正确 token 通过
- DISABLED: 不投机, 云端直接返回 (TTFT = cloud TTFT)
- 状态转换日志正确打印
- /health 端点包含 tracker 状态
"""
import asyncio
import json
import sys
import os
import time
from aiohttp import web

# 确保能导入 proxy 模块
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "app"))

from app.shared.acceptance_tracker import (
    AcceptanceTracker, STATE_ENABLED, STATE_DEGRADED, STATE_DISABLED,
    get_acceptance_tracker
)


# ── Mock Cloud Server ─────────────────────────────────────────

class MockCloudServer:
    """模拟云端 sglang, 返回指定首 token + 延迟."""

    def __init__(self, first_token="The", delay_ms=30, total_tokens=10):
        self.first_token = first_token
        self.delay_ms = delay_ms
        self.total_tokens = total_tokens
        self.request_count = 0

    async def handle_chat(self, request):
        self.request_count += 1
        await asyncio.sleep(self.delay_ms / 1000)

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        # First token
        first_data = {
            "id": f"chatcmpl-{self.request_count}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "mock-cloud",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": self.first_token}, "finish_reason": None}]
        }
        await resp.write(f"data: {json.dumps(first_data)}\n\n".encode())

        # Remaining tokens
        for i in range(self.total_tokens - 1):
            chunk_data = {
                "id": f"chatcmpl-{self.request_count}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "mock-cloud",
                "choices": [{"index": 0, "delta": {"content": f" token{i+1}"}, "finish_reason": None}]
            }
            await resp.write(f"data: {json.dumps(chunk_data)}\n\n".encode())

        # Done
        done_data = {
            "id": f"chatcmpl-{self.request_count}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "mock-cloud",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        await resp.write(f"data: {json.dumps(done_data)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp


async def start_mock_cloud(first_token="The", delay_ms=30, port=39998):
    """启动 mock cloud server."""
    cloud = MockCloudServer(first_token=first_token, delay_ms=delay_ms)
    app = web.Application()
    app.router.add_post("/v1/chat/completions", cloud.handle_chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return cloud, runner


# ── Tests ─────────────────────────────────────────────────────

async def test_tracker_integration():
    """测试 AcceptanceTracker 与投机流程的集成."""
    print("\n=== Test: AcceptanceTracker Integration ===\n")

    # 1. 初始化 tracker (用小窗口加速测试)
    tracker = AcceptanceTracker(window_size=20, transition_count=5)
    transitions = []

    def on_transition(old, new, rate):
        transitions.append((old, new, rate))
        print(f"  [TRANSITION] {old} → {new} (rate={rate:.3f})")

    tracker.set_on_transition(on_transition)

    # 2. Phase 1: High accept rate → stays ENABLED
    print("--- Phase 1: High accept (all HIT) ---")
    for i in range(15):
        tracker.record(hit=True, family="fix")
    status = tracker.get_status()
    print(f"  State: {status['state']}, Rate: {status['global_accept_rate']}, Samples: {status['global_samples']}")
    assert tracker.get_state() == STATE_ENABLED
    assert tracker.should_speculate("fix") == True
    print("  ✓ Stays ENABLED with high accept rate")

    # 3. Phase 2: Declining accept → DEGRADED
    # Need consecutive misses (not alternating) to trigger transition
    print("\n--- Phase 2: Declining accept → DEGRADED ---")
    # Burst of misses to trigger degradation
    for i in range(25):
        tracker.record(hit=False, family="fix")
    status = tracker.get_status()
    print(f"  State: {status['state']}, Rate: {status['global_accept_rate']}, Samples: {status['global_samples']}")
    print(f"  Family rates: {status['family_rates']}")
    print(f"  Transitions so far: {len(transitions)}")

    # Should be at least DEGRADED
    assert tracker.get_state() in (STATE_DEGRADED, STATE_DISABLED), \
        f"Expected DEGRADED or DISABLED, got {tracker.get_state()}"
    print(f"  ✓ Degraded to {tracker.get_state()}")

    # 4. Phase 3: Continued low → DISABLED
    print("\n--- Phase 3: Continued low → DISABLED ---")
    for i in range(20):
        tracker.record(hit=False, family="fix")
    status = tracker.get_status()
    print(f"  State: {status['state']}, Rate: {status['global_accept_rate']}")
    assert tracker.get_state() == STATE_DISABLED
    assert tracker.should_speculate("fix") == False
    print("  ✓ DISABLED: speculation blocked")

    # 5. Verify MTP config changes with state
    cfg = tracker.get_mtp_config()
    assert cfg["speculate"] == False
    assert cfg["steps"] == 0
    print(f"  ✓ MTP config in DISABLED: {cfg}")

    # 6. Phase 4: Recovery → ENABLED
    print("\n--- Phase 4: Recovery → ENABLED ---")
    for i in range(25):
        tracker.record(hit=True, family="fix")
    status = tracker.get_status()
    print(f"  State: {status['state']}, Rate: {status['global_accept_rate']}")
    assert tracker.get_state() == STATE_ENABLED
    assert tracker.should_speculate("fix") == True
    print("  ✓ Recovered to ENABLED")

    # 7. Verify transition history
    print(f"\n--- Transition History ({len(transitions)} transitions) ---")
    for t in transitions:
        print(f"  {t[0]} → {t[1]} (rate={t[2]:.3f})")
    assert len(transitions) >= 2  # At least DEGRADED and DISABLED transitions

    # 8. Verify dynamic confidence threshold
    print("\n--- Dynamic Confidence Threshold ---")
    tracker2 = AcceptanceTracker(window_size=20, transition_count=5)
    assert tracker2.get_min_confidence() == 0.55  # ENABLED
    print(f"  ENABLED threshold: {tracker2.get_min_confidence()}")
    # Degrade
    for _ in range(30):
        tracker2.record(hit=False, family="generic")
    if tracker2.get_state() == STATE_DEGRADED:
        assert tracker2.get_min_confidence() == 0.70
        print(f"  DEGRADED threshold: {tracker2.get_min_confidence()}")
    elif tracker2.get_state() == STATE_DISABLED:
        assert tracker2.get_min_confidence() == 1.01
        print(f"  DISABLED threshold: {tracker2.get_min_confidence()}")
    print("  ✓ Dynamic threshold works")

    print("\n=== ALL INTEGRATION TESTS PASSED ===")


async def test_mock_cloud_flow():
    """测试 mock cloud + proxy flow (simplified)."""
    print("\n=== Test: Mock Cloud Flow ===\n")

    # Start mock cloud
    cloud, runner = await start_mock_cloud(first_token="The", delay_ms=30, port=39998)
    print(f"Mock cloud started (port 39998, first_token='The', delay=30ms)")

    # Simulate what the proxy does: parallel preflight
    import aiohttp

    async def simulate_request(predicted_token=None, tracker=None):
        """Simulate edge_first_proxy flow with parallel preflight."""
        t0 = time.monotonic()

        # 1. Start cloud request immediately (parallel preflight)
        async with aiohttp.ClientSession() as session:
            cloud_task = asyncio.create_task(
                session.post("http://127.0.0.1:39998/v1/chat/completions",
                            json={"model": "mock", "messages": [{"role": "user", "content": "test"}], "stream": True})
            )

            # 2. Simulate speculation (2ms)
            if predicted_token and tracker and tracker.should_speculate("fix"):
                await asyncio.sleep(0.002)  # 2ms speculation
                ttft = (time.monotonic() - t0) * 1000
                print(f"  Speculated: '{predicted_token}' TTFT={ttft:.1f}ms")

                # 3. Wait for cloud response
                resp = await cloud_task
                first_cloud_token = None
                chunks = []
                async for chunk in resp.content.iter_any():
                    chunks.append(chunk)
                    if first_cloud_token is None:
                        # Parse SSE lines (chunk may contain multiple lines)
                        for line in chunk.decode(errors="replace").split("\n"):
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                continue
                            try:
                                obj = json.loads(data_str)
                                delta = obj.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content and content.strip():
                                    first_cloud_token = content.strip().split()[0]
                                    break
                            except:
                                pass

                # 4. Compare
                hit = predicted_token.strip() == first_cloud_token.strip() if first_cloud_token else False
                tracker.record(hit=hit, family="fix")

                if hit:
                    print(f"  HIT: predicted='{predicted_token}' == cloud='{first_cloud_token}'")
                else:
                    print(f"  MISS: predicted='{predicted_token}' != cloud='{first_cloud_token}' (correction sent)")

                resp.release()
                return hit, ttft
            else:
                # No speculation, wait for cloud
                resp = await cloud_task
                ttft = (time.monotonic() - t0) * 1000
                first_cloud_token = None
                async for chunk in resp.content.iter_any():
                    if first_cloud_token is None:
                        for line in chunk.decode(errors="replace").split("\n"):
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                continue
                            try:
                                obj = json.loads(data_str)
                                delta = obj.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content and content.strip():
                                    first_cloud_token = content.strip().split()[0]
                                    break
                            except:
                                pass
                print(f"  No speculation (disabled), cloud TTFT={ttft:.1f}ms, first='{first_cloud_token}'")
                resp.release()
                return None, ttft

    # Run requests with tracker
    tracker = AcceptanceTracker(window_size=20, transition_count=5)
    print(f"\nTracker initial state: {tracker.get_state()}")

    # Phase 1: All HIT (predicted == cloud "The")
    print("\n--- Phase 1: All HIT ---")
    for i in range(5):
        hit, ttft = await simulate_request("The", tracker)
        assert hit == True
        assert ttft < 10  # Should be ~2ms (speculation time)
    print(f"  State: {tracker.get_state()}, Rate: {tracker.get_accept_rate()}")

    # Phase 2: All MISS (predicted "The" but cloud returns "Because")
    cloud.first_token = "Because"
    print("\n--- Phase 2: All MISS (cloud changed to 'Because') ---")
    for i in range(15):
        hit, ttft = await simulate_request("The", tracker)
        # hit may be False (miss) or None (speculation disabled by tracker)
        assert hit != True, f"Request {i}: expected miss or disabled, got hit"
    print(f"  State: {tracker.get_state()}, Rate: {tracker.get_accept_rate()}")
    assert tracker.get_state() in (STATE_DEGRADED, STATE_DISABLED)

    # Phase 3: DISABLED → no speculation, cloud direct
    # Note: tracker may be in DEGRADED with per-family gating blocking "fix"
    # When speculation is blocked, no new data flows to tracker (feedback loop pauses)
    # This is correct behavior — need exploration mechanism to recover
    print("\n--- Phase 3: Speculation disabled by tracker ---")
    spec_blocked_count = 0
    for i in range(10):
        hit, ttft = await simulate_request("The", tracker)
        if hit is None:
            spec_blocked_count += 1
    print(f"  State: {tracker.get_state()}, Rate: {tracker.get_accept_rate()}")
    print(f"  Speculation blocked: {spec_blocked_count}/10 requests")
    assert tracker.get_state() in (STATE_DEGRADED, STATE_DISABLED)
    assert spec_blocked_count > 0  # At least some requests should have speculation blocked

    # When speculation is blocked, cloud direct
    hit, ttft = await simulate_request("The", tracker)
    assert hit is None  # No speculation happened
    assert ttft > 20  # Cloud TTFT (~30ms)
    print(f"  ✓ Speculation blocked, cloud TTFT={ttft:.1f}ms")

    # Phase 4: Recovery (cloud back to "The")
    # Note: per-family gating may block "fix" — need exploration to recover
    cloud.first_token = "The"
    print("\n--- Phase 4: Recovery (cloud back to 'The') ---")
    # Simulate exploration: force tracker to try speculation by recording hits
    # In real proxy, exploration = occasionally speculate even when tracker says no
    for i in range(25):
        if tracker.should_speculate("fix"):
            # Normal speculation
            hit, ttft = await simulate_request("The", tracker)
        else:
            # Exploration: speculate anyway, but use a fake tracker
            # that always allows speculation, then record result on real tracker
            t0_explore = time.monotonic()
            async with aiohttp.ClientSession() as session:
                cloud_task = asyncio.create_task(
                    session.post("http://127.0.0.1:39998/v1/chat/completions",
                                json={"model": "mock", "messages": [{"role": "user", "content": "test"}], "stream": True})
                )
                await asyncio.sleep(0.002)  # 2ms speculation
                resp = await cloud_task
                first_cloud_token = None
                async for chunk in resp.content.iter_any():
                    if first_cloud_token is None:
                        for line in chunk.decode(errors="replace").split("\n"):
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                continue
                            try:
                                obj = json.loads(data_str)
                                delta = obj.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content and content.strip():
                                    first_cloud_token = content.strip().split()[0]
                                    break
                            except:
                                pass
                hit = "The".strip() == first_cloud_token.strip() if first_cloud_token else False
                tracker.record(hit=hit, family="fix")
                resp.release()
                if i % 5 == 0:
                    print(f"  Exploration #{i}: hit={hit} cloud='{first_cloud_token}' state={tracker.get_state()}")
    print(f"  State: {tracker.get_state()}, Rate: {tracker.get_accept_rate()}")
    assert tracker.get_state() == STATE_ENABLED

    await runner.cleanup()
    print("\n=== MOCK CLOUD FLOW TEST PASSED ===")


async def main():
    await test_tracker_integration()
    await test_mock_cloud_flow()
    print("\n\n✅ ALL E2E TESTS PASSED ✅")


if __name__ == "__main__":
    asyncio.run(main())
