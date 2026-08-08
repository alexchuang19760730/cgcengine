#!/usr/bin/env python3
"""MTP Draft 首 token 预测测试 v2: context-aware + warm cache。每 prompt 跑两次(冷+热)。"""
import os, sys, time, json, urllib.request

os.environ["DSV4_TOKENIZER_PATH"] = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"

REPO = "/Users/alexchuang/Documents/flashkv0516"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from app.servers.edge_first_proxy import (
    _predict_first_token, _record_first_token, _classify_prompt_family,
)

CLOUD_URL = "http://47.95.250.55:30001/v1/chat/completions"
MODEL = "Qwen3-VL-2B-Instruct"

prompts = [
    "France is",
    "Write a short story about a cat",
    "What is the capital of France?",
    "Hello, how are you?",
    "Explain quantum computing in simple terms",
    "Translate 'hello' to French",
    "Write a Python function to reverse a string",
    "What is 2 + 2?",
    "Describe the process of photosynthesis",
    "Who wrote Romeo and Juliet?",
]

print("=== Round 1: Cold (no cache) ===")
cold_correct = 0
for prompt_text in prompts:
    messages = [{"role": "user", "content": prompt_text}]
    family = _classify_prompt_family(messages)
    prompt_hash = str(family.get("prompt_hash") or "")

    t0 = time.monotonic()
    predicted = _predict_first_token(messages)
    pred_ms = (time.monotonic() - t0) * 1000

    payload = json.dumps({"model": MODEL, "messages": messages, "max_tokens": 1, "stream": True}).encode()
    req = urllib.request.Request(CLOUD_URL, data=payload, headers={"Content-Type": "application/json"})
    t1 = time.monotonic()
    actual = ""
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: ") and "content" in line:
                    d = json.loads(line[6:])
                    c = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if c:
                        actual = c
                        break
    except Exception as e:
        actual = f"ERR:{e}"
    cloud_ms = (time.monotonic() - t1) * 1000

    match = predicted and actual and predicted.strip() in actual.strip()
    if match:
        cold_correct += 1
    # Record for warm cache
    _record_first_token(prompt_hash, actual)
    mark = "✓" if match else "✗"
    print(f"  {mark} pred={predicted!r:12s} actual={actual!r:12s} pred={pred_ms:.0f}ms cloud={cloud_ms:.0f}ms | {prompt_text[:40]}")

print(f"\nCold 准确率: {cold_correct}/{len(prompts)} = {cold_correct/len(prompts)*100:.0f}%")

print("\n=== Round 2: Warm (cache hit) ===")
warm_correct = 0
warm_times = []
for prompt_text in prompts:
    messages = [{"role": "user", "content": prompt_text}]

    t0 = time.monotonic()
    predicted = _predict_first_token(messages)
    pred_ms = (time.monotonic() - t0) * 1000
    warm_times.append(pred_ms)

    payload = json.dumps({"model": MODEL, "messages": messages, "max_tokens": 1, "stream": True}).encode()
    req = urllib.request.Request(CLOUD_URL, data=payload, headers={"Content-Type": "application/json"})
    t1 = time.monotonic()
    actual = ""
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data: ") and "content" in line:
                    d = json.loads(line[6:])
                    c = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if c:
                        actual = c
                        break
    except:
        actual = "ERR"
    cloud_ms = (time.monotonic() - t1) * 1000

    match = predicted and actual and predicted.strip() in actual.strip()
    if match:
        warm_correct += 1
    mark = "✓" if match else "✗"
    print(f"  {mark} pred={predicted!r:12s} actual={actual!r:12s} pred={pred_ms:.0f}ms | {prompt_text[:40]}")

print(f"\nWarm 准确率: {warm_correct}/{len(prompts)} = {warm_correct/len(prompts)*100:.0f}%")
print(f"Warm 预测 TTFT: avg={sum(warm_times)/len(warm_times):.0f}ms")
print(f"\n{'='*60}")
cold_acc = cold_correct / len(prompts)
warm_acc = warm_correct / len(prompts)
avg_cloud = 109  # 从之前测试
cold_ttft = 0 * cold_acc + avg_cloud * (1 - cold_acc)
warm_ttft = 0 * warm_acc + avg_cloud * (1 - warm_acc)
print(f"冷启动: 准确率 {cold_acc:.0%}, 平均 TTFT = {cold_ttft:.0f}ms")
print(f"缓存命中: 准确率 {warm_acc:.0%}, 平均 TTFT = {warm_ttft:.0f}ms")
print(f"纯 cloud: TTFT = {avg_cloud}ms")
print(f"\n生产场景 (50% 重复 prompt):")
prod_acc = cold_acc * 0.5 + warm_acc * 0.5
prod_ttft = 0 * prod_acc + avg_cloud * (1 - prod_acc)
print(f"  混合准确率 = {prod_acc:.0%}, 平均 TTFT = {prod_ttft:.0f}ms (vs cloud {avg_cloud}ms, -{(1-prod_ttft/avg_cloud)*100:.0f}%)")
