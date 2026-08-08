#!/usr/bin/env python3
"""MTP Draft 首 token 预测测试: context-aware 预测 vs cloud 实际返回。"""
import os, sys, time, json, urllib.request

os.environ["DSV4_TOKENIZER_PATH"] = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"

REPO = "/Users/alexchuang/Documents/flashkv0516"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from app.servers.edge_first_proxy import _edge_generate_first_token

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

correct = 0
total = 0
pred_times = []
cloud_times = []

for prompt_text in prompts:
    messages = [{"role": "user", "content": prompt_text}]

    # 1. context-aware 预测首 token
    t0 = time.monotonic()
    predicted = _edge_generate_first_token(messages, max_tokens=1)
    pred_ms = (time.monotonic() - t0) * 1000
    pred_times.append(pred_ms)

    # 2. cloud 实际首 token (直连)
    payload = json.dumps({
        "model": MODEL, "messages": messages,
        "max_tokens": 1, "stream": True
    }).encode()
    req = urllib.request.Request(CLOUD_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
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
        actual = f"ERROR: {e}"
    cloud_ms = (time.monotonic() - t1) * 1000
    cloud_times.append(cloud_ms)

    match = predicted and actual and predicted.strip() in actual.strip()
    if match:
        correct += 1
    total += 1
    mark = "✓" if match else "✗"
    print(f"{mark} pred={predicted!r:12s} actual={actual!r:12s} "
          f"pred={pred_ms:.0f}ms cloud={cloud_ms:.0f}ms "
          f"| {prompt_text[:40]}")

print(f"\n{'='*60}")
print(f"准确率: {correct}/{total} = {correct/total*100:.0f}%")
print(f"预测 TTFT: avg={sum(pred_times)/len(pred_times):.0f}ms")
print(f"cloud TTFT: avg={sum(cloud_times)/len(cloud_times):.0f}ms")
print(f"如果预测正确: TTFT = {sum(pred_times)/len(pred_times):.0f}ms")
print(f"如果预测错误: TTFT = {sum(pred_times)/len(pred_times):.0f} + {sum(cloud_times)/len(cloud_times):.0f} = {sum(pred_times)/len(pred_times)+sum(cloud_times)/len(cloud_times):.0f}ms")
acc = correct / total
avg_pred = sum(pred_times) / len(pred_times)
avg_cloud = sum(cloud_times) / len(cloud_times)
print(f"平均 TTFT = {acc*avg_pred:.0f} × {acc:.0%} + {(avg_pred+avg_cloud):.0f} × {1-acc:.0%} = {acc*avg_pred + (1-acc)*(avg_pred+avg_cloud):.0f}ms")
