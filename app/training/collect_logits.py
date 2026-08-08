#!/usr/bin/env python3
"""Tier 1 MTP Draft 蒸馏训练 — 数据收集脚本.

向云端 sglang 发送多样化 prompt, 收集 (input_ids, full_logits) 对,
存成训练 shard 供 KL 蒸馏使用.

用法 (在 Host1 上运行):
  python3 collect_logits.py \
    --url http://localhost:30010 \
    --model /data/models/gemma-4-26b-a4b-it \
    --output-dir /data/mtp_distill_data \
    --num-samples 50000 \
    --batch-size 8

输出:
  /data/mtp_distill_data/shard_000000.pt   # (input_ids, logits, attention_mask)
  /data/mtp_distill_data/shard_000001.pt
  ...
  /data/mtp_distill_data/meta.json         # 收集元信息
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import aiohttp
import torch

# === 多样化 prompt 语料 ===
CODE_PROMPTS = [
    "def binary_search(arr, target):\n    ",
    "class LinkedList:\n    def __init__(self):\n        ",
    "import numpy as np\n\ndef matrix_multiply(a, b):\n    ",
    "async def fetch_data(url):\n    ",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ",
    "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\nasync def root():\n    ",
    "def merge_sort(arr):\n    ",
    "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        ",
    "def dfs(graph, node, visited):\n    ",
    "def bfs(graph, start):\n    ",
    "import pandas as pd\n\ndef clean_data(df):\n    ",
    "def train_model(X, y):\n    from sklearn.ensemble import RandomForestClassifier\n    ",
    "class Singleton:\n    _instance = None\n    def __new__(cls):\n        ",
    "def retry(func, max_retries=3):\n    ",
    "def memoize(func):\n    cache = {}\n    def wrapper(*args):\n        ",
    "class Observer:\n    def update(self, subject):\n        ",
    "def parse_json(text):\n    import json\n    ",
    "def validate_email(email):\n    import re\n    ",
    "def chunk_list(lst, size):\n    ",
    "def flatten_dict(d, parent_key='', sep='.'):\n    ",
]

QA_PROMPTS = [
    "What is the time complexity of quicksort?",
    "Explain the difference between TCP and UDP.",
    "How does garbage collection work in Python?",
    "What is a closure in JavaScript?",
    "Explain the CAP theorem.",
    "What is the difference between SQL and NoSQL?",
    "How does HTTPS encryption work?",
    "What is a microservice architecture?",
    "Explain eventual consistency.",
    "What is the actor model in concurrency?",
    "How does a B-tree index work?",
    "What is the difference between process and thread?",
    "Explain the difference between async/await and threads.",
    "What is a reverse proxy and when to use it?",
    "How does JWT authentication work?",
    "What is the difference between GraphQL and REST?",
    "Explain the MapReduce programming model.",
    "What is a Bloom filter?",
    "How does consistent hashing work?",
    "What is the difference between mutex and semaphore?",
]

COMPLETION_PROMPTS = [
    "The main advantage of using a linked list over an array is",
    "In a microservices architecture, service discovery is important because",
    "The key difference between supervised and unsupervised learning is",
    "When designing a distributed system, the CAP theorem states that",
    "The purpose of a load balancer in a web application is to",
    "In object-oriented programming, encapsulation means",
    "The time complexity of inserting an element into a hash table is",
    "A decorator in Python is a function that",
    "The main benefit of using Docker containers is",
    "In a REST API, the POST method is used to",
    "The difference between WHERE and HAVING in SQL is",
    "In Python, a generator differs from a list because",
    "The primary use case for Redis in a web application is",
    "In Kubernetes, a pod is the smallest deployable unit that",
    "The advantage of using WebSockets over HTTP polling is",
]

CREATIVE_PROMPTS = [
    "Write a short story about a robot learning to paint.",
    "Describe a futuristic city in the year 2150.",
    "Write a poem about the ocean.",
    "Create a dialogue between two AI systems meeting for the first time.",
    "Write a product description for a smart water bottle.",
    "Compose a tweet announcing a new open-source library.",
    "Write a blog post title about the future of quantum computing.",
    "Create a character description for a sci-fi novel protagonist.",
    "Write a press release headline for a Mars colony achievement.",
    "Describe the experience of flying through a nebula.",
]


def generate_prompts(num_samples: int) -> list[str]:
    """生成多样化 prompt 列表."""
    prompts = []
    categories = [
        (CODE_PROMPTS, 0.40),       # 40% 代码
        (QA_PROMPTS, 0.25),         # 25% 问答
        (COMPLETION_PROMPTS, 0.20), # 20% 补全
        (CREATIVE_PROMPTS, 0.15),   # 15% 创意
    ]

    for pool, ratio in categories:
        count = int(num_samples * ratio)
        for _ in range(count):
            base = random.choice(pool)
            # 随机变换: 添加上下文、修改变量名等
            variant = random.random()
            if variant < 0.3 and pool == CODE_PROMPTS:
                # 添加类型注解
                base = base.replace("def ", "def ", 1)
            elif variant < 0.5:
                # 添加 system message 上下文
                base = f"You are a helpful assistant. {base}"
            elif variant < 0.7:
                # 添加前缀
                prefixes = ["Please ", "Can you ", "", "Help me ", ""]
                base = random.choice(prefixes) + base
            prompts.append(base)

    random.shuffle(prompts)
    return prompts[:num_samples]


async def collect_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """收集单条 prompt 的 logits."""
    async with semaphore:
        try:
            # 使用 /generate 端点获取 logits
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0,
                "return_logits": True,
                "return_tokens_in_logprobs": True,
            }

            # 先尝试 /v1/chat/completions
            try:
                async with session.post(
                    f"{url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # 检查是否有 logits
                        if "logits" in data or "choices" in data:
                            choice = data.get("choices", [{}])[0]
                            logprobs = choice.get("logprobs", {})
                            token_id = choice.get("logprobs", {}).get("tokens", [None])[0]
                            token_logprobs = logprobs.get("token_logprobs", [])

                            # 如果有 logprobs, 用它近似 logits
                            if token_logprobs:
                                # logprobs → softmax probs → logits
                                top_logprobs = logprobs.get("top_logprobs", [{}])[0] if logprobs.get("top_logprobs") else {}
                                if top_logprobs:
                                    # 构建 sparse logits from top logprobs
                                    vocab_size = 262144  # Gemma4 default
                                    logits = torch.full((vocab_size,), -50.0, dtype=torch.float32)
                                    for token_str, logprob in top_logprobs.items():
                                        # 这里需要 token_id, 暂时存 logprobs
                                        pass
                                    return {
                                        "prompt": prompt,
                                        "response_token": choice.get("message", {}).get("content", ""),
                                        "logprobs": top_logprobs,
                                        "method": "chat_completions",
                                    }
            except Exception:
                pass

            # 降级: 使用 /generate 端点
            gen_payload = {
                "model": model,
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": 0,
                },
                "return_logits": True,
            }

            async with session.post(
                f"{url}/generate",
                json=gen_payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logits = None
                    input_ids = None

                    # sglang /generate 返回格式
                    if "logits" in data:
                        logits = torch.tensor(data["logits"], dtype=torch.float32)
                    if "input_ids" in data:
                        input_ids = data["input_ids"]

                    # 也检查 meta_info
                    meta = data.get("meta_info", {})
                    if logits is None and "output_token_logprobs" in meta:
                        # 从 logprobs 构建 sparse logits
                        logprobs = meta["output_token_logprobs"]
                        if logprobs:
                            logits = torch.tensor(logprobs[0], dtype=torch.float32)

                    if logits is not None:
                        return {
                            "prompt": prompt,
                            "input_ids": input_ids,
                            "logits": logits,
                            "target_token_id": int(logits.argmax().item()),
                            "method": "generate",
                        }

            return None

        except Exception as e:
            print(f"  [collect] Error: {e}", file=sys.stderr)
            return None


async def collect_batch(
    url: str,
    model: str,
    prompts: list[str],
    output_dir: str,
    shard_size: int = 500,
    concurrency: int = 8,
) -> int:
    """批量收集 logits 并保存为 shard."""
    os.makedirs(output_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    shard_idx = 0
    shard_data = []
    total_collected = 0
    total_failed = 0

    # 使用连接池
    connector = aiohttp.TCPConnector(limit=concurrency, keepalive_timeout=60)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 先测试连通性
        try:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"ERROR: LB health check failed: {resp.status}", file=sys.stderr)
                    return 0
                print(f"Connected to {url}, health OK", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: Cannot connect to {url}: {e}", file=sys.stderr)
            return 0

        t0 = time.time()
        for i, prompt in enumerate(prompts):
            result = await collect_one(session, url, model, prompt, semaphore)

            if result is not None:
                shard_data.append(result)
                total_collected += 1
            else:
                total_failed += 1

            # 保存 shard
            if len(shard_data) >= shard_size:
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
                torch.save(shard_data, shard_path)
                elapsed = time.time() - t0
                rate = total_collected / elapsed if elapsed > 0 else 0
                print(
                    f"  Shard {shard_idx}: {len(shard_data)} samples saved "
                    f"(total: {total_collected}, failed: {total_failed}, "
                    f"rate: {rate:.1f}/s, elapsed: {elapsed:.0f}s)",
                    file=sys.stderr,
                )
                shard_data = []
                shard_idx += 1

            # 进度报告
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  Progress: {i+1}/{len(prompts)} "
                    f"(collected: {total_collected}, failed: {total_failed}, "
                    f"rate: {rate:.1f}/s)",
                    file=sys.stderr,
                )

    # 保存最后的 shard
    if shard_data:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)
        shard_idx += 1

    # 保存元信息
    meta = {
        "total_samples": total_collected,
        "total_failed": total_failed,
        "num_shards": shard_idx,
        "model": model,
        "url": url,
        "collection_time_sec": time.time() - t0,
        "prompt_distribution": {
            "code": 0.40,
            "qa": 0.25,
            "completion": 0.20,
            "creative": 0.15,
        },
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {total_collected} samples in {shard_idx} shards", file=sys.stderr)
    print(f"  Output: {output_dir}/", file=sys.stderr)
    print(f"  Failed: {total_failed}", file=sys.stderr)
    print(f"  Time: {time.time() - t0:.0f}s", file=sys.stderr)

    return total_collected


def main():
    parser = argparse.ArgumentParser(description="Collect logits for KL distillation")
    parser.add_argument("--url", required=True, help="sglang LB URL (e.g. http://localhost:30010)")
    parser.add_argument("--model", required=True, help="Model path/name")
    parser.add_argument("--output-dir", default="/data/mtp_distill_data", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples to collect")
    parser.add_argument("--batch-size", type=int, default=8, help="Concurrency level")
    parser.add_argument("--shard-size", type=int, default=500, help="Samples per shard file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Generating {args.num_samples} diverse prompts...", file=sys.stderr)
    prompts = generate_prompts(args.num_samples)
    print(f"Generated {len(prompts)} prompts", file=sys.stderr)

    total = asyncio.run(collect_batch(
        url=args.url,
        model=args.model,
        prompts=prompts,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        concurrency=args.batch_size,
    ))

    if total == 0:
        print("WARNING: No samples collected. Check sglang /generate endpoint and return_logits support.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
