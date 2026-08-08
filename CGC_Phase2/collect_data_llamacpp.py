#!/usr/bin/env python3
"""Collect MTP training data using llama.cpp (matches inference engine).

Critical: training data must come from the SAME engine used at inference time.
transformers (FP32) hidden states differ from llama.cpp (FP16) hidden states,
causing 0% accept rate when MTP head trained on transformers data is used
with llama.cpp inference.

This script:
  1. Loads FP16 GGUF model via llama.cpp
  2. Generates text autoregressively (greedy)
  3. Collects decode hidden states via llama_get_embeddings_ith(ctx, 0)
  4. Extracts embed/lm_head weights from embed_head.pt (pre-saved)
  5. Saves shards in same format as collect_hidden_states.py

Usage:
  python3 collect_data_llamacpp.py --model-path /path/to/model.gguf \\
    --embed-head /path/to/embed_head.pt \\
    --output-dir ./mtp_train_data/qwen25_llamacpp \\
    --num-samples 200 --gen-length 40
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import llama_cpp

# === Prompt corpus (same as train_qwen25_mtp.py) ===
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
    "export function debounce(fn, delay) {\n  ",
    "const useState = (initial) => {\n  ",
    "async function fetchData(apiUrl) {\n  ",
    "class Vector3 {\n  constructor(x, y, z) {\n    ",
    "function binarySearch(arr, target) {\n  ",
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


def generate_prompts(num_samples: int, seed: int = 42) -> list[str]:
    random.seed(seed)
    prompts = []
    categories = [
        (CODE_PROMPTS, 0.40),
        (QA_PROMPTS, 0.25),
        (COMPLETION_PROMPTS, 0.20),
        (CREATIVE_PROMPTS, 0.15),
    ]
    for pool, ratio in categories:
        count = int(num_samples * ratio)
        for _ in range(count):
            base = random.choice(pool)
            variant = random.random()
            if variant < 0.3:
                prefixes = ["Please ", "Can you ", "", "Help me ", ""]
                base = random.choice(prefixes) + base
            elif variant < 0.5:
                base = f"You are a helpful assistant. {base}"
            prompts.append(base)
    random.shuffle(prompts)
    return prompts[:num_samples]


EOS_TOKENS = {151643, 151645}


def get_last_hidden(ctx: int, n_embd: int) -> np.ndarray:
    """Get hidden state of last decode via llama_get_embeddings_ith(ctx, 0)."""
    emb_ptr = llama_cpp.llama_get_embeddings_ith(ctx, 0)
    if not emb_ptr:
        return np.zeros(n_embd, dtype=np.float32)
    base = ctypes.addressof(emb_ptr.contents)
    arr = ctypes.c_float * n_embd
    return np.array(arr.from_address(base), dtype=np.float32)


def get_last_logits(ctx: int, n_vocab: int) -> np.ndarray:
    """Get logits of last decode."""
    logits_ptr = llama_cpp.llama_get_logits(ctx)
    if not logits_ptr:
        return np.zeros(n_vocab, dtype=np.float32)
    base = ctypes.addressof(logits_ptr.contents)
    arr = ctypes.c_float * n_vocab
    return np.array(arr.from_address(base), dtype=np.float32)


def decode_single(ctx: int, token_id: int, pos: int, seq_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Single token decode, returns (hidden, logits)."""
    token_arr = (llama_cpp.llama_token * 1)(token_id)
    batch = llama_cpp.llama_batch_get_one(token_arr, 1, pos, seq_id)
    ret = llama_cpp.llama_decode(ctx, batch)
    if ret != 0:
        raise RuntimeError(f"llama_decode failed: {ret}")
    return None, None  # will be fetched separately


def collect_one_prompt(
    llm,
    prompt: str,
    n_embd: int,
    n_vocab: int,
    gen_length: int = 40,
    num_chain: int = 4,
    max_input_length: int = 256,
) -> list[dict]:
    """Collect decode hidden states for one prompt using llama.cpp.

    Returns list of chain samples.
    """
    ctx = llm.ctx
    seq_id = 0

    # 1. Tokenize
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    tokens = list(tokens)
    # Filter EOS
    tokens = [t for t in tokens if t not in EOS_TOKENS]
    if len(tokens) < 3:
        return []
    if len(input_ids := tokens) > max_input_length:
        tokens = tokens[:max_input_length]

    n_prompt = len(tokens)

    # 2. Prefill (batch decode)
    batch_size = min(llm.n_batch, n_prompt)
    pos = 0
    while pos < n_prompt:
        end = min(pos + batch_size, n_prompt)
        batch_tokens = tokens[pos:end]
        n_bt = len(batch_tokens)
        token_arr = (llama_cpp.llama_token * n_bt)(*batch_tokens)
        batch = llama_cpp.llama_batch_get_one(token_arr, n_bt, pos, seq_id)
        ret = llama_cpp.llama_decode(ctx, batch)
        if ret != 0:
            return []
        pos = end

    # 3. Get logits for first token
    logits = get_last_logits(ctx, n_vocab)
    first_token = int(logits.argmax())

    # 4. Decode loop — collect hidden states
    # First, decode first_token to get its hidden
    decode_hiddens = []
    decode_tokens = []
    current_token = first_token
    n_past = n_prompt

    for step in range(gen_length):
        # Single token decode
        token_arr = (llama_cpp.llama_token * 1)(current_token)
        batch = llama_cpp.llama_batch_get_one(token_arr, 1, n_past, seq_id)
        ret = llama_cpp.llama_decode(ctx, batch)
        if ret != 0:
            break
        n_past += 1

        # Get hidden state
        hidden = get_last_hidden(ctx, n_embd)
        # Get logits for next token
        logits = get_last_logits(ctx, n_vocab)
        next_token = int(logits.argmax())

        decode_hiddens.append(torch.from_numpy(hidden.copy()).float())
        decode_tokens.append(current_token)

        if next_token in EOS_TOKENS:
            break
        current_token = next_token

    # Rewind KV cache for next prompt
    if hasattr(llama_cpp, 'llama_memory_seq_rm'):
        mem = llama_cpp.llama_get_memory(ctx)
        llama_cpp.llama_memory_seq_rm(mem, seq_id, 0, -1)
    elif hasattr(llama_cpp, 'llama_kv_cache_seq_rm'):
        llama_cpp.llama_kv_cache_seq_rm(ctx, seq_id, 0, -1)
        if hasattr(llama_cpp, 'llama_kv_cache_update'):
            llama_cpp.llama_kv_cache_update(ctx)

    if len(decode_hiddens) < num_chain + 1:
        return []

    # 5. Create chain training samples
    samples = []
    for i in range(len(decode_hiddens) - num_chain):
        end = i + num_chain
        if end >= len(decode_tokens):
            break
        chain_hidden = torch.stack(decode_hiddens[i:end])
        chain_tokens = torch.tensor(decode_tokens[i:end])
        chain_next = torch.tensor(decode_tokens[i + 1:end + 1])

        samples.append({
            "hidden_states": chain_hidden,
            "token_ids": chain_tokens,
            "next_token_ids": chain_next,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Collect MTP training data using llama.cpp")
    parser.add_argument("--model-path", required=True, help="GGUF model path")
    parser.add_argument("--embed-head", required=True, help="embed_head.pt path (from transformers collection)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of prompts")
    parser.add_argument("--num-chain", type=int, default=4, help="Chain length")
    parser.add_argument("--gen-length", type=int, default=40, help="Max tokens per prompt")
    parser.add_argument("--shard-size", type=int, default=500, help="Samples per shard")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model
    print(f"[collect] Loading {args.model_path}...", flush=True)
    llm = llama_cpp.Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_batch=512,
        embedding=True,
        logits_all=False,
        verbose=False,
    )
    n_embd = llm.n_embd()
    n_vocab = llm.n_vocab()
    print(f"[collect] n_embd={n_embd}, n_vocab={n_vocab}", flush=True)

    # 2. Copy embed_head.pt (already has embed + lm_head weights from transformers)
    import shutil
    dst_embed_head = os.path.join(args.output_dir, "embed_head.pt")
    if os.path.exists(args.embed_head) and not os.path.exists(dst_embed_head):
        shutil.copy2(args.embed_head, dst_embed_head)
        print(f"[collect] Copied embed_head.pt", flush=True)
    elif os.path.exists(dst_embed_head):
        print(f"[collect] embed_head.pt already exists", flush=True)

    # 3. Generate prompts
    prompts = generate_prompts(args.num_samples)
    print(f"[collect] {len(prompts)} prompts generated", flush=True)

    # 4. Collect
    shard_idx = 0
    shard_data = []
    total_collected = 0
    total_failed = 0
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        try:
            samples = collect_one_prompt(
                llm, prompt, n_embd, n_vocab,
                gen_length=args.gen_length,
                num_chain=args.num_chain,
            )
            if samples:
                shard_data.extend(samples)
                total_collected += len(samples)
            else:
                total_failed += 1
        except Exception as e:
            total_failed += 1
            if total_failed <= 5:
                print(f"[collect] Error on prompt {i}: {e}", flush=True)

        if len(shard_data) >= args.shard_size:
            shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:06d}.pt")
            torch.save(shard_data, shard_path)
            elapsed = time.time() - t0
            print(f"[collect] Shard {shard_idx}: {len(shard_data)} samples "
                  f"(total: {total_collected}, failed: {total_failed}, "
                  f"elapsed: {elapsed:.0f}s)", flush=True)
            shard_data = []
            shard_idx += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[collect] Progress: {i+1}/{len(prompts)} "
                  f"(samples: {total_collected}, failed: {total_failed}, "
                  f"rate: {(i+1)/elapsed:.1f} prompts/s)", flush=True)

    # Save final shard
    if shard_data:
        shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)
        shard_idx += 1

    # Save metadata
    meta = {
        "model_name": "qwen25-0.5b",
        "model_path": args.model_path,
        "hidden_size": n_embd,
        "vocab_size": n_vocab,
        "num_chain": args.num_chain,
        "gen_length": args.gen_length,
        "total_samples": total_collected,
        "total_failed": total_failed,
        "num_shards": shard_idx,
        "collection_time_sec": time.time() - t0,
        "hidden_type": "decode_llamacpp",
        "engine": "llama.cpp",
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[collect] Done! {total_collected} samples in {shard_idx} shards", flush=True)
    print(f"[collect] Time: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
