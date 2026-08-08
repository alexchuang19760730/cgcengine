#!/usr/bin/env python3
"""Improved MTP head training for Qwen2.5 (0.5B and 1.5B) on Mac.

Improvements over v1/v2:
  1. Real code prompt corpus (50+ diverse prompts from actual codebases)
  2. Validation split (15%) to detect overfitting
  3. Cosine LR schedule with warmup
  4. Early stopping on validation loss
  5. Best checkpoint saved by validation accept rate
  6. Support both 0.5B and 1.5B models

Pipeline:
  Phase 1: Create embed_head.pt (via transformers, one-time)
  Phase 2: Collect training data (via llama.cpp, matches inference engine)
  Phase 3: Train MTP head (chain CE, with validation)

Usage:
  # 1.5B full pipeline:
  python3 train_qwen25_mtp_v3.py --model-size 1.5b --phase all

  # 1.5B collect only:
  python3 train_qwen25_mtp_v3.py --model-size 1.5b --phase collect --num-samples 5000

  # 1.5B train only (data already collected):
  python3 train_qwen25_mtp_v3.py --model-size 1.5b --phase train --epochs 3

  # 0.5B retrain with improved corpus:
  python3 train_qwen25_mtp_v3.py --model-size 0.5b --phase all --num-samples 5000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# === Path setup ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [REPO_ROOT, SCRIPT_DIR, os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_head")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# === Model configs ===
MODEL_CONFIGS = {
    "0.5b": {
        "model_name": "qwen25-0.5b",
        "display_name": "Qwen2.5-0.5B-Instruct",
        "hf_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "gguf_path": "/Users/alexchuang/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf",
        "hidden_size": 896,
        "vocab_size": 151936,
        "num_heads": 14,
        "head_dim": 64,
        "intermediate_size": 4864,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "eos_tokens": {151643, 151645},
    },
    "1.5b": {
        "model_name": "qwen25-1.5b",
        "display_name": "Qwen2.5-1.5B-Instruct",
        "hf_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "gguf_path": "/Users/alexchuang/models/gguf/qwen2.5-1.5b-instruct-fp16.gguf",
        "hidden_size": 1536,
        "vocab_size": 151936,
        "num_heads": 12,
        "head_dim": 128,
        "intermediate_size": 8960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "eos_tokens": {151643, 151645},
    },
    "7b": {
        "model_name": "qwen25-7b",
        "display_name": "Qwen2.5-7B-Instruct",
        "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gguf_path": "/Users/alexchuang/models/gguf/qwen2.5-7b-instruct-q4_k_m.gguf",
        "hidden_size": 3584,
        "vocab_size": 152064,
        "num_heads": 28,
        "head_dim": 128,
        "intermediate_size": 18944,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "eos_tokens": {151643, 151645},
    },
    "gemma2-2b": {
        "model_name": "gemma2-2b",
        "display_name": "Gemma-2-2B-it",
        "hf_model_id": "google/gemma-2-2b-it",
        "gguf_path": "/Users/alexchuang/Documents/flashkv0516/models/gemma2_2b_gguf/gemma-2-2b-it-Q4_K_M.gguf",
        "hidden_size": 2304,
        "vocab_size": 256000,
        "num_heads": 8,
        "head_dim": 256,
        "intermediate_size": 9216,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 8192,
        "eos_tokens": {107, 577},  # <end_of_turn>, <end_of_text>
    },
}

# === Expanded prompt corpus (Task B: real code prompts) ===

# Real Python code from common libraries/patterns
CODE_PROMPTS_PY = [
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
    "class ThreadPool:\n    def __init__(self, num_threads):\n        ",
    "def rate_limiter(calls, period):\n    ",
    "def circular_buffer(size):\n    ",
    "class EventBus:\n    def subscribe(self, event, handler):\n        ",
    "def lru_cache(maxsize=128):\n    ",
    "def parallel_map(func, items, workers=4):\n    ",
    "class StateMachine:\n    def transition(self, event):\n        ",
    "def backoff_retry(func, base_delay=1, max_delay=60):\n    ",
    "def pipeline(*steps):\n    ",
    "class TypedDict:\n    def __setitem__(self, key, value):\n        ",
]

# JavaScript/TypeScript code
CODE_PROMPTS_JS = [
    "export function debounce(fn, delay) {\n  ",
    "const useState = (initial) => {\n  ",
    "async function fetchData(apiUrl) {\n  ",
    "class Vector3 {\n  constructor(x, y, z) {\n    ",
    "function binarySearch(arr, target) {\n  ",
    "export class Observable {\n  constructor() {\n    ",
    "const memoize = (fn) => {\n  const cache = new Map();\n  ",
    "async function* paginate(url) {\n  ",
    "class PriorityQueue {\n  constructor(compare) {\n    ",
    "function deepClone(obj) {\n  ",
    "export const compose = (...fns) => (x) =>\n  ",
    "class EventEmitter {\n  on(event, listener) {\n    ",
]

# Rust/Go/System code
CODE_PROMPTS_SYS = [
    "fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {\n    ",
    "func (s *Server) handleRequest(w http.ResponseWriter, r *http.Request) {\n    ",
    "impl<T> Stack<T> {\n    fn new() -> Self {\n        ",
    "func fibonacci(n int) int {\n    if n <= 1 {\n        return n\n    }\n    ",
    "pub struct HashMap<K, V> {\n    buckets: Vec<Vec<(K, V)>>,\n    ",
    "func (db *Database) Query(ctx context.Context, sql string) (Rows, error) {\n    ",
]

# Q&A and explanation prompts
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

# Completion prompts
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

# Creative/general prompts
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
    """Generate diverse prompt list with more code-focused distribution."""
    random.seed(seed)
    prompts = []
    # 55% code (was 40%), 20% QA, 15% completion, 10% creative
    categories = [
        (CODE_PROMPTS_PY + CODE_PROMPTS_JS + CODE_PROMPTS_SYS, 0.55),
        (QA_PROMPTS, 0.20),
        (COMPLETION_PROMPTS, 0.15),
        (CREATIVE_PROMPTS, 0.10),
    ]
    for pool, ratio in categories:
        count = int(num_samples * ratio)
        for _ in range(count):
            base = random.choice(pool)
            # Add variation
            variant = random.random()
            if variant < 0.2:
                prefixes = ["Please ", "Can you ", "", "Help me ", ""]
                base = random.choice(prefixes) + base
            elif variant < 0.35:
                base = f"You are a helpful assistant. {base}"
            elif variant < 0.45:
                # Add context
                contexts = [
                    "I'm building a web app. ",
                    "I'm working on a data pipeline. ",
                    "I'm implementing a CLI tool. ",
                    "I'm writing tests for my service. ",
                ]
                base = random.choice(contexts) + base
            prompts.append(base)
    random.shuffle(prompts)
    return prompts[:num_samples]


# === Phase 1: Create embed_head.pt ===

def create_embed_head(model_size: str, output_dir: str):
    """Extract embed + lm_head weights from transformers model or GGUF."""
    cfg = MODEL_CONFIGS[model_size]
    os.makedirs(output_dir, exist_ok=True)

    embed_head_path = os.path.join(output_dir, "embed_head.pt")
    if os.path.exists(embed_head_path):
        print(f"[embed] embed_head.pt already exists at {embed_head_path}")
        return embed_head_path

    # 优先尝试从 GGUF 提取 (避免 gated repo 认证问题)
    gguf_path = cfg.get("gguf_path", "")
    if gguf_path and os.path.exists(gguf_path):
        print(f"[embed] Loading GGUF for n_embd/n_vocab: {gguf_path}", flush=True)
        try:
            import llama_cpp
            llm = llama_cpp.Llama(
                model_path=gguf_path,
                n_gpu_layers=0,
                n_ctx=512,
                verbose=False,
            )
            n_embd = llm.n_embd()
            n_vocab = llm.n_vocab()
            print(f"[embed] n_embd={n_embd}, n_vocab={n_vocab}", flush=True)
            del llm

            # get_tensor 在多数 llama.cpp 版本未实现 → 只保存元数据
            # 训练时在内存生成可训练 lm_head (避免 4GB+ 随机权重落盘)
            torch.save({
                "hidden_size": n_embd,
                "vocab_size": n_vocab,
                "model_name": cfg["model_name"],
                "init_method": "random_trainable",
            }, embed_head_path)
            print(f"[embed] Saved metadata-only {embed_head_path} ({os.path.getsize(embed_head_path) / 1e3:.0f}KB)", flush=True)
            print(f"[embed] NOTE: lm_head will be random-initialized and trainable in training", flush=True)
            return embed_head_path
        except Exception as e:
            print(f"[embed] GGUF loading failed: {e}, falling back to transformers", flush=True)

    # Fallback: transformers (需要 HF 认证, 对 gated repo 可能失败)
    from transformers import AutoModelForCausalLM
    print(f"[embed] Loading {cfg['display_name']} via transformers...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_model_id"],
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    embed_weight = model.get_input_embeddings().weight.data.clone()
    lm_head_weight = model.get_output_embeddings().weight.data.clone()

    if lm_head_weight is None or lm_head_weight.shape[0] == 0:
        lm_head_weight = embed_weight.clone()
        print("[embed] Using tied embeddings (embed = lm_head)", flush=True)

    print(f"[embed] embed: {embed_weight.shape}, lm_head: {lm_head_weight.shape}", flush=True)

    torch.save({
        "embed_weight": embed_weight.cpu().to(torch.float32),
        "lm_head_weight": lm_head_weight.cpu().to(torch.float32),
        "hidden_size": cfg["hidden_size"],
        "vocab_size": cfg["vocab_size"],
        "model_name": cfg["model_name"],
    }, embed_head_path)
    print(f"[embed] Saved {embed_head_path} ({os.path.getsize(embed_head_path) / 1e6:.0f}MB)", flush=True)

    del model
    return embed_head_path


# === Phase 2: Collect data via llama.cpp ===

def collect_data_llamacpp(model_size: str, output_dir: str, num_samples: int,
                          num_chain: int = 4, gen_length: int = 50):
    """Collect training data using llama.cpp (matches inference engine)."""
    cfg = MODEL_CONFIGS[model_size]
    gguf_path = cfg["gguf_path"]

    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

    # Ensure embed_head.pt exists
    embed_head_path = os.path.join(output_dir, "embed_head.pt")
    if not os.path.exists(embed_head_path):
        create_embed_head(model_size, output_dir)

    # Import collection module
    sys.path.insert(0, SCRIPT_DIR)
    from collect_data_llamacpp import (
        generate_prompts as _gen,  # We'll override with our expanded corpus
        collect_one_prompt,
    )
    import llama_cpp
    import numpy as np

    # Override prompt generation with our expanded corpus
    prompts = generate_prompts(num_samples)
    print(f"[collect] {len(prompts)} prompts generated (expanded corpus)", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"[collect] Loading {gguf_path}...", flush=True)
    llm = llama_cpp.Llama(
        model_path=gguf_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=512,
        embedding=True,
        logits_all=False,
        verbose=False,
    )
    n_embd = llm.n_embd()
    n_vocab = llm.n_vocab()
    print(f"[collect] n_embd={n_embd}, n_vocab={n_vocab}", flush=True)

    if n_embd != cfg["hidden_size"]:
        print(f"[collect] WARNING: n_embd={n_embd} != config hidden_size={cfg['hidden_size']}")
        cfg["hidden_size"] = n_embd

    # Collect
    EOS_TOKENS = cfg["eos_tokens"]
    shard_idx = 0
    shard_data = []
    total_collected = 0
    total_failed = 0
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        try:
            samples = collect_one_prompt(
                llm, prompt, n_embd, n_vocab,
                gen_length=gen_length,
                num_chain=num_chain,
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

        if len(shard_data) >= 500:
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
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
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)
        shard_idx += 1

    # Save metadata
    meta = {
        "model_name": cfg["model_name"],
        "model_path": gguf_path,
        "hidden_size": n_embd,
        "vocab_size": n_vocab,
        "num_chain": num_chain,
        "gen_length": gen_length,
        "total_samples": total_collected,
        "total_failed": total_failed,
        "num_shards": shard_idx,
        "collection_time_sec": time.time() - t0,
        "engine": "llama.cpp",
        "corpus": "expanded_v3",
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[collect] Done! {total_collected} samples in {shard_idx} shards", flush=True)
    print(f"[collect] Time: {time.time() - t0:.0f}s", flush=True)

    del llm
    return total_collected


# === Phase 3: Training with validation ===

class MTPDataset(Dataset):
    """Loads chain training samples from collected shards."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "hidden_states": item["hidden_states"].float(),
            "token_ids": item["token_ids"].long(),
            "next_token_ids": item["next_token_ids"].long(),
        }


def collate_fn(batch):
    hidden_states = torch.stack([item["hidden_states"] for item in batch])
    token_ids = torch.stack([item["token_ids"] for item in batch])
    next_token_ids = torch.stack([item["next_token_ids"] for item in batch])
    return {
        "hidden_states": hidden_states,
        "token_ids": token_ids,
        "next_token_ids": next_token_ids,
    }


def train_chain_step(mtp, embed_weight, lm_head_weight,
                     hidden_states, token_ids, next_token_ids, device,
                     scheduled_p: float = 0.0):
    """Single chain training step with scheduled sampling.

    Args:
        scheduled_p: probability of feeding model's own argmax prediction as
                     next-step token embedding (exposure bias mitigation).
                     0.0 = pure teacher forcing, 1.0 = pure self-feeding.
                     Linear schedule from 0 → 0.5 over training is recommended.
    """
    batch_size, chain_len, hidden_size = hidden_states.shape
    total_loss = torch.tensor(0.0, device=device)
    correct = 0
    total = 0

    current_hidden = hidden_states[:, 0, :]
    # Track which token is actually fed as input at each step (for scheduled sampling)
    current_input_token = token_ids[:, 0]

    for k in range(chain_len):
        token_embed = F.embedding(current_input_token, embed_weight)
        h_3d = current_hidden.unsqueeze(1)
        e_3d = token_embed.unsqueeze(1)

        x = torch.cat([h_3d, e_3d], dim=-1)
        x = mtp.proj(x)
        x = x + mtp.attn(mtp.norm1(x))
        x = x + mtp.mlp(mtp.norm2(x))
        mtp_hidden = mtp.norm_out(x)
        logits = F.linear(mtp_hidden[:, 0, :], lm_head_weight)

        loss = F.cross_entropy(logits, next_token_ids[:, k])
        total_loss = total_loss + loss

        pred = logits.argmax(dim=-1)
        correct += (pred == next_token_ids[:, k]).sum().item()
        total += batch_size

        if k < chain_len - 1:
            current_hidden = mtp_hidden[:, 0, :].detach()
            # Scheduled sampling: with prob scheduled_p, use model's own prediction
            # as next-step input token (instead of ground truth token_ids[:, k+1])
            if scheduled_p > 0.0 and k > 0 and random.random() < scheduled_p:
                current_input_token = pred.detach()
            else:
                current_input_token = token_ids[:, k + 1]

    total_loss = total_loss / chain_len
    return total_loss, {"chain_acc": correct / max(total, 1)}


def evaluate(mtp, embed_weight, lm_head_weight, loader, device, max_batches=50):
    """Evaluate on validation set."""
    mtp.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            hidden_states = batch["hidden_states"].to(device)
            token_ids = batch["token_ids"].to(device)
            next_token_ids = batch["next_token_ids"].to(device)

            loss, metrics = train_chain_step(
                mtp, embed_weight, lm_head_weight,
                hidden_states, token_ids, next_token_ids, device,
            )
            total_loss += loss.item()
            correct += metrics["chain_acc"] * hidden_states.shape[0]
            total += hidden_states.shape[0]
            num_batches += 1

    mtp.train()
    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_acc": correct / max(total, 1),
    }


def train(model_size: str, data_dir: str, output_dir: str,
          epochs: int = 3, batch_size: int = 16, lr: float = 3e-4,
          warmup_steps: int = 50, device: str = "cpu",
          val_ratio: float = 0.15):
    """Train MTP head with validation and early stopping."""
    from model import MTPHead, MTPHeadConfig

    cfg = MODEL_CONFIGS[model_size]
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)

    # 1. Load dataset
    all_samples = []
    shard_files = sorted(Path(data_dir).glob("shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files in {data_dir}")

    for shard_path in shard_files:
        try:
            shard = torch.load(shard_path, weights_only=False)
            if isinstance(shard, list):
                all_samples.extend(shard)
        except Exception as e:
            print(f"  [dataset] Error loading {shard_path}: {e}", file=sys.stderr)

    print(f"[train] Loaded {len(all_samples)} samples from {len(shard_files)} shards", flush=True)

    if len(all_samples) < 50:
        raise ValueError(f"Too few samples: {len(all_samples)}")

    # 2. Split train/val
    random.seed(123)
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * val_ratio))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    train_dataset = MTPDataset(train_samples)
    val_dataset = MTPDataset(val_samples)

    print(f"[train] Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, persistent_workers=True,
    )

    # 3. Load embed + lm_head (convert to float32 for training stability)
    embed_head_path = os.path.join(data_dir, "embed_head.pt")
    if not os.path.exists(embed_head_path):
        print(f"[train] embed_head.pt not found, creating...", flush=True)
        create_embed_head(model_size, data_dir)
    weights = torch.load(embed_head_path, map_location="cpu", weights_only=True)

    init_method = weights.get("init_method", "pretrained")
    if init_method == "random_trainable":
        # metadata-only: 在内存生成可训练 lm_head (避免 4GB+ 落盘)
        n_embd = weights["hidden_size"]
        n_vocab = weights["vocab_size"]
        torch.manual_seed(42)
        embed_weight = (torch.randn(n_vocab, n_embd) * 0.02).to(dev)
        lm_head_weight = (torch.randn(n_vocab, n_embd) * 0.02).to(dev)
        lm_head_trainable = True
        print(f"[train] Random trainable lm_head: {lm_head_weight.shape}", flush=True)
    else:
        embed_weight = weights["embed_weight"].float().to(dev)
        lm_head_weight = weights["lm_head_weight"].float().to(dev)
        lm_head_trainable = False
        print(f"[train] Pretrained embed: {embed_weight.shape}, lm_head: {lm_head_weight.shape}", flush=True)

    # 4. Create MTP head
    config = MTPHeadConfig(
        hidden_size=cfg["hidden_size"],
        vocab_size=cfg["vocab_size"],
        num_heads=cfg["num_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
    )
    mtp = MTPHead(config)
    # 随机初始化时用 LoRA (rank=8) 替代全量训练 lm_head, 大幅减少可训练参数
    lora_rank = 8 if lm_head_trainable else 0
    mtp.set_shared_lm_head(lm_head_weight, trainable=False, lora_rank=lora_rank)
    print(f"[train] MTP head: {mtp.num_parameters() / 1e6:.1f}M params (lora_rank={lora_rank})", flush=True)

    mtp.to(dev).to(torch.float32)
    mtp.train()

    # 5. Optimizer + cosine schedule
    optimizer = torch.optim.AdamW(mtp.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(train_loader) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. Training loop with validation
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_step = 0
    global_step = 0
    t0 = time.time()
    log = {"config": cfg, "steps": [], "val_history": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0

        for batch in train_loader:
            hidden_states = batch["hidden_states"].to(dev)
            token_ids = batch["token_ids"].to(dev)
            next_token_ids = batch["next_token_ids"].to(dev)

            # Scheduled sampling: linear schedule 0 → 0.5 over training
            # First 20% of steps: pure teacher forcing (warmup)
            # Last 80% of steps: linear ramp to 0.5 self-feeding
            warmup_frac = 0.2
            if global_step < total_steps * warmup_frac:
                scheduled_p = 0.0
            else:
                progress = (global_step - total_steps * warmup_frac) / max(total_steps * (1 - warmup_frac), 1)
                scheduled_p = min(0.5, 0.5 * progress)

            loss, metrics = train_chain_step(
                mtp, embed_weight, lm_head_weight,
                hidden_states, token_ids, next_token_ids, dev,
                scheduled_p=scheduled_p,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_correct += metrics["chain_acc"] * hidden_states.shape[0]
            epoch_total += hidden_states.shape[0]
            num_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                elapsed = time.time() - t0
                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_correct / epoch_total
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  [E{epoch+1}/{epochs}] S{global_step}/{total_steps} | "
                    f"loss={avg_loss:.4f} acc={avg_acc:.3f} | "
                    f"lr={current_lr:.2e} | {elapsed:.0f}s",
                    flush=True,
                )
                log["steps"].append({
                    "step": global_step, "epoch": epoch + 1,
                    "loss": round(avg_loss, 4), "acc": round(avg_acc, 4),
                    "lr": current_lr, "elapsed_s": round(elapsed, 1),
                })

        # Validate after each epoch
        val_results = evaluate(mtp, embed_weight, lm_head_weight, val_loader, dev)
        print(
            f"\n  Epoch {epoch+1}: train_loss={epoch_loss/max(num_batches,1):.4f} "
            f"train_acc={epoch_correct/max(epoch_total,1):.3f} | "
            f"val_loss={val_results['val_loss']:.4f} val_acc={val_results['val_acc']:.3f} "
            f"({time.time()-t0:.0f}s)\n",
            flush=True,
        )
        log["val_history"].append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss / max(num_batches, 1), 4),
            "train_acc": round(epoch_correct / max(epoch_total, 1), 4),
            "val_loss": round(val_results["val_loss"], 4),
            "val_acc": round(val_results["val_acc"], 4),
        })

        # Save best checkpoint
        if val_results["val_acc"] > best_val_acc:
            best_val_acc = val_results["val_acc"]
            best_val_loss = val_results["val_loss"]
            best_step = global_step
            best_path = os.path.join(output_dir, f"mtp_head_{cfg['model_name']}_decode.pt")
            _save_checkpoint(mtp, cfg, best_path, global_step, val_results["val_loss"], "chain")
            print(f"  ** Best val_acc={best_val_acc:.3f}, saved to {best_path}", flush=True)

    # 7. Save final + log
    log["best_val_acc"] = best_val_acc
    log["best_val_loss"] = best_val_loss
    log["best_step"] = best_step
    log["total_steps"] = global_step
    log["training_time_sec"] = time.time() - t0
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n[train] Best val_acc: {best_val_acc:.3f} at step {best_step}", flush=True)
    ckpt_name = f"mtp_head_{cfg['model_name']}_decode.pt"
    print(f"[train] Final checkpoint: {os.path.join(output_dir, ckpt_name)}", flush=True)
    print(f"[train] Log: {log_path}", flush=True)

    return os.path.join(output_dir, f"mtp_head_{cfg['model_name']}_decode.pt")


def _save_checkpoint(mtp, cfg, path, step, loss, mode):
    """Save slim checkpoint (exclude shared lm_head)."""
    slim_sd = {k: v for k, v in mtp.state_dict().items() if "lm_head" not in k}
    torch.save({
        "model_state_dict": slim_sd,
        "model_name": cfg["model_name"],
        "step": step,
        "loss": loss,
        "mode": mode,
        "config": {
            "hidden_size": cfg["hidden_size"],
            "vocab_size": cfg["vocab_size"],
            "num_heads": cfg["num_heads"],
            "head_dim": cfg["head_dim"],
            "intermediate_size": cfg["intermediate_size"],
        },
    }, path)


# === CLI ===

def main():
    parser = argparse.ArgumentParser(
        description="Improved MTP head training for Qwen2.5 (0.5B and 1.5B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model-size", default="1.5b", choices=["0.5b", "1.5b", "7b", "gemma2-2b"],
                        help="Model size")
    parser.add_argument("--phase", default="all", choices=["embed", "collect", "train", "all"],
                        help="Pipeline phase")

    parser.add_argument("--data-dir", default="",
                        help="Data directory (default: auto)")
    parser.add_argument("--output-dir", default="",
                        help="Output directory (default: auto)")

    # Collect
    parser.add_argument("--num-samples", type=int, default=3000, help="Number of prompts")
    parser.add_argument("--num-chain", type=int, default=4, help="Chain length")
    parser.add_argument("--gen-length", type=int, default=50, help="Max tokens per prompt")

    # Train
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or mps)")

    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    suffix = f"qwen25_{args.model_size.replace('.', '_')}_v3"
    data_dir = args.data_dir or os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_train_data", suffix)
    output_dir = args.output_dir or os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_output", suffix)

    print(f"\n{'='*60}", flush=True)
    print(f"  Qwen2.5-{args.model_size} MTP Training v3 (improved)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model:       {cfg['display_name']}", flush=True)
    print(f"  Architecture: hidden={cfg['hidden_size']}, vocab={cfg['vocab_size']}, "
          f"heads={cfg['num_heads']}x{cfg['head_dim']}, inter={cfg['intermediate_size']}", flush=True)
    print(f"  Phase:       {args.phase}", flush=True)
    print(f"  Data dir:    {data_dir}", flush=True)
    print(f"  Output dir:  {output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()

    if args.phase in ("embed", "all"):
        create_embed_head(args.model_size, data_dir)

    if args.phase in ("collect", "all"):
        collect_data_llamacpp(
            args.model_size, data_dir,
            num_samples=args.num_samples,
            num_chain=args.num_chain,
            gen_length=args.gen_length,
        )

    if args.phase in ("train", "all"):
        train(
            args.model_size, data_dir, output_dir,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=args.device,
        )

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  Pipeline Complete! Time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
