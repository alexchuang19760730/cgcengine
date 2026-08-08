#!/usr/bin/env python3
"""Standalone MTP head training for Qwen2.5-0.5B-Instruct on Mac (CPU).

End-to-end pipeline:
  1. Load Qwen2.5-0.5B-Instruct via transformers (auto-download from HF)
  2. Collect decode hidden states (same format as collect_hidden_states.py)
  3. Create MTPHead with Qwen2.5-0.5B config
  4. Train with chain CE mode (proven for DSV4)
  5. Save checkpoint compatible with mtp_verify_loop.py

Qwen2.5-0.5B-Instruct config:
  hidden_size=896, vocab_size=151936, num_heads=14, head_dim=64
  intermediate_size=4864, rope_theta=1000000.0, rms_norm_eps=1e-6
  EOS: 151643 (<endoftext>), 151645 (<im_end>)

Usage:
  python3 train_qwen25_mtp.py                           # Full pipeline
  python3 train_qwen25_mtp.py --phase collect           # Collect only
  python3 train_qwen25_mtp.py --phase train             # Train only
  python3 train_qwen25_mtp.py --num-samples 500         # Quick test
  python3 train_qwen25_mtp.py --epochs 5 --lr 3e-4      # Custom hyperparams
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

# === Qwen2.5-0.5B Config ===
QWEN25_CONFIG = {
    "model_name": "qwen25-0.5b",
    "display_name": "Qwen2.5-0.5B-Instruct",
    "hf_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "hidden_size": 896,
    "vocab_size": 151936,
    "num_heads": 14,
    "head_dim": 64,
    "intermediate_size": 4864,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 32768,
    "eos_tokens": {151643, 151645},  # <endoftext>, <im_end>
}

# === Prompt corpus (reused from collect_hidden_states.py) ===
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
    """Generate diverse prompt list."""
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


# === Data Collection ===

def collect_one_prompt(
    model,
    tokenizer,
    embed_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    prompt: str,
    eos_tokens: set,
    gen_length: int = 50,
    num_chain: int = 4,
    max_input_length: int = 256,
    device: str = "cpu",
) -> list[dict]:
    """Collect decode hidden states for a single prompt.

    Returns list of chain samples:
      [{"hidden_states": [chain, hidden], "token_ids": [chain], "next_token_ids": [chain]}]
    """
    # 1. Tokenize
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [t for t in input_ids if t not in eos_tokens]
    if len(input_ids) < 3:
        return []
    if len(input_ids) > max_input_length:
        input_ids = input_ids[:max_input_length]

    input_tensor = torch.tensor([input_ids], device=device)

    # 2. Prefill
    try:
        with torch.no_grad():
            outputs = model(input_tensor, output_hidden_states=True, use_cache=True)
    except Exception:
        return []

    kv_cache = outputs.past_key_values
    prefill_last_hidden = outputs.hidden_states[-1][0, -1]  # [hidden]

    # 3. Generate first token (greedy)
    with torch.no_grad():
        first_logits = F.linear(prefill_last_hidden, lm_head_weight)
    first_token = int(first_logits.argmax().item())

    # 4. Decode loop - collect decode hidden states
    decode_hiddens: list[torch.Tensor] = []
    decode_tokens: list[int] = []
    current_token = first_token

    for step in range(gen_length):
        try:
            with torch.no_grad():
                decode_out = model(
                    torch.tensor([[current_token]], device=device),
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                    use_cache=True,
                )
        except Exception:
            break

        decode_hidden = decode_out.hidden_states[-1][0, 0]  # [hidden]
        kv_cache = decode_out.past_key_values

        with torch.no_grad():
            next_logits = F.linear(decode_hidden, lm_head_weight)
        next_token = int(next_logits.argmax().item())

        decode_hiddens.append(decode_hidden.cpu())
        decode_tokens.append(current_token)

        if next_token in eos_tokens:
            break
        current_token = next_token

    # Free KV cache
    del kv_cache, outputs
    if 'decode_out' in locals():
        del decode_out

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


def collect_data(
    output_dir: str,
    num_samples: int = 2000,
    num_chain: int = 4,
    gen_length: int = 50,
    shard_size: int = 500,
    device: str = "cpu",
):
    """Collect decode hidden states from Qwen2.5-0.5B-Instruct."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = QWEN25_CONFIG
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model
    print(f"[collect] Loading {cfg['display_name']} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_model_id"],
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    # 2. Extract embed + lm_head weights
    embed_weight = model.get_input_embeddings().weight.data.clone()
    lm_head_weight = model.get_output_embeddings().weight.data.clone()

    # Check if tied
    if lm_head_weight is None or lm_head_weight.shape[0] == 0:
        lm_head_weight = embed_weight.clone()
        print("[collect] Using tied embeddings (embed = lm_head)", flush=True)

    print(f"[collect] embed: {embed_weight.shape}, lm_head: {lm_head_weight.shape}", flush=True)

    # 3. Save embed + lm_head weights
    embed_head_path = os.path.join(output_dir, "embed_head.pt")
    torch.save({
        "embed_weight": embed_weight.cpu().to(torch.float32),
        "lm_head_weight": lm_head_weight.cpu().to(torch.float32),
        "hidden_size": cfg["hidden_size"],
        "vocab_size": cfg["vocab_size"],
        "model_name": cfg["model_name"],
    }, embed_head_path)
    print(f"[collect] Saved embed_head.pt ({os.path.getsize(embed_head_path) / 1e6:.0f}MB)", flush=True)

    # 4. Generate prompts
    prompts = generate_prompts(num_samples)
    print(f"[collect] {len(prompts)} prompts generated", flush=True)

    # 5. Collect
    shard_idx = 0
    shard_data: list[dict] = []
    total_collected = 0
    total_failed = 0
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        try:
            samples = collect_one_prompt(
                model, tokenizer, embed_weight, lm_head_weight,
                prompt, cfg["eos_tokens"],
                gen_length=gen_length, num_chain=num_chain,
                device=device,
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

        # Save shard
        if len(shard_data) >= shard_size:
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
            torch.save(shard_data, shard_path)
            elapsed = time.time() - t0
            rate = total_collected / elapsed if elapsed > 0 else 0
            print(
                f"[collect] Shard {shard_idx}: {len(shard_data)} samples "
                f"(total: {total_collected}, failed: {total_failed}, "
                f"rate: {rate:.1f}/s, elapsed: {elapsed:.0f}s)",
                flush=True,
            )
            shard_data = []
            shard_idx += 1

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"[collect] Progress: {i + 1}/{len(prompts)} "
                f"(samples: {total_collected}, failed: {total_failed}, "
                f"rate: {rate:.1f} prompts/s)",
                flush=True,
            )

    # Save final shard
    if shard_data:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)
        shard_idx += 1

    # 6. Save metadata
    meta = {
        "model_name": cfg["model_name"],
        "model_path": cfg["hf_model_id"],
        "hidden_size": cfg["hidden_size"],
        "vocab_size": cfg["vocab_size"],
        "num_chain": num_chain,
        "gen_length": gen_length,
        "total_samples": total_collected,
        "total_failed": total_failed,
        "num_shards": shard_idx,
        "collection_time_sec": time.time() - t0,
        "hidden_type": "decode",
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[collect] Done! {total_collected} samples in {shard_idx} shards", flush=True)
    print(f"[collect] Time: {time.time() - t0:.0f}s", flush=True)

    # Free model
    del model
    return total_collected


# === Dataset ===

class MTPDataset(Dataset):
    """Loads chain training samples from collected shards."""

    def __init__(self, data_dir: str):
        self.samples: list[dict] = []
        shard_files = sorted(Path(data_dir).glob("shard_*.pt"))

        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {data_dir}")

        for shard_path in shard_files:
            try:
                shard = torch.load(shard_path, weights_only=False)
                if isinstance(shard, list):
                    self.samples.extend(shard)
                elif isinstance(shard, dict) and "samples" in shard:
                    self.samples.extend(shard["samples"])
            except Exception as e:
                print(f"  [dataset] Error loading {shard_path}: {e}", file=sys.stderr)

        print(f"[dataset] Loaded {len(self.samples)} samples from {len(shard_files)} shards")

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


# === Training: chain mode (CE, proven for DSV4) ===

def train_chain_step(
    mtp: nn.Module,
    embed_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Single chain training step."""
    batch_size, chain_len, hidden_size = hidden_states.shape
    total_loss = torch.tensor(0.0, device=device)
    correct = 0
    total = 0

    current_hidden = hidden_states[:, 0, :]

    for k in range(chain_len):
        token_embed = F.embedding(token_ids[:, k], embed_weight)

        h_3d = current_hidden.unsqueeze(1)
        e_3d = token_embed.unsqueeze(1)

        # Manual forward for chaining
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

    total_loss = total_loss / chain_len
    return total_loss, {"chain_acc": correct / max(total, 1)}


# === Main training function ===

def train(
    data_dir: str,
    output_dir: str,
    mode: str = "chain",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-4,
    warmup_steps: int = 50,
    save_every: int = 1000,
    checkpoint: str = "",
    device: str = "cpu",
):
    """Train MTP head for Qwen2.5-0.5B."""
    from model import MTPHead, MTPHeadConfig

    cfg = QWEN25_CONFIG
    os.makedirs(output_dir, exist_ok=True)

    dev = torch.device(device)
    print(f"[train] Device: {dev}", flush=True)

    # 1. Load dataset
    dataset = MTPDataset(data_dir)
    if len(dataset) < 10:
        print(f"[train] Too few samples ({len(dataset)}), need at least 10. Aborting.", flush=True)
        return None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # 2. Load embed + lm_head weights
    embed_head_path = os.path.join(data_dir, "embed_head.pt")
    if not os.path.exists(embed_head_path):
        raise FileNotFoundError(f"embed_head.pt not found in {data_dir}")

    print(f"[train] Loading embed+head from {embed_head_path}...", flush=True)
    weights = torch.load(embed_head_path, map_location="cpu", weights_only=True)
    embed_weight = weights["embed_weight"].to(dev)
    lm_head_weight = weights["lm_head_weight"].to(dev)
    print(f"[train] embed: {embed_weight.shape}, lm_head: {lm_head_weight.shape}", flush=True)

    # 3. Create MTP head
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
    mtp.set_shared_lm_head(lm_head_weight)
    print(f"[train] MTP head: {mtp.num_parameters() / 1e6:.1f}M trainable params", flush=True)

    # Load checkpoint if provided
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        filtered = {k: v for k, v in sd.items() if "lm_head" not in k}
        mtp.load_state_dict(filtered, strict=False)
        print(f"[train] Loaded checkpoint: {checkpoint}", flush=True)

    mtp.to(dev).to(torch.float32)
    mtp.train()

    # 4. Optimizer + scheduler
    optimizer = torch.optim.AdamW(mtp.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(loader) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Training log
    log = {
        "config": {
            "model_name": cfg["model_name"],
            "display_name": cfg["display_name"],
            "hidden_size": cfg["hidden_size"],
            "vocab_size": cfg["vocab_size"],
            "mode": mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "num_params_m": round(mtp.num_parameters() / 1e6, 1),
            "num_samples": len(dataset),
        },
        "steps": [],
    }

    # 6. Training loop
    global_step = 0
    t0 = time.time()
    avg_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_metrics: dict[str, float] = {}
        num_batches = 0
        random.shuffle(dataset.samples)

        for batch_idx, batch in enumerate(loader):
            hidden_states = batch["hidden_states"].to(dev)
            token_ids = batch["token_ids"].to(dev)
            next_token_ids = batch["next_token_ids"].to(dev)

            # Forward + loss
            if mode == "chain":
                loss, metrics = train_chain_step(
                    mtp, embed_weight, lm_head_weight,
                    hidden_states, token_ids, next_token_ids, dev,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            num_batches += 1
            global_step += 1

            # Log
            if global_step % 20 == 0:
                elapsed = time.time() - t0
                avg = epoch_loss / num_batches
                avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
                current_lr = scheduler.get_last_lr()[0]
                metric_str = " ".join(f"{k}={v:.3f}" for k, v in avg_metrics.items())

                print(
                    f"  [E{epoch+1}/{epochs}] S{global_step}/{total_steps} | "
                    f"loss={avg:.4f} {metric_str} | "
                    f"lr={current_lr:.2e} | {elapsed:.0f}s",
                    flush=True,
                )

                log["steps"].append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": round(avg, 4),
                    **{k: round(v / num_batches, 4) for k, v in epoch_metrics.items()},
                    "lr": current_lr,
                    "elapsed_s": round(elapsed, 1),
                })

            # Checkpoint (keep only latest to save disk)
            if global_step % save_every == 0:
                ckpt_path = os.path.join(output_dir, f"mtp_head_step_{global_step}.pt")
                _save_checkpoint(mtp, cfg, ckpt_path, global_step, epoch_loss / num_batches, mode)
                print(f"  Checkpoint: {ckpt_path}", flush=True)
                # Delete previous intermediate checkpoint to save disk
                prev_step = global_step - save_every
                if prev_step > 0:
                    prev_ckpt = os.path.join(output_dir, f"mtp_head_step_{prev_step}.pt")
                    if os.path.exists(prev_ckpt):
                        os.remove(prev_ckpt)
                        print(f"  Cleaned old checkpoint: step_{prev_step}", flush=True)

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        metric_str = " ".join(f"{k}={v:.3f}" for k, v in avg_metrics.items())
        print(
            f"\n  Epoch {epoch+1}: avg_loss={avg_loss:.4f} {metric_str} "
            f"({time.time()-t0:.0f}s)\n",
            flush=True,
        )

    # 7. Save final checkpoint
    final_path = os.path.join(output_dir, f"mtp_head_{cfg['model_name']}_decode.pt")
    _save_checkpoint(mtp, cfg, final_path, global_step, avg_loss, mode)
    print(f"[train] Final model: {final_path}", flush=True)

    # 8. Evaluation on training data
    mtp.eval()
    eval_correct = 0
    eval_total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 50:
                break
            hidden_states = batch["hidden_states"].to(dev)
            token_ids = batch["token_ids"].to(dev)
            next_token_ids = batch["next_token_ids"].to(dev)

            current_hidden = hidden_states[:, 0, :]
            for k in range(hidden_states.shape[1]):
                token_embed = F.embedding(token_ids[:, k], embed_weight)
                h_3d = current_hidden.unsqueeze(1)
                e_3d = token_embed.unsqueeze(1)
                logits_3d = mtp(h_3d, e_3d)
                logits = logits_3d[:, 0, :]
                pred = logits.argmax(dim=-1)
                eval_correct += (pred == next_token_ids[:, k]).sum().item()
                eval_total += pred.shape[0]
                if k < hidden_states.shape[1] - 1:
                    current_hidden = hidden_states[:, k + 1]

    chain_acc = eval_correct / max(eval_total, 1)
    print(f"[train] Chain accept rate (train data, {hidden_states.shape[1]} steps): "
          f"{eval_correct}/{eval_total} = {chain_acc:.1%}", flush=True)
    print("[train] NOTE: Train data accept rate. Real accept must be tested on NEW prompts.", flush=True)

    # 9. Save training log
    log["final_loss"] = avg_loss
    log["total_steps"] = global_step
    log["training_time_sec"] = time.time() - t0
    log["train_chain_acc"] = chain_acc
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[train] Log: {log_path}", flush=True)

    return final_path


def _save_checkpoint(mtp, cfg, path, step, loss, mode):
    """Save slim checkpoint (exclude shared lm_head to keep file small)."""
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
        description="Standalone MTP head training for Qwen2.5-0.5B-Instruct on Mac (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--phase", default="all", choices=["collect", "train", "all"],
                        help="Pipeline phase")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_train_data", "qwen25"),
                        help="Data directory for collected shards")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_output", "qwen25"),
                        help="Output directory for checkpoints")

    # Collect options
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of prompts to collect")
    parser.add_argument("--num-chain", type=int, default=4, help="Chain length")
    parser.add_argument("--gen-length", type=int, default=50, help="Max tokens per prompt")
    parser.add_argument("--shard-size", type=int, default=500, help="Samples per shard")

    # Train options
    parser.add_argument("--mode", default="chain", choices=["chain"],
                        help="Training mode")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint", default="", help="Existing checkpoint to resume from")

    # Device
    parser.add_argument("--device", default="cpu", help="Device (cpu or mps)")

    args = parser.parse_args()

    cfg = QWEN25_CONFIG
    print(f"\n{'='*60}", flush=True)
    print(f"  Qwen2.5-0.5B MTP Head Training (Standalone, Mac CPU)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model:       {cfg['display_name']}", flush=True)
    print(f"  Architecture: hidden={cfg['hidden_size']}, vocab={cfg['vocab_size']}, "
          f"heads={cfg['num_heads']}x{cfg['head_dim']}", flush=True)
    print(f"  Phase:       {args.phase}", flush=True)
    print(f"  Data dir:    {args.data_dir}", flush=True)
    print(f"  Output dir:  {args.output_dir}", flush=True)
    print(f"  Device:      {args.device}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()

    # === Phase: Collect ===
    if args.phase in ("collect", "all"):
        collect_data(
            output_dir=args.data_dir,
            num_samples=args.num_samples,
            num_chain=args.num_chain,
            gen_length=args.gen_length,
            shard_size=args.shard_size,
            device=args.device,
        )

    # === Phase: Train ===
    if args.phase in ("train", "all"):
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint=args.checkpoint,
            device=args.device,
        )

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  Pipeline Complete! Time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
