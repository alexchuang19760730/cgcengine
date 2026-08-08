"""MLX 数据收集 — 用 MLX model forward 收集 (hidden_state, next_token) 对.

解决 PyTorch→MLX 跨框架数值差异: 训练数据必须用 MLX hidden.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx_lm import load
from mlx_lm.models.qwen3 import create_attention_mask


def collect_hidden_states_mlx(
    model,
    tokenizer,
    text: str,
    max_length: int = 512,
) -> list[dict]:
    """用 MLX model forward 收集 hidden_states + next_token 对.

    Returns:
        samples: [{hidden_state: np.ndarray, token_id: int, next_token_id: int}, ...]
    """
    # Tokenize
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    # 过滤 special tokens (对齐推理)
    input_ids = [t for t in input_ids if t not in (151644, 151645)]
    if len(input_ids) < 2:
        return []

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    y = mx.array(input_ids, mx.uint32)  # [seq_len]

    # 获取 language_model + inner model
    lm = model.language_model
    inner = lm.model  # Qwen3Model
    embed = inner.embed_tokens

    # 手动 forward (不调 norm), 获取 norm 前 hidden (和推理一致)
    h = embed(y[None])  # [1, seq_len, hidden]
    # 创建 cache (不需要持久, 只为 forward)
    cache = [None] * len(inner.layers)
    mask = create_attention_mask(h, cache[0])
    for layer, c in zip(inner.layers, cache):
        h = layer(h, mask, c)
    # h: [1, seq_len, hidden] norm 前

    # 转为 numpy
    hidden_np = np.array(h[0])  # [seq_len, hidden] float32

    samples = []
    for i in range(len(input_ids) - 1):
        samples.append({
            "hidden_state": hidden_np[i],  # [hidden]
            "token_id": input_ids[i],
            "next_token_id": input_ids[i + 1],
        })

    return samples


def collect_from_corpus_mlx(
    model,
    tokenizer,
    corpus_path: str,
    output_dir: str,
    max_samples: int = 50000,
    max_length: int = 512,
    shard_size: int = 10000,
) -> int:
    """从语料收集训练数据 (MLX hidden)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载语料
    print(f"[collect] Loading corpus: {corpus_path}")
    with open(corpus_path) as f:
        corpus = [json.loads(line) for line in f]
    print(f"[collect] Corpus: {len(corpus)} entries")

    all_samples = []
    shard_idx = 0
    total = 0

    t0 = time.time()

    for i, entry in enumerate(corpus):
        if total >= max_samples:
            break

        # 提取 text
        if "text" in entry:
            text = entry["text"]
        elif "messages" in entry:
            # chat format
            msgs = entry["messages"]
            text_parts = []
            for msg in msgs:
                if msg["role"] == "user":
                    text_parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    text_parts.append(f"Assistant: {msg['content']}")
            text = "\n".join(text_parts)
        elif "instruction" in entry:
            text = entry.get("instruction", "")
            if entry.get("input"):
                text += "\n" + entry["input"]
            if entry.get("output"):
                text += "\n" + entry["output"]
        else:
            continue

        if not text or len(text) < 10:
            continue

        # 收集 hidden states
        try:
            samples = collect_hidden_states_mlx(model, tokenizer, text, max_length)
            all_samples.extend(samples)
            total += len(samples)
        except Exception as e:
            continue

        # 分片保存
        if len(all_samples) >= shard_size:
            shard_path = output_path / f"shard_{shard_idx:04d}.pt"
            torch.save(all_samples, shard_path)
            print(f"[collect] Shard {shard_idx}: {len(all_samples)} samples → {shard_path}")
            all_samples = []
            shard_idx += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f"[collect] {i+1}/{len(corpus)} entries, {total} samples, {rate:.0f} samples/s")

    # 保存剩余
    if all_samples:
        shard_path = output_path / f"shard_{shard_idx:04d}.pt"
        torch.save(all_samples, shard_path)
        print(f"[collect] Final shard {shard_idx}: {len(all_samples)} samples → {shard_path}")

    elapsed = time.time() - t0
    print(f"[collect] Done: {total} samples in {elapsed:.1f}s ({total/elapsed:.0f} samples/s)")
    return total


if __name__ == "__main__":
    import sys

    model_path = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mtp_corpus.jsonl"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mtp_mlx_data"
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5000  # 默认 5K (Mac 上慢)

    print(f"[main] Model: {model_path}")
    print(f"[main] Corpus: {corpus_path}")
    print(f"[main] Output: {output_dir}")
    print(f"[main] Max samples: {max_samples}")

    # 加载模型
    print("[main] Loading MLX model...")
    model, tokenizer = load(model_path)
    print(f"[main] Model loaded: {len(model.language_model.model.layers)} layers")

    # 检查语料
    if not Path(corpus_path).exists():
        print(f"[main] Corpus not found: {corpus_path}")
        print("[main] Downloading Alpaca...")
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        with open(corpus_path, "w") as f:
            for entry in ds:
                f.write(json.dumps({
                    "instruction": entry["instruction"],
                    "input": entry.get("input", ""),
                    "output": entry["output"],
                }) + "\n")
        print(f"[main] Corpus saved: {corpus_path}")

    # 收集
    total = collect_from_corpus_mlx(
        model, tokenizer, corpus_path, output_dir, max_samples
    )
    print(f"\n[main] Total: {total} samples")
    print(f"[main] Output: {output_dir}")
