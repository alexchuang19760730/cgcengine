"""链式 MTP head fine-tune — 多卡并行版本.

架构:
  Phase 1 (收集): 8 进程并行, 每进程 1 GPU 处理 1/8 corpus, 收集 chained samples 存 shard.
  Phase 2 (训练): 主进程合并所有 shards, 单卡训练 MTP head (60M 参数, 单卡足够).

用法:
  python train_chained_multi.py --world-size 8 --max-samples 50000 --epochs 3

对比单卡串行: 50K corpus 收集约 2.8h → 8 卡并行约 20min.
"""
from __future__ import annotations

import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional
import sys
import random


# ============ 复用原版逻辑 ============

def load_base_model(model_path: str, device: str = "cuda"):
    """加载 base model (统一加载器)."""
    import sys
    sys.path.insert(0, "/root/flashkv0516/app/shared")
    from model_loader import load_base_model as _load
    return _load(model_path, device=device)


def collect_chained_data(
    base_model,
    tokenizer,
    mtp_head,
    text: str,
    num_chain: int = 4,
    max_length: int = 512,
    device: str = "cuda",
) -> list[dict]:
    """链式数据收集 — 模拟推理时的链式 draft."""
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = [t for t in input_ids if t not in (151644, 151645)]
    if len(input_ids) < num_chain + 1:
        return []

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = base_model(input_tensor, output_hidden_states=True, use_cache=False)

    hidden_states_all = outputs.hidden_states[-1][0]  # [seq_len, hidden]
    embed_weights = base_model.model.language_model.embed_tokens.weight

    samples = []
    mtp_head.eval()

    for i in range(len(input_ids) - num_chain):
        current_hidden = hidden_states_all[i]
        chain_hidden = []
        chain_tokens = []
        chain_next_tokens = []

        for k in range(num_chain):
            token_id = input_ids[i + k]
            next_token_id = input_ids[i + k + 1]
            token_embed = embed_weights[token_id]

            with torch.no_grad():
                h_3d = current_hidden.unsqueeze(0).unsqueeze(0)
                e_3d = token_embed.unsqueeze(0).unsqueeze(0)
                concat_input = torch.cat([h_3d, e_3d], dim=-1)
                x = mtp_head.proj(concat_input)
                x = mtp_head.norm1(x)
                x = x + mtp_head.attn(x)
                x = mtp_head.norm2(x)
                x = x + mtp_head.mlp(x)
                mtp_out = mtp_head.norm_out(x)

            chain_hidden.append(current_hidden.cpu())
            chain_tokens.append(token_id)
            chain_next_tokens.append(next_token_id)

            current_hidden = mtp_out[0, 0]

        samples.append({
            "hidden_states": torch.stack(chain_hidden),
            "token_ids": torch.tensor(chain_tokens),
            "next_token_ids": torch.tensor(chain_next_tokens),
        })

    return samples


def extract_text(entry: dict) -> str:
    """从多种 corpus 格式提取 text."""
    if "text" in entry and entry["text"]:
        return entry["text"]
    if "messages" in entry:
        return " ".join(m.get("content", "") for m in entry["messages"])
    if "instruction" in entry:
        text = entry.get("instruction", "")
        if entry.get("input"):
            text += "\n" + entry["input"]
        if entry.get("output"):
            text += "\n" + entry["output"]
        return text
    return ""


def load_mtp_head(mtp_checkpoint: str, device: str, lm_head_weight):
    """加载 MTP head."""
    import sys
    sys.path.insert(0, "/root/flashkv0516/CGC_Phase2/mtp_head")
    from model import MTPHead, MTPHeadConfig
    mtp = MTPHead(MTPHeadConfig())
    mtp.set_shared_lm_head(lm_head_weight)
    ckpt = torch.load(mtp_checkpoint, weights_only=False, map_location="cpu")
    mtp.load_state_dict(ckpt["model_state_dict"], strict=False)
    mtp.to(device).to(torch.bfloat16)
    return mtp


# ============ Phase 1: 多卡并行收集 ============

def collect_worker(
    rank: int,
    world_size: int,
    corpus_path: str,
    base_model_path: str,
    mtp_checkpoint: str,
    shard_dir: str,
    num_chain: int,
    max_samples_per_worker: int,
):
    """每个 worker 进程: 1 GPU, 处理 corpus 的 1/world_size, 存 shard."""
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    print(f"[rank {rank}] Loading base model on {device}...", flush=True)
    base_model, tokenizer = load_base_model(base_model_path, device)
    base_model.eval()

    embed_weights = base_model.model.language_model.embed_tokens.weight
    lm_head = getattr(base_model, "lm_head", getattr(base_model.model.language_model, "lm_head", None))
    lm_head_weight = lm_head.weight if hasattr(lm_head, "weight") else embed_weights

    mtp = load_mtp_head(mtp_checkpoint, device, lm_head_weight)
    mtp.eval()

    # 读 corpus 并切分
    with open(corpus_path) as f:
        all_lines = f.readlines()
    total = len(all_lines)
    start = rank * total // world_size
    end = (rank + 1) * total // world_size
    my_lines = all_lines[start:end]
    print(f"[rank {rank}] corpus[{start}:{end}] = {len(my_lines)} entries, target {max_samples_per_worker} samples", flush=True)

    all_samples = []
    t0 = time.time()
    for i, line in enumerate(my_lines):
        if len(all_samples) >= max_samples_per_worker:
            break
        try:
            entry = json.loads(line)
        except Exception:
            continue
        text = extract_text(entry)
        if not text or len(text) < 20:
            continue
        try:
            samples = collect_chained_data(base_model, tokenizer, mtp, text, num_chain, device=device)
            all_samples.extend(samples)
        except Exception as e:
            continue
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(my_lines) - i - 1) / rate if rate > 0 else 0
            print(f"[rank {rank}] {i+1}/{len(my_lines)} entries, {len(all_samples)} samples, {rate:.1f} entry/s, ETA {eta:.0f}s", flush=True)

    shard_path = f"{shard_dir}/shard_{rank}.pt"
    torch.save(all_samples, shard_path)
    elapsed = time.time() - t0
    print(f"[rank {rank}] DONE: {len(all_samples)} samples in {elapsed:.0f}s -> {shard_path}", flush=True)


# ============ Phase 2: 单卡训练 ============

def train_on_shards(
    base_model_path: str,
    mtp_checkpoint: str,
    shard_dir: str,
    output_dir: str,
    num_chain: int = 4,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    device: str = "cuda:0",
):
    """合并所有 shards, 单卡训练 MTP head."""
    print(f"[train] Loading base model on {device} for lm_head/embed...", flush=True)
    base_model, tokenizer = load_base_model(base_model_path, device)
    base_model.eval()

    embed_weights = base_model.model.language_model.embed_tokens.weight
    lm_head = getattr(base_model, "lm_head", getattr(base_model.model.language_model, "lm_head", None))
    lm_head_weight = lm_head.weight if hasattr(lm_head, "weight") else embed_weights

    mtp = load_mtp_head(mtp_checkpoint, device, lm_head_weight)
    mtp.train()

    # 合并 shards
    shard_paths = sorted(Path(shard_dir).glob("shard_*.pt"))
    print(f"[train] Merging {len(shard_paths)} shards...", flush=True)
    all_samples = []
    for p in shard_paths:
        shard = torch.load(p, weights_only=False)
        print(f"  {p.name}: {len(shard)} samples", flush=True)
        all_samples.extend(shard)
    print(f"[train] Total: {len(all_samples)} samples", flush=True)

    if len(all_samples) == 0:
        print("[train] No samples! Aborting.", flush=True)
        return

    # 训练
    optimizer = torch.optim.AdamW(mtp.parameters(), lr=lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(all_samples) // batch_size
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    avg_loss = 0.0
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        random.shuffle(all_samples)
        t0 = time.time()

        for i in range(0, len(all_samples), batch_size):
            batch = all_samples[i:i + batch_size]
            if len(batch) < 2:
                continue

            hidden = torch.stack([s["hidden_states"] for s in batch]).to(device)
            tokens = torch.stack([s["token_ids"] for s in batch]).to(device)
            next_tokens = torch.stack([s["next_token_ids"] for s in batch]).to(device)

            # 转 bfloat16 对齐
            hidden = hidden.to(torch.bfloat16)

            loss = 0
            current_hidden = hidden[:, 0, :]

            for k in range(num_chain):
                token_embed = F.embedding(tokens[:, k], embed_weights)
                concat_input = torch.cat(
                    [current_hidden.unsqueeze(1), token_embed.unsqueeze(1)], dim=-1
                )
                x = mtp.proj(concat_input)
                x = mtp.norm1(x)
                x = x + mtp.attn(x)
                x = mtp.norm2(x)
                x = x + mtp.mlp(x)
                mtp_hidden = mtp.norm_out(x)

                logits = F.linear(mtp_hidden[:, 0, :], lm_head_weight)
                loss += F.cross_entropy(logits, next_tokens[:, k])

                current_hidden = mtp_hidden[:, 0, :].detach()

            loss = loss / num_chain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                elapsed = time.time() - t0
                rate = num_batches / elapsed
                eta = (len(all_samples) // batch_size - num_batches) / rate if rate > 0 else 0
                log(f"  Epoch {epoch+1} batch {num_batches}/{len(all_samples)//batch_size}: loss={loss.item():.4f}, {rate:.1f} batch/s, ETA {eta:.0f}s")

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0
        log(f"[train] Epoch {epoch+1}: avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")

    # 保存
    output_path = Path(output_dir) / "mtp_head_chained.pt"
    torch.save({
        "model_state_dict": mtp.state_dict(),
        "epoch": epochs,
        "loss": avg_loss,
        "num_chain": num_chain,
        "training": "chained_multi",
        "num_samples": len(all_samples),
    }, output_path)
    log(f"[train] Saved: {output_path}")

    # 评估
    mtp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for s in all_samples[:200]:
            hidden = s["hidden_states"][0].to(device).to(torch.bfloat16)
            token_embed = F.embedding(s["token_ids"][0].to(device), embed_weights)
            mtp_out = mtp(hidden.unsqueeze(0).unsqueeze(0), token_embed.unsqueeze(0).unsqueeze(0))
            pred = mtp_out[0, 0].argmax().item()
            if pred == s["next_token_ids"][0].item():
                correct += 1
            total += 1
    if total > 0:
        log(f"[train] Single token accuracy: {correct}/{total} = {correct/total:.1%}")

    # 链式 accept rate 评估
    chain_correct = 0
    chain_total = 0
    with torch.no_grad():
        for s in all_samples[:200]:
            hidden_states = s["hidden_states"].to(device).to(torch.bfloat16)
            token_ids = s["token_ids"].to(device)
            next_token_ids = s["next_token_ids"].to(device)
            current_hidden = hidden_states[0]
            for k in range(num_chain):
                token_embed = F.embedding(token_ids[k], embed_weights)
                h_3d = current_hidden.unsqueeze(0).unsqueeze(0)
                e_3d = token_embed.unsqueeze(0).unsqueeze(0)
                concat_input = torch.cat([h_3d, e_3d], dim=-1)
                x = mtp.proj(concat_input)
                x = mtp.norm1(x)
                x = x + mtp.attn(x)
                x = mtp.norm2(x)
                x = x + mtp.mlp(x)
                mtp_out = mtp.norm_out(x)
                logits = F.linear(mtp_out[0, 0], lm_head_weight)
                pred = logits.argmax().item()
                if pred == next_token_ids[k].item():
                    chain_correct += 1
                chain_total += 1
                current_hidden = mtp_out[0, 0]
    if chain_total > 0:
        log(f"[train] Chain accept rate ({num_chain} steps): {chain_correct}/{chain_total} = {chain_correct/chain_total:.1%}")

    return output_path


# ============ Main ============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/data2/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--mtp-checkpoint", default="/data/mtp_head_output/mtp_head_final.pt")
    parser.add_argument("--corpus", default="/data/mtp_corpus.jsonl")
    parser.add_argument("--output", default="/data/mtp_head_chained_50k_multi")
    parser.add_argument("--num-chain", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-dir", default="/data/mtp_chain_shards")
    parser.add_argument("--mode", default="all", choices=["all", "collect", "train"])
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.shard_dir).mkdir(parents=True, exist_ok=True)

    max_per_worker = args.max_samples // args.world_size + 1

    if args.mode in ("all", "collect"):
        # 清理旧 shards
        for old in Path(args.shard_dir).glob("shard_*.pt"):
            old.unlink()

        print(f"[main] Phase 1: collect with {args.world_size} GPUs, ~{max_per_worker} samples/worker", flush=True)
        t0 = time.time()
        mp.spawn(
            collect_worker,
            args=(args.world_size, args.corpus, args.base_model, args.mtp_checkpoint,
                  args.shard_dir, args.num_chain, max_per_worker),
            nprocs=args.world_size,
            join=True,
        )
        print(f"[main] Phase 1 done in {time.time()-t0:.0f}s", flush=True)

    if args.mode in ("all", "train"):
        print(f"[main] Phase 2: train on shards", flush=True)
        train_on_shards(
            args.base_model, args.mtp_checkpoint, args.shard_dir, args.output,
            args.num_chain, args.epochs, args.batch_size,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
