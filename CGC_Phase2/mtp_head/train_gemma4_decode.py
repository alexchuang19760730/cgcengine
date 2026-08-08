"""Gemma4-26B-A4B Decode-mode MTP 训练.

适配 Gemma4 (Gemma4ForConditionalGeneration):
  - hidden_size=2816, vocab_size=262144, head_dim=256
  - text_model.embed_tokens (非 language_model.embed_tokens)
  - EOS tokens: [1, 106]
  - MoE 128 experts (load 时需 trust_remote_code)

关键修复: 用 decode hidden (非 prefill hidden) 收集训练数据.

用法:
  # sglang 在 GPU 0-3, 训练用 GPU 4-7
  python train_gemma4_decode.py --world-size 4 --gpu-base 4 --max-samples 20000 --epochs 3

  # 或全 GPU (需先停 sglang)
  python train_gemma4_decode.py --world-size 8 --max-samples 50000 --epochs 3
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


def load_base_model(model_path: str, device: str = "cuda"):
    """加载 Gemma4 base model + tokenizer."""
    sys.path.insert(0, "/root/flashkv0516/app/shared")
    from model_loader import load_base_model as _load, get_embed_weight, get_lm_head_weight
    model, tokenizer = _load(model_path, device=device)
    # 验证 embed/lm_head 可访问
    embed = get_embed_weight(model)
    lm_head = get_lm_head_weight(model)
    if embed is None:
        raise RuntimeError("Cannot find embed_tokens for Gemma4")
    if lm_head is None:
        print("[WARN] lm_head not found, will use embed_weights (tied)")
    return model, tokenizer


def extract_text(entry: dict) -> str:
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
    """加载 Gemma4 MTP head."""
    sys.path.insert(0, "/root/flashkv0516/CGC_Phase2/mtp_head")
    from model import create_mtp_head_for_gemma4_26b
    mtp = create_mtp_head_for_gemma4_26b()
    mtp.set_shared_lm_head(lm_head_weight)
    if mtp_checkpoint and os.path.exists(mtp_checkpoint):
        ckpt = torch.load(mtp_checkpoint, weights_only=False, map_location="cpu")
        mtp.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[mtp] Loaded checkpoint: {mtp_checkpoint}")
    else:
        print("[mtp] No checkpoint, starting from scratch")
    mtp.to(device).to(torch.bfloat16)
    return mtp


def collect_decode_data(
    base_model,
    tokenizer,
    mtp_head,
    text: str,
    num_chain: int = 4,
    gen_length: int = 50,
    device: str = "cuda",
) -> list[dict]:
    """Decode-mode 数据收集 — 从生成序列中收集 decode hidden.

    训练样本: (decode_hidden[i], token[i]) -> token[i+1]
    """
    sys.path.insert(0, "/root/flashkv0516/app/shared")
    from model_loader import get_embed_weight, get_lm_head_weight

    input_ids = tokenizer.encode(text, add_special_tokens=False)
    # Gemma4 EOS tokens: [1, 106]
    eos_tokens = {1, 106}
    # Also filter Qwen special tokens if any leaked through
    eos_tokens.update({151644, 151645})
    input_ids = [t for t in input_ids if t not in eos_tokens]
    if len(input_ids) < 3:
        return []

    if len(input_ids) > 256:
        input_ids = input_ids[:256]

    input_tensor = torch.tensor([input_ids], device=device)
    embed_weights = get_embed_weight(base_model)
    lm_head_weight = get_lm_head_weight(base_model)
    if lm_head_weight is None:
        lm_head_weight = embed_weights  # tied embeddings

    # 1. Prefill (use_cache=True)
    with torch.no_grad():
        outputs = base_model(input_tensor, output_hidden_states=True, use_cache=True)
    kv_cache = outputs.past_key_values
    prefill_last_hidden = outputs.hidden_states[-1][0, -1]  # [hidden]

    # 2. 生成第一个 token
    with torch.no_grad():
        first_logits = F.linear(prefill_last_hidden, lm_head_weight)
    first_token = int(first_logits.argmax().item())

    # 3. 逐 token decode forward, 收集 decode hidden
    decode_hiddens = []
    decode_tokens = []
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 1

    current_token = first_token
    for step in range(gen_length):
        with torch.no_grad():
            decode_out = base_model(
                torch.tensor([[current_token]], device=device),
                past_key_values=kv_cache,
                output_hidden_states=True,
                use_cache=True,
            )
        decode_hidden = decode_out.hidden_states[-1][0, 0]  # [hidden] decode hidden
        kv_cache = decode_out.past_key_values

        with torch.no_grad():
            next_logits = F.linear(decode_hidden, lm_head_weight)
        next_token = int(next_logits.argmax().item())

        decode_hiddens.append(decode_hidden)
        decode_tokens.append(current_token)

        if next_token == eos_token_id or next_token in eos_tokens:
            break
        current_token = next_token

    if len(decode_hiddens) < num_chain + 1:
        return []

    # 4. 构造链式训练样本
    samples = []
    mtp_head.eval()

    for i in range(len(decode_hiddens) - num_chain):
        chain_hidden = []
        chain_tokens = []
        chain_next_tokens = []

        current_hidden = decode_hiddens[i]  # decode hidden (正确!)

        for k in range(num_chain):
            token_id = decode_tokens[i + k]
            next_token_id = decode_tokens[i + k + 1] if i + k + 1 < len(decode_tokens) else token_id

            with torch.no_grad():
                token_embed = embed_weights[token_id]
                h_3d = current_hidden.unsqueeze(0).unsqueeze(0)
                e_3d = token_embed.unsqueeze(0).unsqueeze(0)
                concat = torch.cat([h_3d, e_3d], dim=-1)
                x = mtp_head.proj(concat)
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
    gen_length: int,
    gpu_base: int,
):
    actual_gpu = gpu_base + rank
    torch.cuda.set_device(actual_gpu)
    device = f"cuda:{actual_gpu}"

    print(f"[rank {rank}] Loading Gemma4 on {device}...", flush=True)
    base_model, tokenizer = load_base_model(base_model_path, device)
    base_model.eval()

    sys.path.insert(0, "/root/flashkv0516/app/shared")
    from model_loader import get_embed_weight, get_lm_head_weight
    embed_weights = get_embed_weight(base_model)
    lm_head_weight = get_lm_head_weight(base_model)
    if lm_head_weight is None:
        lm_head_weight = embed_weights

    mtp = load_mtp_head(mtp_checkpoint, device, lm_head_weight)
    mtp.eval()

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
            samples = collect_decode_data(
                base_model, tokenizer, mtp, text,
                num_chain=num_chain, gen_length=gen_length, device=device,
            )
            all_samples.extend(samples)
        except Exception as e:
            continue
        if (i + 1) % 50 == 0:
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
    print(f"[train] Loading Gemma4 on {device} for lm_head/embed...", flush=True)
    base_model, tokenizer = load_base_model(base_model_path, device)
    base_model.eval()

    sys.path.insert(0, "/root/flashkv0516/app/shared")
    from model_loader import get_embed_weight, get_lm_head_weight
    embed_weights = get_embed_weight(base_model)
    lm_head_weight = get_lm_head_weight(base_model)
    if lm_head_weight is None:
        lm_head_weight = embed_weights

    mtp = load_mtp_head(mtp_checkpoint, device, lm_head_weight)
    mtp.train()

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

    output_path = Path(output_dir) / "mtp_head_gemma4_decode.pt"
    torch.save({
        "model_state_dict": mtp.state_dict(),
        "epoch": epochs,
        "loss": avg_loss,
        "num_chain": num_chain,
        "training": "gemma4_decode_chained_multi",
        "num_samples": len(all_samples),
        "hidden_type": "decode",
        "model": "gemma4-26b-a4b",
        "hidden_size": 2816,
        "vocab_size": 262144,
    }, output_path)
    log(f"[train] Saved: {output_path}")

    # 评估
    mtp.eval()
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
        log(f"[train] Chain accept rate ({num_chain} steps, train data): {chain_correct}/{chain_total} = {chain_correct/chain_total:.1%}")
        log(f"[train] NOTE: this is on TRAINING data. Real accept must be tested on NEW prompts.")

    # 导出 slim 版 (不含 lm_head)
    slim_sd = {k: v for k, v in mtp.state_dict().items() if "lm_head" not in k and "embed" not in k}
    slim_path = Path(output_dir) / "mtp_head_gemma4_decode_slim.pt"
    torch.save({
        "model_state_dict": slim_sd,
        "num_chain": num_chain,
        "training": "gemma4_decode_chained_multi",
        "hidden_type": "decode",
        "model": "gemma4-26b-a4b",
        "hidden_size": 2816,
        "vocab_size": 262144,
    }, slim_path)
    log(f"[train] Saved slim (no lm_head): {slim_path} ({slim_path.stat().st_size / 1e6:.0f}MB)")

    return output_path


# ============ Main ============

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma4 MTP Head Decode Training")
    parser.add_argument("--base-model", default="/data/models/gemma-4-26b-a4b-it")
    parser.add_argument("--mtp-checkpoint", default="", help="Existing MTP checkpoint (empty=start from scratch)")
    parser.add_argument("--corpus", default="/data/mtp_corpus.jsonl")
    parser.add_argument("--output", default="/data/mtp_head_gemma4_decode")
    parser.add_argument("--num-chain", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--gpu-base", type=int, default=0, help="Base GPU index (0 for full, 4 if sglang on 0-3)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gen-length", type=int, default=50, help="Tokens to generate per corpus entry")
    parser.add_argument("--shard-dir", default="/data/mtp_gemma4_shards")
    parser.add_argument("--mode", default="all", choices=["all", "collect", "train"])
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.shard_dir).mkdir(parents=True, exist_ok=True)

    max_per_worker = args.max_samples // args.world_size + 1

    if args.mode in ("all", "collect"):
        for old in Path(args.shard_dir).glob("shard_*.pt"):
            old.unlink()

        print(f"[main] Phase 1: Gemma4 decode-mode collect with {args.world_size} GPUs "
              f"(GPU {args.gpu_base}-{args.gpu_base + args.world_size - 1}), "
              f"~{max_per_worker} samples/worker, gen_length={args.gen_length}", flush=True)
        t0 = time.time()
        mp.spawn(
            collect_worker,
            args=(args.world_size, args.corpus, args.base_model, args.mtp_checkpoint,
                  args.shard_dir, args.num_chain, max_per_worker, args.gen_length,
                  args.gpu_base),
            nprocs=args.world_size,
            join=True,
        )
        print(f"[main] Phase 1 done in {time.time()-t0:.0f}s", flush=True)

    if args.mode in ("all", "train"):
        train_device = f"cuda:{args.gpu_base}"
        print(f"[main] Phase 2: train on shards (device={train_device})", flush=True)
        train_on_shards(
            base_model_path=args.base_model,
            mtp_checkpoint=args.mtp_checkpoint,
            shard_dir=args.shard_dir,
            output_dir=args.output,
            num_chain=args.num_chain,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=train_device,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
