#!/usr/bin/env python3
"""Train DSV4 MTP head using collected decode hidden states.

This script runs on Mac (CPU/MPS) using:
1. Hidden states collected by collect_dsv4_hidden.py on Host1
2. DSV4 embed + lm_head weights extracted from the converted checkpoint
3. MTP head architecture from model_registry

Usage:
    python train_dsv4_mtp_local.py \
        --hidden-dir /tmp/dsv4_hidden \
        --embed-head-path /tmp/dsv4_embed_head.pt \
        --output-dir /tmp/dsv4_mtp_output \
        --num-chain 4 --epochs 5 --batch-size 16
"""
from __future__ import annotations

import os
import sys
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "app", "shared"),
          os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_head")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def load_hidden_data(hidden_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load collected hidden states and token IDs from shard files."""
    hidden_list = []
    token_list = []
    shard_paths = sorted(Path(hidden_dir).glob("hidden_shard_*.pt"))
    if not shard_paths:
        # Also try shard_*.pt format
        shard_paths = sorted(Path(hidden_dir).glob("shard_*.pt"))

    print(f"Loading {len(shard_paths)} shards from {hidden_dir}...")
    for p in shard_paths:
        data = torch.load(p, weights_only=False)
        if isinstance(data, dict):
            hidden_list.append(data["hidden_states"])
            token_list.append(data["token_ids"])
        elif isinstance(data, list):
            # Already in chain format
            return None, data  # Handle separately
        print(f"  {p.name}: {data['hidden_states'].shape[0] if isinstance(data, dict) else len(data)} samples")

    if not hidden_list:
        raise RuntimeError(f"No hidden state shards found in {hidden_dir}")

    all_hidden = torch.cat(hidden_list, dim=0)  # [N, hidden_size]
    all_tokens = torch.cat(token_list, dim=0)   # [N]
    print(f"Total: {all_hidden.shape[0]} hidden states, hidden_size={all_hidden.shape[1]}")
    return all_hidden, all_tokens


def create_chain_samples(
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    num_chain: int = 4,
) -> List[Dict]:
    """Convert flat hidden states + token IDs into chain training samples.

    For each starting position i, create a chain of length num_chain:
    - Step k: hidden=hidden_states[i+k], token=token_ids[i+k], next_token=token_ids[i+k+1]
    """
    N = hidden_states.shape[0]
    samples = []

    for i in range(N - num_chain):
        chain_hidden = hidden_states[i:i + num_chain]       # [num_chain, hidden_size]
        chain_tokens = token_ids[i:i + num_chain]            # [num_chain]
        chain_next = token_ids[i + 1:i + num_chain + 1]     # [num_chain]

        samples.append({
            "hidden_states": chain_hidden,
            "token_ids": chain_tokens,
            "next_token_ids": chain_next,
        })

    return samples


def train_dsv4_mtp(
    hidden_dir: str,
    embed_head_path: str,
    output_dir: str,
    num_chain: int = 4,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-5,
    device: str = "cpu",
    mtp_checkpoint: str = "",
):
    """Train DSV4 MTP head on Mac."""
    from model import MTPHead, MTPHeadConfig

    print(f"[train] Device: {device}")
    print(f"[train] num_chain={num_chain}, epochs={epochs}, batch_size={batch_size}, lr={lr}")

    # 1. Load hidden states
    all_hidden, all_tokens = load_hidden_data(hidden_dir)
    if all_hidden is None:
        # Already in chain format
        all_samples = all_tokens
        print(f"[train] Loaded {len(all_samples)} pre-chain samples")
    else:
        # 2. Create chain samples
        all_samples = create_chain_samples(all_hidden, all_tokens, num_chain)
        print(f"[train] Created {len(all_samples)} chain samples from {all_hidden.shape[0]} hidden states")

    if len(all_samples) < 10:
        print(f"[train] Too few samples ({len(all_samples)}), need at least 10. Aborting.")
        return

    # 3. Load embed + lm_head weights (float32 for CPU speed; skip set_shared_lm_head to save 2GB)
    print(f"[train] Loading embed+head from {embed_head_path}...")
    weights = torch.load(embed_head_path, map_location=device, weights_only=True)
    embed_weight = weights["embed.weight"].to(device).to(torch.float32)
    lm_head_weight = weights["head.weight"].to(device).to(torch.float32)
    # Free the raw bfloat16 weights
    del weights
    import gc; gc.collect()
    print(f"[train] embed: {embed_weight.shape}, lm_head: {lm_head_weight.shape}")

    # 4. Create MTP head — use smaller head_dim for efficient Mac training
    # DSV4 base model uses head_dim=512 with MLA, but MTP head uses standard MHA
    # so we use head_dim=128 (standard) to keep the head lightweight (~200M params)
    mtp_config = MTPHeadConfig(
        hidden_size=4096,
        vocab_size=129280,
        num_heads=8,           # 8 heads × 128 dim = 1024 (reasonable)
        head_dim=128,          # Standard head_dim, not DSV4's 512
        intermediate_size=11264,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=4096,
    )
    mtp = MTPHead(mtp_config)
    # Don't call set_shared_lm_head — it duplicates 2GB weight.
    # We use lm_head_weight directly via F.linear in training/eval.
    if mtp_checkpoint and os.path.exists(mtp_checkpoint):
        ckpt = torch.load(mtp_checkpoint, map_location="cpu", weights_only=False)
        mtp.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[train] Loaded MTP checkpoint: {mtp_checkpoint}")
    else:
        print("[train] Starting from scratch (no checkpoint)")
    mtp.to(device).to(torch.float32)
    mtp.train()

    # 5. Training
    optimizer = torch.optim.AdamW(mtp.parameters(), lr=lr, betas=(0.9, 0.95))
    total_steps = epochs * len(all_samples) // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[train] Starting training: {len(all_samples)} samples, {epochs} epochs, {device}")

    avg_loss = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        random.shuffle(all_samples)
        t0 = time.time()

        for i in range(0, len(all_samples), batch_size):
            batch = all_samples[i:i + batch_size]
            if len(batch) < 2:
                continue

            hidden = torch.stack([s["hidden_states"] for s in batch]).to(device)  # [B, num_chain, hidden]
            tokens = torch.stack([s["token_ids"] for s in batch]).to(device)      # [B, num_chain]
            next_tokens = torch.stack([s["next_token_ids"] for s in batch]).to(device)  # [B, num_chain]
            hidden = hidden.to(torch.float32)

            loss = 0.0
            current_hidden = hidden[:, 0, :]  # [B, hidden]

            for k in range(num_chain):
                token_embed = F.embedding(tokens[:, k], embed_weight)  # [B, hidden] float32
                concat_input = torch.cat(
                    [current_hidden.unsqueeze(1), token_embed.unsqueeze(1)], dim=-1
                )  # [B, 1, 2*hidden]
                x = mtp.proj(concat_input)
                # Pre-norm: residual from pre-norm x, sublayer input is normed
                x = x + mtp.attn(mtp.norm1(x))
                x = x + mtp.mlp(mtp.norm2(x))
                mtp_hidden = mtp.norm_out(x)  # [B, 1, hidden]

                logits = F.linear(mtp_hidden[:, 0, :], lm_head_weight)  # [B, vocab] float32
                loss = loss + F.cross_entropy(logits, next_tokens[:, k])
                current_hidden = mtp_hidden[:, 0, :].detach()

            loss = loss / num_chain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 20 == 0:
                elapsed = time.time() - t0
                rate = num_batches / max(elapsed, 0.01)
                eta = (len(all_samples) // batch_size - num_batches) / max(rate, 0.01)
                log(f"  Epoch {epoch+1} batch {num_batches}/{len(all_samples)//batch_size}: "
                    f"loss={loss.item():.4f}, {rate:.1f} batch/s, ETA {eta:.0f}s")

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0
        log(f"[train] Epoch {epoch+1}: avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")

    # 6. Save (exclude shared lm_head to keep file small)
    slim_sd = {k: v for k, v in mtp.state_dict().items() if "lm_head" not in k}
    output_path = Path(output_dir) / "mtp_head_dsv4_decode.pt"
    torch.save({
        "model_state_dict": slim_sd,
        "epoch": epochs,
        "loss": avg_loss,
        "num_chain": num_chain,
        "training": "dsv4_decode_chained",
        "num_samples": len(all_samples),
        "hidden_type": "decode",
        "model": "DeepSeek-V4-Flash",
        "model_name": "dsv4",
        "hidden_size": 4096,
        "vocab_size": 129280,
    }, output_path)
    log(f"[train] Saved: {output_path} ({output_path.stat().st_size / 1e6:.0f}MB)")

    # 7. Evaluate on training data
    mtp.eval()
    chain_correct = 0
    chain_total = 0
    with torch.no_grad():
        for s in all_samples[:min(200, len(all_samples))]:
            hidden_states = s["hidden_states"].to(device).to(torch.float32)
            token_ids = s["token_ids"].to(device)
            next_token_ids = s["next_token_ids"].to(device)
            current_hidden = hidden_states[0]
            for k in range(num_chain):
                token_embed = F.embedding(token_ids[k], embed_weight)
                h_3d = current_hidden.unsqueeze(0).unsqueeze(0)
                e_3d = token_embed.unsqueeze(0).unsqueeze(0)
                concat_input = torch.cat([h_3d, e_3d], dim=-1)
                x = mtp.proj(concat_input)
                x = x + mtp.attn(mtp.norm1(x))
                x = x + mtp.mlp(mtp.norm2(x))
                mtp_out = mtp.norm_out(x)
                logits = F.linear(mtp_out[0, 0], lm_head_weight)
                pred = logits.argmax().item()
                if pred == next_token_ids[k].item():
                    chain_correct += 1
                chain_total += 1
                current_hidden = mtp_out[0, 0]

    if chain_total > 0:
        log(f"[train] Chain accept rate ({num_chain} steps, train data): "
            f"{chain_correct}/{chain_total} = {chain_correct/chain_total:.1%}")
        log("[train] NOTE: Train data accept rate. Real accept must be tested on NEW prompts.")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train DSV4 MTP head locally (Mac)")
    parser.add_argument("--hidden-dir", required=True, help="Directory with hidden state shards")
    parser.add_argument("--embed-head-path", required=True, help="Path to dsv4_embed_head.pt")
    parser.add_argument("--output-dir", default="/tmp/dsv4_mtp_output", help="Output directory")
    parser.add_argument("--num-chain", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--mtp-checkpoint", default="", help="Existing MTP checkpoint")
    args = parser.parse_args()

    train_dsv4_mtp(
        hidden_dir=args.hidden_dir,
        embed_head_path=args.embed_head_path,
        output_dir=args.output_dir,
        num_chain=args.num_chain,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        mtp_checkpoint=args.mtp_checkpoint,
    )


if __name__ == "__main__":
    main()
