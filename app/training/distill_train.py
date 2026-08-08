#!/usr/bin/env python3
"""Universal MTP Draft training — KL distillation + chain CE.

FIXED version: uses REAL decode hidden states from collect_hidden_states.py
(removes the broken torch.randn() placeholder).

Trains MTP head for ANY registered model using:
  - model_registry for architecture config (no hardcoded sizes)
  - CGC_Phase2/mtp_head/model.py for canonical MTPHead definition
  - model_loader for universal embed/lm_head extraction

Training modes:
  - "chain" (default): CE loss with multi-step chain training (proven for DSV4)
  - "kl": KL(teacher||student) + CE, single-step per position (better distribution)
  - "hybrid": KL for step 0 + CE chain for steps 1+ (best of both)

Teacher logits are recomputed from (hidden_states, lm_head_weight) during training —
no need to store full logits in shards.

Usage (on GPU server):
  python3 distill_train.py \
    --model gemma4 \
    --data-dir /data/mtp_train_data/gemma4 \
    --output-dir /data/mtp_output/gemma4 \
    --epochs 3 \
    --batch-size 32 \
    --lr 1e-4 \
    --mode chain

  # KL distillation mode
  python3 distill_train.py --model gemma4 --mode kl --alpha 0.7 --beta 0.3 --temperature 2.0

  # Resume from checkpoint
  python3 distill_train.py --model gemma4 --checkpoint /data/mtp_output/gemma4/mtp_head_step_1000.pt

Output:
  /data/mtp_output/gemma4/mtp_head_<model>_decode.pt  # Final checkpoint (DraftRegistry-compatible)
  /data/mtp_output/gemma4/training_log.json            # Training log
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
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "app", "shared"),
          os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_head")]:
    if p not in sys.path:
        sys.path.insert(0, p)


# === Dataset: loads REAL hidden states from collected shards ===

class MTPDataset(Dataset):
    """Loads chain training samples from collect_hidden_states.py output.

    Each shard is a list of:
      {"hidden_states": [chain, hidden], "token_ids": [chain], "next_token_ids": [chain]}
    """

    def __init__(self, data_dir: str):
        self.samples: list[dict] = []
        shard_files = sorted(Path(data_dir).glob("shard_*.pt"))

        # Also check worker subdirectories (multi-GPU collection)
        if not shard_files:
            worker_dirs = sorted(Path(data_dir).glob("worker_*"))
            for wd in worker_dirs:
                shard_files.extend(sorted(wd.glob("shard_*.pt")))

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
            "hidden_states": item["hidden_states"].float(),      # [chain, hidden]
            "token_ids": item["token_ids"].long(),               # [chain]
            "next_token_ids": item["next_token_ids"].long(),     # [chain]
        }


def collate_fn(batch):
    """Batch chain samples."""
    hidden_states = torch.stack([item["hidden_states"] for item in batch])   # [B, chain, hidden]
    token_ids = torch.stack([item["token_ids"] for item in batch])           # [B, chain]
    next_token_ids = torch.stack([item["next_token_ids"] for item in batch]) # [B, chain]
    return {
        "hidden_states": hidden_states,
        "token_ids": token_ids,
        "next_token_ids": next_token_ids,
    }


# === KL Distillation Loss ===

class KLDistillLoss(nn.Module):
    """KL(p_teacher || p_student) + CE(p_student, target).

    Loss = alpha * KL(p_teacher || p_student) * T^2 + beta * CE

    KL teaches the student the full probability distribution (soft labels).
    CE maintains argmax correctness (hard labels).
    Teacher logits are recomputed from (hidden_states, lm_head_weight).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,   # [B, vocab]
        teacher_logits: torch.Tensor,    # [B, vocab]
        target_tokens: torch.Tensor,     # [B]
    ) -> dict:
        T = self.temperature

        # KL divergence: KL(p_teacher || p_student)
        p_teacher = F.softmax(teacher_logits / T, dim=-1)
        log_p_student = F.log_softmax(student_logits / T, dim=-1)
        kl_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

        # Cross-entropy with hard labels
        ce_loss = F.cross_entropy(student_logits, target_tokens)

        total = self.alpha * kl_loss + self.beta * ce_loss

        return {
            "total": total,
            "kl": kl_loss.item(),
            "ce": ce_loss.item(),
        }


# === Training: chain mode (CE, proven for DSV4) ===

def train_chain_step(
    mtp: nn.Module,
    embed_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    hidden_states: torch.Tensor,   # [B, chain, hidden]
    token_ids: torch.Tensor,       # [B, chain]
    next_token_ids: torch.Tensor,  # [B, chain]
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Single chain training step.

    Step 0: use real decode hidden, predict next_token[0]
    Step k: use MTP output from step k-1 (detached), predict next_token[k]

    Returns (loss, metrics_dict).
    """
    batch_size, chain_len, hidden_size = hidden_states.shape
    total_loss = torch.tensor(0.0, device=device)
    correct = 0
    total = 0

    current_hidden = hidden_states[:, 0, :]  # [B, hidden] — real decode hidden

    for k in range(chain_len):
        # Get token embedding
        token_embed = F.embedding(token_ids[:, k], embed_weight)  # [B, hidden]

        # MTP forward (single position)
        h_3d = current_hidden.unsqueeze(1)  # [B, 1, hidden]
        e_3d = token_embed.unsqueeze(1)    # [B, 1, hidden]

        # Manual forward to get intermediate hidden for chaining
        x = torch.cat([h_3d, e_3d], dim=-1)
        x = mtp.proj(x)
        x = x + mtp.attn(mtp.norm1(x))
        x = x + mtp.mlp(mtp.norm2(x))
        mtp_hidden = mtp.norm_out(x)  # [B, 1, hidden]

        # Compute logits via shared lm_head
        logits = F.linear(mtp_hidden[:, 0, :], lm_head_weight)  # [B, vocab]

        # CE loss
        loss = F.cross_entropy(logits, next_token_ids[:, k])
        total_loss = total_loss + loss

        # Accuracy
        pred = logits.argmax(dim=-1)
        correct += (pred == next_token_ids[:, k]).sum().item()
        total += batch_size

        # Chain: next step uses MTP output (detached)
        if k < chain_len - 1:
            current_hidden = mtp_hidden[:, 0, :].detach()

    total_loss = total_loss / chain_len
    return total_loss, {"chain_acc": correct / max(total, 1)}


# === Training: KL mode (single-step, KL+CE) ===

def train_kl_step(
    mtp: nn.Module,
    embed_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    hidden_states: torch.Tensor,   # [B, chain, hidden]
    token_ids: torch.Tensor,       # [B, chain]
    next_token_ids: torch.Tensor,  # [B, chain]
    device: torch.device,
    criterion: KLDistillLoss,
) -> tuple[torch.Tensor, dict]:
    """KL distillation training step (single position, all chain positions treated independently).

    For each position k: use real decode hidden[k] (not chained),
    compute teacher_logits = lm_head(hidden[k]), student_logits = mtp(hidden[k], embed[k]).
    """
    batch_size, chain_len, hidden_size = hidden_states.shape

    # Use only step 0 for KL (real hidden, not chained)
    # Could extend to all positions, but step 0 is most important for first-token prediction
    current_hidden = hidden_states[:, 0, :]  # [B, hidden]
    token_embed = F.embedding(token_ids[:, 0], embed_weight)  # [B, hidden]

    # Teacher logits (from base model's lm_head applied to real hidden)
    with torch.no_grad():
        teacher_logits = F.linear(current_hidden, lm_head_weight)  # [B, vocab]

    # Student logits (from MTP head)
    h_3d = current_hidden.unsqueeze(1)
    e_3d = token_embed.unsqueeze(1)
    student_logits_3d = mtp(h_3d, e_3d)  # [B, 1, vocab]
    student_logits = student_logits_3d[:, 0, :]  # [B, vocab]

    # KL + CE loss
    loss_dict = criterion(student_logits, teacher_logits, next_token_ids[:, 0])

    # Also compute chain CE for remaining steps (optional, improves multi-step)
    chain_loss = torch.tensor(0.0, device=device)
    current = current_hidden.detach()
    correct = 0
    for k in range(chain_len):
        token_embed_k = F.embedding(token_ids[:, k], embed_weight)
        h = current.unsqueeze(1)
        e = token_embed_k.unsqueeze(1)
        x = torch.cat([h, e], dim=-1)
        x = mtp.proj(x)
        x = x + mtp.attn(mtp.norm1(x))
        x = x + mtp.mlp(mtp.norm2(x))
        mtp_h = mtp.norm_out(x)
        logits = F.linear(mtp_h[:, 0, :], lm_head_weight)
        chain_loss = chain_loss + F.cross_entropy(logits, next_token_ids[:, k])
        pred = logits.argmax(dim=-1)
        correct += (pred == next_token_ids[:, k]).sum().item()
        current = mtp_h[:, 0, :].detach()

    chain_loss = chain_loss / chain_len

    # Combine: KL loss for step 0 + chain CE for all steps
    total_loss = loss_dict["total"] + 0.3 * chain_loss

    return total_loss, {
        "kl": loss_dict["kl"],
        "ce": loss_dict["ce"],
        "chain_acc": correct / max(batch_size * chain_len, 1),
    }


# === Main training function ===

def train(
    model_name: str,
    data_dir: str,
    output_dir: str,
    mode: str = "chain",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
    alpha: float = 0.7,
    beta: float = 0.3,
    temperature: float = 2.0,
    warmup_steps: int = 100,
    save_every: int = 500,
    checkpoint: str = "",
    device: str = "cuda",
):
    """Main training function.

    Args:
        model_name: model_registry name (gemma4, dsv4, qwen3vl)
        data_dir: directory with collected shards + embed_head.pt
        output_dir: output directory for checkpoints
        mode: "chain" | "kl" | "hybrid"
        epochs: training epochs
        batch_size: batch size
        lr: learning rate
        alpha: KL loss weight (for kl/hybrid mode)
        beta: CE loss weight (for kl/hybrid mode)
        temperature: distillation temperature
        checkpoint: existing checkpoint to resume from
        device: "cuda" or "cpu"
    """
    from app.shared.model_registry import get_model_config
    from model import create_mtp_head_by_model_name

    cfg = get_model_config(model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Device
    if device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA not available, using CPU", file=sys.stderr)
        device = "cpu"
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
        num_workers=4 if device == "cuda" else 0,
        pin_memory=device == "cuda",
        drop_last=True,
    )

    # 2. Load embed + lm_head weights
    embed_head_path = os.path.join(data_dir, "embed_head.pt")
    if not os.path.exists(embed_head_path):
        # Check worker_0
        embed_head_path = os.path.join(data_dir, "worker_0", "embed_head.pt")
    if not os.path.exists(embed_head_path):
        raise FileNotFoundError(f"embed_head.pt not found in {data_dir}")

    print(f"[train] Loading embed+head from {embed_head_path}...", flush=True)
    weights = torch.load(embed_head_path, map_location="cpu", weights_only=True)
    embed_weight = weights["embed_weight"].to(dev).to(torch.float32)
    lm_head_raw = weights.get("lm_head_weight")
    if lm_head_raw is None and bool(weights.get("lm_head_tied_to_embed")):
        lm_head_raw = weights["embed_weight"]
    if lm_head_raw is None:
        raise RuntimeError(f"lm_head_weight missing in {embed_head_path}")
    lm_head_weight = lm_head_raw.to(dev).to(torch.float32)
    print(f"[train] embed: {embed_weight.shape}, lm_head: {lm_head_weight.shape}", flush=True)

    # 3. Create MTP head (from canonical model.py, using model_registry config)
    mtp = create_mtp_head_by_model_name(model_name)
    mtp.set_shared_lm_head(lm_head_weight)
    print(f"[train] MTP head: {mtp.num_parameters() / 1e6:.1f}M trainable params", flush=True)
    print(f"[train]   hidden={cfg.hidden_size}, vocab={cfg.vocab_size}, "
          f"heads={cfg.num_heads}x{cfg.head_dim}", flush=True)

    # Load checkpoint if provided
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        mtp.load_state_dict(ckpt["model_state_dict"], strict=False)
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

    # 5. Loss
    criterion = KLDistillLoss(alpha=alpha, beta=beta, temperature=temperature) if mode != "chain" else None

    # 6. Training log
    log = {
        "config": {
            "model_name": model_name,
            "display_name": cfg.display_name,
            "hidden_size": cfg.hidden_size,
            "vocab_size": cfg.vocab_size,
            "mode": mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "alpha": alpha,
            "beta": beta,
            "temperature": temperature,
            "num_params_m": round(mtp.num_parameters() / 1e6, 1),
            "num_samples": len(dataset),
        },
        "steps": [],
    }

    # 7. Training loop
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
            elif mode in ("kl", "hybrid"):
                loss, metrics = train_kl_step(
                    mtp, embed_weight, lm_head_weight,
                    hidden_states, token_ids, next_token_ids,
                    dev, criterion,
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
            if global_step % 50 == 0:
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

            # Checkpoint
            if global_step % save_every == 0:
                ckpt_path = os.path.join(output_dir, f"mtp_head_step_{global_step}.pt")
                _save_checkpoint(mtp, cfg, ckpt_path, global_step, epoch_loss / num_batches, mode)
                print(f"  Checkpoint: {ckpt_path}", flush=True)

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        metric_str = " ".join(f"{k}={v:.3f}" for k, v in avg_metrics.items())
        print(
            f"\n  Epoch {epoch+1}: avg_loss={avg_loss:.4f} {metric_str} "
            f"({time.time()-t0:.0f}s)\n",
            flush=True,
        )

    # 8. Save final checkpoint (DraftRegistry-compatible)
    final_path = os.path.join(output_dir, cfg.get_checkpoint_path("/data").replace("/data/", ""))
    # Use simpler path: output_dir/mtp_head_<model>_decode.pt
    final_path = os.path.join(output_dir, f"mtp_head_{model_name}_decode.pt")
    _save_checkpoint(mtp, cfg, final_path, global_step, avg_loss, mode)
    print(f"[train] Final model: {final_path}", flush=True)

    # 9. Evaluation on training data
    mtp.eval()
    eval_correct = 0
    eval_total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 50:  # Evaluate on first 50 batches
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
                # Use real hidden for eval (not chained) to measure per-position accuracy
                if k < hidden_states.shape[1] - 1:
                    current_hidden = hidden_states[:, k + 1]

    chain_acc = eval_correct / max(eval_total, 1)
    print(f"[train] Chain accept rate (train data, {hidden_states.shape[1]} steps): "
          f"{eval_correct}/{eval_total} = {chain_acc:.1%}", flush=True)
    print("[train] NOTE: Train data accept rate. Real accept must be tested on NEW prompts.", flush=True)

    # 10. Save training log
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
    slim_sd = {}
    for key, value in mtp.state_dict().items():
        if "lm_head" in key:
            continue
        tensor = value.detach().cpu()
        # Save trainable floating tensors in fp16 to reduce checkpoint size on disk.
        if torch.is_floating_point(tensor):
            tensor = tensor.to(torch.float16)
        slim_sd[key] = tensor
    torch.save({
        "model_state_dict": slim_sd,
        "model_name": cfg.name,
        "step": step,
        "loss": loss,
        "mode": mode,
        "checkpoint_dtype": "float16",
        "config": {
            "hidden_size": cfg.hidden_size,
            "vocab_size": cfg.vocab_size,
            "num_heads": cfg.num_heads,
            "head_dim": cfg.head_dim,
            "intermediate_size": cfg.intermediate_size,
        },
    }, path)


# === CLI ===

def main():
    parser = argparse.ArgumentParser(
        description="Universal MTP Draft training (KL distillation + chain CE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chain training (CE, proven for DSV4)
  python3 distill_train.py --model gemma4 --data-dir /data/mtp_train_data/gemma4

  # KL distillation
  python3 distill_train.py --model gemma4 --mode kl --alpha 0.7 --beta 0.3

  # Resume from checkpoint
  python3 distill_train.py --model gemma4 --checkpoint /data/mtp_output/gemma4/mtp_head_step_1000.pt
        """,
    )
    parser.add_argument("--model", required=True, help="Model name: gemma4 | dsv4 | qwen3vl")
    parser.add_argument("--data-dir", required=True, help="Directory with collected shards + embed_head.pt")
    parser.add_argument("--output-dir", default="", help="Output dir (default: /data/mtp_output/<model>)")
    parser.add_argument("--mode", default="chain", choices=["chain", "kl", "hybrid"],
                        help="Training mode: chain=CE(proven), kl=KL+CE, hybrid=both")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.7, help="KL loss weight")
    parser.add_argument("--beta", type=float, default=0.3, help="CE loss weight")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint", default="", help="Existing checkpoint to resume from")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()

    from app.shared.model_registry import get_model_config
    cfg = get_model_config(args.model)
    output_dir = args.output_dir or cfg.get_mtp_output_dir()

    print(f"=== Universal MTP Draft Training ===", flush=True)
    print(f"  Model: {cfg.display_name} ({cfg.name})", flush=True)
    print(f"  Data: {args.data_dir}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Mode: {args.mode}", flush=True)
    if args.mode != "chain":
        print(f"  Loss: {args.alpha}*KL + {args.beta}*CE (T={args.temperature})", flush=True)
    print(f"  Architecture: hidden={cfg.hidden_size}, vocab={cfg.vocab_size}, "
          f"heads={cfg.num_heads}x{cfg.head_dim}", flush=True)

    train(
        model_name=cfg.name,
        data_dir=args.data_dir,
        output_dir=output_dir,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        checkpoint=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
