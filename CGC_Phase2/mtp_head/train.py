"""训练 MTP Head.

用法:
  python train.py \
    --base-model /data2/models/Qwen3-VL-2B-Instruct \
    --data-dir /data/mtp_training_data \
    --output-dir /data/mtp_head_output \
    --epochs 3 --batch-size 32 --lr 1e-4

训练数据格式 (collect_data.py 生成):
  shard_XXXXX.pt: list[{hidden_state: Tensor[hidden], token_id: int, next_token_id: int}]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MTPHead, MTPHeadConfig, create_mtp_head_for_qwen3vl_2b


class MTPDataset(Dataset):
    """MTP 训练数据集."""

    def __init__(self, data_dir: str, max_samples: int = None):
        self.data_dir = data_dir
        self.shards = sorted(Path(data_dir).glob("shard_*.pt"))
        if not self.shards:
            raise FileNotFoundError(f"No shard files found in {data_dir}")

        # 加载所有样本到内存 (假设数据量 < 50GB)
        print(f"[dataset] Loading {len(self.shards)} shards from {data_dir}")
        self.samples = []
        for shard_path in self.shards:
            shard_data = torch.load(shard_path, weights_only=False)
            self.samples.extend(shard_data)
            print(f"[dataset] Loaded {shard_path.name}: {len(shard_data)} samples")

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"[dataset] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "hidden_state": sample["hidden_state"],  # [hidden]
            "token_id": sample["token_id"],
            "next_token_id": sample["next_token_id"],
        }


def collate_fn(batch, embed_weight: torch.Tensor):
    """Collate: 获取 token embeddings.

    Args:
        batch: list of dicts
        embed_weight: base model embed_tokens weight [vocab, hidden]
    """
    hidden_states = torch.stack([b["hidden_state"] for b in batch])  # [batch, hidden]
    token_ids = torch.tensor([b["token_id"] for b in batch])  # [batch]
    next_token_ids = torch.tensor([b["next_token_id"] for b in batch])  # [batch]

    # 移到与 embed_weight 相同的 device
    device = embed_weight.device
    hidden_states = hidden_states.to(device)
    token_ids = token_ids.to(device)
    next_token_ids = next_token_ids.to(device)

    # 获取 token embeddings
    token_embeddings = F.embedding(token_ids, embed_weight)  # [batch, hidden]

    return {
        "hidden_states": hidden_states.unsqueeze(1),  # [batch, 1, hidden]
        "token_embeddings": token_embeddings.unsqueeze(1),  # [batch, 1, hidden]
        "labels": next_token_ids.unsqueeze(1),  # [batch, 1]
    }


def train(
    base_model_path: str,
    data_dir: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
    warmup_steps: int = 100,
    save_every: int = 1000,
    device: str = "cuda",
):
    """训练 MTP Head."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 base model (统一加载器,支持所有模型类型)
    print(f"[train] Loading base model from {base_model_path}")
    import sys as _sys
    _sys.path.insert(0, "/root/flashkv0516")
    from app.shared.model_loader import load_base_model, get_embed_weight, get_lm_head_weight

    base_model, tokenizer = load_base_model(base_model_path, device=device)

    # 统一获取 embed_tokens + lm_head 权重 (通过 model_loader)
    embed_weight = get_embed_weight(base_model)
    lm_head_weight = get_lm_head_weight(base_model)

    if embed_weight is None or lm_head_weight is None:
        raise RuntimeError("Cannot find embed_tokens or lm_head (model_loader failed)")

    print(f"[train] embed_weight: {tuple(embed_weight.shape)}")
    print(f"[train] lm_head_weight: {tuple(lm_head_weight.shape)}")

    # 释放 base model (只保留权重)
    del base_model
    torch.cuda.empty_cache()

    # 2. 创建 MTP head
    print("[train] Creating MTP head")
    mtp_head = create_mtp_head_for_qwen3vl_2b()
    mtp_head.set_shared_lm_head(lm_head_weight)
    mtp_head = mtp_head.to(device).to(torch.bfloat16)

    print(f"[train] MTP head parameters: {mtp_head.num_parameters() / 1e6:.2f}M")

    # 3. 加载数据
    dataset = MTPDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # CUDA 不支持 fork 多进程
        collate_fn=lambda b: collate_fn(b, embed_weight.to(device).to(torch.bfloat16)),
    )

    # 4. 优化器
    optimizer = torch.optim.AdamW(
        mtp_head.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Cosine learning rate schedule with warmup
    total_steps = len(dataloader) * epochs
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    scheduler = LambdaLR(optimizer, lr_lambda)

    # 5. 训练循环
    print(f"[train] Starting training: {epochs} epochs, {total_steps} steps")
    global_step = 0
    losses = []

    for epoch in range(epochs):
        mtp_head.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):
            hidden_states = batch["hidden_states"].to(device)
            token_embeddings = batch["token_embeddings"].to(device)
            labels = batch["labels"].to(device)  # [batch, 1]

            # Forward
            logits = mtp_head(hidden_states, token_embeddings)  # [batch, 1, vocab]
            logits = logits[:, -1, :]  # [batch, vocab]

            # Loss (cross-entropy)
            loss = F.cross_entropy(logits, labels[:, 0])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mtp_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            losses.append(loss.item())

            if step % 50 == 0:
                avg_loss = epoch_loss / epoch_steps
                lr_current = scheduler.get_last_lr()[0]
                print(
                    f"[train] Epoch {epoch+1}/{epochs} Step {step}/{len(dataloader)} "
                    f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} lr={lr_current:.2e}",
                    flush=True,
                )

            # 保存 checkpoint
            if global_step % save_every == 0:
                ckpt_path = os.path.join(output_dir, f"mtp_head_step{global_step}.pt")
                torch.save({
                    "model_state_dict": mtp_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": global_step,
                    "loss": loss.item(),
                }, ckpt_path)
                print(f"[train] Saved checkpoint: {ckpt_path}")

        # Epoch 结束
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"[train] Epoch {epoch+1} done. avg_loss={avg_epoch_loss:.4f}")

        # 保存 epoch checkpoint
        ckpt_path = os.path.join(output_dir, f"mtp_head_epoch{epoch+1}.pt")
        torch.save({
            "model_state_dict": mtp_head.state_dict(),
            "step": global_step,
            "epoch": epoch + 1,
            "loss": avg_epoch_loss,
        }, ckpt_path)
        print(f"[train] Saved epoch checkpoint: {ckpt_path}")

    # 6. 保存最终模型
    final_path = os.path.join(output_dir, "mtp_head_final.pt")
    torch.save({
        "model_state_dict": mtp_head.state_dict(),
        "config": MTPHeadConfig().__dict__,
        "step": global_step,
        "epochs": epochs,
    }, final_path)
    print(f"[train] Training complete. Final model: {final_path}")

    # 保存 loss 曲线
    loss_path = os.path.join(output_dir, "losses.json")
    with open(loss_path, "w") as f:
        json.dump({"losses": losses, "final_loss": losses[-1]}, f)
    print(f"[train] Loss curve: {loss_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MTP Head")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--data-dir", required=True, help="Training data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        base_model_path=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        device=args.device,
    )


if __name__ == "__main__":
    import json
    main()
