import os, sys
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#!/usr/bin/env python3
"""MoT-h 瑷撶反鑵虫湰 (CPU / MPS / CUDA 涓夊緦绔?.

鐢ㄦ硶:
  # CPU (椹楄瓑鐢? 鎱?
  py train_mot_h.py --data train.pt --output mot_h.pt --epochs 10 --device cpu

  # Mac MPS (鎺ㄨ枽)
  py train_mot_h.py --data train.pt --output mot_h.pt --epochs 10 --device mps

  # CUDA (闆?GPU)
  py train_mot_h.py --data train.pt --output mot_h.pt --epochs 10 --device cuda

鏁告摎鏍煎紡 (collect_batch.py 杓稿嚭鐨?train.pt):
  {
    "pairs": [
      {"h_src": [seq, 2816], "h_tgt": [seq, 2048], "seq_len": int, ...},
      ...
    ],
    "meta": {"src_dim": 2816, "tgt_dim": 2048, ...}
  }

瑷撶反绛栫暐:
  - 鎼嶅け: L_CC (hidden MSE) + L_consistency (cosine sim)
  - 鍎寲鍣? AdamW, lr=1e-4, weight_decay=0.01
  - 瑾垮害: cosine warmup 鈫?decay
  - 姊害瑁佸壀: 1.0
  - batch: 闅ㄦ鍙?N 鍊?token 绐楀彛 (閬垮厤椤瓨鐖?
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 鍕曟厠鍔犲叆 MoT-h 璺緫
_HERE = os.path.dirname(os.path.abspath(__file__))
_MOT_H_PATH = os.path.abspath(os.path.join(_HERE, "..", "..", "CGC_Phase2", "mot_h"))
if _MOT_H_PATH not in sys.path:
    sys.path.insert(0, _MOT_H_PATH)

from mot_h import MoTH, MoTHConfig  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 瑷倷閬告搰
# ---------------------------------------------------------------------------
def select_device(name: str) -> torch.device:
    """閬告搰瑷撶反瑷倷."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# 鏁告摎闆?
# ---------------------------------------------------------------------------
class HiddenPairDataset(Dataset):
    """寰?collect_batch.py 杓稿嚭鐨?pairs 妲嬮€犺〒绶撮泦.

    姣忓€?sample 鏄竴鍊嬪浐瀹氶暦搴︾殑绐楀彛 [window_seq_len, hidden_dim],
    寰炲師濮?seq 涓毃姗熸埅鍙?
    """

    def __init__(self, pairs: list[dict], window_seq_len: int = 128,
                  src_dim: int = 2816, tgt_dim: int = 2048):
        self.pairs = pairs
        self.window_seq_len = window_seq_len
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim

        # 闋愯▓绠楁瘡鍊?pair 鑳界敘鐢熷灏戠獥鍙?
        self.samples = []
        for pair_idx, pair in enumerate(pairs):
            seq_len = pair["seq_len"]
            if seq_len >= window_seq_len:
                n_windows = seq_len // window_seq_len
                for w in range(n_windows):
                    self.samples.append((pair_idx, w * window_seq_len))
            elif seq_len > 16:  # 澶煭涔熶繚鐣? padding
                self.samples.append((pair_idx, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pair_idx, offset = self.samples[idx]
        pair = self.pairs[pair_idx]
        h_src = pair["h_src"]  # [seq, src_dim]
        h_tgt = pair["h_tgt"]  # [seq, tgt_dim]

        # 鎴彇绐楀彛
        if h_src.shape[0] >= self.window_seq_len:
            h_src_w = h_src[offset:offset + self.window_seq_len]
            h_tgt_w = h_tgt[offset:offset + self.window_seq_len]
        else:
            # padding
            pad_src = torch.zeros(self.window_seq_len - h_src.shape[0], self.src_dim)
            pad_tgt = torch.zeros(self.window_seq_len - h_tgt.shape[0], self.tgt_dim)
            h_src_w = torch.cat([h_src, pad_src], dim=0)
            h_tgt_w = torch.cat([h_tgt, pad_tgt], dim=0)

        return h_src_w, h_tgt_w


# ---------------------------------------------------------------------------
# 瑷撶反閰嶇疆
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    data_path: str = "train.pt"
    output_path: str = "mot_h.pt"
    device: str = "auto"
    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    window_seq_len: int = 128
    translator_hidden: int = 4096
    num_translators: int = 3
    top_k: int = 2
    log_every: int = 20
    save_every: int = 1  # 姣?N 鍊?epoch save 涓€娆?
    val_split: float = 0.05  # 5% 椹楄瓑闆?


# ---------------------------------------------------------------------------
# 瑷撶反杩村湀
# ---------------------------------------------------------------------------
def cosine_with_warmup(step: int, total_steps: int, warmup: int, lr_max: float) -> float:
    """Cosine warmup 瑾垮害."""
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * lr_max * (1 + math.cos(math.pi * progress))


def train_step(model: MoTH, batch_src: torch.Tensor, batch_tgt: torch.Tensor,
                device: torch.device) -> tuple[torch.Tensor, dict]:
    """鍠瑷撶反.

    Args:
        batch_src: [B, seq, src_dim]
        batch_tgt: [B, seq, tgt_dim]

    Returns:
        (loss, metrics)
    """
    B, S, D_src = batch_src.shape
    _, _, D_tgt = batch_tgt.shape

    # 鎿村睍鐐?[B, 1, seq, src_dim] (window 缍害=1, 鏈堡)
    src_4d = batch_src.unsqueeze(1).to(device)
    tgt_2d = batch_tgt.to(device)

    # 鍓嶅悜
    out, gate_weights = model(src_4d)  # out: [B, seq, tgt_dim]
    out = out.to(device)

    # L_CC: hidden state MSE
    loss_cc = F.mse_loss(out, tgt_2d)

    # L_cos: cosine similarity loss (榧撳嫷鏂瑰悜灏嶉綂)
    cos_sim = F.cosine_similarity(out, tgt_2d, dim=-1).mean()
    loss_cos = 1.0 - cos_sim

    # L_div: 灏堝澶氭ǎ鎬ф鍓?(閬垮厤鎵€鏈?translator 杓稿嚭涓€妯?
    # gate entropy 榧撳嫷澶氭ǎ鍖栭伕鎿?
    gate_entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1).mean()
    loss_div = -0.01 * gate_entropy  # 璨犺櫉: 榧撳嫷楂樼喌

    # 绺芥悕澶?
    loss = loss_cc + 0.1 * loss_cos + loss_div

    metrics = {
        "loss": loss.item(),
        "cc": loss_cc.item(),
        "cos": loss_cos.item(),
        "div": loss_div.item(),
        "gate_entropy": gate_entropy.item(),
        "cos_sim": cos_sim.item(),
    }
    return loss, metrics


def evaluate(model: MoTH, val_loader: DataLoader, device: torch.device) -> dict:
    """椹楄瓑闆嗚浼?"""
    model.eval()
    total_loss = 0
    total_cos = 0
    total_mse = 0
    n = 0
    with torch.no_grad():
        for batch_src, batch_tgt in val_loader:
            B, S, _ = batch_src.shape
            src_4d = batch_src.unsqueeze(1).to(device)
            tgt_2d = batch_tgt.to(device)
            out, _ = model(src_4d)
            loss = F.mse_loss(out, tgt_2d)
            cos = F.cosine_similarity(out, tgt_2d, dim=-1).mean()
            total_loss += loss.item() * B
            total_cos += cos.item() * B
            total_mse += loss.item() * B
            n += B
    model.train()
    return {
        "val_loss": total_loss / n,
        "val_cos_sim": total_cos / n,
        "val_mse": total_mse / n,
    }


def train(cfg: TrainConfig):
    """涓昏〒绶存祦绋?"""
    device = select_device(cfg.device)
    logger.info("瑷倷: %s", device)
    if device.type == "mps":
        logger.info("  (Mac MPS 鈥?姣?CPU 蹇?10-20x)")
    elif device.type == "cuda":
        logger.info("  (CUDA GPU)")
    else:
        logger.info("  (CPU 鈥?鏈冩瘮杓冩參, 寤鸿鐢?mps 鎴?cuda)")

    # 杓夊叆鏁告摎
    logger.info("杓夊叆鏁告摎: %s", cfg.data_path)
    data = torch.load(cfg.data_path, map_location="cpu")
    pairs = data["pairs"]
    meta = data["meta"]
    src_dim = meta["src_dim"]
    tgt_dim = meta["tgt_dim"]
    logger.info("  pairs=%d, total_tokens=%d, src_dim=%d, tgt_dim=%d",
                len(pairs), meta["total_tokens"], src_dim, tgt_dim)

    # 鍒囧垎 train / val
    n_val = max(1, int(len(pairs) * cfg.val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    logger.info("  train=%d, val=%d", len(train_pairs), len(val_pairs))

    # 鏁告摎闆?
    train_ds = HiddenPairDataset(train_pairs, cfg.window_seq_len, src_dim, tgt_dim)
    val_ds = HiddenPairDataset(val_pairs, cfg.window_seq_len, src_dim, tgt_dim)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=0)
    logger.info("  train samples=%d, val samples=%d", len(train_ds), len(val_ds))
    logger.info("  batch=%d, window=%d", cfg.batch_size, cfg.window_seq_len)

    # 妯″瀷
    mot_cfg = MoTHConfig(
        src_hidden_size=src_dim,
        tgt_hidden_size=tgt_dim,
        src_num_layers=30,  # Gemma4
        tgt_num_layers=40,  # Qwen3.6
        translator_hidden=cfg.translator_hidden,
        num_translators=cfg.num_translators,
        top_k=cfg.top_k,
        window_size=4,
    )
    model = MoTH(mot_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("MoT-h 鍙冩暩閲? %s (%.1fM)", f"{n_params:,}", n_params / 1e6)

    # 鍎寲鍣?
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)
    logger.info("绺?steps: %d (epochs=%d 脳 batches=%d)",
                total_steps, cfg.epochs, len(train_loader))

    # 瑷撶反杩村湀
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(cfg.epochs):
        epoch_loss = 0
        epoch_mse = 0
        epoch_cos = 0
        n_batches = 0

        for batch_src, batch_tgt in train_loader:
            # 瑾垮害瀛哥繏鐜?
            lr = cosine_with_warmup(global_step, total_steps, cfg.warmup_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss, metrics = train_step(model, batch_src, batch_tgt, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            epoch_loss += metrics["loss"]
            epoch_mse += metrics["cc"]
            epoch_cos += metrics["cos_sim"]
            n_batches += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = global_step / elapsed
                logger.info(
                    "[epoch %d step %d/%d] loss=%.4f mse=%.4f cos=%.4f "
                    "lr=%.2e | %.1f step/s | ETA %.0fs",
                    epoch + 1, global_step, total_steps,
                    metrics["loss"], metrics["cc"], metrics["cos_sim"],
                    lr, steps_per_sec,
                    (total_steps - global_step) / steps_per_sec)

        # epoch 绲愭潫
        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_cos = epoch_cos / n_batches
        logger.info("=" * 60)
        logger.info("[epoch %d 瀹屾垚] avg_loss=%.4f mse=%.4f cos_sim=%.4f",
                     epoch + 1, avg_loss, avg_mse, avg_cos)

        # 椹楄瓑
        val_metrics = evaluate(model, val_loader, device)
        logger.info("[val] loss=%.4f mse=%.4f cos_sim=%.4f",
                     val_metrics["val_loss"], val_metrics["val_mse"],
                     val_metrics["val_cos_sim"])

        # 淇濆瓨 best
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_path = cfg.output_path.replace(".pt", "_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": mot_cfg.__dict__,
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "train_meta": meta,
            }, save_path)
            logger.info("  鉁?best model saved: %s (val_loss=%.4f)",
                         save_path, best_val_loss)

        # 瀹氭湡淇濆瓨
        if (epoch + 1) % cfg.save_every == 0:
            save_path = cfg.output_path.replace(".pt", f"_e{epoch+1}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": mot_cfg.__dict__,
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "train_meta": meta,
            }, save_path)
            logger.info("  checkpoint: %s", save_path)

    # 鏈€绲備繚瀛?
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": mot_cfg.__dict__,
        "epoch": cfg.epochs,
        "best_val_loss": best_val_loss,
        "train_meta": meta,
    }, cfg.output_path)
    logger.info("=" * 60)
    logger.info("瑷撶反瀹屾垚! 绺借€楁檪: %.1fs", time.time() - t0)
    logger.info("  best_val_loss: %.4f", best_val_loss)
    logger.info("  鏈€绲傛ā鍨? %s", cfg.output_path)
    logger.info("=" * 60)

    # 闋愭湡璩噺瑭曚及
    if best_val_loss < 1e-3:
        logger.info("鉁?val_mse < 1e-3 鈥?鎺ヨ繎鐒℃悕缈昏")
    elif best_val_loss < 1e-2:
        logger.info("鈿狅笍  val_mse < 1e-2 鈥?鍙敤, 浣嗘湁杓曞井鍝佽唱鎼嶅け")
    else:
        logger.info("鉂?val_mse >= 1e-2 鈥?闇€瑕佹洿澶氭暩鎿氭垨鏇撮暦瑷撶反")


# ---------------------------------------------------------------------------
# 涓荤▼寮?
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MoT-h 瑷撶反")
    parser.add_argument("--data", required=True, help="train.pt 璺緫")
    parser.add_argument("--output", default="mot_h.pt", help="杓稿嚭璺緫")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="瑷撶反瑷倷 (default: auto)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window-seq-len", type=int, default=128,
                        help="瑷撶反绐楀彛闀峰害 (default: 128)")
    parser.add_argument("--translator-hidden", type=int, default=4096,
                        help="translator 闅辫棌缍害 (default: 4096, CPU 鍙檷鍒?1024)")
    parser.add_argument("--num-translators", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = TrainConfig(
        data_path=args.data,
        output_path=args.output,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        window_seq_len=args.window_seq_len,
        translator_hidden=args.translator_hidden,
        num_translators=args.num_translators,
        top_k=args.top_k,
        val_split=args.val_split,
        log_every=args.log_every,
    )
    train(cfg)


if __name__ == "__main__":
    main()
