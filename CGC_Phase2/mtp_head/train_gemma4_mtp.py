#!/usr/bin/env python3
"""端側 MPS finetune gemma4 官方 MTP assistant head.

數據: 引擎 plain-decode dump (TrainingDataDump 格式, /Volumes/AlexZhuang/g4_mtp_train/*.bin)
模型: gemma4_assistant.Gemma4Assistant (已 100% 對齊引擎 Metal forward)
目標: 提高 draft0 接受率 (CE on 主模型 greedy next token)
凍結: embed_tokens (tied lm_head, 268M params) — 只訓 transformer + projections (~150M)
"""
from __future__ import annotations

import glob
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from compare_gemma4_head import read_dump, HD
from gemma4_assistant import load_gemma4_assistant

HEAD_DIR = "/Users/alexchuang/Documents/flashkv0516/models/gemma-4-mtp-head"
DUMP_DIR = "/Volumes/AlexZhuang/g4_mtp_train"
CKPT = "/Users/alexchuang/Documents/flashkv0516/temp/g4_mtp_head_ft.pt"
TRAIN_CTX = 32           # 每樣本 KV 最多保留最近 N token (記憶體/速度折衷)
SUBSAMPLE = 2            # 每 N 條取 1 條 (相鄰 decode step 高度相關)
BATCH = 1
GRAD_ACC = 8
EPOCHS = 12
LR = 2e-5
MAX_EPOCHS = int(os.environ.get("G4_MAX_EPOCHS", "0"))  # 0 = 全跑
VAL_FILES = ["/Volumes/AlexZhuang/g4_mtp_train/eval_ab.bin"]  # 新鮮 prompt (泛化測試)


def truncate_kv(kv: dict):
    """把 sliding/full KV 截斷到最近 TRAIN_CTX token (依 logical order)。"""
    out = {}
    for kind in ("sliding", "full"):
        d = kv.get(kind)
        if d is None or d["nk"] == 0 or d["pos"] == 0 or d["startSlot"] < 0:
            continue
        pos = d["pos"]
        keep = min(pos, TRAIN_CTX)
        # 存 fp16 (dataset 記憶體減半), forward 時再升 fp32
        k = d["k"].reshape(pos, d["nk"], d["hd"]).astype(np.float16)[-keep:]
        v = d["v"].reshape(pos, d["nk"], d["hd"]).astype(np.float16)[-keep:]
        out[kind] = dict(k=torch.tensor(k), v=torch.tensor(v), seq_len=keep, window=1024)
    return out


class Dataset:
    def __init__(self, paths: list[str], subsample: int = 1):
        self.samples = []  # (hidden[2816] f32, ctx, next, kv dict)
        self.add_paths(paths, subsample)
        print(f"dataset: {len(self.samples)} samples from {len(paths)} files", flush=True)

    def __len__(self):
        return len(self.samples)

    def sample_batches(self, batch: int):
        idx = list(range(len(self.samples)))
        random.shuffle(idx)
        for i in range(0, len(idx), batch):
            yield [self.samples[j] for j in idx[i:i + batch]]

    def add_paths(self, paths: list[str], subsample: int):
        """解析 dump 檔, 每 subsample 條取 1 條。"""
        for p in paths:
            if not os.path.exists(p):
                continue
            n = 0
            for h, ctx, nxt, _d, emb, kv in read_dump(p):
                if n % subsample:
                    n += 1
                    continue
                n += 1
                self.samples.append((
                    torch.tensor(h, dtype=torch.float32),
                    int(ctx), int(nxt),
                    torch.tensor(emb, dtype=torch.float32),
                    truncate_kv(kv)))


def run_batch(model, batch, train: bool, device: str):
    """逐樣本 forward + CE; 回傳 (loss, correct, total)。"""
    loss_sum = torch.tensor(0.0)
    correct = total = 0
    for h, ctx, nxt, emb, kv in batch:
        h = h.to(device)
        emb = emb.to(device)
        kv = {k: dict(v, k=v["k"].to(device).float(), v=v["v"].to(device).float()) for k, v in kv.items()}
        pos = kv.get("sliding", {}).get("seq_len", 0) or kv.get("full", {}).get("seq_len", 0)
        token, logits, _nb = model(
            h, torch.tensor(ctx, dtype=torch.long).to(device),
            target_token_embedding=emb, kv=kv, position=int(pos), return_logits=True)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([nxt], dtype=torch.long).to(device))
        if train:
            (loss / GRAD_ACC).backward()
        loss_sum = loss_sum + loss.detach()
        correct += int(token.item()) == nxt
        total += 1
    return loss_sum / max(len(batch), 1), correct, total


def evaluate(model, ds, device: str):
    model.eval()
    losses = []
    correct = total = 0
    with torch.no_grad():
        for batch in ds.sample_batches(32):
            loss, c, t = run_batch(model, batch, train=False, device=device)
            losses.append(loss.item())
            correct += c
            total += t
    model.train()
    return float(np.mean(losses)), correct, total


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}", flush=True)

    model, info = load_gemma4_assistant(HEAD_DIR, device="cpu")
    print(f"loaded head: missing={len(info['missing'])} unexpected={len(info['unexpected'])}", flush=True)
    model = model.float().to(device)
    model.train()

    # 凍結 tied embed (lm_head)
    for p in model.embed_tokens.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.1f}M", flush=True)

    files = sorted(glob.glob(os.path.join(DUMP_DIR, "train_p*.bin"))) + \
            sorted(glob.glob(os.path.join(DUMP_DIR, "train_d*.bin"))) + \
            sorted(glob.glob(os.path.join(DUMP_DIR, "eval_*.bin")))
    val_files = [f for f in files if os.path.basename(f) in [os.path.basename(x) for x in VAL_FILES]]
    train_files = [f for f in files if f not in val_files]
    print(f"train files: {len(train_files)} val files: {len(val_files)}", flush=True)

    train_ds = Dataset(train_files, subsample=SUBSAMPLE)
    val_ds = Dataset(val_files, subsample=1) if val_files else None

    # resume 支援: 若 ckpt 存在, 從上次完成的 epoch 繼續
    start_ep = 0
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location="cpu")
        epoch_marker = sd.pop("epoch", None)
        model.load_state_dict(sd)
        if epoch_marker is not None:
            start_ep = int(epoch_marker) + 1
        print(f"resumed from epoch {start_ep}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    end_ep = MAX_EPOCHS if MAX_EPOCHS > 0 else EPOCHS

    # baseline eval
    if val_ds:
        vl, vc, vt = evaluate(model, val_ds, device)
        print(f"[init] val loss={vl:.3f} accept={vc}/{vt}={vc/max(vt,1)*100:.1f}%", flush=True)

    step = 0
    t0 = time.time()
    for ep in range(start_ep, end_ep):
        ep_loss = []
        ep_correct = ep_total = 0
        for bi, batch in enumerate(train_ds.sample_batches(BATCH)):
            loss, c, t = run_batch(model, batch, train=True, device=device)
            ep_loss.append(loss.item())
            ep_correct += c
            ep_total += t
            step += 1
            if step % GRAD_ACC == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        sched.step()
        ml = float(np.mean(ep_loss))
        ma = ep_correct / max(ep_total, 1) * 100
        msg = f"[ep {ep+1}/{end_ep}] loss={ml:.3f} train_accept={ma:.1f}% ({step} steps, {time.time()-t0:.0f}s)"
        if val_ds:
            vl, vc, vt = evaluate(model, val_ds, device)
            msg += f" | val loss={vl:.3f} accept={vc}/{vt}={vc/max(vt,1)*100:.1f}%"
        print(msg, flush=True)
        sd = model.state_dict()
        sd["epoch"] = ep
        torch.save(sd, CKPT)

    print(f"DONE. checkpoint: {CKPT}", flush=True)


if __name__ == "__main__":
    main()
