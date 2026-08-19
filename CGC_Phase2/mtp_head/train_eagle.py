#!/usr/bin/env python3
"""EAGLE-2-style rollout training for the gemma4 MTP assistant head.

Data: engine chain dumps (TrainingDataDump.appendChain) — each record holds the
verify span's row hiddens (backbone hidden after each drafted prefix) and the
target's greedy predictions per row, i.e. the exact conditional supervision for
a self-consistent draft chain.

Objective per chain position i (draft i):
    L_i = CE(head(d_i), predictions[i]) + LAMBDA * MSE(h_hat_i, rows[i+1])
  pos 0    input: (true backbone hidden, target embed of ctx)
  pos i>0  input: (head's own predicted hidden h_hat_{i-1}, target embed of
                   the recorded draft_{i-1})  — the self-consistent rollout
  all pos  share the same KV snapshot (single-position head, engine semantics)

Eval: full-chain rollout on held-out chain dumps (fresh prompt). row share =
1 + avg accepted, where draft i accepted iff head argmax == predictions[i].
Note: for drafts the head has not seen in the dump, eval falls back to the
head's own embed+post_projection (target embeds are only recorded for the
collected drafts) — a small distribution gap, mostly affecting pos>=1.
"""
from __future__ import annotations

import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_chain import read_chain, HD  # noqa: E402
from gemma4_assistant import load_gemma4_assistant  # noqa: E402

HEAD_DIR = "/Users/alexchuang/Documents/flashkv0516/models/gemma-4-mtp-head"
CKPT = "/Volumes/AlexZhuang/g4_eagle_head.pt"
TRAIN_FILES = sorted(
    f"/Volumes/AlexZhuang/g4_mtp_train/chain_p{i:02d}.bin" for i in range(8))
EVAL_FILES = ["/Volumes/AlexZhuang/g4_mtp_train/chain_fresh.bin"]

TRAIN_CTX = 32          # KV truncation (memory)
EVAL_CTX = 0            # full KV for eval
LAMBDA = 0.5
LR = 5e-5
EPOCHS = int(os.environ.get("G4_EAGLE_EPOCHS", "6"))
DEVICE = "mps"


def build_kv(rec, ctx_len: int, device):
    kv = {}
    for kind, key in (("sliding", "sliding"), ("full", "full")):
        d = rec.get(key)
        if not d or d["pos"] <= 0 or d["nk"] <= 0 or d["hd"] <= 0:
            continue
        keep = min(d["pos"], ctx_len) if ctx_len > 0 else d["pos"]
        k = d["k"][-keep:].astype(np.float32)
        v = d["v"][-keep:].astype(np.float32)
        kv[kind] = dict(k=torch.tensor(k, device=device),
                        v=torch.tensor(v, device=device),
                        seq_len=keep, window=1024)
    return kv


def kv_position(kv) -> int:
    pos = 0
    for d in kv.values():
        pos = max(pos, d["seq_len"])
    return pos


def rollout_positions(model, rec, device, train: bool, ctx_len: int):
    """Run the draft chain of one record; return (logits, h_hat, tgt_tok, tgt_feat) per position."""
    kv = build_kv(rec, ctx_len, device)
    pos = kv_position(kv)
    h_in = torch.tensor(rec["hidden"], dtype=torch.float32, device=device)
    emb = torch.tensor(rec["embed_ctx"], dtype=torch.float32, device=device)
    tok = torch.tensor(rec["ctx"], dtype=torch.long, device=device)
    rows = torch.tensor(rec["rows"], dtype=torch.float32, device=device)  # (B+2, HD)
    out = []
    with torch.set_grad_enabled(train):
        for i in range(len(rec["drafts"])):
            pred_tok, logits, h_hat = model(
                h_in, tok, target_token_embedding=emb,
                kv=kv, position=pos, return_logits=True)
            out.append((logits, h_hat, rec["predictions"][i], rows[i + 1]))
            h_in = h_hat.detach()  # rollout input: own predicted hidden (stop-grad)
            emb = torch.tensor(rec["emb_drafts"][i], dtype=torch.float32, device=device)
            tok = torch.tensor(rec["drafts"][i], dtype=torch.long, device=device)
    return out


def run_epoch(model, recs, device, train: bool):
    loss_sum = 0.0
    n_pos = 0
    ce_correct = 0
    ce_total = 0
    random.shuffle(recs)
    for rec in recs:
        positions = rollout_positions(model, rec, device, train, TRAIN_CTX if train else 0)
        loss = torch.tensor(0.0, device=device)
        for logits, h_hat, tgt_tok, tgt_feat in positions:
            ce = F.cross_entropy(logits.unsqueeze(0),
                                 torch.tensor([tgt_tok], dtype=torch.long, device=device))
            mse = F.mse_loss(h_hat, tgt_feat)
            loss = loss + ce + LAMBDA * mse
            ce_correct += int(logits.argmax(-1).item()) == tgt_tok
            ce_total += 1
        if train:
            (loss / max(len(positions), 1)).backward()
        loss_sum += loss.item()
        n_pos += max(len(positions), 1)
    return loss_sum / max(n_pos, 1), ce_correct, ce_total


def eval_chain(model, recs, device, max_pos=4):
    """Full-chain rollout: own drafts + own-embed fallback for unseen tokens."""
    model.eval()
    total_steps = 0
    total_accepted = 0
    with torch.no_grad():
        for rec in recs:
            kv = build_kv(rec, EVAL_CTX, device)
            pos = kv_position(kv)
            h_in = torch.tensor(rec["hidden"], dtype=torch.float32, device=device)
            emb = torch.tensor(rec["embed_ctx"], dtype=torch.float32, device=device)
            tok = torch.tensor(rec["ctx"], dtype=torch.long, device=device)
            accepted = 0
            for i in range(min(len(rec["drafts"]), max_pos)):
                pred_tok, _logits, h_hat = model(
                    h_in, tok, target_token_embedding=emb,
                    kv=kv, position=pos, return_logits=True)
                if pred_tok.item() == rec["predictions"][i]:
                    accepted += 1
                else:
                    break
                h_in = h_hat
                if i + 1 < len(rec["drafts"]):
                    # use the recorded target embedding when the head's draft
                    # matches the recorded one; otherwise own-embed fallback
                    if pred_tok.item() == rec["drafts"][i]:
                        emb = torch.tensor(rec["emb_drafts"][i],
                                           dtype=torch.float32, device=device)
                    else:
                        emb = None
                    tok = pred_tok
            total_steps += 1
            total_accepted += accepted
    model.train()
    if total_steps == 0:
        return 1.0, 0.0, 0
    return 1 + total_accepted / total_steps, total_accepted / total_steps, total_steps


def main():
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device(DEVICE)
    model, _info = load_gemma4_assistant(HEAD_DIR)
    model = model.to(device).train()

    train_recs = []
    for f in TRAIN_FILES:
        if not os.path.exists(f):
            continue
        recs = read_chain(f)
        print(f"{os.path.basename(f)}: {len(recs)} records", flush=True)
        train_recs.extend(recs)
    eval_recs = []
    for f in EVAL_FILES:
        if os.path.exists(f):
            eval_recs.extend(read_chain(f))
    print(f"train={len(train_recs)} eval={len(eval_recs)}", flush=True)

    row_share, p0, n = eval_chain(model, eval_recs, device)
    print(f"[epoch init] eval_row_share={row_share:.3f} (n={n})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR / 10)
    start_epoch = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        # ckpt stores only trainable params (embed_tokens is frozen)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"resumed from epoch {start_epoch - 1}", flush=True)

    for ep in range(start_epoch, EPOCHS):
        t0 = time.time()
        model.train()
        loss, correct, total = run_epoch(model, train_recs, device, train=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        row_share, p0, n = eval_chain(model, eval_recs, device)
        print(f"[ep {ep}] loss={loss:.4f} ce_acc={correct}/{total} "
              f"eval_row_share={row_share:.3f} ({time.time() - t0:.0f}s)", flush=True)
        saveable = {k: v for k, v in model.state_dict().items()
                    if not k.startswith("embed_tokens.")}
        tmp = CKPT + ".tmp"
        torch.save({"model": saveable, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep}, tmp)
        os.replace(tmp, CKPT)  # atomic: a kill can never corrupt the main ckpt

    row_share, p0, n = eval_chain(model, eval_recs, device)
    print(f"DONE final_row_share={row_share:.3f} p0={p0:.3f} n={n}", flush=True)


if __name__ == "__main__":
    main()
