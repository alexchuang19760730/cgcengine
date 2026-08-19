#!/usr/bin/env python3
"""對拍: Python 同構 Gemma4Assistant vs 引擎.

Dump 格式 (每條記錄):
  hidden[2816] f32 | ctx i32 | next i32 | draft0 i32 | embed[2816] f32
  [sliding: pos, nk, hd, stride, off, k[nk*stride] f16, v...]
  [full:    pos, nk, hd, stride, off, k[nk*stride] f16, v...]

embed = 主模型 embed_tokens(ctx) = 引擎 draft0 的 proxyEmbed (initialTokenEmbedding)。
Python 前向必須用它, 不是 head.embed_tokens + post_projection。

關鍵對照:
  - 引擎 accept = (draft0 == next)
  - Python align = (py_pred == draft0)  ← 同輸入同 forward 應同輸出 (架構對齊)
"""
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gemma4_assistant import load_gemma4_assistant

DUMP = "/Users/alexchuang/Documents/flashkv0516/temp/g4_mtp_dump.bin"
HEAD_DIR = "/Users/alexchuang/Documents/flashkv0516/models/gemma-4-mtp-head"
HD = 2816


def read_dump(path: str):
    data = open(path, "rb").read()
    records = []
    base = 0
    while base < len(data):
        h = np.frombuffer(data, dtype=np.float32, count=HD, offset=base)
        base += HD * 4
        ctx, nxt, draft0 = struct.unpack("<iii", data[base:base + 12])
        base += 12
        emb = np.frombuffer(data, dtype=np.float32, count=HD, offset=base)
        base += HD * 4
        kv = {}
        for name in ("sliding", "full"):
            pos, nk, hd, stride, ringCap, startSlot = struct.unpack("<iiiiii", data[base:base + 24])
            base += 24
            if nk > 0 and pos > 0 and startSlot >= 0:
                kc = pos * nk * hd
                k = np.frombuffer(data, dtype=np.float16, count=kc, offset=base)
                base += kc * 2
                v = np.frombuffer(data, dtype=np.float16, count=kc, offset=base)
                base += kc * 2
            else:
                k = v = np.zeros(0, dtype=np.float16)
            kv[name] = dict(pos=pos, nk=nk, hd=hd, stride=stride,
                            ringCap=ringCap, startSlot=startSlot, k=k, v=v)
        records.append((h, ctx, nxt, draft0, emb, kv))
    return records


def build_kv_tensor(kv: dict):
    out = {}
    for kind in ("sliding", "full"):
        d = kv.get(kind)
        if d is None or d["nk"] == 0 or d["pos"] == 0 or d["startSlot"] < 0:
            continue
        nk, hd, pos = d["nk"], d["hd"], d["pos"]
        # dump 已把 ring wrap 攤成邏輯順序: [valid, nk, hd] token-major
        # einsum "shd" 需要 [s=pos, h=nk, d=hd] — 直接 reshape, 不轉置
        k = d["k"].reshape(pos, nk, hd).astype(np.float32)   # [pos, nk, hd]
        v = d["v"].reshape(pos, nk, hd).astype(np.float32)
        out[kind] = dict(k=torch.tensor(k), v=torch.tensor(v), seq_len=pos, window=1024)
    return out


def main():
    print("loading head...", flush=True)
    head, info = load_gemma4_assistant(HEAD_DIR, device="cpu")
    print(f"missing={len(info['missing'])} unexpected={len(info['unexpected'])}", flush=True)

    records = read_dump(DUMP)
    print(f"dump: {len(records)} records", flush=True)

    head = head.float().eval()

    py_ok_vs_next = 0
    py_ok_vs_draft0 = 0
    eng_ok = 0
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for i, (h, ctx, nxt, draft0, emb, kv) in enumerate(records):
            bh = torch.tensor(h, dtype=torch.float32)
            tgt = torch.tensor(emb, dtype=torch.float32)
            kv_map = build_kv_tensor(kv)
            pos = kv.get("sliding", {}).get("pos", 0) or kv.get("full", {}).get("pos", 0)
            pred = int(head(bh, torch.tensor(ctx, dtype=torch.long),
                            target_token_embedding=tgt,
                            kv=kv_map, position=int(pos)).item())
            total += 1
            eng_acc = draft0 == nxt
            py_acc = pred == nxt
            py_al = pred == draft0
            if eng_acc:
                eng_ok += 1
            if py_acc:
                py_ok_vs_next += 1
            if py_al:
                py_ok_vs_draft0 += 1
            print(f"  [{total}] py={pred} eng_draft={draft0} next={nxt} "
                  f"{'E!' if eng_acc else ''}{'P!' if py_acc else ''}{'A' if py_al else ''}", flush=True)
    dt = time.time() - t0
    print(f"\n=== RESULT ===\n"
          f"engine accept (draft0==next): {eng_ok}/{total} = {eng_ok/total*100:.1f}%\n"
          f"python accept (py==next):     {py_ok_vs_next}/{total} = {py_ok_vs_next/total*100:.1f}%\n"
          f"python==engine-draft0 (align): {py_ok_vs_draft0}/{total} = {py_ok_vs_draft0/total*100:.1f}%\n"
          f"({dt:.1f}s)")


if __name__ == "__main__":
    main()
