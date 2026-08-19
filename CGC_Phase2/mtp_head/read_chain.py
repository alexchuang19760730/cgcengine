#!/usr/bin/env python3
"""Parse the EAGLE-2 chain-dump binary format written by TrainingDataDump.appendChain.

Record layout (all little-endian):
    B i32
    hidden[2816] f32          # true backbone hidden at the committed position
    embed_ctx[2816] f32       # target embedding of ctx
    ctx i32
    drafts[B] i32
    predictions[B+1] i32      # target greedy after span[0...i]
    rowHidden[(B+2)*2816] f16 # backbone hidden at chunk position i (rows[0]=current;
                              # feature target for draft i-1 is rows[i])
    embed_drafts[B][2816] f32 # target embeddings of each draft token
    [sliding KV: pos i32, nk i32, hd i32, stride i32, ringCap i32, startSlot i32,
                 k[pos*nk*hd] f16, v[pos*nk*hd] f16]
    [full KV:    same header + k/v]
"""
from __future__ import annotations

import struct

import numpy as np

HD = 2816  # backbone hidden size (gemma4)

# Cached embed lookups so repeated draft tokens share work (none: pure parse).


def _read_kv(raw, off):
    pos, nk, hd, stride, ring_cap, start_slot = struct.unpack_from("<6i", raw, off)
    off += 24
    n = pos * nk * hd
    if n > 0:
        k = np.frombuffer(raw, dtype=np.float16, count=n, offset=off).reshape(pos, nk, hd)
        off += n * 2
        v = np.frombuffer(raw, dtype=np.float16, count=n, offset=off).reshape(pos, nk, hd)
        off += n * 2
    else:
        k = np.zeros((0, nk, hd), dtype=np.float16)
        v = np.zeros((0, nk, hd), dtype=np.float16)
    return dict(pos=pos, nk=nk, hd=hd, k=k, v=v, ring_cap=ring_cap, start_slot=start_slot), off


def read_chain(path: str) -> list[dict]:
    raw = open(path, "rb").read()
    recs = []
    off = 0
    while off + 4 <= len(raw):
        (B,) = struct.unpack_from("<i", raw, off)
        off += 4
        if B < 0 or B > 16:
            break  # bad alignment guard
        hidden = np.frombuffer(raw, dtype=np.float32, count=HD, offset=off)
        off += HD * 4
        embed_ctx = np.frombuffer(raw, dtype=np.float32, count=HD, offset=off)
        off += HD * 4
        (ctx,) = struct.unpack_from("<i", raw, off)
        off += 4
        drafts = np.frombuffer(raw, dtype=np.int32, count=B, offset=off).tolist()
        off += B * 4
        predictions = np.frombuffer(raw, dtype=np.int32, count=B + 1, offset=off).tolist()
        off += (B + 1) * 4
        rows = np.frombuffer(raw, dtype=np.float16, count=(B + 2) * HD, offset=off)
        rows = rows.reshape(B + 2, HD).astype(np.float32)
        off += (B + 2) * HD * 2
        emb_drafts = []
        for _ in range(B):
            e = np.frombuffer(raw, dtype=np.float32, count=HD, offset=off)
            off += HD * 4
            emb_drafts.append(e)
        sliding, off = _read_kv(raw, off)
        full, off = _read_kv(raw, off)
        recs.append(dict(
            hidden=hidden, embed_ctx=embed_ctx, ctx=ctx,
            drafts=drafts, predictions=predictions, rows=rows,
            emb_drafts=emb_drafts, sliding=sliding, full=full))
    return recs
    return recs


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        recs = read_chain(p)
        print(f"{p}: {len(recs)} records")
        if recs:
            r = recs[0]
            print("  B=%d ctx=%d drafts=%s preds=%s" % (
                len(r["drafts"]), r["ctx"], r["drafts"], r["predictions"]))
            print("  hidden finite=%s rows finite=%s kv pos=%d/%d" % (
                np.isfinite(r["hidden"]).all(), np.isfinite(r["rows"]).all(),
                r["sliding"]["pos"], r["full"]["pos"]))
