#!/usr/bin/env python3
"""sim_gguf_cache.py — replay real routing trace through production-style expert cache.

Answers: with llama.cpp IQ2/IQ3 GGUF expert streaming on this 16GB Mac,
what hit rate and fill-bytes-per-step would a bounded pool achieve?

Semantics (fixed 2026-08-14):
  - hot pool is PER-LAYER: top-K experts of EACH layer are pinned (production
    HOT_POOL_PROFILE_SIZE semantics = K per layer => K*40 pinned pairs), NOT
    the top-K global pairs (that was a bug: it pinned only layer-0's experts).
  - global LRU for the rest (1 slot = 1 (layer, expert) pair)
  - optional adaptive mode: per-layer LRUs sized proportional to each layer's
    observed miss share (two-pass), the "dynamic pool" variant.

Inputs:
  trace:  temp/qwen36_bench/r3_trace.txt  (layer_XX.bin,phase,step,hits,exp1..exp8)
  profile: top*.json (per-layer ordered expert lists) -> hot pool selection
  --layers: number of MoE layers per token (default 40; paper traces differ)
  --warm-trace: flat trace (e.g. prefill phase) replayed to pre-fill the pool
                BEFORE the measured trace, without counting toward stats
                (pool-warmup experiment: paper insight Ob3, prefill->decode).
  --n-exp: experts per MoE layer (qwen36/DSV3=256, gemma4=128)
  --ssd-gbs: measured SSD read throughput for ms/step (default 4.8 was the
                warm page-cache figure; F_NOCACHE cold read on this Mac M4
                measured 2026-08-15: gemma4 4.37 @12t, qwen36 3.04 @16t).
"""
import argparse
import json, sys
from collections import OrderedDict, defaultdict

_ap = argparse.ArgumentParser(add_help=True)
_ap.add_argument("trace", nargs="?", default="temp/qwen36_bench/r3_trace.txt")
_ap.add_argument("profile", nargs="?", default="")
_ap.add_argument("quant", nargs="?", default="iq2")
_ap.add_argument("--layers", type=int, default=40)
_ap.add_argument("--warm-trace", default="", help="flat trace used to pre-fill the pool before replay (stats not counted)")
_ap.add_argument("--n-exp", type=int, default=256, help="experts per MoE layer (qwen36=256, gemma4=128)")
_ap.add_argument("--ssd-gbs", type=float, default=4.8, help="measured SSD read throughput GB/s for ms/step estimate")
_a = _ap.parse_args()

TRACE = _a.trace
PROFILE = _a.profile or \
    "prime-agent-worktrees/qwen36-r3q4la_ga_e4h3.gturbo/profiles/top112_code_prose.json"
WARM_TRACE = _a.warm_trace

N_LAYERS, N_EXP = _a.layers, _a.n_exp
B_EXP = {
    "iq2": 876544,        # qwen36 IQ2_XXS gate+up+down bytes/expert (real GGUF)
    "iq3": 1015808,       # qwen36 IQ3_XXS (real GGUF)
    "gemma4_iq3": 2385152,  # gemma4-26B IQ3_S: ffn_down 1115136 + ffn_gate_up 1270016 (measured 2026-08-15)
}

# ---- load trace, group into tokens (N_LAYERS consecutive lines = 1 token) ----
def _load_tokens(path):
    out = []
    lines = [ln.strip() for ln in open(path) if ln.strip()]
    for i in range(0, len(lines), N_LAYERS):
        tok = []
        for ln in lines[i:i + N_LAYERS]:
            p = ln.split(",")
            layer = int(p[0].split("_")[1].split(".")[0])
            exps = [int(x) for x in p[4].split()]
            tok.append((layer, exps))
        if len(tok) == N_LAYERS:
            out.append(tok)
    return out

tokens = _load_tokens(TRACE)
print(f"trace: {len(tokens)} tokens x {N_LAYERS} layers")

warm_tokens = []
if WARM_TRACE:
    warm_tokens = _load_tokens(WARM_TRACE)
    print(f"warm-trace: {len(warm_tokens)} tokens pre-filled before replay (stats excluded)")

# ---- load profile (per-layer ordered top-K); missing/mismatched -> empty hot ----
per_layer_hot = [[] for _ in range(N_LAYERS)]
try:
    prof = json.load(open(PROFILE))
    if len(prof) == N_LAYERS:
        per_layer_hot = prof  # list of N_LAYERS lists
    else:
        print(f"warn: profile has {len(prof)} layers, expected {N_LAYERS}; using empty hot pool", file=sys.stderr)
except Exception as exc:
    print(f"warn: no usable profile ({exc}); using empty hot pool (pure LRU mode)", file=sys.stderr)

def run_sim(hot_per_layer, lru_size, mode="global", adaptive_pass=None):
    """hot_per_layer: K pinned per layer. lru_size: extra LRU slots.
    mode: 'global' LRU across layers, or 'perlayer' LRU (sized by miss share)."""
    if mode == "global":
        hot = set()
        for li, lst in enumerate(per_layer_hot):
            for e in lst[:hot_per_layer]:
                hot.add((li, e))
        lru = OrderedDict()
        def lru_get(key):
            if key in lru:
                lru.move_to_end(key)
                return True
            lru[key] = True
            lru.move_to_end(key)
            if len(lru) > lru_size:
                lru.popitem(last=False)
            return False
    else:  # per-layer dynamic LRU
        hot = set()
        for li, lst in enumerate(per_layer_hot):
            for e in lst[:hot_per_layer]:
                hot.add((li, e))
        # adaptive_pass: per-layer slot budget (list of ints)
        lrus = [OrderedDict() for _ in range(N_LAYERS)]
        caps = adaptive_pass or [lru_size // N_LAYERS] * N_LAYERS
        def lru_get(key):
            li = key[0]
            l = lrus[li]
            if key in l:
                l.move_to_end(key)
                return True
            l[key] = True
            l.move_to_end(key)
            if len(l) > caps[li]:
                l.popitem(last=False)
            return False

    # ---- warm phase: pre-fill pool from warm trace, no stats counted ----
    for tok in warm_tokens:
        for layer, exps in tok:
            for e in exps:
                key = (layer, e)
                if key in hot:
                    continue
                lru_get(key)

    reqs = hits = 0
    miss_bytes = 0
    per_token_miss = []
    per_layer_miss = defaultdict(int)
    per_layer_req = defaultdict(int)
    for tok in tokens:
        tmiss = 0
        for layer, exps in tok:
            per_layer_req[layer] += len(exps)
            for e in exps:
                key = (layer, e)
                reqs += 1
                if key in hot:
                    hits += 1
                    continue
                if lru_get(key):
                    hits += 1
                    continue
                per_layer_miss[layer] += 1
                tmiss += 1
        per_token_miss.append(tmiss)
        miss_bytes += tmiss * B_EXP[quant]
    hit_rate = hits / reqs
    avg_miss = sum(per_token_miss) / len(per_token_miss)
    avg_bytes = miss_bytes / len(per_token_miss)
    return hit_rate, avg_miss, avg_bytes, per_layer_miss, per_layer_req

# adaptive allocation: pass 1 global LRU -> per-layer miss share -> cap_i = floor(total_slots * miss_i/share_total)
def adaptive_caps(total_slots, hot_per_layer):
    _, _, _, pl_miss, _ = run_sim(hot_per_layer, total_slots, mode="global")
    total = sum(pl_miss.values())
    return [int(total_slots * pl_miss[l] / total) for l in range(N_LAYERS)]

quant = _a.quant
B = B_EXP[quant]
SSD_GBS = _a.ssd_gbs
print(f"quant: {quant.upper()} ({B} B/expert) · SSD read: {SSD_GBS} GB/s (measured F_NOCACHE cold)")

# 3GB budget in pairs
G3 = int(3 * 1024**3 / B)
print(f"3GB pool = {G3} (layer,expert) pairs\n")
print(f"{'config':<42}{'hit%':>7}{'miss/step':>10}{'fill MB/step':>13}{'fill ms/step@' + f'{SSD_GBS}GB/s':>22}")
def report(label, hot, lru_slots, mode="global", caps=None):
    if caps is None:
        hr, am, ab, _, _ = run_sim(hot, lru_slots, mode="global")
    else:
        hr, am, ab, _, _ = run_sim(hot, lru_slots, mode="perlayer", adaptive_pass=caps)
    ms = ab / 1e6 / SSD_GBS  # measured SSD cold pread bandwidth (F_NOCACHE bench 2026-08-15)
    print(f"{label:<46}{hr*100:>6.1f}%{am:>10.1f}{ab/1e6:>13.2f}{ms:>21.1f}")

# within 3GB budget: hot_pairs + lru_slots <= G3 (fair comparison)
n_pairs = 40 if N_LAYERS <= 0 else N_LAYERS
report(f"hot=0 + global LRU {G3} (pure LRU 3GB)", 0, G3)
report(f"hot=90/layer ({90*n_pairs} pairs) + LRU {max(0,G3-90*n_pairs)}", 90, max(0, G3-90*n_pairs))
report(f"hot=60/layer ({60*n_pairs} pairs) + LRU {max(0,G3-60*n_pairs)}", 60, max(0, G3-60*n_pairs))
report(f"hot=30/layer ({30*n_pairs} pairs) + LRU {max(0,G3-30*n_pairs)}", 30, max(0, G3-30*n_pairs))
# adaptive per-layer dynamic (hot=0, 同預算) — the 'dynamic pool' variant
caps0 = adaptive_caps(G3, 0)
report(f"adaptive per-layer LRU {G3} slots (hot=0)", 0, G3, caps=caps0)
# adaptive + 小 hot
capsA = adaptive_caps(max(0,G3-30*n_pairs), 30)
report(f"adaptive per-layer LRU {max(0,G3-30*n_pairs)} (hot=30/layer)", 30, max(0,G3-30*n_pairs), caps=capsA)
