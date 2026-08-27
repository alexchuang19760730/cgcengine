#!/usr/bin/env python3
# DFlash draft-side latency measurement (Road C cost quantification)
# context features -> 6-layer draft -> block of 16 hidden states
import sys, time, statistics
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0516')
import mlx.core as mx
import mlx.nn as nn

from dflash_mlx.runtime.loading import load_draft_bundle

print("=== loading draft (z-lab Qwen3.6-35B-A3B-DFlash, local) ===", flush=True)
t0 = time.time()
model, meta = load_draft_bundle("/Users/alexchuang/Documents/flashkv0516/models/dflash35", lazy=False)
model.eval()
print(f"loaded in {time.time()-t0:.1f}s | layers={len(model.layers)} meta={meta['config'].get('hidden_size')}", flush=True)

fc_out = model.fc.weight.shape[0]      # projected dim (5760)
draft_hidden = model.layers[0].hidden_size if hasattr(model.layers[0], 'hidden_size') else 2048
print(f"fc: {model.fc.weight.shape} -> projected dim {fc_out}, draft hidden {draft_hidden}", flush=True)

# --- realistic noise_embedding: target embed of block tokens [1, 16, 2048] ---
B, BLOCK = 1, 16
noise = mx.random.normal((B, BLOCK, 2048))

def measure(context_len, reps=8, warmup=2):
    # projected target hidden: [1, T, fc_out]
    ctx = mx.random.normal((B, context_len, fc_out))
    # warmup
    for _ in range(warmup):
        h = model.forward_projected_context(noise_embedding=noise, draft_context=ctx)
        mx.eval(h)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        h = model.forward_projected_context(noise_embedding=noise, draft_context=ctx)
        mx.eval(h)
        times.append((time.perf_counter() - t0) * 1000)
    ms = statistics.median(times)
    return ms, h.shape

print("\n=== block generation latency (one block = 16 tokens) ===", flush=True)
print(f"{'ctx_len':>8} | {'block_ms':>9} | {'per_token_ms':>13} | out_shape", flush=True)
for T in (32, 128, 256, 512, 1024):
    ms, shape = measure(T)
    print(f"{T:>8} | {ms:>8.1f}ms | {ms/16:>12.2f}ms | {shape}", flush=True)

# --- projection cost (8 target layers) ---
print("\n=== project_target_hidden cost (8 layers x [1,2048]) ===", flush=True)
th = mx.random.normal((B, 2048))
times = []
for _ in range(5):
    t0 = time.perf_counter()
    p = model.project_target_hidden(th)
    mx.eval(p)
    times.append((time.perf_counter()-t0)*1000)
print(f"project: {statistics.median(times):.2f}ms -> {p.shape}", flush=True)

# --- full pipeline per verify cycle: project + block ---
print("\n=== per verify cycle total (project 8 + block 16 @ T=256) ===", flush=True)
ms_b, _ = measure(256, reps=5)
ms_p = statistics.median(times)
print(f"block(T=256) {ms_b:.1f}ms + project {ms_p:.2f}ms = {ms_b+ms_p:.1f}ms per verify cycle", flush=True)
print(f"draft tokens/cycle=16 -> draft-side ~{(ms_b+ms_p)/16:.2f}ms/token", flush=True)
