#!/usr/bin/env python3
"""Diagnostic: compare training data hidden states with live llama.cpp hidden states,
and test MTP head predictions directly.

This isolates WHERE the train/inference mismatch occurs:
  1. Are training-data hidden states the same as live llama.cpp hidden states?
  2. Does the MTP head produce correct predictions on training data?
  3. Does the MTP head produce correct predictions on live hidden states?
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import llama_cpp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [REPO_ROOT, SCRIPT_DIR, os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_head")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mtp_head.model import MTPHead, MTPHeadConfig

MODEL_PATH = "/Users/alexchuang/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf"
MTP_CKPT = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_output/qwen25_llamacpp/mtp_head_qwen25-0.5b_decode.pt"
EMBED_HEAD = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_train_data/qwen25_llamacpp/embed_head.pt"
DATA_DIR = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_train_data/qwen25_llamacpp"

HIDDEN = 896
VOCAB = 151936
N_HEADS = 14
HEAD_DIM = 64
INTER = 4864


def get_hidden(ctx, n_embd):
    ptr = llama_cpp.llama_get_embeddings_ith(ctx, 0)
    base = ctypes.addressof(ptr.contents)
    arr = ctypes.c_float * n_embd
    return np.array(arr.from_address(base), dtype=np.float32)


def get_logits(ctx, n_vocab):
    ptr = llama_cpp.llama_get_logits(ctx)
    base = ctypes.addressof(ptr.contents)
    arr = ctypes.c_float * n_vocab
    return np.array(arr.from_address(base), dtype=np.float32)


def decode_single(ctx, token_id, pos, seq_id=0):
    token_arr = (llama_cpp.llama_token * 1)(token_id)
    batch = llama_cpp.llama_batch_get_one(token_arr, 1, pos, seq_id)
    ret = llama_cpp.llama_decode(ctx, batch)
    if ret != 0:
        raise RuntimeError(f"decode failed: {ret}")
    return get_hidden(ctx, HIDDEN), get_logits(ctx, VOCAB)


def main():
    # 1. Load training data
    print("=" * 60)
    print("  1. Load training data")
    print("=" * 60)
    shard = torch.load(os.path.join(DATA_DIR, "shard_000000.pt"), weights_only=False)
    print(f"  Shards loaded: {len(shard)} samples")
    sample = shard[0]
    print(f"  Sample 0: hidden={sample['hidden_states'].shape}, tokens={sample['token_ids']}, next={sample['next_token_ids']}")

    # 2. Load embed + lm_head
    print("\n" + "=" * 60)
    print("  2. Load embed + lm_head weights")
    print("=" * 60)
    eh = torch.load(EMBED_HEAD, map_location="cpu", weights_only=True)
    embed_w = eh["embed_weight"].float()
    lm_head_w = eh["lm_head_weight"].float()
    print(f"  embed: {embed_w.shape}, lm_head: {lm_head_w.shape}")
    print(f"  tied: {torch.equal(embed_w, lm_head_w)}")

    # 3. Load MTP head
    print("\n" + "=" * 60)
    print("  3. Load MTP head")
    print("=" * 60)
    config = MTPHeadConfig(
        hidden_size=HIDDEN, vocab_size=VOCAB, num_heads=N_HEADS,
        head_dim=HEAD_DIM, intermediate_size=INTER,
    )
    mtp = MTPHead(config)
    ckpt = torch.load(MTP_CKPT, weights_only=False, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    filtered = {k: v for k, v in sd.items() if "lm_head" not in k}
    mtp.load_state_dict(filtered, strict=False)
    mtp.set_shared_lm_head(lm_head_w)
    mtp.eval()
    print(f"  Loaded {len(filtered)} tensors, step={ckpt.get('step', '?')}")

    # 4. Test MTP on training data (step 0 only)
    print("\n" + "=" * 60)
    print("  4. Test MTP on TRAINING DATA (step 0, first 100 samples)")
    print("=" * 60)
    correct = 0
    total = 0
    for i in range(min(100, len(shard))):
        s = shard[i]
        h = s["hidden_states"][0].float().unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        t = s["token_ids"][0].long()
        e = embed_w[t].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        with torch.no_grad():
            logits = mtp(h, e)  # [1, 1, vocab]
        pred = int(logits[0, 0].argmax().item())
        target = int(s["next_token_ids"][0].item())
        if pred == target:
            correct += 1
        total += 1
    print(f"  Step-0 accuracy on training data: {correct}/{total} = {correct/total:.1%}")

    # 5. Load llama.cpp and generate live hidden states
    print("\n" + "=" * 60)
    print("  5. Load llama.cpp and generate live hidden states")
    print("=" * 60)
    llm = llama_cpp.Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=512,
        embedding=True,
        logits_all=False,
        verbose=False,
    )
    ctx = llm.ctx
    seq_id = 0

    # Use a simple prompt
    prompt = "Write a Python function to check if a number is prime:"
    tokens = list(llm.tokenize(prompt.encode(), add_bos=True, special=True))
    tokens = [t for t in tokens if t not in {151643, 151645}]
    n_prompt = len(tokens)
    print(f"  Prompt: {prompt}")
    print(f"  Tokens: {n_prompt}")

    # Prefill
    pos = 0
    batch_size = min(llm.n_batch, n_prompt)
    while pos < n_prompt:
        end = min(pos + batch_size, n_prompt)
        bt = tokens[pos:end]
        n_bt = len(bt)
        arr = (llama_cpp.llama_token * n_bt)(*bt)
        batch = llama_cpp.llama_batch_get_one(arr, n_bt, pos, seq_id)
        ret = llama_cpp.llama_decode(ctx, batch)
        if ret != 0:
            print(f"  Prefill failed at pos={pos}, ret={ret}")
            return
        pos = end

    # Get first token
    logits = get_logits(ctx, VOCAB)
    first_token = int(logits.argmax())
    print(f"  First token: {first_token} (id), logits argmax")

    # 6. Decode tokens and compare MTP vs target
    print("\n" + "=" * 60)
    print("  6. Live test: MTP vs target model (20 steps)")
    print("=" * 60)

    current_token = first_token
    n_past = n_prompt
    live_correct = 0
    live_total = 0

    # Also compare hidden state norms
    train_hiddens_norms = [float(shard[i]["hidden_states"][0].norm()) for i in range(min(20, len(shard)))]

    for step in range(20):
        # Decode current_token
        hidden_np, logits_np = decode_single(ctx, current_token, n_past, seq_id)
        n_past += 1

        hidden_t = torch.from_numpy(hidden_np.copy()).float()
        hidden_norm = float(hidden_t.norm())

        # Target prediction
        target_token = int(np.argmax(logits_np))

        # MTP prediction
        h_3d = hidden_t.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        e_3d = embed_w[current_token].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        with torch.no_grad():
            mtp_logits = mtp(h_3d, e_3d)  # [1, 1, vocab]
        mtp_pred = int(mtp_logits[0, 0].argmax().item())

        # Also check: what does lm_head(hidden) predict? (without MTP, just raw hidden → lm_head)
        raw_logits = F.linear(hidden_t, lm_head_w)
        raw_pred = int(raw_logits.argmax().item())

        match = "MATCH" if mtp_pred == target_token else "MISS"
        raw_match = "MATCH" if raw_pred == target_token else "MISS"

        if mtp_pred == target_token:
            live_correct += 1
        live_total += 1

        print(f"  Step {step:2d}: token={current_token:6d} | hidden_norm={hidden_norm:.2f} | "
              f"target={target_token:6d} mtp={mtp_pred:6d} [{match}] | "
              f"raw_lm_head={raw_pred:6d} [{raw_match}] | "
              f"train_norm_avg={np.mean(train_hiddens_norms):.2f}")

        if target_token in {151643, 151645}:
            print("  EOS reached")
            break
        current_token = target_token

    print(f"\n  Live step-0 accuracy: {live_correct}/{live_total} = {live_correct/live_total:.1%}")

    # 7. Compare training hidden vs live hidden directly
    print("\n" + "=" * 60)
    print("  7. Direct hidden state comparison (training vs live)")
    print("=" * 60)

    # Find a training sample where token_ids[0] matches our first_token
    match_found = False
    for i in range(len(shard)):
        s = shard[i]
        if int(s["token_ids"][0].item()) == first_token:
            train_hidden = s["hidden_states"][0].float()
            print(f"  Found matching training sample at index {i}")
            print(f"  Training hidden norm: {float(train_hidden.norm()):.4f}")
            print(f"  Live hidden norm:     {float(hidden_t.norm()):.4f}")

            # Compute cosine similarity
            cos_sim = F.cosine_similarity(train_hidden.unsqueeze(0), hidden_t.unsqueeze(0)).item()
            print(f"  Cosine similarity: {cos_sim:.6f}")

            # Compute L2 distance
            l2 = float((train_hidden - hidden_t).norm())
            print(f"  L2 distance: {l2:.4f}")

            # Element-wise stats
            diff = (train_hidden - hidden_t).abs()
            print(f"  Element diff: mean={float(diff.mean()):.6f}, max={float(diff.max()):.6f}")

            match_found = True
            break

    if not match_found:
        print(f"  No training sample with token_id={first_token} found")
        # Just compare first training sample's hidden norm with live
        train_h = shard[0]["hidden_states"][0].float()
        print(f"  Training sample 0: token={int(shard[0]['token_ids'][0].item())}, hidden_norm={float(train_h.norm()):.4f}")
        print(f"  Live (last): token={current_token}, hidden_norm={float(hidden_t.norm()):.4f}")

        # Show distribution of training hidden norms
        train_norms = [float(shard[i]["hidden_states"][0].norm()) for i in range(min(200, len(shard)))]
        print(f"  Training hidden norms: mean={np.mean(train_norms):.2f}, std={np.std(train_norms):.2f}, "
              f"min={np.min(train_norms):.2f}, max={np.max(train_norms):.2f}")

    # 8. Check if raw hidden → lm_head gives correct prediction (bypass MTP entirely)
    print("\n" + "=" * 60)
    print("  8. Raw hidden → lm_head accuracy (bypass MTP, sanity check)")
    print("=" * 60)
    # This tests if llama_get_embeddings_ith returns the correct hidden state
    # that, when passed through lm_head, gives the same prediction as llama_get_logits
    raw_correct = 0
    raw_total = 0

    # Rewind and redo
    if hasattr(llama_cpp, 'llama_memory_seq_rm'):
        mem = llama_cpp.llama_get_memory(ctx)
        llama_cpp.llama_memory_seq_rm(mem, seq_id, 0, -1)

    # Re-prefill
    pos = 0
    while pos < n_prompt:
        end = min(pos + batch_size, n_prompt)
        bt = tokens[pos:end]
        n_bt = len(bt)
        arr = (llama_cpp.llama_token * n_bt)(*bt)
        batch = llama_cpp.llama_batch_get_one(arr, n_bt, pos, seq_id)
        llama_cpp.llama_decode(ctx, batch)
        pos = end

    logits = get_logits(ctx, VOCAB)
    current_token = int(logits.argmax())
    n_past = n_prompt

    for step in range(20):
        hidden_np, logits_np = decode_single(ctx, current_token, n_past, seq_id)
        n_past += 1

        target = int(np.argmax(logits_np))
        hidden_t = torch.from_numpy(hidden_np.copy()).float()
        raw_pred = int(F.linear(hidden_t, lm_head_w).argmax().item())

        if raw_pred == target:
            raw_correct += 1
        raw_total += 1

        if target in {151643, 151645}:
            break
        current_token = target

    print(f"  Raw hidden → lm_head accuracy: {raw_correct}/{raw_total} = {raw_correct/raw_total:.1%}")
    if raw_correct / raw_total < 0.9:
        print("  [WARN] Low accuracy! llama_get_embeddings_ith may not return the correct hidden state.")
        print("         The hidden state might be pre-norm or post-norm, not matching lm_head input.")
    else:
        print("  [OK] Hidden states are correct - lm_head(hidden) matches logits.argmax()")

    print("\n" + "=" * 60)
    print("  Diagnostic complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
