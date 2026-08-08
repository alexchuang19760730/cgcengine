#!/usr/bin/env python3
"""Test REAL chain accuracy on training data (using MTP output, not real hidden).

This tells us if the bottleneck is:
1. Overfitting (high train, low test) → need more data
2. Chain degradation (low chain acc even on train) → architecture issue
"""
import os, sys, torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [REPO_ROOT, SCRIPT_DIR, os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_head")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mtp_head.model import MTPHead, MTPHeadConfig

DATA_DIR = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_train_data/qwen25_llamacpp"
MTP_CKPT = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_output/qwen25_llamacpp/mtp_head_qwen25-0.5b_decode.pt"
EMBED_HEAD = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_train_data/qwen25_llamacpp/embed_head.pt"

HIDDEN = 896; VOCAB = 151936; N_HEADS = 14; HEAD_DIM = 64; INTER = 4864

def main():
    # Load data
    shard = torch.load(os.path.join(DATA_DIR, "shard_000000.pt"), weights_only=False)
    print(f"Loaded {len(shard)} samples")

    # Load weights
    eh = torch.load(EMBED_HEAD, map_location="cpu", weights_only=True)
    embed_w = eh["embed_weight"].float()
    lm_head_w = eh["lm_head_weight"].float()

    # Load MTP
    config = MTPHeadConfig(hidden_size=HIDDEN, vocab_size=VOCAB, num_heads=N_HEADS,
                           head_dim=HEAD_DIM, intermediate_size=INTER)
    mtp = MTPHead(config)
    ckpt = torch.load(MTP_CKPT, weights_only=False, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    filtered = {k: v for k, v in sd.items() if "lm_head" not in k}
    mtp.load_state_dict(filtered, strict=False)
    mtp.set_shared_lm_head(lm_head_w)
    mtp.eval()

    # Test 1: Step-0 accuracy (use real hidden for each step)
    print("\n=== Test 1: Step-0 accuracy (real hidden, each step independent) ===")
    correct = 0; total = 0
    for i in range(min(200, len(shard))):
        s = shard[i]
        for k in range(s["hidden_states"].shape[0]):
            h = s["hidden_states"][k].float().unsqueeze(0).unsqueeze(0)
            t = s["token_ids"][k].long()
            e = embed_w[t].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = mtp(h, e)
            pred = int(logits[0, 0].argmax().item())
            target = int(s["next_token_ids"][k].item())
            if pred == target: correct += 1
            total += 1
    print(f"  Step-0 (real hidden): {correct}/{total} = {correct/total:.1%}")

    # Test 2: Chain accuracy (step 0 real, steps 1-3 use MTP output)
    print("\n=== Test 2: Chain accuracy (step 0 real, 1-3 MTP output) ===")
    chain_correct = [0, 0, 0, 0]
    chain_total = [0, 0, 0, 0]
    for i in range(min(200, len(shard))):
        s = shard[i]
        current_hidden = s["hidden_states"][0].float().unsqueeze(0).unsqueeze(0)
        for k in range(s["hidden_states"].shape[0]):
            t = s["token_ids"][k].long()
            e = embed_w[t].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                # Forward
                x = torch.cat([current_hidden, e], dim=-1)
                x = mtp.proj(x)
                h = x + mtp.attn(mtp.norm1(x))
                h = h + mtp.mlp(mtp.norm2(h))
                mtp_hidden = mtp.norm_out(h)
                logits = mtp.lm_head(mtp_hidden)

            pred = int(logits[0, 0].argmax().item())
            target = int(s["next_token_ids"][k].item())
            if pred == target: chain_correct[k] += 1
            chain_total[k] += 1

            # Chain: use MTP output for next step
            current_hidden = mtp_hidden.detach()

    for k in range(4):
        if chain_total[k] > 0:
            print(f"  Step {k}: {chain_correct[k]}/{chain_total[k]} = {chain_correct[k]/chain_total[k]:.1%}")
    overall = sum(chain_correct) / max(sum(chain_total), 1)
    print(f"  Overall chain: {overall:.1%}")

    # Test 3: What if we use the REAL next hidden (oracle) for chain?
    print("\n=== Test 3: Oracle chain (always use real hidden) ===")
    oracle_correct = [0, 0, 0, 0]
    oracle_total = [0, 0, 0, 0]
    for i in range(min(200, len(shard))):
        s = shard[i]
        for k in range(s["hidden_states"].shape[0]):
            h = s["hidden_states"][k].float().unsqueeze(0).unsqueeze(0)
            t = s["token_ids"][k].long()
            e = embed_w[t].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = mtp(h, e)
            pred = int(logits[0, 0].argmax().item())
            target = int(s["next_token_ids"][k].item())
            if pred == target: oracle_correct[k] += 1
            oracle_total[k] += 1
    for k in range(4):
        if oracle_total[k] > 0:
            print(f"  Step {k}: {oracle_correct[k]}/{oracle_total[k]} = {oracle_correct[k]/oracle_total[k]:.1%}")

if __name__ == "__main__":
    main()
