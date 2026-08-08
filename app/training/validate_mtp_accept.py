#!/usr/bin/env python3
"""Offline MTP Head Accept Rate Validator.

Validates trained MTP head without deploying the full base model.
Uses collected hidden states (from collect_hidden_states.py) to simulate
speculative decoding chain and compute accept rate at each step.

Usage:
  python3 validate_mtp_accept.py \
    --model dsv4 \
    --checkpoint /data/mtp_output/dsv4/mtp_head_dsv4_decode.pt \
    --data-dir /data/mtp_train_data/dsv4 \
    --device cuda \
    --num-samples 500 \
    --chain-length 5

  python3 validate_mtp_accept.py \
    --model qwen3vl \
    --checkpoint /data/mtp_output/qwen3vl/mtp_head_qwen3vl_decode.pt \
    --data-dir /data/mtp_train_data/qwen3vl \
    --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# === Path setup ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "app", "shared"),
          os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_head")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def load_mtp_head(model_name: str, checkpoint_path: str, embed_weight: torch.Tensor,
                  lm_head_weight: torch.Tensor, device: torch.device):
    """Load trained MTP head from checkpoint."""
    from model import create_mtp_head_by_model_name

    mtp = create_mtp_head_by_model_name(model_name)
    mtp.set_shared_lm_head(lm_head_weight)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        mtp.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[validate] Loaded checkpoint: {checkpoint_path}")
        if "epoch" in ckpt:
            print(f"[validate]   epoch={ckpt['epoch']}, step={ckpt.get('step', '?')}")
        if "metrics" in ckpt:
            print(f"[validate]   training metrics: {ckpt['metrics']}")
    else:
        # Direct state dict
        mtp.load_state_dict(ckpt, strict=False)
        print(f"[validate] Loaded direct state dict: {checkpoint_path}")

    mtp.to(device).to(torch.float32)
    mtp.eval()
    return mtp


def validate_accept_rate(
    mtp: nn.Module,
    embed_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    hidden_states: torch.Tensor,   # [B, chain, hidden]
    token_ids: torch.Tensor,       # [B, chain]
    next_token_ids: torch.Tensor,  # [B, chain]
    device: torch.device,
    chain_length: int = 5,
):
    """Simulate speculative decoding chain and compute accept rate.

    For each sample:
      Step 0: feed real hidden_state[0] + token_embed[0] → predict next_token[0]
      Step k: feed MTP output from step k-1 + token_embed[k] → predict next_token[k]

    Accept rate at step k = fraction where argmax(logits) == next_token_ids[k]
    """
    batch_size, max_chain, hidden_size = hidden_states.shape
    chain_len = min(chain_length, max_chain)

    # Per-step correct counts
    step_correct = [0] * chain_len
    step_total = [0] * chain_len

    # Chain accept: how many consecutive correct predictions (like sglang accept length)
    chain_accept_lengths = []  # accept length per sample

    # Top-k accuracy (k=1, 2, 5)
    topk_correct = {1: [0] * chain_len, 2: [0] * chain_len, 5: [0] * chain_len}

    current_hidden = hidden_states[:, 0, :].to(device)  # [B, hidden]

    with torch.no_grad():
        for k in range(chain_len):
            # Get token embedding
            token_embed = F.embedding(token_ids[:, k].to(device), embed_weight)

            # MTP forward
            h_3d = current_hidden.unsqueeze(1)  # [B, 1, hidden]
            e_3d = token_embed.unsqueeze(1)    # [B, 1, hidden]

            # Manual forward (same as training)
            x = torch.cat([h_3d, e_3d], dim=-1)
            x = mtp.proj(x)
            x = x + mtp.attn(mtp.norm1(x))
            x = x + mtp.mlp(mtp.norm2(x))
            mtp_hidden = mtp.norm_out(x)  # [B, 1, hidden]

            # Compute logits
            logits = F.linear(mtp_hidden[:, 0, :], lm_head_weight)  # [B, vocab]

            # Predictions
            pred = logits.argmax(dim=-1)  # [B]
            targets = next_token_ids[:, k].to(device)

            correct = (pred == targets)
            step_correct[k] = correct.sum().item()
            step_total[k] = batch_size

            # Top-k
            for k_val in [1, 2, 5]:
                _, topk_idx = torch.topk(logits, k_val, dim=-1)
                topk_correct[k_val][k] = (topk_idx == targets.unsqueeze(-1)).any(dim=-1).sum().item()

            # Chain: next step uses MTP output
            if k < chain_len - 1:
                current_hidden = mtp_hidden[:, 0, :]

    # Compute chain accept lengths (per-sample, for accurate accept length distribution)
    # For each sample, count how many consecutive steps were correct from step 0
    for b in range(batch_size):
        accept_len = 0
        current_hidden_b = hidden_states[b, 0, :].to(device)  # [hidden] — 1D
        for k in range(chain_len):
            token_embed_b = F.embedding(token_ids[b, k].unsqueeze(0).to(device), embed_weight)  # [1, hidden]
            h_3d = current_hidden_b.view(1, 1, -1)  # [1, 1, hidden] — always 3D
            e_3d = token_embed_b.unsqueeze(1)       # [1, 1, hidden] — 3D
            with torch.no_grad():
                x = torch.cat([h_3d, e_3d], dim=-1)  # [1, 1, 2*hidden]
                x = mtp.proj(x)                       # [1, 1, hidden]
                x = x + mtp.attn(mtp.norm1(x))
                x = x + mtp.mlp(mtp.norm2(x))
                mtp_hidden = mtp.norm_out(x)           # [1, 1, hidden]
                logits = F.linear(mtp_hidden[0, 0, :], lm_head_weight)  # [vocab]
                pred = logits.argmax(dim=-1)
                target = next_token_ids[b, k].to(device)
                if pred.item() == target.item():
                    accept_len += 1
                    current_hidden_b = mtp_hidden[0, 0, :]  # [hidden] — keep 1D
                else:
                    break
        chain_accept_lengths.append(accept_len)

    # Summary
    results = {
        "step_accuracy": [],
        "topk_accuracy": {1: [], 2: [], 5: []},
        "chain_accept_rate": 0.0,
        "avg_accept_length": 0.0,
        "total_samples": batch_size,
        "chain_length": chain_len,
    }

    for k in range(chain_len):
        acc = step_correct[k] / max(step_total[k], 1)
        results["step_accuracy"].append(round(acc, 4))
        for k_val in [1, 2, 5]:
            results["topk_accuracy"][k_val].append(round(topk_correct[k_val][k] / max(step_total[k], 1), 4))

    # Chain accept rate = fraction of samples that accepted >= 1 token
    chain_accept_lengths_t = torch.tensor(chain_accept_lengths)
    results["chain_accept_rate"] = round((chain_accept_lengths_t >= 1).float().mean().item(), 4)
    results["avg_accept_length"] = round(chain_accept_lengths_t.float().mean().item(), 4)

    # Accept length distribution
    dist = {}
    for l in range(chain_len + 1):
        dist[f"accept_{l}"] = round((chain_accept_lengths_t == l).float().mean().item(), 4)
    results["accept_length_distribution"] = dist

    # Expected speedup = 1 + avg_accept_length (each accepted token saves one forward pass)
    results["expected_speedup"] = round(1 + results["avg_accept_length"], 2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Offline MTP Head Accept Rate Validator")
    parser.add_argument("--model", required=True, choices=["dsv4", "qwen3vl", "gemma4"],
                        help="Model name")
    parser.add_argument("--checkpoint", required=True, help="Path to trained MTP head checkpoint")
    parser.add_argument("--data-dir", required=True, help="Directory with shard_*.pt and embed_head.pt")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to validate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--chain-length", type=int, default=5, help="Max chain length to validate")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    args = parser.parse_args()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[validate] CUDA not available, using CPU", file=sys.stderr)
        args.device = "cpu"
    dev = torch.device(args.device)
    print(f"[validate] Device: {dev}", flush=True)

    if dev.type == "cuda":
        gpu_id = torch.cuda.current_device()
        print(f"[validate] GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}", flush=True)

    # 1. Load embed + lm_head weights
    embed_head_path = os.path.join(args.data_dir, "embed_head.pt")
    if not os.path.exists(embed_head_path):
        embed_head_path = os.path.join(args.data_dir, "worker_0", "embed_head.pt")
    if not os.path.exists(embed_head_path):
        raise FileNotFoundError(f"embed_head.pt not found in {args.data_dir}")

    print(f"[validate] Loading embed+head from {embed_head_path}...", flush=True)
    weights = torch.load(embed_head_path, map_location="cpu", weights_only=True)
    embed_weight = weights["embed_weight"].to(dev)
    lm_head_weight = weights["lm_head_weight"].to(dev)
    print(f"[validate] embed: {embed_weight.shape}, lm_head: {lm_head_weight.shape}", flush=True)

    # 2. Load MTP head
    mtp = load_mtp_head(args.model, args.checkpoint, embed_weight, lm_head_weight, dev)
    print(f"[validate] MTP head: {mtp.num_parameters() / 1e6:.1f}M params", flush=True)

    # 3. Load shard data
    shard_files = sorted(Path(args.data_dir).glob("shard_*.pt"))
    # Also check worker subdirectories
    if not shard_files:
        worker_dirs = sorted(Path(args.data_dir).glob("worker_*"))
        for wd in worker_dirs:
            shard_files.extend(sorted(wd.glob("shard_*.pt")))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {args.data_dir}")

    print(f"[validate] Found {len(shard_files)} shard files", flush=True)

    all_samples = []
    for shard_path in shard_files:
        try:
            shard = torch.load(shard_path, weights_only=False)
            if isinstance(shard, list):
                all_samples.extend(shard)
            elif isinstance(shard, dict) and "samples" in shard:
                all_samples.extend(shard["samples"])
        except Exception as e:
            print(f"  [validate] Error loading {shard_path}: {e}", file=sys.stderr)

    print(f"[validate] Loaded {len(all_samples)} total samples", flush=True)

    # Subsample
    if len(all_samples) > args.num_samples:
        import random
        random.seed(42)
        all_samples = random.sample(all_samples, args.num_samples)
        print(f"[validate] Subsampled to {len(all_samples)} samples", flush=True)

    # 4. Run validation in batches
    all_results = []
    total_samples_done = 0
    start_time = time.time()

    for i in range(0, len(all_samples), args.batch_size):
        batch_samples = all_samples[i:i + args.batch_size]
        batch_size = len(batch_samples)

        hidden_states = torch.stack([s["hidden_states"].float() for s in batch_samples])
        token_ids = torch.stack([s["token_ids"].long() for s in batch_samples])
        next_token_ids = torch.stack([s["next_token_ids"].long() for s in batch_samples])

        results = validate_accept_rate(
            mtp, embed_weight, lm_head_weight,
            hidden_states, token_ids, next_token_ids,
            dev, chain_length=args.chain_length,
        )
        all_results.append(results)
        total_samples_done += batch_size

        if (i // args.batch_size) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[validate] {total_samples_done}/{len(all_samples)} samples "
                  f"({elapsed:.1f}s) — step0_acc={results['step_accuracy'][0]:.3f}", flush=True)

    # 5. Aggregate results
    n_batches = len(all_results)

    # Find the minimum chain length across all batches (some batches may have shorter chains)
    min_chain = min(len(r["step_accuracy"]) for r in all_results) if all_results else 0
    print(f"[validate] Min chain length across batches: {min_chain}", flush=True)

    agg = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "num_samples": total_samples_done,
        "chain_length": min_chain,
        "step_accuracy": [],
        "topk_accuracy": {1: [], 2: [], 5: []},
        "chain_accept_rate": 0.0,
        "avg_accept_length": 0.0,
        "expected_speedup": 0.0,
        "accept_length_distribution": {},
    }

    for k in range(min_chain):
        accs = [r["step_accuracy"][k] for r in all_results if k < len(r["step_accuracy"])]
        agg["step_accuracy"].append(round(sum(accs) / len(accs), 4))
        for k_val in [1, 2, 5]:
            accs_k = [r["topk_accuracy"][k_val][k] for r in all_results if k < len(r["topk_accuracy"][k_val])]
            agg["topk_accuracy"][k_val].append(round(sum(accs_k) / len(accs_k), 4))

    car = [r["chain_accept_rate"] for r in all_results]
    aal = [r["avg_accept_length"] for r in all_results]
    es = [r["expected_speedup"] for r in all_results]
    agg["chain_accept_rate"] = round(sum(car) / len(car), 4)
    agg["avg_accept_length"] = round(sum(aal) / len(aal), 4)
    agg["expected_speedup"] = round(sum(es) / len(es), 2)

    # Aggregate accept length distribution
    for l in range(min_chain + 1):
        key = f"accept_{l}"
        vals = [r["accept_length_distribution"].get(key, 0) for r in all_results]
        agg["accept_length_distribution"][key] = round(sum(vals) / len(vals), 4)

    elapsed = time.time() - start_time

    # 6. Print results
    print("\n" + "=" * 70)
    print(f"  MTP HEAD ACCEPT RATE VALIDATION — {args.model.upper()}")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples: {total_samples_done} | Chain length: {min_chain}")
    print(f"  Time: {elapsed:.1f}s")
    print("-" * 70)
    print(f"\n  Per-Step Accuracy (argmax):")
    for k in range(min_chain):
        print(f"    Step {k}: {agg['step_accuracy'][k]:.4f} ({agg['step_accuracy'][k]*100:.1f}%)"
              f"  | top-2: {agg['topk_accuracy'][2][k]:.4f}"
              f"  | top-5: {agg['topk_accuracy'][5][k]:.4f}")

    print(f"\n  Chain Accept Rate (>=1 correct): {agg['chain_accept_rate']:.4f} ({agg['chain_accept_rate']*100:.1f}%)")
    print(f"  Average Accept Length: {agg['avg_accept_length']:.2f} tokens")
    print(f"  Expected Speedup: {agg['expected_speedup']}x")

    print(f"\n  Accept Length Distribution:")
    for l in range(min_chain + 1):
        key = f"accept_{l}"
        pct = agg["accept_length_distribution"].get(key, 0) * 100
        bar = "#" * int(pct / 2)
        print(f"    accept_{l}: {pct:5.1f}%  {bar}")

    print("\n" + "=" * 70)

    # 7. Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n  Results saved to: {args.output}")

    return agg


if __name__ == "__main__":
    main()
