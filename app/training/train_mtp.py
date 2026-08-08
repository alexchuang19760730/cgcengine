#!/usr/bin/env python3
"""Unified MTP Draft training pipeline — one command for any model.

Orchestrates: model_registry lookup -> collect data -> train -> evaluate
Works for ALL registered models (Gemma4, DSV4, Qwen3-VL) with zero hardcoded params.

Usage:
  # Full pipeline (collect + train):
  python3 train_mtp.py --model gemma4

  # Collect only:
  python3 train_mtp.py --model gemma4 --phase collect --num-samples 50000

  # Train only (data already collected):
  python3 train_mtp.py --model gemma4 --phase train --mode kl

  # Multi-GPU collect + train:
  python3 train_mtp.py --model gemma4 --world-size 4 --gpu-base 4

  # Custom paths:
  python3 train_mtp.py --model dsv4 --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2

  # Quick test (small dataset):
  python3 train_mtp.py --model qwen3vl --num-samples 100 --epochs 1

Pipeline:
  1. Look up model_registry for config (hidden_size, vocab_size, model_path, etc.)
  2. Collect decode hidden states (collect_hidden_states.py)
  3. Train MTP head with KL distillation + chain CE (distill_train.py)
  4. Save checkpoint to DraftRegistry-compatible path
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Path setup
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "app", "shared")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def run_phase(script_name: str, args: list[str]) -> int:
    """Run a phase script as subprocess."""
    script_path = os.path.join(REPO_ROOT, "app", "training", script_name)
    cmd = [sys.executable, script_path] + args
    print(f"\n{'='*60}", flush=True)
    print(f"  Running: {' '.join(cmd)}", flush=True)
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def _resolve_train_device(device_hint: str) -> str:
    lowered = str(device_hint or "").strip().lower()
    if "cuda" in lowered:
        return "cuda"
    if "mps" in lowered or "metal" in lowered:
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Unified MTP Draft training pipeline — one command for any model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 train_mtp.py --model gemma4                           # Full pipeline
  python3 train_mtp.py --model gemma4 --phase collect           # Collect only
  python3 train_mtp.py --model gemma4 --phase train --mode kl   # KL training
  python3 train_mtp.py --model dsv4 --world-size 4 --gpu-base 4 # Multi-GPU
        """,
    )

    # === Required ===
    parser.add_argument("--model", required=True,
                        help="Model name: gemma4 | dsv4 | qwen3vl (or alias: g4, ds, q3)")

    # === Phase selection ===
    parser.add_argument("--phase", default="all", choices=["collect", "train", "all"],
                        help="Pipeline phase: collect=data collection, train=training, all=both")

    # === Model path ===
    parser.add_argument("--model-path", default="",
                        help="Model path (default: from model_registry)")

    # === Collect options ===
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of prompts to collect")
    parser.add_argument("--num-chain", type=int, default=4, help="Chain length for training samples")
    parser.add_argument("--gen-length", type=int, default=50, help="Max tokens to generate per prompt")
    parser.add_argument("--shard-size", type=int, default=500, help="Samples per shard file")

    # === GPU options ===
    parser.add_argument("--device", default="cuda:0", help="Single-GPU device")
    parser.add_argument("--world-size", type=int, default=0, help="Multi-GPU worker count (0=single)")
    parser.add_argument("--gpu-base", type=int, default=0, help="Base GPU index for multi-GPU")

    # === Train options ===
    parser.add_argument("--mode", default="chain", choices=["chain", "kl", "hybrid"],
                        help="Training mode: chain=CE(proven), kl=KL+CE, hybrid=both")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.7, help="KL loss weight")
    parser.add_argument("--beta", type=float, default=0.3, help="CE loss weight")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--checkpoint", default="", help="Existing checkpoint to resume from")

    # === Output ===
    parser.add_argument("--data-dir", default="", help="Data directory (default: auto from registry)")
    parser.add_argument("--output-dir", default="", help="Output directory (default: auto from registry)")

    args = parser.parse_args()

    # === Resolve model config ===
    from app.shared.model_registry import get_model_config
    cfg = get_model_config(args.model)
    model_path = args.model_path or cfg.base_model_path
    data_dir = args.data_dir or cfg.get_shard_dir()
    output_dir = args.output_dir or cfg.get_mtp_output_dir()

    # === Print plan ===
    print(f"\n{'='*60}", flush=True)
    print(f"  CGC Unified MTP Draft Training Pipeline", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model:       {cfg.display_name} ({cfg.name})", flush=True)
    print(f"  Model path:  {model_path}", flush=True)
    print(f"  Architecture: hidden={cfg.hidden_size}, vocab={cfg.vocab_size}, "
          f"heads={cfg.num_heads}x{cfg.head_dim}", flush=True)
    if cfg.is_moe:
        print(f"  MoE:         {cfg.n_routed_experts} experts, {cfg.num_experts_per_tok}/tok", flush=True)
    print(f"  Phase:       {args.phase}", flush=True)
    print(f"  Data dir:    {data_dir}", flush=True)
    print(f"  Output dir:  {output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()

    # === Phase: Collect ===
    if args.phase in ("collect", "all"):
        collect_args = [
            "--model", cfg.name,
            "--model-path", model_path,
            "--output-dir", data_dir,
            "--num-samples", str(args.num_samples),
            "--num-chain", str(args.num_chain),
            "--gen-length", str(args.gen_length),
            "--shard-size", str(args.shard_size),
        ]
        if args.world_size > 1:
            collect_args.extend(["--world-size", str(args.world_size),
                                 "--gpu-base", str(args.gpu_base)])
        else:
            collect_args.extend(["--device", args.device])

        rc = run_phase("collect_hidden_states.py", collect_args)
        if rc != 0:
            print(f"ERROR: Data collection failed (exit={rc})", flush=True)
            sys.exit(rc)

        # Verify data was collected
        embed_head_path = os.path.join(data_dir, "embed_head.pt")
        if not os.path.exists(embed_head_path):
            embed_head_path = os.path.join(data_dir, "worker_0", "embed_head.pt")
        if not os.path.exists(embed_head_path):
            print(f"ERROR: embed_head.pt not found after collection", flush=True)
            sys.exit(1)

        shard_count = len(list(Path(data_dir).glob("shard_*.pt")))
        worker_shards = len(list(Path(data_dir).glob("worker_*/shard_*.pt")))
        total_shards = shard_count + worker_shards
        print(f"\n[pipeline] Collected {total_shards} shards", flush=True)

    # === Phase: Train ===
    if args.phase in ("train", "all"):
        train_args = [
            "--model", cfg.name,
            "--data-dir", data_dir,
            "--output-dir", output_dir,
            "--mode", args.mode,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--alpha", str(args.alpha),
            "--beta", str(args.beta),
            "--temperature", str(args.temperature),
        ]
        if args.checkpoint:
            train_args.extend(["--checkpoint", args.checkpoint])
        if args.world_size == 0:
            train_args.extend(["--device", _resolve_train_device(args.device)])

        rc = run_phase("distill_train.py", train_args)
        if rc != 0:
            print(f"ERROR: Training failed (exit={rc})", flush=True)
            sys.exit(rc)

    # === Summary ===
    elapsed = time.time() - t0
    checkpoint_path = os.path.join(output_dir, f"mtp_head_{cfg.name}_decode.pt")
    log_path = os.path.join(output_dir, "training_log.json")

    print(f"\n{'='*60}", flush=True)
    print(f"  Pipeline Complete!", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model:       {cfg.display_name}", flush=True)
    print(f"  Time:        {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Checkpoint:  {checkpoint_path}", flush=True)
    print(f"  Log:         {log_path}", flush=True)

    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        if "final_loss" in log:
            print(f"  Final loss:  {log['final_loss']:.4f}", flush=True)
        if "train_chain_acc" in log:
            print(f"  Chain acc:   {log['train_chain_acc']:.1%}", flush=True)
        print(f"  Total steps: {log.get('total_steps', 0)}", flush=True)

    print(f"\n  Next steps:", flush=True)
    print(f"  1. Deploy to sglang: --speculative-algorithm NEXTN --speculative-draft-model-path {output_dir}", flush=True)
    print(f"  2. Register in DraftRegistry: registry.register('{cfg.name}', '{output_dir}', tier=1, trained=True)", flush=True)
    print(f"  3. Test accept rate: send prompts to sglang with speculative decoding enabled", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
