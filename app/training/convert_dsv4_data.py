#!/usr/bin/env python3
"""Convert DSV4 flat hidden states to chain training format + extract weights.

DSV4 uses a custom inference framework (not standard transformers), so
collect_dsv4_hidden.py saves flat format:
  {"hidden_states": [N, hidden], "token_ids": [N]}

This script converts to chain format expected by distill_train.py:
  {"hidden_states": [chain, hidden], "token_ids": [chain], "next_token_ids": [chain]}

Also extracts embed_weight and lm_head_weight from the DSV4 model.

Usage:
    cd /data/models/DeepSeek-V4-Flash-UD-IQ2/inference
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29502 \
        /tmp/cgc_test/app/training/convert_dsv4_data.py \
        --input-dir /data/mtp_train_data/dsv4_test \
        --output-dir /data/mtp_train_data/dsv4_chain \
        --ckpt-path /dev/shm/dsv4_converted \
        --config /data/models/DeepSeek-V4-Flash-UD-IQ2/inference/config.json \
        --num-chain 4
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.distributed as dist

# Add DSV4 inference dir to path
DSV4_INFER_DIR = "/data/models/DeepSeek-V4-Flash-UD-IQ2/inference"
sys.path.insert(0, DSV4_INFER_DIR)

from model import Transformer, ModelArgs
from safetensors.torch import load_model


def convert_flat_to_chain(flat_hidden, flat_tokens, num_chain=4):
    """Convert flat hidden states + tokens to chain training samples.

    Args:
        flat_hidden: [N, hidden] tensor
        flat_tokens: [N] tensor
        num_chain: chain length

    Returns:
        list of {"hidden_states": [chain, hidden], "token_ids": [chain], "next_token_ids": [chain]}
    """
    samples = []
    n = len(flat_hidden)

    for i in range(n - num_chain):
        end = i + num_chain
        chain_hidden = flat_hidden[i:end]           # [chain, hidden]
        chain_tokens = flat_tokens[i:end]            # [chain]
        chain_next = flat_tokens[i + 1:end + 1]     # [chain] shifted

        samples.append({
            "hidden_states": chain_hidden,
            "token_ids": chain_tokens,
            "next_token_ids": chain_next,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Convert DSV4 data + extract weights")
    parser.add_argument("--input-dir", required=True, help="Dir with flat hidden_shard_*.pt")
    parser.add_argument("--output-dir", required=True, help="Output dir for chain format")
    parser.add_argument("--ckpt-path", required=True, help="DSV4 converted checkpoint dir")
    parser.add_argument("--config", required=True, help="DSV4 inference config.json")
    parser.add_argument("--num-chain", type=int, default=4, help="Chain length")
    args = parser.parse_args()

    # Setup distributed (for model loading)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")

    global print
    if rank != 0:
        print = lambda *_, **__: None

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    # 1. Load DSV4 model to extract embed/lm_head weights
    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))
    model_args.max_batch_size = 1

    print(f"Loading DSV4 model for weight extraction...")
    with torch.device("cuda"):
        model = Transformer(model_args)

    ckpt_file = os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    load_model(model, ckpt_file, strict=False)
    print("Model loaded!")

    # 2. Extract embed_weight and lm_head_weight
    # DSV4 embed: model.embed (nn.Embedding)
    embed_weight = model.embed.weight.detach()  # [vocab/TP, hidden] or [vocab, hidden]

    # DSV4 lm_head: model.head - check for weight
    lm_head_weight = None
    if hasattr(model.head, 'weight'):
        lm_head_weight = model.head.weight.detach()
    elif hasattr(model.head, 'lm_head'):
        lm_head_weight = model.head.lm_head.weight.detach()
    else:
        # Try to get it from get_logits
        print(f"head type: {type(model.head)}, attrs: {[a for a in dir(model.head) if not a.startswith('_')]}")
        # Check for common weight names
        for attr in ['weight', 'lm_head', 'out_proj', 'decoder']:
            if hasattr(model.head, attr):
                w = getattr(model.head, attr)
                if hasattr(w, 'weight'):
                    lm_head_weight = w.weight.detach()
                    break
                elif isinstance(w, torch.Tensor):
                    lm_head_weight = w.detach()
                    break

    if lm_head_weight is None:
        # Use tied embeddings
        print("Using tied embeddings (embed = lm_head)")
        lm_head_weight = embed_weight

    # Gather weights from all GPUs if tensor parallel
    if world_size > 1:
        # For embed: vocab is split across GPUs, need to gather along dim 0
        gathered_embed = [torch.zeros_like(embed_weight) for _ in range(world_size)]
        dist.all_gather(gathered_embed, embed_weight)
        embed_weight = torch.cat(gathered_embed, dim=0)

        gathered_lm_head = [torch.zeros_like(lm_head_weight) for _ in range(world_size)]
        dist.all_gather(gathered_lm_head, lm_head_weight)
        lm_head_weight = torch.cat(gathered_lm_head, dim=0)

    print(f"embed_weight: {embed_weight.shape}, lm_head_weight: {lm_head_weight.shape}")

    # 3. Save embed_head.pt
    os.makedirs(args.output_dir, exist_ok=True)
    embed_head_path = os.path.join(args.output_dir, "embed_head.pt")
    torch.save({
        "embed_weight": embed_weight.cpu().to(torch.float32),
        "lm_head_weight": lm_head_weight.cpu().to(torch.float32),
        "hidden_size": model_args.dim,
        "vocab_size": model_args.vocab_size,
        "model_name": "dsv4",
    }, embed_head_path)
    print(f"Saved embed_head.pt ({os.path.getsize(embed_head_path) / 1e9:.1f}GB)")

    # 4. Convert flat hidden states to chain format
    shard_files = sorted(Path(args.input_dir).glob("hidden_shard_*.pt"))
    print(f"Found {len(shard_files)} flat shards")

    chain_shard_idx = 0
    total_samples = 0

    for sf in shard_files:
        data = torch.load(sf, weights_only=False)
        flat_hidden = data["hidden_states"]  # [N, hidden]
        flat_tokens = data["token_ids"]      # [N]

        # Create chain samples
        chain_samples = convert_flat_to_chain(flat_hidden, flat_tokens, args.num_chain)

        if chain_samples:
            chain_path = os.path.join(args.output_dir, f"shard_{chain_shard_idx:06d}.pt")
            torch.save(chain_samples, chain_path)
            total_samples += len(chain_samples)
            print(f"  Chain shard {chain_shard_idx}: {len(chain_samples)} samples from {sf.name}")
            chain_shard_idx += 1

    # 5. Save metadata
    meta = {
        "model_name": "dsv4",
        "model_path": args.ckpt_path,
        "hidden_size": model_args.dim,
        "vocab_size": model_args.vocab_size,
        "num_chain": args.num_chain,
        "total_samples": total_samples,
        "num_shards": chain_shard_idx,
        "source": "convert_dsv4_data.py",
        "hidden_type": "decode",
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {total_samples} chain samples in {chain_shard_idx} shards")
    print(f"Output: {args.output_dir}/")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
