"""链式 MTP head fine-tune — 训练时模拟链式 draft.

修改: 不只预测 1 个 token, 而是链式预测 N 个:
  step 1: hidden = target_hidden → MTP → pred1, mtp_out1
  step 2: hidden = mtp_out1 → MTP → pred2, mtp_out2
  step 3: hidden = mtp_out2 → MTP → pred3
  loss = sum(cross_entropy(pred_k, target_token[i+k]))

解决链式退化: MTP 学会用自己的输出作为下一步输入.
"""
from __future__ import annotations

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import sys
import numpy as np


def load_base_model(model_path: str, device: str = "cuda"):
    """加载 base model (统一加载器)."""
    import sys
    sys.path.insert(0, "/root/flashkv0516/app/shared")
    from model_loader import load_base_model as _load
    return _load(model_path, device=device)


def collect_chained_data(
    base_model,
    tokenizer,
    mtp_head,  # 当前 MTP head (用于链式 forward)
    text: str,
    num_chain: int = 4,  # 链式步数
    max_length: int = 512,
    device: str = "cuda",
) -> list[dict]:
    """链式数据收集 — 模拟推理时的链式 draft.

    Returns:
        samples: [{hidden_states, token_ids, next_token_ids, mtp_outputs}, ...]
        hidden_states: [num_chain, hidden] (target hidden + MTP outputs)
        token_ids: [num_chain] (当前 token)
        next_token_ids: [num_chain] (target next token)
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = [t for t in input_ids if t not in (151644, 151645)]
    if len(input_ids) < num_chain + 1:
        return []

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = base_model(input_tensor, output_hidden_states=True, use_cache=False)

    hidden_states_all = outputs.hidden_states[-1][0]  # [seq_len, hidden]
    embed_weights = base_model.model.language_model.embed_tokens.weight  # [vocab, hidden]

    samples = []
    mtp_head.eval()

    for i in range(len(input_ids) - num_chain):
        # step 1: target hidden
        current_hidden = hidden_states_all[i]  # [hidden]
        chain_hidden = []
        chain_tokens = []
        chain_next_tokens = []

        for k in range(num_chain):
            token_id = input_ids[i + k]
            next_token_id = input_ids[i + k + 1]
            token_embed = embed_weights[token_id]  # [hidden]

            # MTP forward (不含 lm_head, 返回 hidden)
            with torch.no_grad():
                # 手动 forward 到 norm_out (不调 lm_head), 输入需 3D
                h_3d = current_hidden.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
                e_3d = token_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
                concat_input = torch.cat([h_3d, e_3d], dim=-1)  # [1, 1, 2*hidden]
                x = mtp_head.proj(concat_input)
                x = mtp_head.norm1(x)
                x = x + mtp_head.attn(x)
                x = mtp_head.norm2(x)
                x = x + mtp_head.mlp(x)
                mtp_out = mtp_head.norm_out(x)  # [1, 1, hidden]

            chain_hidden.append(current_hidden.cpu())
            chain_tokens.append(token_id)
            chain_next_tokens.append(next_token_id)

            # 链式: 下一步用 MTP 输出作为 hidden
            current_hidden = mtp_out[0, 0]  # [hidden]

        samples.append({
            "hidden_states": torch.stack(chain_hidden),  # [num_chain, hidden]
            "token_ids": torch.tensor(chain_tokens),  # [num_chain]
            "next_token_ids": torch.tensor(chain_next_tokens),  # [num_chain]
        })

    return samples


def train_chained_mtp(
    base_model_path: str,
    mtp_checkpoint: str,
    corpus_path: str,
    output_dir: str,
    num_chain: int = 4,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 5e-5,  # fine-tune 用更小 lr
    device: str = "cuda",
    max_samples: int = 10000,
):
    """Fine-tune MTP head 支持链式 draft."""
    print(f"[chain-train] Loading base model: {base_model_path}")
    base_model, tokenizer = load_base_model(base_model_path, device)
    base_model.eval()

    # 获取 embed + lm_head
    embed_weights = base_model.model.language_model.embed_tokens.weight
    lm_head = getattr(base_model, "lm_head", getattr(base_model.model.language_model, "lm_head", None))
    lm_head_weight = lm_head.weight if hasattr(lm_head, "weight") else embed_weights

    print(f"[chain-train] Loading MTP head: {mtp_checkpoint}")
    import sys
    sys.path.insert(0, "/root/flashkv0516/CGC_Phase2/mtp_head")
    from model import MTPHead, MTPHeadConfig
    mtp = MTPHead(MTPHeadConfig())
    mtp.set_shared_lm_head(lm_head_weight)
    ckpt = torch.load(mtp_checkpoint, weights_only=False, map_location="cpu")
    mtp.load_state_dict(ckpt["model_state_dict"], strict=False)
    mtp.to(device).to(torch.bfloat16)  # 对齐 base model dtype
    mtp.train()

    # 收集链式数据
    print(f"[chain-train] Collecting chained data from {corpus_path}")
    with open(corpus_path) as f:
        corpus = [json.loads(line) for line in f]

    all_samples = []
    for i, entry in enumerate(corpus):
        if len(all_samples) >= max_samples:
            break
        # 提取 text (支持多种格式)
        if "text" in entry:
            text = entry["text"]
        elif "messages" in entry:
            text = " ".join(m["content"] for m in entry["messages"])
        elif "instruction" in entry:
            text = entry.get("instruction", "")
            if entry.get("input"): text += "\n" + entry["input"]
            if entry.get("output"): text += "\n" + entry["output"]
        else:
            continue
        if len(text) < 20:
            continue
        samples = collect_chained_data(base_model, tokenizer, mtp, text, num_chain, device=device)
        all_samples.extend(samples)
        if (i + 1) % 100 == 0:
            print(f"  {i+1} entries, {len(all_samples)} samples")

    print(f"[chain-train] Total: {len(all_samples)} samples")

    # 训练
    optimizer = torch.optim.AdamW(mtp.parameters(), lr=lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(all_samples) // batch_size)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # 打乱
        import random
        random.shuffle(all_samples)

        for i in range(0, len(all_samples), batch_size):
            batch = all_samples[i:i + batch_size]
            if len(batch) < 2:
                continue

            # 构造 batch
            # hidden: [batch, num_chain, hidden]
            # tokens: [batch, num_chain]
            # next_tokens: [batch, num_chain]
            hidden = torch.stack([s["hidden_states"] for s in batch]).to(device)
            tokens = torch.stack([s["token_ids"] for s in batch]).to(device)
            next_tokens = torch.stack([s["next_token_ids"] for s in batch]).to(device)

            # 链式 forward (手动, 不用 mtp() 因需分离 hidden 和 logits)
            loss = 0
            current_hidden = hidden[:, 0, :]  # [batch, hidden] (target hidden)

            for k in range(num_chain):
                token_embed = F.embedding(tokens[:, k], embed_weights)  # [batch, hidden]
                # 手动 forward (3D, 含梯度)
                concat_input = torch.cat([current_hidden.unsqueeze(1), token_embed.unsqueeze(1)], dim=-1)  # [batch, 1, 2*hidden]
                x = mtp.proj(concat_input)
                x = mtp.norm1(x)
                x = x + mtp.attn(x)
                x = mtp.norm2(x)
                x = x + mtp.mlp(x)
                mtp_hidden = mtp.norm_out(x)  # [batch, 1, hidden] (不含 lm_head)

                # lm_head → logits
                logits = F.linear(mtp_hidden[:, 0, :], lm_head_weight)  # [batch, vocab]
                loss += F.cross_entropy(logits, next_tokens[:, k])

                # 链式: 下一步用 MTP hidden (detach 避免梯度爆炸)
                current_hidden = mtp_hidden[:, 0, :].detach()  # [batch, hidden]

            loss = loss / num_chain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                print(f"  Epoch {epoch+1} batch {num_batches}: loss={loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        print(f"[chain-train] Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    # 保存
    output_path = Path(output_dir) / "mtp_head_chained.pt"
    torch.save({
        "model_state_dict": mtp.state_dict(),
        "epoch": epochs,
        "loss": avg_loss,
        "num_chain": num_chain,
        "training": "chained",
    }, output_path)
    print(f"[chain-train] Saved: {output_path}")

    # 评估
    mtp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, s in enumerate(all_samples[:100]):
            hidden = s["hidden_states"][0].to(device)  # target hidden
            token_embed = F.embedding(s["token_ids"][0].to(device), embed_weights)
            mtp_out = mtp(hidden.unsqueeze(0).unsqueeze(0), token_embed.unsqueeze(0).unsqueeze(0))
            pred = mtp_out[0, 0].argmax().item()
            if pred == s["next_token_ids"][0].item():
                correct += 1
            total += 1

    if total > 0:
        print(f"[chain-train] Single token accuracy: {correct}/{total} = {correct/total:.1%}")
    else:
        print(f"[chain-train] No evaluation samples")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/data2/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--mtp-checkpoint", default="/data/mtp_head_output/mtp_head_final.pt")
    parser.add_argument("--corpus", default="/data/mtp_corpus.jsonl")
    parser.add_argument("--output", default="/data/mtp_head_chained_output")
    parser.add_argument("--num-chain", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()

    train_chained_mtp(
        args.base_model, args.mtp_checkpoint, args.corpus,
        args.output, args.num_chain, args.epochs,
        max_samples=args.max_samples,
    )
