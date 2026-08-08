"""数据收集: 用 base model (Qwen3-VL-2B) 生成 MTP head 训练数据.

收集 (hidden_state, current_token, next_token) 三元组:
  - hidden_state: base model 最后一层输出 [seq, hidden]
  - current_token: 当前 token id
  - next_token: 下一个 token id (label)

输出: safetensors 格式, 分片存储
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


def collect_from_text(
    base_model,
    tokenizer,
    text: str,
    device: str = "cuda",
    max_length: int = 512,
    text_model=None,
) -> list[dict]:
    """从单条文本收集训练样本.

    Returns:
        samples: [{hidden_state: Tensor, token_id: int, next_token_id: int}, ...]
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) < 2:
        return []

    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    input_tensor = torch.tensor([input_ids], device=device)

    # 用 text_model (VL 的 language_model) 或 base_model forward
    model = text_model or base_model
    with torch.no_grad():
        outputs = model(
            input_tensor,
            output_hidden_states=True,
            use_cache=False,
        )

    # 最后一层 hidden states [1, seq, hidden]
    hidden_states = outputs.hidden_states[-1][0]  # [seq, hidden]

    samples = []
    for i in range(len(input_ids) - 1):
        samples.append({
            "hidden_state": hidden_states[i].cpu(),  # [hidden]
            "token_id": input_ids[i],
            "next_token_id": input_ids[i + 1],
        })

    return samples


def collect_from_corpus(
    base_model,
    tokenizer,
    corpus_path: str,
    output_dir: str,
    device: str = "cuda",
    max_samples: int = 500000,
    max_length: int = 512,
    shard_size: int = 10000,
):
    """从语料文件收集训练数据.

    Args:
        corpus_path: 语料文件路径 (JSONL 格式, 每行 {"text": "..."} 或 {"messages": [...]})
        output_dir: 输出目录
        max_samples: 最大样本数
        max_length: 单条文本最大 token 长度
        shard_size: 每个分片的样本数
    """
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []
    shard_idx = 0
    total = 0

    print(f"[collect] Loading corpus from {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if total >= max_samples:
                break

            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # 提取文本
            if "messages" in data:
                # chat 格式
                text = tokenizer.apply_chat_template(
                    data["messages"], tokenize=False, add_generation_prompt=False
                )
            elif "text" in data:
                text = data["text"]
            elif "content" in data:
                text = data["content"]
            else:
                continue

            if len(text.strip()) < 10:
                continue

            # 收集样本
            samples = collect_from_text(base_model, tokenizer, text, device, max_length, text_model=locals().get("text_model"))
            all_samples.extend(samples)
            total += len(samples)

            # 分片存储
            if len(all_samples) >= shard_size:
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
                torch.save(all_samples[:shard_size], shard_path)
                print(f"[collect] Saved shard {shard_idx}: {shard_size} samples → {shard_path}")
                all_samples = all_samples[shard_size:]
                shard_idx += 1

            if line_num % 100 == 0:
                print(f"[collect] Processed {line_num} lines, {total} samples", flush=True)

    # 保存剩余
    if all_samples:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
        torch.save(all_samples, shard_path)
        print(f"[collect] Saved final shard {shard_idx}: {len(all_samples)} samples → {shard_path}")

    print(f"[collect] Done. Total {total} samples in {shard_idx + 1} shards → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Collect MTP head training data")
    parser.add_argument("--model-path", required=True, help="Base model path (Qwen3-VL-2B)")
    parser.add_argument("--corpus-path", required=True, help="Corpus JSONL file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-samples", type=int, default=500000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 加载 base model (统一加载器,支持所有模型类型)
    print(f"[collect] Loading base model from {args.model_path}")
    import sys as _sys
    _sys.path.insert(0, "/root/flashkv0516")
    from app.shared.model_loader import load_base_model, get_text_model

    base_model, tokenizer = load_base_model(args.model_path, device=args.device)
    text_model = get_text_model(base_model)
    base_model.eval()

    # 收集数据
    collect_from_corpus(
        base_model=base_model,
        tokenizer=tokenizer,
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        device=args.device,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
