"""准备 MTP head 训练语料.

从 HuggingFace 下载 Alpaca 数据集 (~52K 样本, ~50MB) 并转换为 JSONL 格式.

用法:
  python prepare_corpus.py --output /data/mtp_corpus.jsonl

输出格式 (JSONL):
  {"text": "Below is an instruction..."}
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def prepare_alpaca(output_path: str, max_samples: int = 52000):
    """下载 Alpaca 数据集并转换为 JSONL."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[prepare] Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    print(f"[prepare] Downloading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"[prepare] Loaded {len(ds)} samples")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for item in ds:
            if count >= max_samples:
                break

            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            output_text = item.get("output", "").strip()

            if not instruction or not output_text:
                continue

            # 构造完整文本 (Alpaca prompt format)
            if input_text:
                full_text = (
                    f"Below is an instruction that describes a task, paired with an input "
                    f"that provides further context. Write a response that appropriately "
                    f"completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n"
                    f"### Response:\n{output_text}"
                )
            else:
                full_text = (
                    f"Below is an instruction that describes a task. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Response:\n{output_text}"
                )

            # 也保存 messages 格式
            messages = [
                {"role": "user", "content": instruction + (f"\n\n{input_text}" if input_text else "")},
                {"role": "assistant", "content": output_text},
            ]

            f.write(json.dumps({"text": full_text, "messages": messages}, ensure_ascii=False) + "\n")
            count += 1

    print(f"[prepare] Saved {count} samples to {output_path}")
    return count


def prepare_openorca(output_path: str, max_samples: int = 100000):
    """下载 OpenOrca 数据集子集."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[prepare] Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    print(f"[prepare] Downloading OpenOrca dataset (streaming)...")
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for item in ds:
            if count >= max_samples:
                break

            question = item.get("question", "").strip()
            response = item.get("response", "").strip()

            if not question or not response:
                continue

            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

            if count % 10000 == 0:
                print(f"[prepare] Processed {count} samples", flush=True)

    print(f"[prepare] Saved {count} samples to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare MTP training corpus")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--source", choices=["alpaca", "openorca"], default="alpaca")
    parser.add_argument("--max-samples", type=int, default=52000)
    args = parser.parse_args()

    if args.source == "alpaca":
        prepare_alpaca(args.output, args.max_samples)
    elif args.source == "openorca":
        prepare_openorca(args.output, args.max_samples)

    print(f"\n[prepare] Done. Corpus ready at {args.output}")
    print(f"[prepare] Next step:")
    print(f"  bash launch_mtp_train.sh {args.output}")


if __name__ == "__main__":
    main()
