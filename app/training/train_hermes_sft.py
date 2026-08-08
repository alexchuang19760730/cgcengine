#!/usr/bin/env python3
"""Hermes Router SFT 训练 — 用 LoRA 微调小模型学习路由决策.

任务: 给定 4D 矩阵 + 平台 benchmark + pipeline 摘要 → 输出 JSON 路由决策
基座: Qwen3-VL-2B-bf16 (本地已有) 或 Qwen2.5-0.5B (需下载)
方法: LoRA r=16, 3 epochs, cosine LR

Usage:
    # 训练
    python3 -m app.training.train_hermes_sft --epochs 3 --batch-size 4

    # 评估
    python3 -m app.training.train_hermes_sft --eval-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "hermes_sft")

# === 模型配置 ===
MODEL_CONFIGS = {
    "qwen3vl-2b": {
        "hf_model_path": "/Users/alexchuang/models/Qwen3-VL-2B-bf16",
        "max_length": 2048,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "qwen25-0.5b": {
        "hf_model_path": "/Users/alexchuang/Documents/flashkv0516/models/qwen25_0.5b_hf",
        "max_length": 1024,
        "lora_r": 16,
        "lora_alpha": 32,
    },
}

# === 路由模式标签 ===
ROUTING_MODES = ["cache_hit", "local_only", "edge_draft", "cloud_mtp", "cloud_only"]
MODE_TO_ID = {m: i for i, m in enumerate(ROUTING_MODES)}


class HermesSFTDataset(Dataset):
    """Hermes SFT 数据集 — 加载 JSONL, apply chat template."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path) as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data)

        print(f"[dataset] Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]["messages"]

        # 用 tokenizer 的 chat template 构建输入
        # 只对 system + user 部分计算 loss (assistant 部分才是目标)
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:2],  # system + user
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages,  # system + user + assistant
            tokenize=False,
            add_generation_prompt=False,
        )

        # tokenize
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # 截断
        full_ids = full_ids[: self.max_length]

        # labels: prompt 部分 mask 为 -100, 只对 assistant 输出计算 loss
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        labels = labels[: len(full_ids)]

        # padding
        attention_mask = [1] * len(full_ids)
        pad_len = self.max_length - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_model_and_tokenizer(model_key: str = "qwen3vl-2b"):
    """加载基座模型 + tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    config = MODEL_CONFIGS[model_key]
    model_path = config["hf_model_path"]

    print(f"[model] Loading {model_key} from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    use_mps = os.environ.get("HERMES_USE_MPS", "0") == "1" and torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cpu")
    if use_mps:
        model.to(device)
    print(f"[model] Loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer, config


def setup_lora(model, lora_r: int = 16, lora_alpha: int = 32):
    """配置 LoRA."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model
    except ImportError:
        print("[lora] peft not available, training full model")
        return model


def train(
    model_key: str = "qwen3vl-2b",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    train_path: str = "",
    eval_path: str = "",
):
    """SFT 训练."""
    if not train_path:
        train_path = os.path.join(DATA_DIR, "hermes_sft_train_v4.jsonl")
    if not eval_path:
        eval_path = os.path.join(DATA_DIR, "hermes_sft_eval_v4.jsonl")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    model, tokenizer, config = load_model_and_tokenizer(model_key)
    model = setup_lora(model, config["lora_r"], config["lora_alpha"])

    # 2. 加载数据
    train_dataset = HermesSFTDataset(train_path, tokenizer, config["max_length"])
    eval_dataset = HermesSFTDataset(eval_path, tokenizer, config["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # 3. 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    # 4. cosine LR schedule
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 5. 训练循环
    device = next(model.parameters()).device
    best_eval_loss = float("inf")
    best_eval_acc = 0.0

    print(f"\n[train] Starting training: {epochs} epochs, {len(train_loader)} steps/epoch, device={device}")
    print(f"[train] Total samples: {len(train_dataset)} train, {len(eval_dataset)} eval")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step_count += 1

            if (step + 1) % 10 == 0:
                avg_loss = epoch_loss / step_count
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | "
                      f"loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"{elapsed:.0f}s")

        avg_train_loss = epoch_loss / max(step_count, 1)

        # 6. 评估
        eval_loss, eval_acc, mode_acc = evaluate(model, eval_loader, device, tokenizer)

        print(f"\n[epoch {epoch+1}] train_loss={avg_train_loss:.4f} | "
              f"eval_loss={eval_loss:.4f} | mode_acc={eval_acc:.2%}")
        print(f"  Per-mode: {mode_acc}")

        # 7. 保存最佳
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_eval_loss = eval_loss
            save_path = os.path.join(OUTPUT_DIR, "best")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  [save] Best model saved to {save_path} (acc={eval_acc:.2%})")

    print(f"\n[train] Done. Best eval: loss={best_eval_loss:.4f}, acc={best_eval_acc:.2%}")
    return best_eval_acc


def evaluate(model, eval_loader, device, tokenizer):
    """评估: loss + mode 分类准确率."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_mode = 0
    mode_stats = {m: {"correct": 0, "total": 0} for m in ROUTING_MODES}

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch["input_ids"].shape[0]
            total_samples += batch["input_ids"].shape[0]

            # 解析预测的 mode
            for i in range(batch["input_ids"].shape[0]):
                # 找到 assistant 输出部分 (labels != -100 的部分)
                labels = batch["labels"][i]
                valid = (labels != -100).nonzero(as_tuple=True)[0]
                if len(valid) == 0:
                    continue

                # 生成预测
                input_ids = batch["input_ids"][i:i+1, :valid[0]+1]
                attn = batch["attention_mask"][i:i+1, :valid[0]+1]
                generated = model.generate(
                    input_ids=input_ids.to(device),
                    attention_mask=attn.to(device),
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
                pred_text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)

                # 解析 ground truth
                gt_text = tokenizer.decode(labels[valid[0]:], skip_special_tokens=True)

                try:
                    pred_decision = json.loads(pred_text)
                    gt_decision = json.loads(gt_text)
                    pred_mode = pred_decision.get("mode", "")
                    gt_mode = gt_decision.get("mode", "")

                    if pred_mode == gt_mode:
                        correct_mode += 1
                    if gt_mode in mode_stats:
                        mode_stats[gt_mode]["total"] += 1
                        if pred_mode == gt_mode:
                            mode_stats[gt_mode]["correct"] += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    avg_loss = total_loss / max(total_samples, 1)
    mode_acc = correct_mode / max(total_samples, 1)
    per_mode = {m: f"{s['correct']}/{s['total']}" for m, s in mode_stats.items() if s["total"] > 0}

    return avg_loss, mode_acc, per_mode


def evaluate_rule_based(eval_path: str = ""):
    """规则基线: 用 HermesRouter 原始逻辑评估 (作为 SFT 的对照)."""
    if not eval_path:
        eval_path = os.path.join(DATA_DIR, "hermes_sft_eval_v4.jsonl")

    from app.shared.hermes_router import (
        SystemProfile, PlatformBenchmark, ProfileBinding,
        FourDMatrix, TenStepPipeline, HermesRouter, Bootstrap, StateABI,
    )

    correct = 0
    total = 0
    mode_stats = {m: {"correct": 0, "total": 0} for m in ROUTING_MODES}

    with open(eval_path) as f:
        for line in f:
            data = json.loads(line)
            user_data = json.loads(data["messages"][1]["content"])
            gt_decision = json.loads(data["messages"][2]["content"])
            gt_mode = gt_decision["mode"]

            # 用 pipeline 的 step_7.5_route 作为规则预测
            route_step = user_data.get("ten_step_pipeline_summary", {}).get("step_7.5_route", {})
            pred_mode = route_step.get("mode", "cloud_only")

            # cache_hit 特殊处理
            if user_data.get("request_context", {}).get("cache_hit", False):
                pred_mode = "cache_hit"
            if not user_data.get("request_context", {}).get("online", True):
                pred_mode = "local_only"

            if pred_mode == gt_mode:
                correct += 1
            total += 1
            if gt_mode in mode_stats:
                mode_stats[gt_mode]["total"] += 1
                if pred_mode == gt_mode:
                    mode_stats[gt_mode]["correct"] += 1

    acc = correct / max(total, 1)
    per_mode = {m: f"{s['correct']}/{s['total']}" for m, s in mode_stats.items() if s["total"] > 0}
    print(f"\n[rule-based] Mode accuracy: {acc:.2%} ({correct}/{total})")
    print(f"  Per-mode: {per_mode}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hermes Router SFT Training")
    parser.add_argument("--model", type=str, default="qwen3vl-2b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-path", type=str, default="")
    parser.add_argument("--eval-path", type=str, default="")
    parser.add_argument("--eval-only", action="store_true", help="Only run rule-based eval")
    parser.add_argument("--rule-baseline", action="store_true", help="Run rule-based baseline")
    args = parser.parse_args()

    if args.rule_baseline or args.eval_only:
        evaluate_rule_based(args.eval_path)
    else:
        train(
            model_key=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_path=args.train_path,
            eval_path=args.eval_path,
        )
        # 训练后也跑规则基线对照
        print("\n" + "=" * 60)
        print("Rule-based baseline for comparison:")
        evaluate_rule_based(args.eval_path)
