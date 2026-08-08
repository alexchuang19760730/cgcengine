"""评估 MTP Head.

用法:
  python eval.py \
    --base-model /data2/models/Qwen3-VL-2B-Instruct \
    --mtp-checkpoint /data/mtp_head_output/mtp_head_final.pt \
    --test-prompts /data/test_prompts.jsonl

评估指标:
  1. 单 token 预测准确率 (首包用)
  2. 投机 accept rate (decode 用, N=4/10)
  3. forward 时间 (Metal GPU 估算)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MTPHead, MTPHeadConfig, create_mtp_head_for_qwen3vl_2b


def load_mtp_head(checkpoint_path: str, base_model, device: str = "cuda") -> MTPHead:
    """加载训练好的 MTP head."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    mtp_head = create_mtp_head_for_qwen3vl_2b()

    # 获取 base model 的 lm_head 权重
    lm_head_weight = None
    for attr_path in [("language_model", "lm_head"), ("lm_head",)]:
        try:
            obj = base_model
            for attr in attr_path:
                obj = getattr(obj, attr)
            lm_head_weight = obj.weight
            break
        except AttributeError:
            continue

    if lm_head_weight is None:
        raise RuntimeError("Cannot find lm_head in base model")

    mtp_head.set_shared_lm_head(lm_head_weight)
    mtp_head.load_state_dict(ckpt["model_state_dict"], strict=False)
    mtp_head = mtp_head.to(device).to(torch.bfloat16)
    mtp_head.eval()

    print(f"[eval] Loaded MTP head from {checkpoint_path}")
    print(f"[eval] Step: {ckpt.get('step')}, Loss: {ckpt.get('loss')}")
    return mtp_head


def eval_single_token_accuracy(
    mtp_head: MTPHead,
    base_model,
    tokenizer,
    test_texts: list[str],
    device: str = "cuda",
    max_length: int = 512,
) -> dict:
    """评估单 token 预测准确率 (首包用).

    对每个 text:
      1. base model forward → hidden_states (最后一层)
      2. MTP head(hidden_states[:-1], embed(input_ids[:-1])) → predict next token
      3. 对比 predict vs actual next token
    """
    mtp_head.eval()

    # 统一获取 embed_tokens + text_model
    from app.shared.model_loader import get_embed_weight, get_text_model
    embed_weight = get_embed_weight(base_model)
    text_model = get_text_model(base_model)

    correct = 0
    total = 0
    top5_correct = 0

    with torch.no_grad():
        for text in test_texts:
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(input_ids) < 2:
                continue
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            input_tensor = torch.tensor([input_ids], device=device)

            # Base model forward
            outputs = text_model(input_tensor, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[-1]  # [1, seq, hidden]

            # MTP head 预测 (除最后一个 token)
            # 输入: hidden_states[:-1], embed(input_ids[:-1])
            # 目标: input_ids[1:]
            hs_input = hidden_states[:, :-1, :]  # [1, seq-1, hidden]
            token_embeds = F.embedding(
                torch.tensor(input_ids[:-1], device=device),
                embed_weight,
            ).unsqueeze(0)  # [1, seq-1, hidden]

            logits = mtp_head(hs_input, token_embeds)  # [1, seq-1, vocab]
            preds = logits.argmax(dim=-1)[0]  # [seq-1]

            # 对比
            targets = torch.tensor(input_ids[1:], device=device)
            correct += (preds == targets).sum().item()
            total += len(targets)

            # Top-5 accuracy
            top5_preds = logits.topk(5, dim=-1).indices[0]  # [seq-1, 5]
            top5_correct += sum(
                targets[i].item() in top5_preds[i].tolist()
                for i in range(len(targets))
            )

    accuracy = correct / total if total > 0 else 0
    top5_accuracy = top5_correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "total_tokens": total,
        "correct": correct,
    }


def eval_spec_decode_accept_rate(
    mtp_head: MTPHead,
    base_model,
    tokenizer,
    test_prompts: list[str],
    device: str = "cuda",
    num_draft_tokens: int = 4,
    max_new_tokens: int = 50,
) -> dict:
    """评估投机 decode accept rate.

    模拟投机 decode:
      1. base model 生成 next token (target)
      2. MTP head 生成 draft tokens
      3. base model 验证 draft tokens
      4. 统计 accept rate
    """
    mtp_head.eval()

    # 获取 embed_weight
    from app.shared.model_loader import get_embed_weight, get_text_model
    embed_weight = get_embed_weight(base_model)
    text_model = get_text_model(base_model)

    total_draft = 0
    total_accept = 0

    with torch.no_grad():
        for prompt in test_prompts:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) < 2:
                continue

            # 生成 max_new_tokens 个 token (投机)
            current_ids = input_ids[:]
            generated = 0

            while generated < max_new_tokens:
                input_tensor = torch.tensor([current_ids], device=device)

                # Base model forward (获取 hidden + 实际 next tokens)
                outputs = text_model(
                    input_tensor,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]  # [1, seq, hidden]
                base_logits = outputs.logits[:, -1, :]  # [1, vocab]
                base_next = base_logits.argmax(dim=-1)  # [1]

                # MTP head 生成 draft (从最后一个 hidden state)
                last_hidden = hidden_states[:, -1:, :]  # [1, 1, hidden]
                last_token_embed = F.embedding(
                    torch.tensor([current_ids[-1]], device=device),
                    embed_weight,
                ).unsqueeze(0)  # [1, 1, hidden]

                draft_logits = mtp_head(last_hidden, last_token_embed)  # [1, 1, vocab]
                draft_token = draft_logits.argmax(dim=-1)[0, 0].item()

                # 验证 draft (简化: 只验证 1 个 token)
                total_draft += 1
                if draft_token == base_next[0].item():
                    total_accept += 1
                    current_ids.append(draft_token)
                    generated += 1
                else:
                    # Reject, use base model token
                    current_ids.append(base_next[0].item())
                    generated += 1

    accept_rate = total_accept / total_draft if total_draft > 0 else 0

    return {
        "accept_rate": accept_rate,
        "total_draft": total_draft,
        "total_accept": total_accept,
        "num_draft_tokens": num_draft_tokens,
    }


def eval_forward_time(mtp_head: MTPHead, device: str = "cuda", warmup: int = 10, runs: int = 100):
    """评估 MTP head forward 时间."""
    mtp_head.eval()

    # 模拟输入 (batch=1, seq=1, decode step)
    hidden_states = torch.randn(1, 1, 2048, device=device, dtype=torch.bfloat16)
    token_embeddings = torch.randn(1, 1, 2048, device=device, dtype=torch.bfloat16)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = mtp_head(hidden_states, token_embeddings)

    # Measure
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = mtp_head(hidden_states, token_embeddings)
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - t0

    return {
        "forward_time_ms": elapsed / runs * 1000,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MTP Head")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--mtp-checkpoint", required=True)
    parser.add_argument("--test-prompts", help="JSONL file with test prompts")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 加载 base model (统一加载器)
    import sys as _sys
    _sys.path.insert(0, "/root/flashkv0516")
    from app.shared.model_loader import load_base_model, get_text_model, get_embed_weight

    base_model, tokenizer = load_base_model(args.base_model, device=args.device)
    text_model = get_text_model(base_model)
    base_model.eval()

    # 加载 MTP head
    mtp_head = load_mtp_head(args.mtp_checkpoint, base_model, args.device)

    # 测试数据
    if args.test_prompts and os.path.exists(args.test_prompts):
        with open(args.test_prompts) as f:
            test_data = [json.loads(line) for line in f]
        test_texts = [d.get("text") or d.get("content") or d.get("prompt", "") for d in test_data]
        test_prompts = test_texts[:50]  # 限制 50 个
    else:
        # 默认测试集
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In machine learning, a neural network is",
            "def fibonacci(n):",
            "The capital of France is",
            "Once upon a time, there was a",
        ] * 10
        test_prompts = test_texts

    # 1. 单 token 准确率
    print("\n[eval] Evaluating single token accuracy...")
    acc_result = eval_single_token_accuracy(mtp_head, base_model, tokenizer, test_texts, args.device)
    print(f"  Accuracy: {acc_result['accuracy']:.2%} ({acc_result['correct']}/{acc_result['total_tokens']})")
    print(f"  Top-5 Accuracy: {acc_result['top5_accuracy']:.2%}")

    # 2. 投机 accept rate
    print("\n[eval] Evaluating speculative decode accept rate...")
    spec_result = eval_spec_decode_accept_rate(
        mtp_head, base_model, tokenizer, test_prompts, args.device, num_draft_tokens=4
    )
    print(f"  Accept rate: {spec_result['accept_rate']:.2%} ({spec_result['total_accept']}/{spec_result['total_draft']})")

    # 3. Forward 时间
    print("\n[eval] Evaluating forward time...")
    time_result = eval_forward_time(mtp_head, args.device)
    print(f"  Forward time: {time_result['forward_time_ms']:.2f}ms")

    # 汇总
    print("\n" + "=" * 60)
    print("MTP Head Evaluation Summary")
    print("=" * 60)
    print(f"Single token accuracy:   {acc_result['accuracy']:.2%}")
    print(f"Top-5 accuracy:          {acc_result['top5_accuracy']:.2%}")
    print(f"Spec decode accept rate: {spec_result['accept_rate']:.2%}")
    print(f"Forward time:            {time_result['forward_time_ms']:.2f}ms")
    print("=" * 60)

    # 保存结果
    result = {
        "single_token_accuracy": acc_result,
        "spec_decode_accept_rate": spec_result,
        "forward_time": time_result,
    }
    result_path = os.path.join(os.path.dirname(args.mtp_checkpoint), "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[eval] Results saved to {result_path}")


if __name__ == "__main__":
    main()
