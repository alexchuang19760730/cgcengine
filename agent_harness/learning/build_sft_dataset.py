#!/usr/bin/env python3
"""
把 freebuff2api 生成的成功轨迹整理成 SFT/LoRA 数据集（MLX-LM 格式）。

输入：tb run 的输出目录（output_path/<run_id>/），每个任务目录下有
    results.json              （is_resolved 等）
    agent-logs/trajectory.jsonl （CodebuffApiAgent 记录的逐步行）
只保留 is_resolved=true 的任务，把轨迹转成 messages 对话格式。

输出：train.jsonl / valid.jsonl（MLX-LM lora 训练可直接使用）：
    {"messages": [
        {"role": "system",    "content": "<agent 系统提示>"},
        {"role": "user",      "content": "TASK: <任务指令>"},
        {"role": "user",      "content": "--- terminal (step 0) ---\\n<obs>"},
        {"role": "assistant", "content": "<模型回复>"},
        ...
    ]}

用法：
    python build_sft_dataset.py <run_dir> [--out-dir sft_data] [--val-frac 0.1] [--max-examples 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 与 CodebuffApiAgent 的 SYSTEM_PROMPT 保持一致，保证训练/推理分布对齐
DEFAULT_SYSTEM = (
    "You are an expert terminal agent inside a Linux shell. "
    "Complete the given task by running shell commands.\n"
    "Rules:\n"
    "- Use your terminal tool to run commands; read the terminal output between steps and adapt.\n"
    "- Prefer small, verifiable steps over one giant script.\n"
    "- Never modify files outside the task workspace.\n"
    "- When the task is complete, reply with a short summary and stop."
)


def find_trials(run_dir: Path) -> list[dict]:
    """扫描 output_path/<task_id>/<trial_name>/ 下的 results.json + trajectory。"""
    trials = []
    results = sorted(run_dir.rglob("results.json"))
    for rj in results:
        try:
            result = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        trial_dir = rj.parent
        traj_path = trial_dir / "agent-logs" / "trajectory.jsonl"
        if not traj_path.exists():
            continue
        trajectory = []
        try:
            for line in traj_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    trajectory.append(json.loads(line))
        except Exception:
            continue
        fm = str(result.get("failure_mode") or "")
        # runner_error 可能带完整 HTTP 错误信息（如 502/429），只留短标签
        if len(fm) > 80:
            fm = fm[:80]
        trials.append(
            {
                "task_id": result.get("task_id"),
                "instruction": result.get("instruction"),
                "is_resolved": result.get("is_resolved"),
                "failure_mode": fm,
                "trajectory": trajectory,
                "trial_dir": trial_dir,
            }
        )
    return trials


def to_messages(
    instruction: str, trajectory: list[dict], max_steps: int = 0
) -> list[dict]:
    """轨迹 → messages 对话（每步一个 user 终端块 + 一个 assistant 回复）。

    max_steps>0 时只保留最后 max_steps 步（系统提示 + TASK 始终保留）。
    原因：agent 的 obs 是全量累积 transcript，长任务样例可达上万 token，
    训练前需要截到模型上下文能容纳的长度。
    """
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": f"TASK: {instruction}"},
    ]
    recs = trajectory[-max_steps:] if max_steps > 0 else trajectory
    prev_commands: list[str] = []
    for rec in recs:
        obs = rec.get("obs") or ""
        reply = rec.get("reply") or ""
        if obs:
            # 与 CodebuffApiAgent._terminal_message 同格式：上一步命令回显
            # 显式附在 obs 后（obs 被截断时不会丢），训练/推理分布对齐
            content = f"--- terminal (step {rec.get('step')}) ---\n{obs}"
            if prev_commands:
                content += "\n\n[Commands you just ran: " + "; ".join(prev_commands) + "]"
            messages.append({"role": "user", "content": content})
        if reply:
            messages.append({"role": "assistant", "content": reply})
        prev_commands = [str(c) for c in (rec.get("commands") or [])]
    return messages


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 SFT/LoRA 数据集")
    ap.add_argument("run_dir", type=Path, help="tb 的输出目录（output_path/<run_id>）")
    ap.add_argument("--out-dir", type=Path, default=Path("sft_data"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-examples", type=int, default=0, help="0=不限制")
    ap.add_argument(
        "--max-steps", type=int, default=0, help="每条轨迹只保留最后 N 步（0=全保留，训练建议 3~6）"
    )
    ap.add_argument(
        "--include-failed",
        action="store_true",
        help="保留失败轨迹作为反例（label=negative，含 no_command/超时等 failure_mode）",
    )
    ap.add_argument(
        "--max-failed-examples", type=int, default=0, help="反例上限（0=不限制）"
    )
    args = ap.parse_args()

    trials = find_trials(args.run_dir)
    passed = [t for t in trials if t.get("is_resolved") is True]
    failed = [t for t in trials if t.get("is_resolved") is False]
    if args.max_examples > 0:
        passed = passed[: args.max_examples]
    if args.max_failed_examples > 0:
        failed = failed[: args.max_failed_examples]

    def _example(t: dict, label: str) -> dict:
        msgs = to_messages(
            t.get("instruction") or "", t.get("trajectory") or [], args.max_steps
        )
        notes = sorted(
            {str(r.get("note") or "") for r in t.get("trajectory") or [] if r.get("note")}
        )
        return {
            "task_id": t["task_id"],
            "n_steps": len(t.get("trajectory") or []),
            "label": label,
            "failure_mode": str(t.get("failure_mode") or ""),
            "trajectory_notes": notes,
            "messages": msgs,
        }

    # 采样打散（正例 + 反例一起混入，label 字段可区分）
    examples = []
    import random

    rng = random.Random(42)
    for t in passed:
        examples.append(_example(t, "positive"))
    if args.include_failed:
        for t in failed:
            examples.append(_example(t, "negative"))
    else:
        print(f"（--include-failed 未开启，跳过 {len(failed)} 条失败轨迹）")

    if not examples:
        print(f"没有可用轨迹：total={len(trials)} passed={len(passed)} failed={len(failed)}")
        return 1
    n_neg = sum(1 for e in examples if e["label"] == "negative")
    if n_neg and not any(e["label"] == "positive" for e in examples):
        print("warning: 数据集只有反例，没有正例（不建议纯反例训练）")
    rng.shuffle(examples)

    n_val = max(1, round(len(examples) * args.val_frac))
    val, train = examples[:n_val], examples[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    valid_path = args.out_dir / "valid.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for ex in train:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with valid_path.open("w", encoding="utf-8") as fh:
        for ex in val:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_steps = sum(t["n_steps"] for t in examples)
    print(
        f"trials={len(trials)} passed={len(passed)} failed={len(failed)} "
        f"-> train={len(train)} valid={len(val)} "
        f"pos={len(passed)} neg={n_neg} avg_steps={n_steps / max(1, len(examples)):.1f}"
    )
    if n_neg:
        from collections import Counter

        modes = Counter(e["failure_mode"] for e in examples if e["label"] == "negative")
        print("   反例 failure_mode:", dict(modes))
    print(f"train: {train_path.absolute()}")
    print(f"valid: {valid_path.absolute()}")
    print("   每条样例带 label(positive/negative) + failure_mode 字段，可用 DPO 等偏好训练")
    return 0


if __name__ == "__main__":
    sys.exit(main())
