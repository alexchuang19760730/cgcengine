#!/usr/bin/env python3
"""
从一轮 tb run 的输出目录提取失败任务（供 refine_harness.sh 使用）。

用法：
    python extract_failures.py <run_dir> [--out failures.json]

run_dir = tb 的 --output-path/<run_id>（含各任务的 results.json 与汇总 results.json）。
布局容错：递归扫描所有 results.json，取 is_resolved=False 的 TrialResults。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def collect_run(run_dir: Path) -> dict[str, dict]:
    """返回 {task_id: TrialResults dict}（含 trial_dir，供 refine 读完整轨迹）。

    优先递归找每个任务目录下的 results.json（能定位 commands.txt / panes /
    agent-logs 等完整轨迹文件），汇总 results.json 只做兜底。
    """
    per_task: dict[str, dict] = {}
    for rj in sorted(run_dir.rglob("results.json")):
        if rj.parent == run_dir:
            continue  # 汇总 results.json（在 run 根目录）不是单任务结果
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("task_id") and data["task_id"] not in per_task:
            data = dict(data)
            data["trial_dir"] = str(rj.parent.resolve())
            per_task[data["task_id"]] = data
    # 兜底：汇总 results.json（无逐任务目录时）
    summary = run_dir / "results.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            for r in data.get("results", []):
                if r.get("task_id") and r["task_id"] not in per_task:
                    per_task.setdefault(r["task_id"], r)
        except Exception:
            pass
    return per_task


def main() -> int:
    ap = argparse.ArgumentParser(description="提取 Terminal-Bench 失败任务")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("failures.json"))
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        print(f"error: {args.run_dir} 不是目录", file=sys.stderr)
        return 2

    per_task = collect_run(args.run_dir)
    failures = []
    for task_id in sorted(per_task):
        r = per_task[task_id]
        if r.get("is_resolved") is False:
            failures.append(
                {
                    "task_id": task_id,
                    "instruction": r.get("instruction"),
                    "failure_mode": str(r.get("failure_mode")),
                    "trial_dir": r.get("trial_dir"),
                    "recording_path": r.get("recording_path"),
                    "total_input_tokens": r.get("total_input_tokens"),
                    "total_output_tokens": r.get("total_output_tokens"),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_pass = sum(1 for r in per_task.values() if r.get("is_resolved") is True)
    print(f"tasks={len(per_task)} passed={n_pass} failed={len(failures)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
