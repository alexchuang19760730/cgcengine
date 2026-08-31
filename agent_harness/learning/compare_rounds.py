#!/usr/bin/env python3
"""
对比两轮评估结果（学习前后）。

用法：
    python compare_rounds.py <round1_run_dir> <round2_run_dir>

run_dir = tb 的 --output-path/<run_id>（含汇总 results.json）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_run(run_dir: Path) -> dict[str, bool | None]:
    per: dict[str, bool | None] = {}
    summary = run_dir / "results.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            for r in data.get("results", []):
                if r.get("task_id") and r.get("task_id") not in per:
                    per[r["task_id"]] = r.get("is_resolved")
        except Exception:
            pass
    # 兜底：递归
    for rj in sorted(run_dir.rglob("results.json")):
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("task_id") and data["task_id"] not in per:
            per[data["task_id"]] = data.get("is_resolved")
    return per


def score(per: dict[str, bool | None]) -> tuple[int, int]:
    n = sum(1 for v in per.values() if v is True)
    t = sum(1 for v in per.values() if v is not None)
    return n, t


def main() -> int:
    ap = argparse.ArgumentParser(description="对比两轮 Terminal-Bench 结果")
    ap.add_argument("round1", type=Path)
    ap.add_argument("round2", type=Path)
    args = ap.parse_args()

    r1 = load_run(args.round1)
    r2 = load_run(args.round2)
    if not r1 or not r2:
        print("error: 某一轮没有结果（检查 run_dir 是否含 results.json）", file=sys.stderr)
        return 2

    all_tasks = sorted(set(r1) | set(r2))
    print(f"{'task':<42}{'round1':>8}{'round2':>8}{'delta':>8}")
    print("-" * 66)
    for t in all_tasks:
        a, b = r1.get(t), r2.get(t)
        mark = lambda v: ("pass" if v is True else "fail" if v is False else "  - ")
        delta = "" if a is None or b is None else ("+" if b and not a else "")
        print(f"{t:<42}{mark(a):>8}{mark(b):>8}{delta:>8}")

    n1, t1 = score(r1)
    n2, t2 = score(r2)
    acc1 = n1 / t1 if t1 else 0.0
    acc2 = n2 / t2 if t2 else 0.0
    print("-" * 66)
    print(f"accuracy  round1: {acc1:.1%} ({n1}/{t1})")
    print(f"accuracy  round2: {acc2:.1%} ({n2}/{t2})")
    print(f"delta: {acc2 - acc1:+.1%}   resolved: {n1} -> {n2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
