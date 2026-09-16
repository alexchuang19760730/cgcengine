#!/usr/bin/env python3
"""
教训级归因验证：把 harness 条目变化与任务成败做相关分析，
找出无用 / 有害条目并输出回滚建议。

方法（诚实声明：这是相关性启发，不是因果）：
- 两轮用**同一任务集**（TB_TASK_IDS 固定），round1 用 harness H1，round2 用 H2
  （H2 = H1 + 两轮之间 /refine 的条目变化）。
- 每个 changed entry（created/updated，必然出现在 H2 里）：
    improvement = round1 失败且 round2 通过的任务数（该条目可能帮了忙）
    regression  = round1 通过且 round2 失败的任务数（该条目可能闯了祸）
- 分类（--min-observations 默认 2，避免单任务噪声）：
    useful   : improvement >= min_obs 且 regression == 0
    harmful  : regression >= min_obs 且 regression > improvement  → 建议回滚
    mixed    : 两边都有可观计数 → 人工复查
    useless  : 两边都是 0 → 无观测效果，可考虑清理
- harmful 条目的 refinement_id 是可回滚单位（prime-agent /refine rollback）。

用法：
    python attribution.py --before <H1 目录/文件> --after <H2 目录/文件> \
        --round1 <round1 run_dir> --round2 <round2 run_dir> \
        [--min-observations 2] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 直接以脚本方式运行（run_round.sh 这样调用）时，保证能 import 到 learning.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning.harness_diff import diff_states, load_state  # noqa: E402
from learning.compare_rounds import load_run  # noqa: E402

LABEL = {"useful": "有用", "harmful": "有害", "mixed": "混合", "useless": "无用"}


def classify(improvement: int, regression: int, min_obs: int) -> str:
    if regression >= min_obs and regression > improvement:
        return "harmful"
    if improvement >= min_obs and regression == 0:
        return "useful"
    if improvement >= min_obs and regression >= min_obs:
        return "mixed"
    return "useless"


def main() -> int:
    ap = argparse.ArgumentParser(description="教训级归因验证")
    ap.add_argument("--before", type=Path, required=True, help="round1 用的 harness 状态")
    ap.add_argument("--after", type=Path, required=True, help="round2 用的 harness 状态")
    ap.add_argument("--round1", type=Path, required=True)
    ap.add_argument("--round2", type=Path, required=True)
    ap.add_argument("--min-observations", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    r1 = load_run(args.round1)
    r2 = load_run(args.round2)
    if not r1 or not r2:
        print("error: 某轮没有结果（检查 run_dir 是否含 results.json）", file=sys.stderr)
        return 2

    tasks = sorted(set(r1) & set(r2))
    if not tasks:
        print("error: 两轮没有重叠任务，无法归因（TB_TASK_IDS 必须固定同一任务集）", file=sys.stderr)
        return 3

    deltas = {"PP": 0, "PF": 0, "FP": 0, "FF": 0}
    delta_by_task = {}
    for t in tasks:
        a, b = r1[t], r2[t]
        if a is None or b is None:
            continue
        key = ("P" if a else "F") + ("P" if b else "F")
        deltas[key] = deltas.get(key, 0) + 1
        delta_by_task[t] = key

    before = load_state(args.before)
    after = load_state(args.after)
    diff = diff_states(before, after)
    changed = diff["created"] + diff["updated"]

    improvements = [t for t, d in delta_by_task.items() if d == "FP"]
    regressions = [t for t, d in delta_by_task.items() if d == "PF"]

    entries = []
    for c in changed:
        eid, kind, title = c.get("id"), c.get("kind"), c.get("title")
        cls = classify(len(improvements), len(regressions), args.min_observations)
        entries.append(
            {
                "id": eid,
                "kind": kind,
                "title": title,
                "refinement_id": c.get("refinement_id"),
                "class": cls,
                "label": LABEL[cls],
                "improvement_tasks": list(improvements),
                "regression_tasks": list(regressions),
            }
        )

    # 同一 refinement 会改多条条目——按 refinement 聚合，回滚粒度是 refinement
    by_ref: dict[str, dict] = {}
    for e in entries:
        rid = e["refinement_id"] or "(no-ref)"
        agg = by_ref.setdefault(
            rid,
            {
                "refinement_id": rid,
                "class": "useless",
                "label": "无用",
                "entries": [],
                "improvement_tasks": set(),
                "regression_tasks": set(),
            },
        )
        agg["entries"].append(e)
        agg["improvement_tasks"] |= set(e["improvement_tasks"])
        agg["regression_tasks"] |= set(e["regression_tasks"])
        # refinement 级分类 = 最严重条目：harmful > mixed > useful > useless
        order = {"harmful": 3, "mixed": 2, "useful": 1, "useless": 0}
        if order[e["class"]] > order[agg["class"]]:
            agg["class"], agg["label"] = e["class"], e["label"]

    for r in by_ref.values():
        r["improvement_tasks"] = sorted(r["improvement_tasks"])
        r["regression_tasks"] = sorted(r["regression_tasks"])
    refinements = sorted(by_ref.values(), key=lambda x: x["refinement_id"] or "")
    rollback = [r for r in refinements if r["class"] == "harmful" and r["refinement_id"] != "(no-ref)"]

    report = {
        "tasks": tasks,
        "deltas": deltas,
        "min_observations": args.min_observations,
        "changes": diff,
        "entries": entries,
        "refinements": refinements,
        "rollback": [r["refinement_id"] for r in rollback],
        "caveat": "相关性启发，非因果；需固定同一任务集、多轮累积再采信",
    }

    print("=== 任务成败变化 (round1 -> round2) ===")
    for t in tasks:
        print(f"  {t:<28} {delta_by_task.get(t, '?')}")
    print(f"  PP={deltas['PP']} PF(回归)={deltas['PF']} FP(改进)={deltas['FP']} FF={deltas['FF']}")
    print()
    print("=== 条目级归因 ===")
    for e in entries:
        print(
            f"  [{e['label']}] {e['kind']:<8} {e['id']:<28} "
            f"improve={len(e['improvement_tasks'])} regress={len(e['regression_tasks'])} "
            f"ref={e['refinement_id'] or '-'}"
        )
    print()
    print("=== refinement 聚合（回滚单位）===")
    for r in refinements:
        print(
            f"  [{r['label']}] {r['refinement_id']:<40} "
            f"entries={len(r['entries'])} improve={len(r['improvement_tasks'])} "
            f"regress={len(r['regression_tasks'])}"
        )
    print()
    if rollback:
        print(f"== 建议回滚 {len(rollback)} 个有害 refinement: "
              f"{[r['refinement_id'] for r in rollback]}")
        print("   执行: bash learning/rollback_harmful.sh <report.json>")
    else:
        print("== 无有害 refinement 需要回滚")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
