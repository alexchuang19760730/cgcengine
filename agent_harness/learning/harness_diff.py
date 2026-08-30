#!/usr/bin/env python3
"""
harness_state 条目级 diff（教训级归因的数据源）。

解析 prime-agent Continual Harness 的两个状态快照（harness_state.json +
refinements.jsonl），输出条目级变化，并溯源到创建/修改它的 refinement id
（回滚单位是 refinement，见 RefinementResult）。

用法：
    python harness_diff.py <before_dir> <after_dir> [--out diff.json]

before/after 目录 = 含 harness_state.json（及可选 refinements.jsonl）的目录。
也可以直接传 harness_state.json 文件路径。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KINDS = ("prompt", "memory", "skill", "subagent")
KIND_LABEL = {"prompt": "prompt", "memory": "记忆", "skill": "skill", "subagent": "subagent"}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_state(dir_or_file: Path) -> dict:
    """返回 {entries: {kind: {id: entry}}, refinements: [RefinementResult]}。"""
    p = Path(dir_or_file)
    state_path = p if p.is_file() else p / "harness_state.json"
    state = _read_json(state_path) or {}
    entries = state.get("entries") or {}
    refinements: list[dict] = []
    # 快照可能是 harness_state.before.json + refinements.before.jsonl（refine_harness.sh 的命名）
    base = state_path.stem  # harness_state.before / harness_state
    base = base.removeprefix("harness_state").lstrip(".")
    candidates = [
        state_path.parent / "refinements.jsonl",
        state_path.parent / f"refinements.{base}.jsonl" if base else None,
    ]
    refin_path = next((c for c in candidates if c and c.exists()), None)
    if refin_path and refin_path.exists():
        try:
            for line in refin_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    refinements.append(json.loads(line))
        except Exception:
            pass
    return {"entries": entries, "refinements": refinements}


def entry_index(entries: dict) -> dict[str, dict]:
    """{id: entry}（含 kind 字段）。"""
    idx: dict[str, dict] = {}
    for kind in KINDS:
        for eid, entry in (entries.get(kind) or {}).items():
            e = dict(entry)
            e.setdefault("kind", kind)
            e.setdefault("id", eid)
            idx[eid] = e
    return idx


def refinement_entry_map(refinements: list[dict]) -> dict[str, str]:
    """{entry_id: 最后触碰它的 refinement_id}（appliedEdits 里 id 字段）。"""
    m: dict[str, str] = {}
    for ref in refinements:
        rid = ref.get("id")
        for edit in ref.get("appliedEdits") or []:
            if edit.get("applied") is False:
                continue
            eid = edit.get("id")
            if eid and rid:
                m[eid] = rid
    return m


def diff_states(before: dict, after: dict) -> dict:
    """条目级 diff：created / updated / deleted，附 refinement 溯源。"""
    b_idx = entry_index(before["entries"])
    a_idx = entry_index(after["entries"])
    b_ref = refinement_entry_map(before["refinements"])
    a_ref = refinement_entry_map(after["refinements"])
    # 合并两边的溯源（before 的 created 可能来自更早 refinement）
    ref_map = {**b_ref, **a_ref}

    created, updated, deleted = [], [], []
    for eid, a_e in sorted(a_idx.items()):
        b_e = b_idx.get(eid)
        if b_e is None:
            created.append(
                {
                    "id": eid,
                    "kind": a_e.get("kind"),
                    "title": a_e.get("title"),
                    "refinement_id": ref_map.get(eid),
                }
            )
        elif (b_e.get("version") or 0) != (a_e.get("version") or 0) or (
            b_e.get("content") or ""
        ) != (a_e.get("content") or ""):
            updated.append(
                {
                    "id": eid,
                    "kind": a_e.get("kind"),
                    "title": a_e.get("title"),
                    "version_before": b_e.get("version"),
                    "version_after": a_e.get("version"),
                    "refinement_id": ref_map.get(eid),
                }
            )
    for eid, b_e in sorted(b_idx.items()):
        if eid not in a_idx:
            deleted.append(
                {
                    "id": eid,
                    "kind": b_e.get("kind"),
                    "title": b_e.get("title"),
                    "refinement_id": ref_map.get(eid),
                }
            )
    return {"created": created, "updated": updated, "deleted": deleted}


def _fmt(items: list[dict]) -> str:
    out = []
    for it in items:
        rid = it.get("refinement_id") or "-"
        out.append(
            f"  {it.get('kind','?'):<8} {it.get('id','?'):<28} "
            f"{str(it.get('title'))[:40]:<42} ref={rid}"
        )
    return "\n".join(out) if out else "  (无)"


def main() -> int:
    ap = argparse.ArgumentParser(description="harness 条目级 diff")
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    before = load_state(args.before)
    after = load_state(args.after)
    diff = diff_states(before, after)

    print("=== harness 条目变化 ===")
    print(f"created ({len(diff['created'])}):")
    print(_fmt(diff["created"]))
    print(f"updated ({len(diff['updated'])}):")
    print(_fmt(diff["updated"]))
    print(f"deleted ({len(diff['deleted'])}):")
    print(_fmt(diff["deleted"]))
    print(f"refinements: before={len(before['refinements'])} after={len(after['refinements'])}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"diff -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
