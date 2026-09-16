#!/usr/bin/env python3
"""sft_prime — the prime-agent projection: (evidence -> lesson) and (state -> next arm).

WHY THESE TWO SHAPES AND NOT "ALL THE TRACES"
--------------------------------------------
The loop's output is not a conversation, it is a set of judgements. A judgement has two halves
worth training on, and the plan names both (§6.2):

  evidence -> lesson   the generalisation step. Input is the observation that COST something
                       (`because`, with its numbers), output is the reusable rule. This is what
                       makes a later session cheaper.
  state -> next arm    the choice step. Input is a decision's question plus the readings it
                       leaned on PLUS what has already been ruled out, output is `action`.
                       Including `ruled_out` is the point: the schema itself calls the negative
                       knowledge "the single most reused thing in this loop", and a model that
                       only sees the winning hypothesis re-walks dead ends.

TWO THINGS ARE DELIBERATELY NOT TRAINED HERE
--------------------------------------------
* `judgement == "refuted"` decisions are EXCLUDED from the positive set. The decision schema
  says this field is "the anti-training-poison gate". A refuted hypothesis that becomes a
  positive example teaches the model to re-walk the dead end the loop already paid for.
* lessons whose `superseded_by` is set are excluded from `evidence -> lesson`, for the same
  reason. `--include-superseded` exists only to make the exclusion VISIBLE (it prints how many
  were held back); it does not make them usable.

Usage:
    python3 agent_harness/engine_loop/sft_prime/build_sft_prime.py [--out-dir DIR] [--val-frac F]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sft_common as C  # noqa: E402


def evidence_to_lesson(lessons: list[dict], sys_text: str) -> list[dict]:
    rows = []
    for l in lessons:
        if l.get("superseded_by"):
            continue  # a replaced rule must not be taught as current
        user = [
            "以下是一條引擎調校迴圈裡「付過代價」的觀測。請把它一般化成**一條可重用的規訓**——",
            "換一個問題也適用，而且讀者手上沒有這筆 episode 也必須能照著做。",
            "",
            f"觀測（because）：\n{l['because']}",
        ]
        ce = l.get("counterexample_observed")
        if ce:
            user += ["", f"它實際被違反的紀錄（counterexample_observed）：\n{ce}"]
        if l.get("applies_to"):
            user += ["", "它管轄的檔案：" + ", ".join(f"`{p}`" for p in l["applies_to"])]
        rows.append({
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": "\n".join(user)},
                {"role": "assistant", "content": l["rule"]},
            ],
            "kind": "evidence_to_lesson",
            "source_id": l["lesson_id"],
            "class": l["class"],
            "label": "positive",
        })
    return rows


def state_to_next_arm(decisions: list[dict], by_ep: dict[str, dict], sys_text: str) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for d in decisions:
        if d.get("judgement") == "refuted":
            skipped += 1
            continue  # anti-training-poison gate (decision.schema.json)
        user = [
            "以下是引擎調校迴圈此刻的狀態。決定**下一個該做什麼**：一個具體的動作，",
            "不要複述狀態。已經被排除的路線不要再走一次。",
            "",
            C.render_state_for_decision(d, by_ep),
        ]
        rows.append({
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": "\n".join(user)},
                {"role": "assistant", "content": d["action"]},
            ],
            "kind": "state_to_next_arm",
            "source_id": d["decision_id"],
            "judgement": d["judgement"],
            "confidence": d.get("confidence"),
            "ruled_out_n": len(d.get("ruled_out") or []),
            "label": "positive",
        })
    return rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--system", choices=("full", "none"), default="full",
                    help="full = CONVENTIONS.md byte-identical (the plan's requirement); "
                         "none = omit it (for a dataset you intend to prompt differently at inference)")
    args = ap.parse_args()

    sys_text, sys_sha = C.charter()
    if args.system == "none":
        sys_text = ""

    lessons = C.load(C.LESSONS)
    decisions = C.load(C.DECISIONS)
    episodes = C.load(C.EPISODES)
    by_ep = {e["episode_id"]: e for e in episodes}

    rows = evidence_to_lesson(lessons, sys_text)
    n_l = len(rows)
    st_rows, skipped = state_to_next_arm(decisions, by_ep, sys_text)
    rows += st_rows

    train, valid = C.split(rows, args.val_frac)
    C.write_jsonl(args.out_dir / "train.jsonl", train)
    C.write_jsonl(args.out_dir / "valid.jsonl", valid)

    C.report(args.out_dir, train, valid,
             f"evidence_to_lesson={n_l} state_to_next_arm={len(st_rows)} "
             f"| excluded: refuted_decisions={skipped} "
             f"superseded_lessons={sum(1 for l in lessons if l.get('superseded_by'))}")
    print(f"  system prompt: CONVENTIONS.md {'(omitted)' if not sys_text else f'{len(sys_text)} B, sha256[:16]={sys_sha}'}")
    print(f"  provenance: reconstructed from traces/*.jsonl; this is NOT a recorded transcript")
    if not rows:
        print("  [error] no rows -- nothing to train on", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
