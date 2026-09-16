#!/usr/bin/env python3
"""Collect the evidence block that engine-side /refine is fed.

WHY THIS IS ITS OWN FILE
------------------------
The tb_loop version inlines its evidence gathering inside a python heredoc in the shell script,
which makes the only way to check it "run the whole refine and see". This loop's whole thesis is
that a step nobody can test is a step nobody can trust, so the gathering is separated and can be
run on its own:

    python3 agent_harness/engine_loop/distill/collect_evidence.py --stats
    python3 agent_harness/engine_loop/distill/collect_evidence.py            # prints the block

WHAT IT COLLECTS, AND WHY EACH PART IS HERE
-------------------------------------------
1. **Episodes no decision cites.** These are the observations nothing has been concluded from --
   raw material for a new decision. (`--stats` reports how many, so "0" is distinguishable from
   "we never looked".)
2. **The newest decisions**, including their `ruled_out`. Negative knowledge is what stops the
   next session from re-walking a dead end; feeding only the conclusions loses exactly the part
   the schema calls most reused.
3. **The lessons**, because a new lesson must not duplicate an existing one, and because a
   refinement that contradicts a lesson is a regression the reviewer needs to see.
4. **The round's documents** (whitepapers + the daily memory sections), listed by PATH, not
   pasted. The memory file is 1000+ lines and grows all day; pasting it is how a consumer
   silently truncates.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent                       # agent_harness/engine_loop
REPO = ENGINE.parent.parent
TRACES = ENGINE / "traces"
MEMORY = REPO / ".workbuddy" / "memory"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true", help="print counts only, not the evidence block")
    ap.add_argument("--next-ids", action="store_true",
                    help="print the next free lesson_id per class, so a refinement cannot collide")
    ap.add_argument("--max-episodes", type=int, default=25, help="uncited episodes to include")
    ap.add_argument("--max-decisions", type=int, default=8)
    args = ap.parse_args()

    episodes = load(TRACES / "episodes.jsonl")
    decisions = load(TRACES / "decisions.jsonl")
    lessons = load(TRACES / "lessons.jsonl")

    if args.next_ids:
        # The id space is a CLOSED series: validate.py rejects duplicate ids, and selftest.py
        # injects a duplicate on purpose to prove the rejection works. Telling the distiller the
        # next free number per class is what stops a refinement from re-using an id that already
        # means something else -- a collision that would only surface as "two different rules under
        # one name", i.e. every later reference to that id becomes ambiguous.
        highest: dict[str, int] = {}
        for l in lessons:
            parts = l["lesson_id"].split("-")          # eng-<abbr>-<4 digits>
            if len(parts) != 3:
                continue
            try:
                highest[parts[1]] = max(highest.get(parts[1], 0), int(parts[2]))
            except ValueError:
                continue
        if not highest:
            print("  (no lessons yet)")
            return 0
        # 用哨兵包起來：這一段是**唯一**可以取用的 id 來源。沒有哨兵的話，模型（與測試）
        # 會從文件中第一個看到的 id 抓——而證據區列了 100+ 個「已經在用」的 id，
        # 於是撞號變成很可能而不是很不可能。這個缺陷是 distill/selftest.py 抓到的。
        print("<!-- NEXT-FREE-IDS-BEGIN -->")
        for abbr in sorted(highest):
            print(f"  eng-{abbr}-{highest[abbr] + 1:04d}")
        print("<!-- NEXT-FREE-IDS-END -->")
        return 0

    cited = {ev.get("episode_id") for d in decisions for ev in d["evidence"] if ev.get("episode_id")}
    uncited = [e for e in episodes if e["episode_id"] not in cited]
    docs = sorted((REPO / "docs").glob("AGENT_HARNESS_*.html"))
    mem = sorted(MEMORY.glob("*.md"))

    if args.stats:
        print(f"  episodes               : {len(episodes)}  (uncited by any decision: {len(uncited)})")
        print(f"    of which quotable    : {sum(1 for e in episodes if e.get('usable_as_evidence'))}")
        print(f"    of which no build fp : {sum(1 for e in episodes if not e.get('build'))}")
        print(f"  decisions              : {len(decisions)}  {dict(Counter(d['judgement'] for d in decisions))}")
        print(f"    ruled_out entries    : {sum(len(d.get('ruled_out') or []) for d in decisions)}")
        print(f"  lessons                : {len(lessons)}  {dict(sorted(Counter(l['class'] for l in lessons).items()))}")
        print(f"    superseded           : {sum(1 for l in lessons if l.get('superseded_by'))}")
        print(f"  round documents (paths): {len(docs)}")
        print(f"  memory files (paths)   : {len(mem)}")
        return 0

    out = []
    out.append("### 1. 還沒有任何 decision 引用過的 episode（未被結論過的觀測）")
    out.append(f"（共 {len(uncited)} 筆，列出最新的 {min(len(uncited), args.max_episodes)} 筆）")
    for e in uncited[-args.max_episodes:]:
        obs = e.get("obs") or {}
        tps = obs.get("decode_tps") or {}
        out.append(
            f"- {e['episode_id']}  profile={e['profile']} arm={e['arm']['name']}  "
            f"decode_tps_median={tps.get('median')} n_rounds={tps.get('n_rounds')}  "
            f"build={'yes' if e.get('build') else 'NULL'}  quotable={e.get('usable_as_evidence')}\n"
            f"  goal: {e['goal']}"
        )

    out.append("")
    out.append("### 2. 最新的 decision（含它們已經排除掉什麼——負面知識是最常被重用的部分）")
    for d in decisions[-args.max_decisions:]:
        out.append(f"- {d['decision_id']} [{d['judgement']}/{d.get('confidence')}] {d['question']}")
        out.append(f"  conclusion: {d['conclusion']}")
        for r in (d.get("ruled_out") or []):
            out.append(f"  RULED OUT: {r['claim']}  <-  {r['why_false']}")

    out.append("")
    out.append("### 3. 既有的 lesson（新教訓不得與它們重複或矛盾；矛盾要先說明誰取代誰）")
    for l in lessons:
        sup = f"  [SUPERSEDED BY {l['superseded_by']}]" if l.get("superseded_by") else ""
        out.append(f"- {l['lesson_id']} [{l['class']}]{sup} {l['rule'][:160]}")

    out.append("")
    out.append("### 4. 本輪的文件（只給路徑——當日日誌 1000+ 行，貼進來會被靜默截斷）")
    for p in docs:
        out.append(f"- {p.relative_to(REPO)}")
    for p in mem:
        out.append(f"- {p.relative_to(REPO)}  (用 engine_loop/memory/INDEX.jsonl 按段落定位，不要整檔讀)")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
