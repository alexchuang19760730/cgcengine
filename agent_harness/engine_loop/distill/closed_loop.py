#!/usr/bin/env python3
"""The E2 closed-loop comparison: does a CONVENTIONS.md change alter behaviour on NEW questions?

WHY THIS FILE EXISTS AT ALL
---------------------------
`CONVENTIONS.md` §E says editing the charter changes what both loops do, therefore every edit must
go through a closed-loop comparison (PLAN §6.3). The 2026-09-16 D6 amendment asserted that its edit
"changes no runtime behaviour" -- and an assertion with no artefact behind it is exactly the kind
of sentence this repo writes rules against. So the debt is: RUN the comparison, and record the
difference (a null result is a result; PLAN §9 says so explicitly).

THREE ARMS BECAME FOUR, BECAUSE THE FIRST VERSION DID NOT ISOLATE WHAT IT CLAIMED
---------------------------------------------------------------------------------
The first version compared `e688f346e^` (pre-amendment) against HEAD. `--dry-run` showed that
diff is +35/-3 lines and most of the additions are NOT the D6 amendment -- they are the later E1
update to D1. So "A vs B" was measuring two edits at once. **A pair that does not isolate its
variable is not a comparison; it is two changes wearing one label.**

    A_preD6        `e688f346e^`  charter   -- immediately before the amendment
    B_postD6       `e688f346e`   charter   -- isolates THE D6 AMENDMENT      (A vs B)
    C_head         HEAD / live   charter   -- isolates everything after D6   (B vs C)
    D_head_mem     HEAD / live   charter + harness_engine/memories/engine/   (C vs D)

Only A vs B answers the debt's question. The other two pairs exist so that a difference found in
A vs B cannot be misattributed to the later edit or to the injected lessons.

Charters are read LIVE from git (or from disk for HEAD), never cached to a file: a cached copy of
the charter is a second copy of the charter, and a second copy's failure mode is silent.

WHAT IS VERIFIABLE WITHOUT A MODEL, AND WHAT IS NOT
---------------------------------------------------
`--dry-run` needs no model: it builds every arm's prompt, prints sizes/sha256, and diffs each
adjacent pair so you can SEE that each pair isolates what it claims. That is the mechanical half.

The model half needs `CLOSED_LOOP_MODEL_CMD` -- a command that reads a prompt on stdin and writes
an answer on stdout. It has no default, on purpose: which model answers the questions IS the
experiment, and a script that silently picks one turns an experiment into a habit.

    python3 agent_harness/engine_loop/distill/closed_loop.py --dry-run
    CLOSED_LOOP_MODEL_CMD='llm -m <model>' python3 .../closed_loop.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
REPO = ENGINE.parent.parent
CHARTER = ENGINE.parent / "CONVENTIONS.md"
MEM_DIR = ENGINE / "harness_engine" / "memories" / "engine"
QUESTIONS = HERE / "closed_loop_questions.md"
OUT_ROOT = HERE / "closed_loop_out"

# The amendment landed in this commit; its parent carries the pre-amendment charter.
D6_COMMIT = "e688f346e"
PRE_D6_REF = f"{D6_COMMIT}^:agent_harness/CONVENTIONS.md"


def charter_pre_d6() -> str:
    r = subprocess.run(["git", "-C", str(REPO), "show", PRE_D6_REF],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot read the pre-D6 charter ({PRE_D6_REF}): {r.stderr.strip()[:200]}\n"
                         f"  The comparison NEEDS both charters. Do not substitute a cached copy.")
    return r.stdout


def load_questions() -> list[tuple[str, str]]:
    """Parse the question file: lines starting with `Q<n>.` begin a question."""
    text = QUESTIONS.read_text(encoding="utf-8")
    out, cur_id, cur = [], None, []
    for line in text.splitlines():
        m = re.match(r"^(Q\d+)\.\s*(.*)$", line)
        if m:
            if cur_id:
                out.append((cur_id, " ".join(cur).strip()))
            cur_id, cur = m.group(1), [m.group(2)]
        elif cur_id and line.strip() and not line.startswith(("##", "|", "---")):
            cur.append(line.strip())
    if cur_id:
        out.append((cur_id, " ".join(cur).strip()))
    if not out:
        raise SystemExit(f"no questions parsed from {QUESTIONS} -- refusing to run an empty comparison")
    return out


def memories_block() -> str:
    if not MEM_DIR.is_dir():
        return ""
    files = sorted(MEM_DIR.glob("*.md"))
    if not files:
        return ""
    parts = ["", "## 這個 scope 已累積的規訓（harness state：每一條都是一個 lesson）", ""]
    for p in files:
        first = p.read_text(encoding="utf-8").splitlines()[0]
        parts.append(f"- {first}")
    return "\n".join(parts)


def build_prompt(charter: str, mems: str, q: str) -> str:
    return (charter + mems + "\n\n---\n\n"
            "## 現在要回答的問題\n\n"
            f"{q}\n\n"
            "**只回答「下一個具體動作」以及你據以判斷的欄位／判準。不要複述問題，不要客套。**\n"
            "若你認為這個問題在現有證據下無法回答，就明說「無法回答」並指出缺什麼。\n")


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def charter_at(ref: str) -> str:
    """A committed charter, read from git. Never cached to disk."""
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:agent_harness/CONVENTIONS.md"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot read CONVENTIONS.md at {ref}: {r.stderr.strip()[:200]}")
    return r.stdout


def _diff_stat(a: str, b: str) -> tuple[int, int]:
    u = _unified(a, b).splitlines()
    return (sum(1 for l in u if l.startswith("+") and not l.startswith("+++")),
            sum(1 for l in u if l.startswith("-") and not l.startswith("---")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="build every arm's prompt, print sizes/sha256, and diff each adjacent pair; no model")
    ap.add_argument("--only", default=None, help="run a single question id, e.g. Q2")
    args = ap.parse_args()

    live = CHARTER.read_text(encoding="utf-8")
    pre = charter_at(PRE_D6_REF if ":" not in PRE_D6_REF else D6_COMMIT + "^")
    post_d6 = charter_at(D6_COMMIT)
    mems = memories_block()
    qs = load_questions()
    if args.only:
        qs = [q for q in qs if q[0] == args.only]
        if not qs:
            raise SystemExit(f"no such question: {args.only}")

    # Four arms. Only A vs B answers the D6 debt; the other pairs exist so a difference in A vs B
    # cannot be blamed on the later E1 edit or on the injected lessons.
    arms = [
        ("A_preD6", pre, ""),
        ("B_postD6", post_d6, ""),
        ("C_head", live, ""),
        ("D_head_mem", live, mems),
    ]
    pairs = [("A_preD6", "B_postD6", "D6 修訂（本欠帳要回答的就是這一對）"),
             ("B_postD6", "C_head", "D6 之後落地的其他改動（含 E1 的 D1 更新）"),
             ("C_head", "D_head_mem", "注入 harness_engine 的 lesson")]

    print(f"  questions: {len(qs)}  ({', '.join(q for q, _ in qs)})")
    for name, ch, mm in arms:
        sample = build_prompt(ch, mm, "Q?")
        print(f"  arm {name:12s} charter={len(ch):7d} B sha256[:16]={sha16(ch)}  memories={len(mm):6d} B"
              f"  prompt≈{len(sample):7d} B")

    print("  --- 每一對只隔離一件事（這一節是儀器自檢；第一版就是在這裡被抓到沒有隔離）---")
    by_name = {n: c for n, c, _ in arms}
    for a, b, what in pairs:
        add, rem = _diff_stat(by_name[a], by_name[b])
        print(f"    {a:11s} -> {b:11s}  +{add:3d}/-{rem:3d} 行, {len(by_name[b]) - len(by_name[a]):+6d} B   {what}")

    if args.dry_run:
        print()
        print("  --- A vs B 的實際內容（前 6 行）---")
        a_only = [l for l in _unified(by_name["A_preD6"], by_name["B_postD6"]).splitlines()
                  if l.startswith("+") and not l.startswith("+++")]
        for l in a_only[:6]:
            print(f"    B 新增: {l[1:][:110]}")
        print(f"    ... 共 {len(a_only)} 行新增")
        print()
        print("  dry-run 到此為止：以上是「每個變數有沒有真的進到 prompt、進了多少」——問題的機械面。")
        print("  行為面需要模型回答問題，該步驟需要 CLOSED_LOOP_MODEL_CMD（不給預設值）。")
        return 0

    cmd = os.environ.get("CLOSED_LOOP_MODEL_CMD", "").strip()
    if not cmd:
        print("  error: CLOSED_LOOP_MODEL_CMD is unset. Which model answers IS the experiment,",
              file=sys.stderr)
        print("         so this script will not pick one. e.g. CLOSED_LOOP_MODEL_CMD='llm -m <model>'",
              file=sys.stderr)
        return 2

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / ts
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid, q in qs:
        for name, ch, mm in arms:
            prompt = build_prompt(ch, mm, q)
            r = subprocess.run(shlex.split(cmd), input=prompt, capture_output=True, text=True)
            rows.append({
                "question_id": qid, "arm": name,
                "rc": r.returncode,
                "answer": (r.stdout or "").strip() or None,
                "stderr_tail": (r.stderr or "")[-300:] or None,
                "prompt_bytes": len(prompt), "prompt_sha256_16": sha16(prompt),
                "charter_sha256_16": sha16(ch), "memories_bytes": len(mm),
            })
            print(f"    {qid} {name:14s} rc={r.returncode} answer={len(r.stdout or '')} B")
    (out / "answers.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

    # The comparison itself: same question, different arm -> are the answers identical?
    by_q: dict[str, dict[str, str]] = {}
    for x in rows:
        by_q.setdefault(x["question_id"], {})[x["arm"]] = x["answer"] or ""
    compare = []
    for qid, per in sorted(by_q.items()):
        a = per.get("A_preD6", "")
        b = per.get("B_postD6", "")
        c = per.get("C_head", "")
        d = per.get("D_head_mem", "")
        compare.append({
            "question_id": qid,
            # 這一對是欠帳的答案：
            "A_vs_B_identical": a == b,
            "A_vs_B_chars": f"{len(a)} -> {len(b)}",
            "A_vs_B_verdict": ("D6 修訂在這一題上是惰性的" if a == b
                               else "D6 修訂改變了這一題的答案"),
            # 另兩對是反歸因用的，不回答欠帳：
            "B_vs_C_identical": b == c,
            "C_vs_D_identical": c == d,
        })
    (out / "compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    n_inert = sum(1 for c in compare if c["A_vs_B_identical"])
    n_cd = sum(1 for c in compare if c["C_vs_D_identical"])
    print(f"  A vs B（D6 本體）: {n_inert}/{len(compare)} 題答案完全相同 -> "
          f"{'這個問題集上找不到 D6 的行為效應' if n_inert == len(compare) else '有差異，逐題看 compare.json'}")
    print(f"  C vs D（注入 lesson）: {n_cd}/{len(compare)} 題相同（反歸因用，不回答欠帳）")
    print(f"  out: {out.relative_to(REPO)}")
    print("  注意：字串相同只是自動部分。逐題的承重點在 closed_loop_questions.md 的表格裡，"
          "要人工核對——兩份答案可以字面不同而判準相同，也可以字面相同而都漏掉承重點。")
    return 0


def _unified(a: str, b: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                       "A_preD6", "B_postD6", n=0))


if __name__ == "__main__":
    raise SystemExit(main())
