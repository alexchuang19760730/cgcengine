#!/usr/bin/env python3
"""The closed-loop comparison: does a charter change -- or an injected lesson -- alter behaviour?

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
    D_head_mem     HEAD / live   charter + harness_engine memories          (C vs D)

A vs B answers the D6 debt. C vs D answers E3 (PLAN §9: "round1 (no lesson) vs round2"). The
middle pair exists so that a difference in either cannot be misattributed to the other.

Charters are read LIVE from git (or from disk for HEAD), never cached to a file: a cached copy of
the charter is a second copy of the charter, and a second copy's failure mode is silent.

HOW THE LESSON SET IS CHOSEN, AND WHY IT IS A PARAMETER
-------------------------------------------------------
PLAN §6.3 says round2 injects "今天定稿的 8 條 lesson". That 8 is a hand-written number, and the
plan is already inconsistent with itself about it: §4.3 -- the table §11 points at as the source of
those lessons -- NOW LISTS TEN. The file order of `lessons.jsonl` is the append-only record of the
order they were written in, and its first eight entries are exactly §4.3's first eight rows, so
`--memories-scope first:8` is the faithful reading of "the eight that were finalised that day".

It is a parameter rather than a hardcoded eight because the choice IS the experiment. Injecting all
106 lessons means an observed improvement could have come from any of them (the cause is smeared);
injecting the eight matches the plan but discards the 98 written since. Both are defensible; a
script that quietly picks one turns a decision into a habit.

    --memories-scope all              every live lesson (the default: "what the harness knows now")
    --memories-scope none             empty (identical to arm C -- kept so the default is not magic)
    --memories-scope first:<k>        the first k entries of lessons.jsonl, in file order
    --memories-scope class:<abbr>     one class, e.g. `class:mh` or `class:measurement-hygiene`
    --memories-scope ids:<path>       an explicit list (one id per line, `#` comments allowed)

SELECTION USES THE RECORD; CONTENT USES THE PROJECTION. `lessons.jsonl` is the authority, so the
ids are chosen from it -- but the injected text comes from `harness_engine/memories/engine/`,
because that projection is the form the harness actually carries. A selected id with no memory
file is a HARD ERROR: silently skipping it would make "the filter is out of sync with the
projection" look exactly like "that lesson does not apply here".

WHY --reps DEFAULTS TO 3, AND WHY THE ARMS ARE INTERLEAVED
----------------------------------------------------------
PLAN §9's acceptance sentence is "at least one lesson is shown to improve things AND IS
REPRODUCIBLE". One sample per arm cannot establish reproducibility: a difference between two
single answers is indistinguishable from sampling noise. So the default is 3 reps per
(question, arm) -- the same "interleaved, >=3 rounds" standard that `eng-mh-0003` imposes on
throughput numbers, applied to the thing that measures the harness itself.

The reps are INTERLEAVED and the arm order ROTATES each round, so no arm is systematically first
or last in the sequence. That matters for the same reason it matters in `ab_interleave.py`: a
machine-side or server-side drift correlated with position would otherwise land entirely on one
arm. The actual per-round order is written to `manifest.json`.

Consequence: **a question whose arm disagrees with ITSELF across reps is not comparable at all.**
`compare.json` therefore reports arm stability first and marks a question `comparable: false` when
any arm is unstable, and the cross-arm verdicts for that question must be read as void. Reporting
"the lesson helped" on a question where one arm answers differently each time is the mistake this
column exists to prevent.

WHAT IS VERIFIABLE WITHOUT A MODEL, AND WHAT IS NOT
---------------------------------------------------
`--dry-run` needs no model: it resolves the scope (and prints the ids it selected), builds every
arm's prompt, prints sizes/sha256, and diffs each adjacent pair so you can SEE that each pair
isolates what it claims. That is the mechanical half.

The model half needs `CLOSED_LOOP_MODEL_CMD` -- a command that reads a prompt on stdin and writes
an answer on stdout. It has no default, on purpose: which model answers the questions IS the
experiment, and a script that silently picks one turns an experiment into a habit.

    python3 agent_harness/engine_loop/distill/closed_loop.py --dry-run
    python3 agent_harness/engine_loop/distill/closed_loop.py --dry-run --memories-scope first:8
    CLOSED_LOOP_MODEL_CMD='llm -m <model>' python3 .../closed_loop.py --memories-scope first:8

The model command is recorded in `manifest.json` as a sha256 ONLY. Manifests get committed; a
shell command line can carry an API key, and a committed key cannot be un-committed.

TWO ENVIRONMENT VARIABLES EXIST ONLY SO `closed_loop_selftest.py` CAN RUN
------------------------------------------------------------------------
`CLOSED_LOOP_OUT` redirects the run directory; `CLOSED_LOOP_MEM_DIR` redirects the projection that
scope ids are resolved against. Without them the selftest would have to write into the real
`closed_loop_out/` and would have to actually damage the memories directory to prove the
missing-file guard fires -- and a guard you can only test by breaking something is a guard that
stops getting tested.
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
LESSONS = ENGINE / "traces" / "lessons.jsonl"
MEM_DIR = Path(os.environ.get("CLOSED_LOOP_MEM_DIR")
               or (ENGINE / "harness_engine" / "memories" / "engine"))
QUESTIONS = HERE / "closed_loop_questions.md"
OUT_ROOT = Path(os.environ.get("CLOSED_LOOP_OUT") or (HERE / "closed_loop_out"))

# The amendment landed in this commit; its parent carries the pre-amendment charter.
D6_COMMIT = "e688f346e"

# class abbreviation -> the `class` value stored in lessons.jsonl (PLAN §7's table).
CLASS_ABBR = {
    "mh": "measurement-hygiene", "lf": "log-forensics", "diag": "diagnosis",
    "gate": "gate-integrity", "src": "source-reading", "bound": "honest-bounds",
    "perf": "performance", "smoke": "smoke",
}


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def nbytes(s: str) -> int:
    """Sizes are reported in BYTES, not characters.

    The charter and the memories are both mostly CJK, so a character count understates the payload
    by roughly a factor of three -- and a size label that measures something other than what it
    names is exactly the class of error this repo writes rules against.
    """
    return len(s.encode("utf-8"))


def rel(p: Path) -> str:
    """Display a path relative to the repo when it is inside it, absolute when it is not.

    `Path.relative_to` RAISES for a path outside the repo -- which is precisely the case when
    `closed_loop_selftest.py` redirects the projection or the output directory. The first version
    of the missing-file guard crashed while rendering its own message, so from the outside "the
    guard fired" and "the script is broken" were the same event: a traceback and a non-zero exit.
    A guard whose message cannot be delivered is not a guard.
    """
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def normalise(s: str) -> str:
    """Whitespace-insensitive comparison form.

    An exact byte comparison would call two answers different because a model re-wrapped a line,
    and every one of those false differences costs a human review. Collapsing whitespace is the
    cheapest normalisation that cannot hide a change of MEANING; both forms are recorded so the
    looser one never silently substitutes for the stricter one.
    """
    return re.sub(r"\s+", " ", s).strip()


def charter_at(ref: str) -> str:
    """A committed charter, read from git. Never cached to disk."""
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:agent_harness/CONVENTIONS.md"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot read CONVENTIONS.md at {ref}: {r.stderr.strip()[:200]}\n"
                         f"  The comparison NEEDS both charters. Do not substitute a cached copy.")
    return r.stdout


def load_lessons() -> list[dict]:
    if not LESSONS.is_file():
        raise SystemExit(f"missing {rel(LESSONS)} -- the lesson set has no authority "
                         f"without it; refusing to select from the derived copies")
    return [json.loads(l) for l in LESSONS.read_text(encoding="utf-8").splitlines() if l.strip()]


def resolve_scope(spec: str) -> tuple[list[str], str]:
    """`--memories-scope` value -> (ordered lesson_ids, human-readable description of the choice).

    Order is preserved from the file because `first:<k>` means "the k that were written first",
    and that ordering is a fact about history rather than about the id numbers (the first eight
    span six different classes, so sorting by id would silently select a different set).
    """
    spec = spec.strip()
    lessons = load_lessons()
    live = [l for l in lessons if not l.get("superseded_by")]
    n_sup = len(lessons) - len(live)

    if spec == "none":
        return [], f"none -- no lessons injected ({n_sup} superseded records ignored)"
    if spec == "all":
        return [l["lesson_id"] for l in live], f"all -- {len(live)} live lessons ({n_sup} superseded)"

    if spec.startswith("first:"):
        raw = spec.split(":", 1)[1].strip()
        if not raw.isdigit() or int(raw) <= 0:
            raise SystemExit(f"--memories-scope first:<k> needs a positive integer, got {raw!r}")
        k = int(raw)
        if k > len(lessons):
            raise SystemExit(f"--memories-scope first:{k} but lessons.jsonl holds only {len(lessons)}")
        head = lessons[:k]
        sel = [l["lesson_id"] for l in head if not l.get("superseded_by")]
        note = f"first:{k} -- {len(sel)} lessons, in file order (the k-th is {head[-1]['lesson_id']})"
        if len(sel) != len(head):
            note += f"; {len(head) - len(sel)} of them already superseded and therefore excluded"
        return sel, note

    if spec.startswith("class:"):
        key = spec.split(":", 1)[1].strip()
        cls = CLASS_ABBR.get(key, key)
        if cls not in set(CLASS_ABBR.values()):
            raise SystemExit(f"unknown class {key!r}. known: "
                             + ", ".join(sorted(CLASS_ABBR)) + " (or a full name)")
        sel = [l["lesson_id"] for l in live if l["class"] == cls]
        if not sel:
            raise SystemExit(f"class {cls!r} has no live lesson -- refusing to run an empty arm")
        return sel, f"class:{key} -- {len(sel)} lessons of class {cls}"

    if spec.startswith("ids:"):
        p = Path(spec.split(":", 1)[1].strip())
        if not p.is_file():
            raise SystemExit(f"--memories-scope ids:<path>: no such file: {p}")
        wanted = [l.split("#", 1)[0].strip() for l in p.read_text(encoding="utf-8").splitlines()]
        wanted = [x for x in wanted if x]
        if not wanted:
            raise SystemExit(f"{p} lists no ids -- refusing to run an empty arm")
        known = {l["lesson_id"] for l in lessons}
        unknown = [x for x in wanted if x not in known]
        if unknown:
            raise SystemExit(f"{p} names ids that are not in lessons.jsonl: {unknown[:5]}"
                             + (" ..." if len(unknown) > 5 else "")
                             + "\n  A typo must fail here, not quietly shrink the arm.")
        superseded = [x for x in wanted if x not in {l["lesson_id"] for l in live}]
        kept = [x for x in wanted if x not in set(superseded)]
        note = f"ids:{p} -- {len(kept)} requested"
        if superseded:
            note += f"; {len(superseded)} already superseded and therefore excluded: {superseded}"
        return kept, note

    raise SystemExit(
        f"unrecognised --memories-scope: {spec!r}\n"
        f"  forms: all | none | first:<k> | class:<abbr|full-name> | ids:<path>")


def memories_block(ids: list[str]) -> str:
    """The injectable block for the selected ids. Hard-errors on a projection gap."""
    if not ids:
        return ""
    missing = [i for i in ids if not (MEM_DIR / f"{i}.md").is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} selected lesson(s) have no memory file under "
            f"{rel(MEM_DIR)}: {missing[:5]}" + (" ..." if len(missing) > 5 else "")
            + "\n  Run `harness_engine/build_memories.py` first. Refusing to run an arm that "
              "silently drops lessons: a filter out of sync with the projection is "
              "indistinguishable from 'those lessons do not apply'.")
    parts = ["", "## 這個 scope 已累積的規訓（harness state：每一條都是一個 lesson）", ""]
    for i in ids:
        first = (MEM_DIR / f"{i}.md").read_text(encoding="utf-8").splitlines()[0]
        parts.append(f"- {first}")
    return "\n".join(parts)


def build_prompt(charter: str, mems: str, q: str) -> str:
    return (charter + mems + "\n\n---\n\n"
            "## 現在要回答的問題\n\n"
            f"{q}\n\n"
            "**只回答「下一個具體動作」以及你據以判斷的欄位／判準。不要複述問題，不要客套。**\n"
            "若你認為這個問題在現有證據下無法回答，就明說「無法回答」並指出缺什麼。\n")


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


def _unified(a: str, b: str, la: str = "a", lb: str = "b") -> str:
    import difflib
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), la, lb, n=0))


def _diff_stat(a: str, b: str) -> tuple[int, int]:
    u = _unified(a, b).splitlines()
    return (sum(1 for l in u if l.startswith("+") and not l.startswith("+++")),
            sum(1 for l in u if l.startswith("-") and not l.startswith("---")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the scope, build every arm's prompt, print sizes/sha256 and the "
                         "adjacent diffs; no model")
    ap.add_argument("--only", default=None, help="run a single question id, e.g. Q2")
    ap.add_argument("--memories-scope", default="all",
                    help="which lessons arm D injects: all | none | first:<k> | class:<abbr> | ids:<path>")
    ap.add_argument("--reps", type=int, default=3,
                    help="calls per (question, arm), interleaved with a rotating arm order (default 3; "
                         "1 cannot establish the reproducibility PLAN §9 asks for)")
    args = ap.parse_args()

    if args.reps < 1:
        raise SystemExit(f"--reps must be >= 1, got {args.reps}")

    live = CHARTER.read_text(encoding="utf-8")
    pre = charter_at(f"{D6_COMMIT}^")
    post_d6 = charter_at(D6_COMMIT)

    scope_ids, scope_note = resolve_scope(args.memories_scope)
    mems = memories_block(scope_ids)

    qs = load_questions()
    if args.only:
        qs = [q for q in qs if q[0] == args.only]
        if not qs:
            raise SystemExit(f"no such question: {args.only}")

    arms = [
        ("A_preD6", pre, ""),
        ("B_postD6", post_d6, ""),
        ("C_head", live, ""),
        ("D_head_mem", live, mems),
    ]
    pairs = [("A_preD6", "B_postD6", "D6 修訂（欠帳要回答的就是這一對）"),
             ("B_postD6", "C_head", "D6 之後落地的其他改動（含 E1 的 D1 更新）"),
             ("C_head", "D_head_mem", "注入 harness_engine 的 lesson（E3 要回答的是這一對）")]

    print(f"  scope: {scope_note}")
    if scope_ids:
        print(f"    ids: {', '.join(scope_ids[:10])}" + (" ..." if len(scope_ids) > 10 else ""))
    print(f"  questions: {len(qs)}  ({', '.join(q for q, _ in qs)})")
    print(f"  reps: {args.reps}  -> total model calls = {len(qs) * len(arms) * args.reps}")
    if args.reps == 1:
        print("    ! reps=1 只能證明「有不一樣」，不能證明「可重現」——PLAN §9 的驗收要求後者。")
    for name, ch, mm in arms:
        sample = build_prompt(ch, mm, "Q?")
        print(f"  arm {name:12s} charter={nbytes(ch):7d} B sha256[:16]={sha16(ch)}"
              f"  memories={nbytes(mm):7d} B  prompt≈{nbytes(sample):7d} B")

    print("  --- 每一對只隔離一件事（這一節是儀器自檢；第一版就是在這裡被抓到沒有隔離）---")
    by_name = {n: c for n, c, _ in arms}
    mems_by_name = {n: m for n, _, m in arms}
    leaks: list[str] = []
    for a, b, what in pairs:
        ca, cb = by_name[a], by_name[b]
        ma, mb = mems_by_name[a], mems_by_name[b]
        add, rem = _diff_stat(ca, cb)
        d_ch, d_mem = nbytes(cb) - nbytes(ca), nbytes(mb) - nbytes(ma)
        print(f"    {a:11s} -> {b:11s}  charter +{add:3d}/-{rem:3d} 行 ({d_ch:+6d} B)  "
              f"memories {d_mem:+7d} B   {what}")
        # The reason this section exists: version 1 did not isolate and looked entirely normal.
        # "Two things moved at once" has to be named HERE, not left for someone to notice in a diff.
        if d_ch and d_mem:
            leaks.append(f"{a} -> {b} 同時動了 charter（{d_ch:+d} B）與 memories（{d_mem:+d} B）")
        if not d_ch and not d_mem:
            leaks.append(f"{a} -> {b} 兩邊完全一樣 ⇒ 這一對不是在比較，是在比對自己")

    if args.dry_run:
        print()
        print("  --- A vs B 的實際內容（前 6 行）---")
        a_only = [l for l in _unified(by_name["A_preD6"], by_name["B_postD6"]).splitlines()
                  if l.startswith("+") and not l.startswith("+++")]
        for l in a_only[:6]:
            print(f"    B 新增: {l[1:][:110]}")
        print(f"    ... 共 {len(a_only)} 行新增")
        print()
        print(f"  --- 隔離自檢：{len(pairs)} 對，"
              + ("每一對都只隔離一件事 [ok]" if not leaks else f"{len(leaks)} 對沒有隔離 [FAIL]") + " ---")
        for l in leaks:
            print(f"    !! {l}")
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
    arm_order = [n for n, _, _ in arms]
    rows: list[dict] = []
    for rep in range(1, args.reps + 1):
        # Rotate so no arm is systematically first or last across the run. Interleaving is what
        # makes "only one thing changed" true; rotation removes position as a candidate cause.
        shift = (rep - 1) % len(arm_order)
        order = arm_order[shift:] + arm_order[:shift]
        for qid, q in qs:
            for name in order:
                ch, mm = by_name[name], mems_by_name[name]
                prompt = build_prompt(ch, mm, q)
                r = subprocess.run(shlex.split(cmd), input=prompt, capture_output=True, text=True)
                rows.append({
                    "question_id": qid, "arm": name, "rep": rep,
                    "rc": r.returncode,
                    "answer": (r.stdout or "").strip() or None,
                    "normalised": normalise(r.stdout or ""),
                    "stderr_tail": (r.stderr or "")[-300:] or None,
                    "prompt_bytes": nbytes(prompt), "prompt_sha256_16": sha16(prompt),
                    "charter_sha256_16": sha16(ch), "memories_bytes": nbytes(mm),
                })
                print(f"    rep{rep} {qid} {name:12s} rc={r.returncode} answer={len(r.stdout or '')} B")
    (out / "answers.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

    # --- the comparison: stability first, then (only where stable) cross-arm identity ---
    per_q: dict[str, dict[str, list[str]]] = {}
    for x in rows:
        per_q.setdefault(x["question_id"], {}).setdefault(x["arm"], []).append(x["normalised"])

    compare = []
    for qid in sorted(per_q):
        stab = {}
        for n in arm_order:
            ans = per_q[qid].get(n, [])
            stab[n] = {"n": len(ans), "distinct": len(set(ans)), "stable": len(set(ans)) == 1}
        comparable = all(stab[n]["stable"] for n in arm_order)

        def pick(n: str) -> str:
            ans = per_q[qid].get(n, [])
            return ans[0] if ans and stab[n]["stable"] else ""

        a, b, c, d = pick("A_preD6"), pick("B_postD6"), pick("C_head"), pick("D_head_mem")
        rec = {
            "question_id": qid,
            "arm_stability": stab,
            "comparable": comparable,
            "note": None if comparable else
                    "至少一個臂自己前後不一致 ⇒ 這一題的跨臂判斷無效（先看是哪個臂、"
                    "以及不一致的是不是『承重點』本身）",
            "A_vs_B_identical": (a == b) if comparable else None,
            "A_vs_B_verdict": (("D6 修訂在這一題上是惰性的" if a == b else "D6 修訂改變了這一題的答案")
                                if comparable else "無法判斷（臂內不穩定）"),
            "B_vs_C_identical": (b == c) if comparable else None,
            "C_vs_D_identical": (c == d) if comparable else None,
            "C_vs_D_verdict": (("注入 lesson 在這一題上沒有改變答案" if c == d
                                else "注入 lesson 改變了這一題的答案")
                               if comparable else "無法判斷（臂內不穩定）"),
        }
        compare.append(rec)
    (out / "compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "ts": ts,
        "scope_spec": args.memories_scope,
        "scope_description": scope_note,
        "scope_ids": scope_ids,
        "n_lessons_injected": len(scope_ids),
        "memories_block_bytes": nbytes(mems),
        "reps": args.reps,
        "questions": [q for q, _ in qs],
        "arms": arm_order,
        "arm_order_by_rep": {r: (arm_order[(r - 1) % len(arm_order):] + arm_order[:(r - 1) % len(arm_order)])
                             for r in range(1, args.reps + 1)},
        "calls": len(rows),
        "charter_sha256_16": {"A_preD6": sha16(pre), "B_postD6": sha16(post_d6), "C_head": sha16(live)},
        "memories_block_sha256_16": sha16(mems),
        # sha256 of the command, never the command: a manifest gets committed and a command line
        # can carry an API key.
        "model_cmd_sha256_16": sha16(cmd),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    n_unstable = sum(1 for c in compare if not c["comparable"])
    if n_unstable:
        print(f"  ! {n_unstable}/{len(compare)} 題有臂內不一致 ⇒ 這些題的跨臂判斷無效")
    comp = [c for c in compare if c["comparable"]]
    if comp:
        n_ab = sum(1 for c in comp if c["A_vs_B_identical"])
        n_cd = sum(1 for c in comp if c["C_vs_D_identical"])
        print(f"  可比較的 {len(comp)} 題中：")
        print(f"    A vs B（D6 本體）      : {n_ab}/{len(comp)} 題答案相同 -> "
              f"{'這個問題集上找不到 D6 的行為效應' if n_ab == len(comp) else '有差異，逐題看 compare.json'}")
        print(f"    C vs D（注入 lesson，E3）: {n_cd}/{len(comp)} 題相同（反歸因用；E3 要的是 "
              f"{len(comp) - n_cd} 題中的『改善』，不是『不同』）")
    print(f"  out: {rel(out)}")
    print("  注意：字串相同只是自動部分。逐題的承重點在 closed_loop_questions.md 的表格裡，"
          "要人工核對——兩份答案可以字面不同而判準相同，也可以字面相同而都漏掉承重點。")
    print("  ★ 「改善」不是「不同」：C vs D 不同只說明 lesson 有影響力，"
          "有沒有改善要看 closed_loop_questions.md 每題列出的承重點有沒有被答到。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
