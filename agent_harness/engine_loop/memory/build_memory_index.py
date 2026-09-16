#!/usr/bin/env python3
"""Index the project memory so the engine loop can actually use it.

The problem this solves. The project memory lives in `.workbuddy/memory/` (one `MEMORY.md` plus a
daily log per working day). It is the densest record this project has — and it is also the least
usable, because it is a single 1000-line file that grows all day: an agent that wants "everything we
know about the mmid assert" has to read the whole thing, which is exactly the kind of read that gets
truncated, and a truncated read is indistinguishable from "that was never written down".

So this script emits a *section-level* index, not a copy:

    agent_harness/engine_loop/memory/INDEX.jsonl      one file row + one row per `##` section

A section row carries the source path, the heading, the line range and the `###` subheadings, so a
consumer can go straight to `.workbuddy/memory/2026-09-15.md:395-435` instead of reading 1026 lines.
`--query` does that lookup for you.

WHY AN INDEX AND NOT A MIRROR. The memory files are written by the host (other WorkBuddy sessions
append to them while this loop runs). A copy under `agent_harness/` would be a second source of
truth for the same bytes, and the failure mode of a second source of truth is silent: the copy keeps
looking authoritative after the original moved on. The project already has this rule for
`scripts/check/*` (index them, never copy them). So the canonical files stay where the host writes
them, this directory holds only derived data, and `--check` re-derives it and fails on drift.

2026-09-16 AMENDMENT (CONVENTIONS.md D6). The rule is about what the loop READS, and it still holds:
this index is how a consumer reaches the memory, and the canonical files stay where the host writes
them. But an index ships *pointers*, not *content*, so there is one consumer it cannot serve at all --
the periodic pusher on the other machine (`agent_harness/scripts/auto_git_push.ps1`, which does
`git add agent_harness` and pushes). A dated, banner-marked snapshot therefore also exists at
`agent_harness/memory/`, with provenance in `SNAPSHOT.jsonl`. This script does not read it and
`--check` does not verify it: if you want a FACT, read `.workbuddy/memory/`. The snapshot exists to
be transported, not to be quoted.

    build_memory_index.py                    regenerate INDEX.jsonl
    build_memory_index.py --check            re-derive and compare; exit 1 on drift
    build_memory_index.py --query mmid -n 8  sections matching a term, best first
    build_memory_index.py --query mmid --full  ... with the matching lines, not just the range
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

assert os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "memory", \
    "build_memory_index.py belongs in agent_harness/engine_loop/memory/"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MEM_DIR = os.path.join(REPO, ".workbuddy", "memory")
DEFAULT_OUT = os.path.join(HERE, "INDEX.jsonl")

HEAD_RE = re.compile(r"^(#{1,3})\s+(.*?)\s*$")


def memory_files() -> list[str]:
    """Every markdown file under .workbuddy/memory/, newest first. Missing directory is not an
    error: the memory root is owned by the host and does not exist in a fresh clone."""
    if not os.path.isdir(MEM_DIR):
        return []
    out = [os.path.join(MEM_DIR, f) for f in os.listdir(MEM_DIR) if f.endswith(".md")]
    out.sort(key=lambda p: os.path.basename(p), reverse=True)
    return out


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for abs_p in memory_files():
        rel = os.path.relpath(abs_p, REPO)
        with open(abs_p, "rb") as fh:
            raw = fh.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        rows.append({
            "row": "file",
            "path": rel,
            "abs": abs_p,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()[:16],
            "lines": len(lines),
            "mtime": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(abs_p))),
        })

        # Section = a `##` heading plus everything up to the next `##` (or EOF). `#` (the file
        # title) and `###` are deliberately not boundaries: `#` occurs once, and subheadings are
        # carried on the parent row so a section stays a unit a reader can consume in one go.
        heads = [(i, len(m.group(1)), m.group(2))
                 for i, line in enumerate(lines)
                 if (m := HEAD_RE.match(line))]
        for n, (i, level, title) in enumerate(heads):
            if level != 2:
                continue
            end = len(lines)
            for j, lv, _ in heads[n + 1:]:
                if lv <= 2:
                    end = j
                    break
            subs = [t for k, lv, t in heads[n + 1:] if lv == 3 and k < end]
            body = "\n".join(lines[i + 1:end])
            rows.append({
                "row": "section",
                "path": rel,
                "level": 2,
                "title": title,
                "line_start": i + 1,          # 1-based, so it can go straight to Read(offset=)
                "line_end": end,              # exclusive
                "lines": end - i - 1,
                "body_bytes": len(body.encode("utf-8")),
                "subs": subs,
            })
    return rows


def cmd_check(out: str) -> int:
    """Re-derive and compare. `--check` has to be able to fail -- a check that only reports what it
    finds is not a check (CONVENTIONS.md B7). Bytes are compared, not existence: the first version of
    index_assets.py --check only asked whether the file was still there and stayed green while the
    file changed under it."""
    if not os.path.exists(out):
        print(f"  [error] no index at {out}")
        return 1
    have = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    want = build_rows()
    hk = [(r["row"], r.get("path"), r.get("title")) for r in have]
    wk = [(r["row"], r.get("path"), r.get("title")) for r in want]
    problems = []
    if len(have) != len(want):
        problems.append(f"row count: index {len(have)} vs derived {len(want)}")
    for a, b in zip(have, want):
        if a != b:
            key = f"{b.get('path')} :: {b.get('title', '(file row)')}"
            differing = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
            problems.append(f"{key}: fields differ {sorted(differing)}")
    if problems:
        print(f"  [error] {len(problems)} drift(s) between INDEX.jsonl and .workbuddy/memory/:")
        for p in problems[:20]:
            print(f"    {p}")
        print("    regenerate with: python3 memory/build_memory_index.py")
        return 1
    n_sec = sum(1 for r in want if r["row"] == "section")
    print(f"  memory index OK: {len(want) - n_sec} file(s), {n_sec} section(s), no drift")
    return 0


def cmd_query(out: str, term: str, limit: int, full: bool) -> int:
    """Rank sections by how often the term occurs, title hits weighted. This is the part that makes
    the index worth having: it answers 'where in 1000 lines is this' without reading any of them."""
    if not os.path.exists(out):
        print(f"  [error] no index at {out}; run build_memory_index.py first", file=sys.stderr)
        return 1
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    t = term.lower()
    hits = []
    for r in rows:
        if r["row"] == "file":
            continue
        abs_p = os.path.join(REPO, r["path"])
        try:
            body = open(abs_p, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        # The index stores ranges, not bodies, so the count is recomputed here. That keeps the
        # index small and means a query can never report a hit in text the file no longer has.
        seg = body[r["line_start"]:r["line_end"]]
        n_body = sum(line.lower().count(t) for line in seg)
        n_title = r["title"].lower().count(t) + sum(s.lower().count(t) for s in r["subs"])
        if n_body or n_title:
            hits.append((n_body + 5 * n_title, n_body, r, seg))
    if not hits:
        print(f"  no section mentions {term!r}")
        return 0
    hits.sort(key=lambda h: -h[0])
    shown = hits[:limit]
    print(f"  {len(hits)} section(s) mention {term!r}; showing {len(shown)}")
    for score, n_body, r, seg in shown:
        print(f"\n  {r['path']}:{r['line_start']}-{r['line_end']}  ({n_body} hit(s), {r['lines']} lines)")
        print(f"    ## {r['title']}")
        for s in r["subs"]:
            print(f"       ### {s}")
        if full:
            for k, line in enumerate(seg):
                if t in line.lower():
                    print(f"    {r['line_start'] + k:>5}| {line[:160]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--query", metavar="TERM")
    ap.add_argument("-n", "--limit", type=int, default=10)
    ap.add_argument("--full", action="store_true",
                    help="with --query, also print the matching lines")
    args = ap.parse_args()

    if args.check:
        return cmd_check(args.out)
    if args.query:
        return cmd_query(args.out, args.query, args.limit, args.full)

    rows = build_rows()
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_file = sum(1 for r in rows if r["row"] == "file")
    n_sec = len(rows) - n_file
    print(f"  wrote {n_file} file(s) + {n_sec} section(s) -> {args.out}")
    if not rows:
        print(f"  [warn] no memory files under {MEM_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
