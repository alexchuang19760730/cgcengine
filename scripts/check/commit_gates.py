#!/usr/bin/env python3
"""Every commit must declare where it stands against the decode-25 gate chain.

THE RULE (user, 2026-09-19 22:17)
  每一次 commit 都要列出跟 gate 的關係，放在狀態裡，格式 `met` / `not met (improve xxx)`.

WHY IT NEEDS A MACHINE. `targets.json` says an objective without an acceptance criterion
"在 portal 上與『永遠進行中』同形". A commit without a gate line has the same shape one level down:
"this work moved the plan" and "this work moved nothing" look identical in `git log`, and the only
way to tell them apart is to re-read a memory file. The repo's own convention is explicit --
**缺口從來不是「沒有檔案」，是沉默** -- so silence must be an error, and "none" must be an
explicit, reasoned act rather than an omission.

WHAT IT ACCEPTS (a `Gates:` block, normally at the end of the commit message; continuation lines
are indented and the block ends at the first non-indented, non-item line)

    Gates: none (docs-only, touches no measured metric)
    Gates: G1 not-met (improve 25.0% -> 12.4%)
    Gates:
      G3 met (union pair L4/L5 < 2x a single layer)
      G4 not-met (blocked-by-G3, unchanged)

RULES
  1. a `Gates:` block must be present                         -> MISSING
  2. either `none (<reason>)` alone, or >= 1 gate entries    -> EMPTY / BAD_NONE
  3. every id must exist in targets.json (all targets)       -> UNKNOWN_GATE
  4. status is `met` or `not-met` (`not met` accepted)       -> BAD_STATUS
  5. `not-met` must carry `(improve ...)`                    -> NO_IMPROVE

USAGE
  python3 scripts/check/commit_gates.py --last                 # check HEAD's message
  python3 scripts/check/commit_gates.py --message-file .git/COMMIT_EDITMSG   # hook mode
  python3 scripts/check/commit_gates.py --message "Gates: none (docs-only)"
  python3 scripts/check/commit_gates.py --list                 # print the known gate ids
  python3 scripts/check/commit_gates.py --install-hook         # install a commit-msg hook
  python3 scripts/check/commit_gates.py --selftest
"""

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGETS = os.path.join(ROOT, "agent_harness", "portal", "targets.json")

RE_GATES_LINE = re.compile(r"^Gates:[ \t]*(?P<inline>.*)$")
RE_ITEM = re.compile(r"^[ \t]+(?P<body>\S.*)$")
RE_ENTRY = re.compile(r"^(?P<id>G\d+)[ \t]+(?P<status>met|not[-\s]?met)\b(?P<rest>.*)$")
RE_NONE = re.compile(r"^none\b(?P<rest>.*)$", re.IGNORECASE)
RE_IMPROVE = re.compile(r"\((?P<body>[^)]*improve[^)]*)\)", re.IGNORECASE)

STATUS_OK = {"met", "not-met", "notmet"}


def known_gates(path=TARGETS):
    """Every gate id declared anywhere in targets.json (any target, chain entries only)."""
    ids = {}
    try:
        doc = json.load(open(path, encoding="utf-8"))
    except Exception:
        return ids
    for t in doc.get("targets", []):
        for g in (t.get("gates") or {}).get("chain", []):
            if isinstance(g, dict) and g.get("id"):
                ids[g["id"]] = "%s / %s" % (t.get("id", "?"), g.get("what", ""))
    return ids


def parse_block(message):
    """-> (found, inline, items). The block runs from `Gates:` to the first non-continuation line."""
    lines = message.splitlines()
    for i, ln in enumerate(lines):
        m = RE_GATES_LINE.match(ln)
        if not m:
            continue
        inline = m.group("inline").strip()
        items = []
        for follow in lines[i + 1:]:
            if follow.strip() == "":
                break
            mi = RE_ITEM.match(follow)
            if not mi:
                break
            body = mi.group("body").strip()
            if body.startswith(("-", "*")):
                body = body[1:].strip()
            items.append(body)
        return True, inline, items
    return False, "", []


def check(message, ids):
    """-> list of (code, detail). Empty list == pass."""
    problems = []
    found, inline, items = parse_block(message)
    if not found:
        return [("MISSING", "no `Gates:` block in the commit message")]

    entries = ([inline] if inline else []) + items
    entries = [e for e in entries if e.strip()]
    if not entries:
        return [("EMPTY", "`Gates:` present but lists nothing (use `none (<reason>)` if unrelated)")]

    if len(entries) == 1 and RE_NONE.match(entries[0]):
        rest = RE_NONE.match(entries[0]).group("rest")
        if not re.search(r"\(.+\)", rest):
            problems.append(("BAD_NONE", "`none` needs a reason: `none (<why it touches no gate>)`"))
        return problems

    for e in entries:
        if RE_NONE.match(e):
            problems.append(("BAD_NONE", "`none` cannot be combined with gate entries: %r" % e))
            continue
        m = RE_ENTRY.match(e)
        if not m:
            problems.append(("BAD_ENTRY", "not `G<id> met` or `G<id> not-met (...)`: %r" % e))
            continue
        gid = m.group("id")
        status = m.group("status").replace(" ", "-").replace("--", "-")
        if gid not in ids:
            problems.append(("UNKNOWN_GATE", "%s is not declared in targets.json (known: %s)"
                             % (gid, ",".join(sorted(ids)) or "none")))
        if status not in STATUS_OK:
            problems.append(("BAD_STATUS", "%s has status %r (want met / not-met)" % (gid, m.group("status"))))
        if status != "met" and not RE_IMPROVE.search(m.group("rest")):
            problems.append(("NO_IMPROVE", "%s is not-met and must carry `(improve ...)`" % gid))
    return problems


# --------------------------------------------------------------------------- selftest
def cmd_selftest(_args):
    ids = {"G0": "x", "G1": "x", "G3": "x", "G4": "x"}
    cases = [
        ("Gates: none (docs-only, touches no measured metric)", True, "explicit none"),
        ("Gates: G1 not-met (improve 25.0% -> 12.4%)", True, "single not-met with improve"),
        ("Gates:\n  G3 met\n  G4 not-met (improve 104.6 -> 98.0 ms, blocked-by-G3)", True, "block form"),
        ("Gates: G3 not met (improve 2.81 -> 2.50 ms)", True, "`not met` spelled with a space"),
        ("nothing here at all", False, "missing block"),
        ("Gates: none", False, "`none` without a reason"),
        ("Gates: G9 met", False, "unknown gate id"),
        ("Gates: G1 not-met", False, "not-met without (improve ...)"),
        ("Gates: G3 maybe", False, "bad status"),
        ("Gates: G3 met\n\nbody continues here", True, "block ends at the blank line"),
        ("Gates: none (no metric touched)\n\nmore body", True, "inline none then body"),
    ]
    fails = 0
    for msg, want_ok, label in cases:
        got = check(msg, ids)
        ok = (not got) == want_ok
        print("  %s  %-42s %s" % ("PASS" if ok else "FAIL", label,
                                  "" if ok else "-> %s" % got))
        if not ok:
            fails += 1
    # the real tree must declare the chain we are checking against
    real = known_gates()
    ok = len(real) >= 8
    print("  %s  %-42s %s" % ("PASS" if ok else "FAIL",
                              "targets.json declares >= 8 gates", "(%d found)" % len(real)))
    fails += 0 if ok else 1
    print()
    print("%d failed" % fails if fails else "all selftest cases behaved")
    return 1 if fails else 0


HOOK = """#!/bin/bash
# installed by scripts/check/commit_gates.py --install-hook
# Enforces: every commit declares its standing against the gate chain in targets.json.
set -u
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
exec python3 "$REPO_ROOT/scripts/check/commit_gates.py" --message-file "$1"
"""


def cmd_install_hook(_args):
    gd = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT,
                        capture_output=True, text=True).stdout.strip()
    if not gd:
        print("ERROR: cannot resolve the git common dir", file=sys.stderr)
        return 2
    gd = gd if os.path.isabs(gd) else os.path.join(ROOT, gd)
    path = os.path.join(gd, "hooks", "commit-msg")
    if os.path.exists(path):
        print("REFUSING: %s already exists -- move it aside first (another session may own it)" % path)
        return 2
    open(path, "w").write(HOOK)
    os.chmod(path, 0o755)
    print("installed %s" % path)
    print("  NOTE: this directory is shared by every worktree of this repo, so this hook now runs")
    print("        for all of them. Remove the file to uninstall.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--message", default=None, help="the commit message itself")
    g.add_argument("--message-file", default=None, help="file holding the commit message (hook mode)")
    g.add_argument("--last", action="store_true", help="use HEAD's commit message")
    ap.add_argument("--list", action="store_true", help="print known gate ids and exit")
    ap.add_argument("--install-hook", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return cmd_selftest(args)
    if args.install_hook:
        return cmd_install_hook(args)

    ids = known_gates()
    if args.list:
        for gid in sorted(ids):
            print("%-4s %s" % (gid, ids[gid]))
        print("(%d gates; source: %s)" % (len(ids), os.path.relpath(TARGETS, ROOT)))
        return 0
    if not ids:
        print("ERROR: no gates found in %s -- cannot check anything" % TARGETS, file=sys.stderr)
        return 2

    if args.message is not None:
        msg = args.message
    elif args.message_file:
        try:
            msg = open(args.message_file, encoding="utf-8", errors="replace").read()
        except OSError as e:
            print("ERROR: cannot read %s: %s" % (args.message_file, e), file=sys.stderr)
            return 2
    elif args.last:
        msg = subprocess.run(["git", "log", "-1", "--format=%B"], cwd=ROOT,
                             capture_output=True, text=True).stdout
    else:
        print("ERROR: give one of --message / --message-file / --last (or --selftest)",
              file=sys.stderr)
        return 2

    problems = check(msg, ids)
    if not problems:
        _found, inline, items = parse_block(msg)
        summary = "; ".join(([inline] if inline else []) + items)
        print("commit-gates: OK  %s" % summary[:120])
        return 0
    print("commit-gates: FAILED -- the message must state its standing against the gate chain.",
          file=sys.stderr)
    print("  Format:  Gates: none (<why it touches no gate>)", file=sys.stderr)
    print("        or  Gates: G3 met; G4 not-met (improve 2.81 -> 2.50 ms)", file=sys.stderr)
    for code, detail in problems:
        print("  [%s] %s" % (code, detail), file=sys.stderr)
    print("  Known gates: %s" % ", ".join(sorted(ids)), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
