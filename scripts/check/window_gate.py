#!/usr/bin/env python3
"""Commit gate: the number of UNGATED server launchers may only go down.

Why this one is hermetic
------------------------
The repo's `precommit_e2e_gate.sh` deliberately is not wired into the hook (`00b70bebc`), and it
needs a live server. This gate needs neither: it reads files, so it costs milliseconds and can run
on every commit without anyone being tempted to `--no-verify` it away.

What it enforces
----------------
`server_window.py audit` finds every script that launches a server and whether it asks the shared
probe first. A launch into a busy box produces a number that describes the neighbour -- measured
spread on identical configs was 17%, and 2.5x on a bad day. That failure is invisible in the number
itself, so it has to be caught in the code.

Rule: no NEW ungated launcher, and the count must not rise. Equal is fine (the remaining ones are
held by another line, see `docs/SERVER_WINDOW_LEDGER_2026-09-19.md`), fewer is fine and is reported
as progress. `--update` re-baselines deliberately; it is a separate command so it cannot happen by
accident inside a commit.

Usage
-----
    python3 scripts/check/window_gate.py check     # exit 0 pass / 1 regression
    python3 scripts/check/window_gate.py update    # re-baseline from the real audit
    python3 scripts/check/window_gate.py selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check"
BASELINE = CHECK / "window_gate_baseline.json"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CHECK / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audit_now() -> dict:
    return _load("sw", "server_window.py").audit()


def ungated_names(rep: dict) -> list[str]:
    return sorted(r["file"] for r in rep.get("ungated", []))


def baseline_doc(rep: dict) -> dict:
    """Written FROM the audit, never typed by hand.

    Hand-typed number blocks are a documented defect on this line (a whitepaper's cross-table was
    typed and wrong), so the baseline is generated and carries the full list, not just a count: a
    count alone cannot tell "someone gated one file and added another" from "nothing changed".
    """
    import time
    return {"note": "ungated server launchers; may only shrink. Regenerate with `update`.",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ungated": ungated_names(rep),
            "gated_count": len(rep.get("gated", [])),
            "launchers": len(rep.get("launchers", []))}


def check(baseline: dict | None = None, rep: dict | None = None) -> tuple[bool, list[str]]:
    """Pure: (ok, messages). A new ungated file or a rising count fails; fewer passes."""
    rep = rep if rep is not None else audit_now()
    baseline = baseline if baseline is not None else load_baseline()
    now = ungated_names(rep)
    was = sorted(baseline.get("ungated", []))
    msgs = []
    added = [f for f in now if f not in was]
    removed = [f for f in was if f not in now]
    ok = not added and len(now) <= len(was)
    if added:
        msgs.append(f"REGRESSION: {len(added)} newly ungated launcher(s): {added}. Each one can "
                    f"measure the neighbour on a busy box. Add a `require_first` call "
                    f"(docs/SERVER_WINDOW_LEDGER_2026-09-19.md) or re-baseline deliberately.")
    if len(now) > len(was):
        msgs.append(f"REGRESSION: ungated count rose {len(was)} -> {len(now)}")
    if removed:
        msgs.append(f"PROGRESS: {len(removed)} launcher(s) now gated: {removed}. "
                    f"Run `update` to lower the baseline so this cannot be given back.")
    if ok and not removed:
        msgs.append(f"ok: {len(now)} ungated launcher(s), unchanged "
                    f"(gated: {len(rep.get('gated', []))}, total: {len(rep.get('launchers', []))})")
    if baseline.get("ungated") and not baseline.get("generated"):
        msgs.append("[baseline] no generation timestamp: it may be hand-typed, which this line has "
                    "been burned by. Re-run `update`.")
    return ok, msgs


def load_baseline() -> dict:
    if not BASELINE.exists():
        return {"ungated": [], "generated": None}
    try:
        return json.loads(BASELINE.read_text())
    except Exception:
        return {"ungated": [], "generated": None}


def cmd_check() -> int:
    bl = load_baseline()
    if not BASELINE.exists():
        print(f"no baseline at {BASELINE}: run `update` once, deliberately")
        return 1
    ok, msgs = check(bl)
    for m in msgs:
        print(m)
    return 0 if ok else 1


def cmd_update() -> int:
    doc = baseline_doc(audit_now())
    BASELINE.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"baseline written: {BASELINE}")
    print(f"  ungated ({len(doc['ungated'])}): {doc['ungated']}")
    return 0


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            fails.append(name)

    def rep(names):
        return {"ungated": [{"file": n} for n in names], "gated": [{"file": "a.py"}] * 3,
                "launchers": [{"file": n} for n in names] + [{"file": "a.py"}] * 3}

    base = {"ungated": ["x.py", "y.py"], "generated": "2026-09-19 21:00:00"}

    print("a new ungated launcher must fail, even if the count is unchanged")
    ok, m = check(base, rep(["x.py", "y.py", "new.py"]))
    expect("fails", ok, False)
    expect("and names the newcomer", any("new.py" in s for s in m), True)
    ok, m = check(base, rep(["x.py", "y.py", "new.py"][:2]))
    expect("swapping one out for another also fails on the addition",
           check(base, rep(["x.py", "new.py"]))[0], False)

    print("\nsame or fewer must pass; fewer must say so")
    expect("identical passes", check(base, rep(["x.py", "y.py"]))[0], True)
    ok, m = check(base, rep(["x.py"]))
    expect("fewer passes", ok, True)
    expect("and is reported as progress", any("PROGRESS" in s for s in m), True)
    expect("with the instruction to re-baseline", any("update" in s for s in m), True)

    print("\nsilent-reduction traps")
    expect("removing everything then re-adding one still fails",
           check({"ungated": ["x.py"]}, rep([]))[0], True)
    expect("an empty baseline makes any launcher a regression",
           check({"ungated": []}, rep(["x.py"]))[0], False)
    expect("a hand-typed baseline with no timestamp is called out",
           any("hand-typed" in s for s in check({"ungated": ["x.py"]}, rep(["x.py"]))[1]), True)

    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)}): {fails}")
        return 1
    print("SELFTEST PASS -- ungated launchers can only go down")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", default="check", choices=("check", "update", "selftest"))
    args = ap.parse_args(argv)
    return {"check": cmd_check, "update": cmd_update, "selftest": selftest}[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())
