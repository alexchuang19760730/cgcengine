#!/usr/bin/env python3
"""One probe for "is this box actually free", shared by every script that launches a server.

Why this is one module
---------------------
A launch on this machine loads a 13 GB model plus up to 8 GiB of expert-cache pool onto 16 GB of
RAM. Started while another session's server is up, it over-commits and the numbers describe a
thrash regime -- measured spread on identical configs was 17% and 2.5x on a bad day. That is not a
small error: it is the difference between a finding and a coincidence.

It was previously handled in three different ways, or not at all:

  * `decode_window_harness.py` classified processes by EXECUTABLE and documented why (`pgrep -f`
    matched a parallel session's polling shell, whose command line merely *contains* a llama name).
  * `plain_match_window.py` called `_H.llama_pids()`, which the harness does not define, and the
    AttributeError silently dropped it onto a naive `pgrep -f` -- the exact probe the harness had
    already replaced, for the exact bug it documents.
  * Twenty-one other scripts that launch a server did neither.

So the decision lives here once, and `audit` makes it visible which launchers are gated.

A deliberate asymmetry
----------------------
`require_first()` gates only the FIRST launch in a process. A multi-arm run loads, measures and
tears down N servers in one session; re-checking "is the box free" before arm 2 would fail on the
page cache our own arm 1 just filled, i.e. the guard would over-block exactly the runs it exists to
make attributable. Later launches in the same process record the state instead of refusing it.

Usage
-----
    python3 scripts/check/server_window.py check                 # one-shot verdict + exit code
    python3 scripts/check/server_window.py audit                 # which launchers are gated
    python3 scripts/check/server_window.py selftest

    from server_window import require_first          # in a script that launches a server
    require_first(port=8080, need_mb=8000.0, where="arm on")
"""
from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "check" / "decode_window_harness.py"
CHECK_DIR = ROOT / "scripts" / "check"

# The machine needs room for the model AND the pool. 8000 MB of reclaimable was the level at which
# guarded runs stopped failing to load; it is a floor for "no one else is holding this box", not a
# claim that less would never work.
NEED_MB = 8000.0
PROD_PORT = 8080
# Deliberate override: it does NOT silence the report, it only lets the launch proceed.
OVERRIDE_ENV = "CGC_WINDOW_OVERRIDE"

_sample_log: list[dict] = []
_cleared = False


class BusyBox(RuntimeError):
    """A launch was attempted while the box was demonstrably in use by someone else."""


_HARNESS = None


def _load_harness():
    """Load the harness ONCE. Re-executing it per probe made `_probe()` return a fresh function
    object every call, so a guard could not even tell whether it was using the harness's probe or
    its own -- which is how the silent-fallback bug read as agreement. (The selftest caught this.)"""
    global _HARNESS
    if _HARNESS is None:
        spec = importlib.util.spec_from_file_location("cgc_window_harness", HARNESS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _HARNESS = mod
    return _HARNESS


def _probe(name, fallback):
    """Harness probe when it exists, else a fallback that ANNOUNCES itself.

    The silent fallback is what made plain_match_window's guard disagree with the harness. A guard
    that quietly substitutes a different probe cannot be trusted to agree with anything.
    """
    try:
        return getattr(_load_harness(), name)
    except AttributeError:
        print(f"[window] harness has no {name}(); using the local fallback", file=sys.stderr)
        return fallback


def _fallback_llama():
    import subprocess
    out = subprocess.run(["ps", "-Ao", "pid=,args="], capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        exe = args.split()[0].rsplit("/", 1)[-1] if args.split() else ""
        if exe in ("llama-server", "llama-bench"):
            hits.append((int(pid), exe))
    return hits


def quiet(port: int = PROD_PORT, need_mb: float = NEED_MB) -> tuple[bool, str]:
    """(is-quiet, reason). The same three questions, asked once, for every launcher."""
    free = float(_probe("vm_free_mb", lambda: 0.0)())
    busy = _probe("foreign_llama", _fallback_llama)()
    held = _probe("listening", lambda p: False)(port)
    reasons = []
    if held:
        reasons.append(f"port {port} already listened on {held}")
    if busy:
        reasons.append(f"other llama process(es): {busy}")
    if free < need_mb:
        reasons.append(f"reclaimable={free:.0f}MB<{need_mb:.0f}")
    return (not reasons), ("; ".join(reasons) or f"quiet (reclaimable {free:.0f}MB)")


def require(port: int = PROD_PORT, need_mb: float = NEED_MB, *, where: str = "launch",
            override: bool | None = None) -> bool:
    """Refuse to launch into a busy box unless explicitly overridden. Returns is_quiet.

    Raises BusyBox by default: a measurement taken while someone else holds the machine cannot be
    attributed, and the failure is invisible in the numbers themselves.
    """
    global _cleared
    ok, why = quiet(port, need_mb)
    _sample_log.append({"where": where, "port": port, "need_mb": need_mb, "quiet": ok, "why": why})
    if ok:
        _cleared = True
        return True
    if override is None:
        override = os.environ.get(OVERRIDE_ENV) == "1"
    if override:
        print(f"[window] {where}: OVERRIDDEN -- {why}. This run may describe the neighbour; the "
              f"condition is recorded with it.", file=sys.stderr)
        _cleared = True
        return False
    raise BusyBox(f"{where}: {why}. Refusing to launch into a busy box ({OVERRIDE_ENV}=1 to "
                  f"override, which records the condition instead of hiding it).")


def record(port: int = PROD_PORT, need_mb: float = NEED_MB, *, where: str = "check",
           gated: bool = False) -> tuple[bool, str]:
    """Ask and REMEMBER, without the refusal rule -- for guards that prefer to wait over raising.

    `gated=True` marks the sample that actually decided a launch may proceed; that flag is what lets
    provenance() tell "the box was busy but the launch waited for quiet" from "the launch went ahead
    over a busy box".
    """
    ok, why = quiet(port, need_mb)
    _sample_log.append({"where": where, "port": port, "need_mb": need_mb, "quiet": ok, "why": why,
                        "gated": gated})
    return ok, why


def require_first(port: int = PROD_PORT, need_mb: float = NEED_MB, *, where: str = "launch") -> bool:
    """Gate the first launch of a process; later launches in the same run only record.

    See the module docstring: per-launch gating would refuse arm 2 of a multi-arm run because arm 1
    filled the page cache. That is over-blocking, not safety.
    """
    if _cleared:
        ok, why = quiet(port, need_mb)
        _sample_log.append({"where": where, "port": port, "need_mb": need_mb, "quiet": ok,
                            "why": why, "gated": False})
        return ok
    return require(port, need_mb, where=where)


def samples() -> list[dict]:
    """Every window check this process made, for the run's own provenance record."""
    return list(_sample_log)


def provenance(port: int = PROD_PORT, need_mb: float = NEED_MB) -> dict:
    """This run's window evidence, in the form a product should carry.

    Taken IN PROCESS, never stamped on afterwards: the window state at write time is not the window
    state the run measured under, so a post-hoc stamp would be a fabricated provenance -- the same
    shape as every wrong number this line has produced.

    `class` is the answer to "was this number taken on a busy box?":
      clean             every sample quiet, and the gated launch saw a quiet box
      busy-then-cleared the box was busy earlier but the gated launch was quiet
      busy-overridden   proceeded over a busy box (override): may describe the neighbour
      unknown           this process never asked the shared probe -- stated, not left blank
    """
    s = samples()
    gated = [x for x in s if x.get("gated", True)]
    busy = [x for x in s if not x["quiet"]]
    out = {"port": port, "need_mb": need_mb, "n_samples": len(s), "busy_samples": len(busy),
           "samples": s}
    if not s:
        out["class"] = "unknown"
        out["why"] = ("this process never asked the shared probe, so whether the box was busy "
                       "cannot be answered from this product")
    elif busy and any(x["quiet"] for x in gated):
        out["class"] = "busy-then-cleared"
        out["why"] = f"{len(busy)} busy sample(s), but the gated launch saw a quiet box"
    elif busy:
        out["class"] = "busy-overridden"
        out["why"] = (f"{len(busy)} busy sample(s) and no quiet gated launch: this measurement "
                       f"may describe the neighbour")
    else:
        out["class"] = "clean"
        out["why"] = f"every window sample was quiet ({len(s)} check(s))"
    return out


# Keys that mark a dict as a measurement product rather than a config block.
MEASURED_KEYS = ("mean_len", "accept", "m1_numeric_identity", "m2_decision_agreement",
                 "tps", "arms", "cuts", "rows", "requests", "results", "onoff_separated")


# --------------------------------------------------------------------------- audit
# A launcher is a real launch, not a docstring mention. These are the shapes seen in this tree.
LAUNCH_PATTERNS = (
    re.compile(r"subprocess\.(Popen|run|call)\([^\n]*run_server\.sh"),
    re.compile(r"\[\s*[\"']bash[\"']\s*,\s*str\([^\n]*run_server\.sh"),
    re.compile(r"build/bin/llama-(server|bench)"),
    re.compile(r"LAUNCHER\s*=\s*[^\n]*run_server\.sh"),
    re.compile(r"[\"']\./scripts/run_server\.sh[\"']"),
)
GATE_TOKENS = ("server_window", "decode_window_harness", "plain_match_window", "m123_gate_window")


def ownership() -> dict[str, str]:
    """Which launcher scripts another line holds, from the shared ownership document."""
    out: dict[str, str] = {}
    doc = ROOT / "docs" / "SHARED_SRC_OWNERSHIP_2026-09-19.md"
    if not doc.exists():
        return out
    for line in doc.read_text(errors="replace").splitlines():
        if "scripts/check" not in line or "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        names = re.findall(r"([A-Za-z0-9_]+)\.py", cells[0])
        for n in names:
            out[n + ".py"] = cells[2] if len(cells) > 2 else ""
    return out


def audit(paths: list[str] | None = None) -> dict:
    """Which files launch a server, and which of them ask the shared probe first.

    Honest boundary: text matching. It finds the launch shapes present in this tree and a gate
    import by name; a launcher written in a shape not listed here would be missed, so a zero-gate
    file is a prompt to look, not proof of a defect.
    """
    owned = ownership()
    rep = {"launchers": [], "gated": [], "ungated": [], "source": []}
    # Expand globs: passing "*.py" as a literal path made audit silently examine nothing, which is
    # the same shape as every other bug on this line -- a check that cannot fail.
    pats = paths or [str(CHECK_DIR / "*.py")]
    files = []
    for pat in pats:
        files.extend(Path(p) for p in glob.glob(str(pat)))
    files = sorted(set(files))
    for f in files:
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        hits = [p.pattern for p in LAUNCH_PATTERNS if p.search(text)]
        if not hits:
            continue
        if f.name == "decode_window_harness.py":
            # The harness DEFINES the probes. It is the source of truth, not a client of it, so it
            # is neither gated nor ungated -- counting it as ungated made the audit's own summary
            # read as "14 launchers, 1 gated" when two of the gated ones were misclassified.
            rep["source"].append({"file": f.name, "launch_shapes": len(hits)})
            rep["launchers"].append({"file": f.name, "kind": "source"})
            continue
        # Wiring detection is TEXTUAL on purpose. These files load the guard by filename through
        # importlib (or add scripts/check to sys.path and import it), so an AST-only check can miss
        # them entirely -- it did, and reported plain_match_window and m123_gate_window as ungated
        # while they gate on every run.
        gated = [t for t in GATE_TOKENS if t in text and t != f.stem]
        wired = bool(gated) or any(tok in text for tok in
                                   ("server_window", "require_first(", "wait_window(",
                                    "quiet(", "require("))
        row = {"file": f.name, "launch_shapes": len(hits), "gate_tokens": gated,
               "wired": wired, "owner": owned.get(f.name, ""),
               "kind": "gated" if wired else "ungated"}
        rep["launchers"].append(row)
        (rep["gated"] if wired else rep["ungated"]).append(row)
    return rep


def audit_products(paths: list[str]) -> dict:
    """Which products carry their window evidence, and which cannot answer the busy-box question.

    Honest boundary: key matching. It finds products that already carry a window block; a product
    that records the same thing under a different name is counted as missing. The number that
    matters is the second one -- a reading whose provenance cannot be answered from the file.
    """
    rep = {"files": 0, "measured_objects": 0, "with_window": 0, "without": 0,
           "classes": {}, "examples_without": []}

    def walk(obj, path):
        if isinstance(obj, dict):
            if any(k in obj for k in MEASURED_KEYS):
                rep["measured_objects"] += 1
                w = obj.get("window")
                if isinstance(w, dict) and w.get("class"):
                    rep["with_window"] += 1
                    rep["classes"][w["class"]] = rep["classes"].get(w["class"], 0) + 1
                else:
                    rep["without"] += 1
                    if len(rep["examples_without"]) < 12:
                        rep["examples_without"].append({"where": path,
                                                        "keys": sorted(
                                                            k for k in obj if k in MEASURED_KEYS)})
            for k, v in obj.items():
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    files = []
    for pat in paths:
        files.extend(Path(p) for p in glob.glob(str(pat)))
    for f in sorted(set(files)):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        rep["files"] += 1
        walk(data, f.name)
    return rep


def cmd_audit_products(args) -> int:
    rep = audit_products(args.paths)
    print(f"products: {rep['files']} file(s), {rep['measured_objects']} measured object(s)")
    print(f"  carry a window block : {rep['with_window']}  {rep['classes']}")
    print(f"  CANNOT answer the busy-box question : {rep['without']}")
    for e in rep["examples_without"]:
        print(f"    {e['where']}: {e['keys']}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))
        print(f"json -> {args.json}")
    return 0 if rep["without"] == 0 else 1


def cmd_audit(args) -> int:
    rep = audit(args.paths or None)
    print(f"launcher scripts: {len(rep['launchers'])}   gated: {len(rep['gated'])}   "
          f"ungated: {len(rep['ungated'])}   (source of truth: {len(rep['source'])})")
    print("\nGATED (asks the shared probe before launching)")
    for r in rep["gated"]:
        print(f"  {r['file']:<38} shapes={r['launch_shapes']} tokens={','.join(r['gate_tokens'][:3])}")
    print("\nUNGATED (launches without asking -- these are the ones that can measure the neighbour)")
    for r in rep["ungated"]:
        own = f"  [held by: {r['owner']}]" if r["owner"] else ""
        print(f"  {r['file']:<38} shapes={r['launch_shapes']}{own}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"\njson -> {args.json}")
    return 0


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            fails.append(name)

    print("the shared probe must agree with the harness, not silently replace it")
    h = _load_harness()
    expect("harness exports vm_free_mb", hasattr(h, "vm_free_mb"), True)
    expect("harness exports foreign_llama (the executable-classified one)",
           hasattr(h, "foreign_llama"), True)
    expect("harness does NOT export llama_pids (the name that silently fell back)",
           hasattr(h, "llama_pids"), False)
    expect("our probe resolves to the harness's", _probe("vm_free_mb", lambda: -1) is h.vm_free_mb,
           True)

    print("\nrequire() must refuse a busy box and record the reason")
    global quiet, _cleared
    saved = quiet

    def busy(port=PROD_PORT, need_mb=NEED_MB):
        return False, "port 8080 already listened on 1234; reclaimable=100MB<8000"
    quiet, _cleared = busy, False
    try:
        require(8080, 8000.0, where="selftest")
        expect("busy box raises", "no-raise", "BusyBox")
    except BusyBox as exc:
        expect("busy box raises", "BusyBox", "BusyBox")
        expect("the reason travels in the message", "reclaimable=100MB<8000" in str(exc), True)
    try:
        ok = require(8080, 8000.0, where="selftest", override=True)
        expect("override proceeds instead of raising", ok, False)
        expect("override still records the condition as not-quiet",
               samples()[-1]["quiet"], False)
    except BusyBox:
        expect("override proceeds", "raised", "returned")

    print("\nrequire_first must gate once and then record (not over-block)")
    _cleared = False
    try:
        require_first(8080, 8000.0, where="arm 1")
        expect("first launch is gated", "not-gated", "gated")
    except BusyBox:
        expect("first launch is gated", "gated", "gated")
    quiet, _cleared = (lambda *a, **k: (False, "reclaimable=10MB<8000")), True
    try:
        ok = require_first(8080, 8000.0, where="arm 2")
        expect("second launch in the same run does NOT block", ok, False)
        expect("but its state is recorded", samples()[-1].get("gated"), False)
    except BusyBox:
        expect("second launch does not block", "raised", "returned")

    print("\nthe window evidence must be taken in process and must distinguish the four cases")
    global _sample_log
    saved_log = _sample_log
    _sample_log = []
    p = provenance(8080, 8000.0)
    expect("no probe call -> unknown", p["class"], "unknown")
    expect("and it says so instead of leaving it blank", "never asked" in p["why"], True)
    _sample_log = [{"where": "arm on", "port": 8080, "need_mb": 8000.0, "quiet": True,
                    "why": "quiet (reclaimable 9000MB)"}]
    expect("all quiet -> clean", provenance()["class"], "clean")
    _sample_log = [{"where": "wait", "quiet": False, "why": "reclaimable=100MB<8000"},
                   {"where": "arm on", "quiet": True, "why": "quiet"}]
    expect("busy earlier, quiet at the gate -> busy-then-cleared",
           provenance()["class"], "busy-then-cleared")
    _sample_log = [{"where": "arm on", "quiet": False, "why": "override"}]
    expect("proceeded over a busy box -> busy-overridden",
           provenance()["class"], "busy-overridden")
    _sample_log = [{"where": "wait", "quiet": False, "why": "busy", "gated": False},
                   {"where": "cleared", "quiet": True, "why": "quiet", "gated": True}]
    expect("a waiting guard's busy sample does not poison a clean gate",
           provenance()["class"], "busy-then-cleared")
    _sample_log = saved_log
    # record() must log without refusing
    saved_quiet, quiet = quiet, (lambda *a, **k: (False, "busy"))
    ok_r, why_r = record(8080, 8000.0, where="waiter", gated=False)
    expect("record() never raises", (ok_r, why_r), (False, "busy"))
    expect("and its sample is kept with its gated flag", samples()[-1]["gated"], False)
    quiet = saved_quiet

    print("\nproduct audit: a measured object with no window block is countable")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        base = Path(d)
        (base / "stamped.json").write_text(json.dumps(
            {"arm": "on", "mean_len": 2.4,
             "window": {"class": "clean", "n_samples": 2, "samples": []}}))
        (base / "bare.json").write_text(json.dumps({"arm": "on", "mean_len": 2.4}))
        arep = audit_products([str(base / "*.json")])
        expect("two measured objects found", arep["measured_objects"], 2)
        expect("one carries a window block", arep["with_window"], 1)
        expect("one cannot answer the question", arep["without"], 1)
        expect("the offender is named", arep["examples_without"][0]["where"], "bare.json")
        expect("a non-measurement dict is ignored",
               audit_products([str(base / "*.json")])["measured_objects"], 2)

    print("\nthe audit must classify by wiring, not by a docstring mention")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "mentioned.py").write_text('"""calls run_server.sh in prose only"""\n')
        (p / "launcher.py").write_text(
            "import subprocess\nsubprocess.Popen([\"bash\", \"scripts/run_server.sh\"])\n")
        (p / "gated.py").write_text(
            "import server_window\nimport subprocess\n"
            "server_window.require_first(8080)\n"
            "subprocess.Popen([\"bash\", \"scripts/run_server.sh\"])\n")
        rep = audit([str(p / "*.py")])
        names = {r["file"] for r in rep["launchers"]}
        expect("a prose mention is not a launcher", "mentioned.py" in names, False)
        expect("a real launcher is found", "launcher.py" in names, True)
        expect("it is classified ungated", [r["file"] for r in rep["ungated"]], ["launcher.py"])
        expect("the wired one is classified gated", [r["file"] for r in rep["gated"]], ["gated.py"])

    quiet = saved
    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)}): {fails}")
        return 1
    print("SELFTEST PASS -- one probe, one refusal rule, and an audit that can see the difference")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest")
    c = sub.add_parser("check")
    c.add_argument("--port", type=int, default=PROD_PORT)
    c.add_argument("--need-mb", type=float, default=NEED_MB)
    sub.add_parser("status", help="alias of check")
    a = sub.add_parser("audit")
    a.add_argument("paths", nargs="*")
    a.add_argument("--json", default=None)
    ap_ = sub.add_parser("audit-products", help="which products carry their window evidence")
    ap_.add_argument("paths", nargs="*", default=None)
    ap_.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "audit":
        return cmd_audit(args)
    if args.cmd == "audit-products":
        args.paths = args.paths or [
            str(ROOT / "Backup" / "phase_decomp" / "*.json"),
            str(ROOT / "Backup" / "cgc_logs" / "*.json"),
            str(ROOT / "Backup" / "m123_oracle_gate" / "summary_*.json")]
        return cmd_audit_products(args)
    if args.cmd in ("check", "status", None):
        port = getattr(args, "port", PROD_PORT)
        need = getattr(args, "need_mb", NEED_MB)
        ok, why = quiet(port, need)
        print(f"{'QUIET' if ok else 'BUSY'}: {why}")
        return 0 if ok else 3
    ap.error("give a command: check | audit | selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
