#!/usr/bin/env python3
"""One front door for every windowed measurement on this line. `show windows` first.

WHY THIS EXISTS
---------------
Four things on this line each reimplemented "wait until the box is free, then run":

  * `decode_window_harness.py` (defines the probes; lane runner + HTML report)
  * `m123_gate_window.py`      (wraps the M1/M2/M3 oracle gate)
  * `plain_match_window.py`    (wraps plain_match_ab)
  * `server_window.py cmd_wait` (the shared probe itself, with its own wait loop)

They do not agree with each other, and the disagreement is not cosmetic. A launch into a busy box
measures the neighbour: the measured spread on identical configs here is 17%, and 2.5x on a bad day.
That failure is invisible in the number it produces, so it has to be caught before the launch -- and
today whether it is caught depends on which wrapper you happened to pick.

This file does NOT replace those wrappers. It is the FRONT DOOR: one place that knows

  1. which tools need a quiet window at all (the registry),
  2. what the box says right now (one shared probe, asked once),
  3. whether that verdict can be trusted physically, not just by bookkeeping
     (`--certify` runs the throughput sentinel; thermal NOMINAL alone has certified boxes that were
     ~5x slow -- 2026-09-20),
  4. and where the evidence went (`harness_runs.jsonl`, appended per run).

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not run a server itself and it does not redefine any probe. Everything it knows about the
box comes from `server_window` and `window_sentinel`, so this file cannot drift from them.

HONEST BOUNDARY ON PROVENANCE
-----------------------------
The provenance recorded here is what THIS process saw: the box state before the run started, and
after it returned. The child process runs in its own address space. If the child is one of the
gated tools it asks the same probe for itself and carries its own evidence; if it is not, this
record is a statement about the box around the run, NOT about the child's internals. The record
says which case it is (`child_gated`) rather than implying more than it measured.

USAGE
-----
    python3 scripts/check/harness.py show                 # the window board (default)
    python3 scripts/check/harness.py show --json
    python3 scripts/check/harness.py run cb_headroom_probe -- --tag x --band any
    python3 scripts/check/harness.py run profile_duo --certify -- --profile prefill250
    python3 scripts/check/harness.py list                 # the registry
    python3 scripts/check/harness.py audit                # ungated launchers
    python3 scripts/check/harness.py selftest

BENCH (統一量測入口)
--------------------
    python3 scripts/check/harness.py bench \
        --arm "prod-new:!CGC_WAKE_POLL_US=999;CGC_SERVER_MTP=1" \
        --json /tmp/bench_out.json

    - 基底 = prod-new（CGC_DUMP_ENV=1 唯一權威）；arm 語法 PROFILE:!OVERRIDE;KEY=VAL
    - base gate：撞 base 鍵需 ! 宣告（實驗開關白名單 _EXPERIMENT_KNOBS 自動放行 + 記錄）
    - 產物契約：每臂含 base_check + sys_before/after（thermal/swap/pageins/memory_pressure/iostat）
    - 報告必須基於 prod-new + 自己的 env 增量；llama-bench 完整側參數見 _BENCH_DEFAULTS
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS_LOG = ROOT / "Backup" / "phase_decomp" / "harness_runs.jsonl"

PY = sys.executable or "python3"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- registry
# `needs_window` is the whole point of the registry: it is the question no single tool could answer
# for another. True = a busy box corrupts this tool's output. False = it parses files / does
# arithmetic, and a neighbour's server does not change its answer (only its speed). None = not yet
# labelled, which `show` reports as a gap instead of quietly treating as False.
#
# `launches` is not hand-maintained: it is merged from `server_window.audit()`, so a new launcher
# shows up here whether or not anyone remembered to edit this table.
REGISTRY: dict[str, dict] = {
    # --- line A (ace): engine layer, S1 / segment boundaries / per-layer KINDxOP
    "cb_headroom_probe.py": dict(needs_window=True, owner="lineA",
                                 purpose="re-read cb under a declared memory band"),
    "decode_step_profile.py": dict(needs_window=True, owner="lineA",
                                   purpose="per-step DECPROF capture (wait/cb/submit/gpu/union)"),
    "decode_sweep.py": dict(needs_window=True, owner="lineA", purpose="phase decomposition sweep"),
    "http_duo.py": dict(needs_window=True, owner="lineA", purpose="service-path two-axis probe"),
    "joint_reconcile.py": dict(needs_window=False, owner="lineA",
                               purpose="reconcile one step's account from a saved log"),
    "gate_registry.py": dict(needs_window=True, owner="lineA",
                             purpose="G0-G7 gate registry (long probe needs a server)"),
    "prod_profile.py": dict(needs_window=True, owner="lineA", purpose="delivery bars: prefill+decode"),
    "profile_duo.py": dict(needs_window=True, owner="lineA", purpose="llama-bench two-axis (delivery)"),
    "prod_matrix.py": dict(needs_window=True, owner="lineA", purpose="production matrix"),
    "llama_bench_matrix.py": dict(needs_window=True, owner="lineA", purpose="bench matrix resolver"),
    "paired_ab.py": dict(needs_window=True, owner="lineA", purpose="paired A/B (AB/BA interleaved)"),
    "ab_interleave.py": dict(needs_window=True, owner="lineA", purpose="interleaved A/B"),
    "window_sentinel.py": dict(needs_window=True, owner="lineA",
                               purpose="certify a window by measured throughput"),
    "decode_window_harness.py": dict(needs_window=True, owner="lineA",
                                     purpose="two decode lanes, judged report"),
    "server_window.py": dict(needs_window=False, owner="shared",
                             purpose="the shared probe + launcher audit"),
    "window_gate.py": dict(needs_window=False, owner="shared", purpose="commit gate: launchers only go down"),
    # --- line I: instrumentation / cache geometry
    "cb_miss_regression.py": dict(needs_window=True, owner="lineI", purpose="cb miss regression"),
    "m123_oracle_gate.py": dict(needs_window=True, owner="lineI", purpose="M1/M2/M3 oracle gate"),
    "m123_gate_window.py": dict(needs_window=True, owner="lineI", purpose="oracle gate in a window"),
    # --- launchers found by the audit whose owning line is not recorded here. `owner` is left empty
    # on purpose so `show` falls back to the shared ownership document: a wrong line name sends the
    # "please gate this" request to someone who does not own the file.
    "mtp_accept_ab.py": dict(needs_window=True, purpose="MTP draft acceptance per carrier"),
    "mtp_long_sequence_verify.py": dict(needs_window=True, purpose="MTP long-sequence M1/M2/M3 verify"),
    "phase_split_ab.py": dict(needs_window=True, purpose="phase-split prefill A/B: armed slab vs clamp"),
    "plain_match_ab.py": dict(needs_window=True,
                              purpose="does batch-verify compute the same thing as per-token decode"),
    "plain_match_window.py": dict(needs_window=True, purpose="wrapper: plain_match_ab in a window"),
    "pool_curve.py": dict(needs_window=True, purpose="pool-size sweep, one budget per restart"),
    "slab_handoff_ab.py": dict(needs_window=True, purpose="slab to pool handoff A/B"),
    "test_decode_window_harness.py": dict(needs_window=True, purpose="self-test for harness verdicts"),
    # A driver whose `--measure` path launches, and whose `--pair` path does not. It is listed as
    # needing a window because the stance has to hold for the mode that launches: `harness.py
    # selftest` fails if a launcher has no stance at all, which is how this entry came to exist.
    "target25_report.py": dict(needs_window=True,
                               purpose="one-shot driver: measure -> decompose -> HTML report"),
}


def registry_with_audit(sw) -> list[dict]:
    """Hand labels merged with the live audit. Audit wins on `launches` because it reads the tree."""
    rep = sw.audit()
    launched = {r["file"] for r in rep["launchers"]}
    gated = {r["file"] for r in rep["gated"]}
    # The audit puts the file that DEFINES the probes under `source`: it is the truth these gates
    # are checked against, not a client of them. Counting it as an ungated launcher made the board
    # demand that the harness gate itself.
    source = {r["file"] for r in rep["source"]}
    owned = sw.ownership()
    rows = []
    for name in sorted(set(REGISTRY) | launched):
        e = dict(REGISTRY.get(name, {}))
        rows.append({
            "tool": name,
            "needs_window": e.get("needs_window", None),
            "launches_server": name in launched,
            "gated": name in gated,
            "is_source": name in source,
            "owner": e.get("owner") or owned.get(name) or "",
            "purpose": e.get("purpose", ""),
            "labelled": name in REGISTRY,
        })
    return rows


def needs_window_for(tool: str, rows: dict) -> bool:
    """Unlabelled means NEEDS a window.

    The unsafety is asymmetric: treating a measurement that needs quiet as one that does not
    produces a number that looks fine and describes the neighbour, while the reverse only makes
    someone wait. So the default errs toward waiting.
    """
    v = rows.get(tool, {}).get("needs_window", None)
    return True if v is None else bool(v)


# --------------------------------------------------------------------------- box view
def box_view(sw, port: int, need_mb: float) -> dict:
    d = sw.decision(port, need_mb)
    try:
        sentinel_others = _load("ws", "window_sentinel.py").others()
    except Exception:
        sentinel_others = None
    return {"port": port, "need_mb": need_mb, "decision": d,
            "other_session_pids": sentinel_others,
            "when": time.strftime("%Y-%m-%d %H:%M:%S")}


def _verdict(box: dict) -> tuple[str, str]:
    d = box["decision"]
    if box.get("other_session_pids"):
        # str() each item: the pids arrive as ints from one producer and as strings from another.
        return ("BUSY", "another session is running (pids %s)"
                % ", ".join(str(p) for p in box["other_session_pids"]))
    if not d["admits"]:
        why = "; ".join(f for f in [("port held" if d["port_held"] else ""),
                                    ("foreign llama: %s" % llama_desc(d["foreign_llama"])
                                     if d["foreign_llama"] else ""),
                                    ("reclaimable %.0fMB < %.0f" % (d["reclaimable_mb"],
                                                                    d["need_mb"])
                                     if d["reclaimable_mb"] < d["need_mb"] else ""),
                                    ("launcher refuses (%s)" % d["launcher_class"]
                                     if d["launcher_admits"] is False else "")] if f)
        # Say when the launcher would have started anyway: "refused" alone reads as "the box is
        # busy" when the truth is "our bar is stricter than the launcher's", and the two call for
        # opposite responses (wait vs lower the bar deliberately).
        if d["agree"] is False:
            why += " [DISAGREE: the launcher's own probe would admit; binding=%s]" % d["binding"]
        return ("REFUSED", why or ("binding=%s" % d["binding"]))
    if d["agree"] is False:
        return ("ADMITS*", "the two definitions disagree; binding=%s" % d["binding"])
    return ("ADMITS", "both definitions agree the box is quiet")


def llama_desc(items) -> str:
    """Render the shared probe's foreign-llama list as text.

    The producers hand back `(pid, exe)` PAIRS (`server_window._fallback_llama`, `decode_window_
    harness.foreign_llama`), while the fixtures in this file's own selftest used bare strings. So
    `",".join(...)` passed the selftest and raised `TypeError: sequence item 0: expected str
    instance, tuple found` in `show` -- the one command whose job is to say WHO is holding the box --
    precisely when a neighbour was holding it. The formatter takes every shape the probe can return,
    and is used by both readers of this field (the verdict's reason and the board's header).
    """
    out = []
    for it in items or []:
        if isinstance(it, (tuple, list)):
            out.append(" ".join(str(x) for x in it) if it else "")
        else:
            out.append(str(it))
    return ", ".join(x for x in out if x)


def box_lines(box, verdict, why) -> list[str]:
    """`show`'s box header, as data, so a selftest can assert on it without a live probe."""
    d = box["decision"]
    # "-" and not 0 for an absent launcher verdict: the launcher's bar and "no bar published" are
    # different statements, and a 0 would read as "the launcher would refuse anything".
    dash = lambda v: "-" if v is None else str(v)
    return [
        "WINDOW BOARD  %s" % box["when"],
        "  port %-5d %s" % (box["port"], "HELD by %s" % d["port_held"] if d["port_held"] else "free"),
        "  foreign llama: %s" % (llama_desc(d["foreign_llama"]) or "none"),
        "  reclaimable %.0f MB (need %.0f)   launcher %s%% (req %s%%, class %s)" % (
            d["reclaimable_mb"], box["need_mb"], dash(d["launcher_free_pct"]),
            dash(d["launcher_req_pct"]), dash(d["launcher_class"])),
        # `other_session_pids` is None whenever the sentinel is absent -- which is the common case,
        # and it used to reach `join()` as None (same defect class as the pairs above: the fixture
        # passed a list, the real box passed None).
        "  other sessions: %s" % (", ".join(str(p) for p in (box["other_session_pids"] or []))
                                  or "none"),
        "  VERDICT: %s -- %s" % (verdict, why),
    ]


def cmd_show(args) -> int:
    sw = _load("sw", "server_window.py")
    box = box_view(sw, args.port, args.need_mb)
    verdict, why = _verdict(box)
    d = box["decision"]
    rows = registry_with_audit(sw)

    if args.json:
        print(json.dumps({"box": box, "verdict": verdict, "why": why, "tools": rows}, indent=1))
        return 0

    print("\n".join(box_lines(box, verdict, why)))
    print()

    can = verdict == "ADMITS"
    print("%-34s %-7s %-8s %-6s %-6s %s" % ("TOOL", "WINDOW", "LAUNCHES", "GATED", "OWNER", "NOW"))
    print("-" * 100)
    for r in rows:
        nw = {True: "yes", False: "no", None: "?"}[r["needs_window"]]
        if r["needs_window"] is False:
            now = "runnable"
        elif can:
            now = "runnable"
        else:
            now = "blocked"
        if r["is_source"]:
            gate = "src"
        elif r["gated"]:
            gate = "yes"
        else:
            gate = "NO" if r["launches_server"] else "-"
        print("%-34s %-7s %-8s %-6s %-6s %s" % (
            r["tool"], nw, "yes" if r["launches_server"] else "-", gate,
            r["owner"] or "-", now))
    unlabelled = [r["tool"] for r in rows if not r["labelled"]]
    ungated = [r["tool"] for r in rows
               if r["launches_server"] and not r["gated"] and not r["is_source"]]
    print()
    print("ungated launchers (%d): %s" % (len(ungated), ", ".join(ungated) or "none"))
    print("unlabelled in registry (%d): %s" % (len(unlabelled), ", ".join(unlabelled) or "none"))
    if ungated:
        print("  -> a launch from any of these cannot be attributed on a busy box; run them through "
              "`harness.py run` or gate them (docs/SERVER_WINDOW_LEDGER_2026-09-19.md)")
    print("windows: %s" % ("open -- `harness.py run <tool> -- ...`" if can else "closed -- the box "
                           "is not admitting a launch right now"))
    if not can and d["agree"] is False and d["launcher_admits"] is not False:
        print("  note: the launcher's own probe would admit this box. The bar that refused is ours "
              "(--need-mb %.0f). Passing a lower `--need-mb` runs at the launcher's bar instead; "
              "do it deliberately, because the record will say which bar was used." % d["need_mb"])
    return 0


def cmd_list(args) -> int:
    sw = _load("sw", "server_window.py")
    rows = registry_with_audit(sw)
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    for r in rows:
        print("%-34s window=%-5s launches=%-5s gated=%-5s owner=%-6s %s" % (
            r["tool"], {True: "yes", False: "no", None: "?"}[r["needs_window"]],
            "yes" if r["launches_server"] else "-", "yes" if r["gated"] else "no",
            r["owner"] or "-", r["purpose"]))
    return 0


# --------------------------------------------------------------------------- run
def wait_for_window(sw, port: int, need_mb: float, timeout_s: float, poll_s: float,
                    quiet_samples: int = 2) -> tuple[bool, str, float]:
    """Poll the shared probe until it admits, needing `quiet_samples` consecutive quiet readings.

    Two, not one: a single quiet reading here has repeatedly been a trough between two neighbours'
    launches, and the run started on it was measured inside someone else's load.
    """
    t0 = time.time()
    streak = 0
    last = ""
    while True:
        ok, why = sw.record(port, need_mb, where="harness.wait", gated=False)
        if ok:
            streak += 1
            if streak >= quiet_samples:
                return True, why, time.time() - t0
        else:
            streak = 0
            last = why
        if time.time() - t0 > timeout_s:
            return False, last or "timed out without a quiet window", time.time() - t0
        time.sleep(poll_s)


def cmd_run(args) -> int:
    sw = _load("sw", "server_window.py")
    tool = args.tool if args.tool.endswith(".py") else args.tool + ".py"
    path = HERE / tool
    if not path.exists():
        print("no such tool: %s (looked in %s)" % (tool, HERE), file=sys.stderr)
        print("`harness.py list` shows what is registered.", file=sys.stderr)
        return 2

    rows = {r["tool"]: r for r in registry_with_audit(sw)}
    meta = rows.get(tool, {})
    needs = needs_window_for(tool, rows)
    if meta.get("needs_window", None) is None:
        print("[harness] %s is not in the registry: assuming it needs a window. Label it in "
              "REGISTRY if that is wrong." % tool, file=sys.stderr)

    if needs:
        ok, why, waited = wait_for_window(sw, args.port, args.need_mb, args.window_timeout_s,
                                          args.poll_s)
        if not ok:
            print("[harness] no window in %.0fs: %s" % (waited, why), file=sys.stderr)
            print("[harness] nothing was run, so nothing was recorded as data.", file=sys.stderr)
            return 3
        sw.record(args.port, args.need_mb, where="harness.launch", gated=True)
        print("[harness] window admitted after %.0fs" % waited)
    else:
        print("[harness] %s does not need a window (registry); running directly" % tool)

    cert = None
    if args.certify:
        ws = _load("ws", "window_sentinel.py")
        ts, rc = ws.measure()
        ref = None
        try:
            ref = json.loads((HERE / "window_sentinel_ref.json").read_text())
        except Exception:
            pass
        cert = {"measured_ts": ts, "rc": rc, "ref": ref}
        if ts is None or rc != 0:
            print("[harness] sentinel could not measure throughput -- refusing to certify",
                  file=sys.stderr)
            return 4
        floor = (ref or {}).get("median_ts")
        if floor and ts < float(floor) * args.certify_frac:
            print("[harness] REFUSED: %.2f t/s < %.2f x reference %.2f -- this window is degraded "
                  "even though the box looks quiet" % (ts, args.certify_frac, float(floor)),
                  file=sys.stderr)
            return 5
        print("[harness] certified: %.2f t/s (ref %.2f)" % (ts, float(floor or 0)))

    cmd = [PY, str(path)] + list(getattr(args, "rest", []) or [])
    print("[harness] $ %s" % " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT))
    dur = time.time() - t0

    prov = sw.provenance(args.port, args.need_mb)
    rec = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": tool, "argv": cmd,
           "exit": p.returncode, "duration_s": round(dur, 1),
           "needs_window": needs, "child_gated": bool(meta.get("gated")),
           "certified": cert, "window_class": prov["class"], "window_why": prov["why"],
           "decision_now": prov["now"], "owner": meta.get("owner", "")}
    try:
        RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUNS_LOG.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as e:
        print("[harness] could not append the run record: %s" % e, file=sys.stderr)
    print("[harness] exit=%s in %.0fs | window=%s | record: %s" % (
        p.returncode, dur, prov["class"], RUNS_LOG))
    return p.returncode


def cmd_audit(args) -> int:
    wg = _load("wg", "window_gate.py")
    if args.json:
        print(json.dumps(wg.audit_now(), indent=1))
        return 0
    ok, msgs = wg.check()
    for m in msgs:
        print(m)
    return 0 if ok else 1


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print("  %s %s: %r" % ("ok  " if ok else "FAIL", name, got))
        if not ok:
            fails.append(name)

    print("registry is the point of the tool: every launcher must appear, labelled or not")
    sw = _load("sw", "server_window.py")
    rows = registry_with_audit(sw)
    launched = {r["tool"] for r in rows if r["launches_server"]}
    expect("audit found launchers", len(launched) > 0, True)
    expect("every launcher has a window stance", all(
        r["needs_window"] is not None for r in rows if r["launches_server"]), True)

    print("\nthe three stances are distinct; unknown is not silently False")
    expect("no-window tool exists", any(r["needs_window"] is False for r in rows), True)
    expect("yes-window tool exists", any(r["needs_window"] is True for r in rows), True)

    print("\nverdict names the blocker instead of just refusing")
    base = {"decision": {"admits": False, "port_held": "1234", "foreign_llama": [],
                         "reclaimable_mb": 9999.0, "need_mb": 1.0, "launcher_admits": True,
                         "binding": "harness", "agree": True, "launcher_class": "k"},
            "other_session_pids": None, "need_mb": 1.0}
    v, why = _verdict(base)
    expect("refused", v, "REFUSED")
    expect("and says why", "port held" in why, True)

    print("\nthe blocker is RENDERED on the probe's real shape (pid, exe) -- regression")
    # The tuple shape is what `server_window._fallback_llama` / `decode_window_harness.foreign_llama`
    # actually return. `",".join(...)` on it raised TypeError and killed `show` -- so the board
    # crashed exactly when a neighbour held the box, and the fixtures above (bare strings) could
    # never catch it. Both readers of the field are asserted below, on the real shape.
    busy_decision = dict(base["decision"], foreign_llama=[(44076, "llama-server")],
                         launcher_free_pct=91.0, launcher_req_pct=30.0)
    busy_box = {"decision": busy_decision, "other_session_pids": None, "need_mb": 1.0,
                "port": 8080, "when": "fixture"}
    v, why = _verdict(busy_box)
    expect("still refuses", v, "REFUSED")
    expect("the reason names the pid and the exe", "44076" in why and "llama-server" in why, True)
    lines = box_lines(busy_box, v, why)
    expect("show's own header renders the pair", any("44076 llama-server" in ln for ln in lines), True)
    expect("and never prints an empty list as a blocker",
           any("foreign llama: none" in ln for ln in box_lines(
               {"decision": dict(busy_decision, foreign_llama=[]), "other_session_pids": None,
                "need_mb": 1.0, "port": 8080, "when": "fixture"}, "ADMITS", "q")), True)
    expect("the probe's pair shape", llama_desc([(1, "a"), (2, "b")]), "1 a, 2 b")
    expect("the bare-int shape other probes return", llama_desc([999]), "999")
    expect("empty stays empty", llama_desc([]), "")

    print("\nanother session overrides a box that otherwise looks quiet")
    v, _ = _verdict({"decision": dict(base["decision"], admits=True, port_held=False),
                     "other_session_pids": ["999"], "need_mb": 1.0})
    expect("busy wins", v, "BUSY")

    print("\ndisagreement is visible, not averaged away")
    v, _ = _verdict({"decision": dict(base["decision"], admits=True, port_held=False, agree=False),
                     "other_session_pids": [], "need_mb": 1.0})
    expect("admits-with-asterisk", v, "ADMITS*")

    print("\nwait needs a streak, not one lucky reading")
    calls = {"n": 0}

    def fake_record(port, need_mb, where="", gated=False):
        calls["n"] += 1
        return (calls["n"] >= 3), "why%d" % calls["n"]

    real = sw.record
    sw.record = fake_record
    try:
        ok, why, waited = wait_for_window(sw, 8080, 1.0, timeout_s=5, poll_s=0.01,
                                          quiet_samples=2)
        expect("waits for 2 consecutive", (ok, calls["n"]), (True, 4))
    finally:
        sw.record = real

    print("\nthe box view comes from the shared probe's real producer, not from a shape invented here")
    # This check exists because the old selftest built the decision dict as a FIXTURE and asserted on
    # it, so `show` -- this file's own default command -- called `sw.decision()` for months while that
    # function did not exist, and the selftest stayed green. Assert the producer, not a mock of it.
    got = sw.decision(need_mb=1.0)
    for k in ("admits", "harness_admits", "launcher_admits", "agree", "binding", "reclaimable_mb",
              "port_held", "foreign_llama", "launcher_class"):
        expect("decision() carries %s" % k, k in got, True)
    expect("agree is never True without a published launcher verdict",
           got["agree"] is True and got["launcher_admits"] is not None or got["agree"] is not True,
           True)

    print("\nan unregistered tool is treated as needing a window, never as not needing one")
    by = {r["tool"]: r for r in rows}
    expect("known no-window tool -> False", needs_window_for("joint_reconcile.py", by), False)
    expect("known yes-window tool -> True", needs_window_for("profile_duo.py", by), True)
    expect("unregistered -> True (errs toward waiting)",
           needs_window_for("no_such_tool.py", by), True)

    print()
    if fails:
        print("SELFTEST FAIL (%d): %s" % (len(fails), fails))
        return 1
    print("SELFTEST PASS -- one front door, four wrappers behind it")
    return 0


# --------------------------------------------------------------------------- bench
# 統一量測入口：所有報告數字必須基於 prod-new（或顯式 profile）＋自己的 env 增量。
# base gate：arm env 撞到 base 已有鍵但未用 `!` 宣告 override → fail-closed 拒跑。
# 產物契約：每臂 json 注入 base_check + 量測紀律字段（pp/tg + thermal + swap + attribution）。

_BASE_PROFILE_DEFAULT = "prod-new"

# 實驗開關白名單（測試卡 §4 臂差異開關 + 支柱開關）：這些鍵允許覆蓋 base（不需 `!`，自動記錄）。
# 其餘 base 鍵 = 環境/儀器/模型常數，鎖死——要改必須 `!` 顯式宣告。
_EXPERIMENT_KNOBS = {
    # ── dump 內實驗開關（撞 base 自動放行 + 記錄）──
    "CGC_EXPERT_CACHE_BYTES", "LLAMA_EXPERT_CACHE_ALLOW_NGL",
    "CGC_OA_ASYNC", "CGC_GATHER_SLAB_CAP", "CGC_PREFILL_STREAM",
    "CGC_SPAC", "CGC_SPAC_ALPHA", "CGC_MM_BITIDENT",
    # ── 不在 dump 的已知實驗開關（arm 設了 = 新增鍵，gate 不擋、產物 extra_env 記錄）──
    "CGC_EXPERT_SKIP_READRAW", "CGC_POOL_MADVISE", "CGC_SERVER_MTP",
    "CGC_SERVER_MTP_N_MAX", "CGC_SEG_BATCH", "CGC_B_SCHEME",
    "CGC_SLOT_TABLE_GPU", "CGC_SPAC_HOT", "CGC_SERVER_PREFIX_REUSE_CKPT",
    "CGC_SERVER_DENSE_IQ4X", "CGC_SERVER_OA_ASYNC", "CGC_SERVER_NO_SEQ_RM_PROBE",
    "CGC_DOWN_COMBINE", "CGC_FORCE_TEMP0", "CGC_HOOK_PROFILE",  # opt-in 儀器
}

# llama-bench 完整側默認（測試卡 §2，可覆寫但記錄在產物）
_BENCH_DEFAULTS = dict(prompt=2048, gen=128, depths="512", reps=1,
                       warm_skip=64, ctx_size=0, batch=5632, ubatch=5632)


def _parse_arm(spec: str) -> tuple[str, dict, set]:
    """'prod-new:!A=1;B=2' -> ('prod-new', {'A':'1','B':'2'}, {'A'})"""
    if ":" in spec:
        profile, envs = spec.split(":", 1)
    else:
        profile, envs = spec, ""
    env, overrides = {}, set()
    for piece in envs.split(";") if envs else []:
        piece = piece.strip()
        if not piece:
            continue
        declared = piece.startswith("!")
        if declared:
            piece = piece[1:]
        if "=" not in piece:
            raise SystemExit(f"bad arm env piece: {piece!r} (want KEY=VAL or !KEY=VAL)")
        k, v = piece.split("=", 1)
        env[k.strip()] = v.strip()
        if declared:
            overrides.add(k.strip())
    return profile, env, overrides


def _base_gate(profile: str, extra_env: dict, overrides: set) -> tuple[bool, list[str], list[str]]:
    """resolve base vs arm；未宣告的 base 鍵變更 → FAIL（fail-closed）。"""
    matrix = _load("matrix", "llama_bench_matrix.py")
    base = matrix.resolve(profile, {})
    arm = matrix.resolve(profile, extra_env)
    diffs, ovr = [], []
    # 雙向比對：arm 改 base 鍵值、或設 0 讓 base 鍵從 dump 消失（= 值變更），都要歸因
    for k in sorted(set(base["env"]) | set(arm["env"])):
        bv, av = base["env"].get(k), arm["env"].get(k)
        if bv is None:          # arm 新增鍵（base 沒有）
            continue
        if av == bv:
            continue
        if k in overrides:
            ovr.append(f"{k}: {bv!r} -> {av!r} (declared override)")
        elif k in _EXPERIMENT_KNOBS:
            ovr.append(f"{k}: {bv!r} -> {av!r} (experiment knob)")
        else:
            diffs.append(f"{k}: base={bv!r} arm={av!r} (NOT declared, use !)")
    return (len(diffs) == 0), diffs, ovr


def _sys_snapshot() -> dict:
    """測試前/後各採一次：thermal + swap 水位 + pageins/pageouts + memory_pressure + iostat。"""
    snap = {}
    tp = _load("tp_snap", "thermal_pressure.py")
    mp = _load("mpp_snap", "memory_pressure.py")
    try:
        t = tp.stamp()
        snap["thermal"] = {"label": t.get("label"), "lv": t.get("lv"), "t": t.get("t")}
    except Exception as e:
        snap["thermal"] = {"error": str(e)}
    try:
        snap["swap_used_mb"] = mp.swap_used_mb()
    except Exception:
        snap["swap_used_mb"] = None
    try:  # vm_stat 累計 pageins/pageouts（欄位大小寫不敏感）
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
        pi = po = None
        for line in out.stdout.splitlines():
            low = line.lower()
            if low.startswith("pageins:"):
                pi = int(line.split()[1].rstrip("."))
            elif low.startswith("pageouts:"):
                po = int(line.split()[1].rstrip("."))
        snap["pageins"], snap["pageouts"] = pi, po
    except Exception:
        pass
    try:  # memory_pressure 狀態（System-wide memory free percentage 行）
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10)
        lines = (out.stdout or out.stderr or "").splitlines()
        snap["memory_pressure"] = next((l.strip() for l in lines if "percentage" in l.lower()), "")
    except Exception:
        snap["memory_pressure"] = ""
    try:  # iostat 磁盤活動（最後一行 = 一次樣本）
        out = subprocess.run(["iostat", "-d", "1", "1"], capture_output=True, text=True, timeout=10)
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        snap["iostat"] = lines[-1].split() if lines else []
    except Exception:
        snap["iostat"] = []
    return snap


def cmd_bench(args) -> int:
    matrix = _load("matrix", "llama_bench_matrix.py")
    specs = list(args.arm)
    ok_all = True
    gate_report = []
    for spec in specs:
        profile, env, overrides = _parse_arm(spec)
        ok, diffs, ovr = _base_gate(profile, env, overrides)
        gate_report.append({"arm": spec, "profile": profile, "overrides": ovr,
                            "diffs": diffs, "pass": ok})
        if not ok:
            ok_all = False
            print(f"!! base gate FAIL for {spec}:")
            for d in diffs:
                print(f"    {d}")
    if not ok_all:
        print("base gate 未過 — 拒跑。用 !KEY=VAL 顯式宣告覆蓋，或去掉該 env。")
        return 2

    # 測試前系統快照（thermal / swap / pageins / memory_pressure / iostat）
    t_bench0 = time.time()
    before = _sys_snapshot()
    print(f"系統快照 [before] thermal={before.get('thermal')} swap={before.get('swap_used_mb'):.0f} MiB "
          f"pageins={before.get('pageins')} pageouts={before.get('pageouts')} "
          f"pressure={before.get('memory_pressure')} iostat={before.get('iostat')}", flush=True)

    # delegate 給 llama_bench_matrix（量測邏輯不複製；matrix 不自動建 workdir）
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    arms_joined = ",".join(specs)
    cmd = [PY, str(HERE / "llama_bench_matrix.py"),
           "--arms", arms_joined,
           "--prompt", str(args.prompt), "--gen", str(args.gen),
           "--depths", args.depths, "--reps", str(args.reps),
           "--ctx-size", str(args.ctx_size), "--warm-skip", str(args.warm_skip),
           "--workdir", str(args.workdir), "--json", str(args.json_path)]
    import shutil
    print("$ " + shlex_join(cmd) if False else " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        return rc

    # 測試後系統快照 + 速率（wall 用牆鐘，thermal t 是字串不能用來減）
    after = _sys_snapshot()
    wall = time.time() - t_bench0
    rate = {}
    if before.get("pageins") is not None and after.get("pageins") is not None:
        rate["pageins_per_s"] = (after["pageins"] - before["pageins"]) / wall
        rate["pageouts_per_s"] = ((after.get("pageouts") or 0) - (before.get("pageouts") or 0)) / wall
    print(f"系統快照 [after]  thermal={after.get('thermal')} swap={after.get('swap_used_mb'):.0f} MiB "
          f"pageins={after.get('pageins')} pageouts={after.get('pageouts')} "
          f"pressure={after.get('memory_pressure')} iostat={after.get('iostat')}", flush=True)
    if rate:
        print(f"page 速率: in={rate['pageins_per_s']:.1f}/s out={rate['pageouts_per_s']:.1f}/s "
              f"(wall={wall:.0f}s)", flush=True)

    # 產物注入 base_check + sys_before/after（每個 arm 一臂）
    data = json.loads(Path(args.json_path).read_text())
    for arm, report in zip(data, gate_report):
        arm["base_check"] = {"profile": report["profile"], "pass": report["pass"],
                             "overrides": report["overrides"], "diffs": report["diffs"]}
        arm["sys_before"], arm["sys_after"] = before, after
        if rate:
            arm["sys_rate"] = rate
    Path(args.json_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n產物已寫入 {args.json_path}（每臂含 base_check + sys_before/after + 系統指標）\n", flush=True)

    # 量測紀律輸出：pp + tg + thermal + swap（取自產物欄位）
    for arm in data:
        print(f"=== {arm.get('tag')} (base_check: {'PASS' if arm['base_check']['pass'] else 'FAIL'}) ===")
        for r in arm["rows"]:
            kind = "pp" if r.get("n_prompt", 0) > 0 else "tg"
            ts = r.get("avg_ts")
            print(f"  {kind:2s}: {ts:.2f} t/s" if ts is not None else f"  {kind:2s}: (no avg_ts)")
        th = arm.get("thermal", {})
        print(f"  thermal: launch={th.get('launch')} worst={th.get('worst')}")
        m = arm.get("memory", {})
        ls, es, ws = m.get("launch_swap"), m.get("end_swap"), m.get("worst_swap")
        print(f"  swap: launch={ls} end={es} worst={ws} growth={((es or 0)-(ls or 0)):+.0f} MiB")
        print(f"  attribution: {arm.get('attribution')}")
    return 0


import shlex as _shlex
def shlex_join(parts):
    return _shlex.join(parts)


def cmd_verify(args) -> int:
    """統一的「雙輪驗證」入口：每 arm 跑 clean + instrumented 兩輪、過生產級 gate、出 HTML。

    順序：arm_two_pass（雙輪 + gate + 全 log）→ 只要 result.json 在就 arm_report_html。
    arm_two_pass 回 2（gate 硬擋、有臂沒跑）仍生成 HTML（報告標 blocked），最終回非 0。
    """
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_json = run_dir / "result.json"
    report_html = run_dir / "report.html"

    cmd = [PY, str(HERE / "arm_two_pass.py")]
    for a in args.arm:
        cmd += ["--arm", a]
    cmd += ["--prompt", str(args.prompt), "--gen", str(args.gen),
            "--depths", str(args.depths), "--reps", str(args.reps),
            "--warm-skip", str(args.warm_skip), "--ctx-size", str(args.ctx_size),
            "--max-swap-mb", str(args.max_swap_mb), "--cool-s", str(args.cool_s),
            "--run-dir", str(run_dir), "--json", str(result_json)]
    if args.allow_dirty:
        cmd.append("--allow-dirty")

    print("「verify」雙輪驗證（每 arm：clean 無儀器 + instrumented 有儀器）", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))

    if result_json.exists():
        hcmd = [PY, str(HERE / "arm_report_html.py"),
                "--json", str(result_json), "--out", str(report_html)]
        hrc = subprocess.call(hcmd, cwd=str(ROOT))
        if hrc == 0:
            print(f"\nHTML 報告：{report_html}", flush=True)
    else:
        print("\n（無 result.json — 通常是 gate 在跑任何 GPU 前就擋下；未生成 HTML）", flush=True)

    # 0 = 全部臂兩輪通過；非 0 = 有臂被擋/失敗（HTML 仍可看）
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("show", aliases=("windows",), help="the window board (default)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--need-mb", type=float, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("run", help="wait for a window, then run one tool through it")
    run_p = p
    p.add_argument("tool")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--need-mb", type=float, default=0)
    p.add_argument("--window-timeout-s", type=float, default=900)
    p.add_argument("--poll-s", type=float, default=15)
    p.add_argument("--certify", action="store_true",
                   help="measure throughput against the sentinel reference first (refuses a "
                        "degraded window that looks quiet)")
    p.add_argument("--certify-frac", type=float, default=0.85)
    # NOT argparse.REMAINDER: a REMAINDER positional placed after `tool` swallows the harness's own
    # options too (it is greedy from the first token it cannot name), so `--window-timeout-s 4`
    # was handed to the child and the run waited at the default 900 s instead. The split is done
    # by hand at the first bare `--` in main().
    p.set_defaults(func=cmd_run, rest=[])

    p = sub.add_parser("list", help="the registry")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("audit", help="ungated server launchers (window gate)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("selftest")
    p.set_defaults(func=lambda a: selftest())

    p = sub.add_parser("bench", help="統一量測入口：prod-new 基底 + env 增量 + base gate + llama-bench 完整側")
    p.add_argument("--arm", action="append", required=True,
                   help='PROFILE:!OVERRIDE;KEY=VAL 語法（可多次）。! = 顯式覆蓋 base 鍵；未宣告的 base 變更 → 拒跑')
    p.add_argument("--prompt", type=int, default=_BENCH_DEFAULTS["prompt"])
    p.add_argument("--gen", type=int, default=_BENCH_DEFAULTS["gen"])
    p.add_argument("--depths", default=_BENCH_DEFAULTS["depths"])
    p.add_argument("--reps", type=int, default=_BENCH_DEFAULTS["reps"])
    p.add_argument("--warm-skip", type=int, default=_BENCH_DEFAULTS["warm_skip"])
    p.add_argument("--ctx-size", type=int, default=_BENCH_DEFAULTS["ctx_size"])
    p.add_argument("--workdir", default="/tmp/harness_bench")
    p.add_argument("--json", dest="json_path", required=True, help="產物 json 路徑（含 base_check）")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("verify",
        help="雙輪驗證：每 arm clean（無儀器）+ instrumented（有儀器）+ 生產級 gate + HTML 報告")
    p.add_argument("--arm", action="append", required=True,
                   help='PROFILE:!K=V;K=V（可多臂）。每臂強制兩輪、兩輪只差儀器')
    p.add_argument("--prompt", type=int, default=2048)
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--depths", default="512")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--warm-skip", type=int, default=64)
    p.add_argument("--ctx-size", type=int, default=0)
    p.add_argument("--max-swap-mb", type=float, default=1024,
                   help="起測允許的最大 swap（MiB）；生產嚴格可調 512")
    p.add_argument("--cool-s", type=float, default=0,
                   help="兩輪之間冷卻秒數（預設 0，避免拖慢開發）")
    p.add_argument("--allow-dirty", action="store_true",
                   help="gate 未過也硬跑、報告標紅（不建議用於 commit）")
    p.add_argument("--run-dir", default="/tmp/harness_verify")
    p.set_defaults(func=cmd_verify)

    raw = list(sys.argv[1:] if argv is None else argv)
    cmd = raw[0] if raw and not raw[0].startswith("-") else None
    if cmd == "run":
        # Everything after the first bare `--` belongs to the child, verbatim; everything before it
        # is ours. No guessing which flags belong to whom.
        tail = raw[1:]
        if "--" in tail:
            i = tail.index("--")
            head, rest = tail[:i], tail[i + 1:]
        else:
            head, rest = tail, []
        args = run_p.parse_args(head)
        args.rest = rest
    elif cmd is None:
        args = ap.parse_args(raw + ["show"])
    else:
        args = ap.parse_args(raw)
    if not getattr(args, "func", None):
        args = ap.parse_args(raw + ["show"])
    # 0 means "ask the shared probe what it wants", so a caller that passes nothing inherits the
    # same bar the launcher uses instead of silently demanding zero.
    if getattr(args, "need_mb", 0) == 0:
        try:
            args.need_mb = float(_load("sw", "server_window.py").NEED_MB)
        except Exception:
            args.need_mb = 8000.0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
