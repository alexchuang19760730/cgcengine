#!/usr/bin/env python3
"""gate_registry.py -- the G0..G7 gate registry: one machine-readable home.

WHY THIS EXISTS
---------------
G0..G7 used to live only in markdown (docs/CROSS_LINE_STATUS_AND_TARGET_MATH_*.md,
docs/G1_G7_SWEEP_*.md) and in dated memory sections. That had three consequences:

  1. no tool could answer "which gates are open" -- it was a human lookup;
  2. "G2 is met" could be asserted without saying WHICH probe it was read at.
     The 9-record short probe and the 884-record long probe are different tests;
     9/9 does not imply 884/884;
  3. G5 had no object set at all -- only `evidence: "same rule as G2"` -- so it
     could not fail anything. A gate with no object set is an unfilled cell, not
     a gate.

This module is declarative: it names each gate's object set, its criterion, and
the command that re-checks it. It does NOT invent verdicts. `--check` runs the
real gate and records only what the real gate printed.

THE TWO PROBE SIZES ARE NOT INTERCHANGEABLE
-------------------------------------------
  short probe -- "15+27 等於多少？請只輸出答案" + 48 tokens -> 9 oracle records.
                 Enough to see that a knob MOVED the logits. NOT enough to say
                 a choice survives: greedy decoding is chaotic and five steps is
                 five coin flips.
  long probe  -- the 巴黎 prompt + 400 tokens -> 884 oracle records.
                 This is the sample size the LAYER_CAPS verdict was read at.

Any claim of bitwise equivalence must say which one it was read at.

G2 AND G5 CANNOT BE "CLOSED"
----------------------------
They are constraints, not levers: they are re-run per change. Recording them as
"met" once would be a category error. Their status means:
  open            -- nothing has been checked against them yet in this baseline
  open-passing    -- checked, PASS, at the probe recorded in status.json
  open-failing    -- checked, FAIL

USAGE
    python3 scripts/check/gate_registry.py --list
    python3 scripts/check/gate_registry.py --check G2                # short probe
    python3 scripts/check/gate_registry.py --check G2 --probe long   # 884 records
    python3 scripts/check/gate_registry.py --check G5                # long probe only
    python3 scripts/check/gate_registry.py --selftest
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "Backup" / "gate_registry"
STATUS_PATH = RESULT_DIR / "status.json"

# The 884-record long probe. The reference .cap records only `probe_answer`, not
# the prompt, so the prompt had to be recovered from a run script (2026-09-20).
# Without the exact prompt the token sequence differs and the 884 common keys
# do not exist -- the comparison silently becomes "only_A=884, only_B=884".
LONG_PROBE_PROMPT = "請用繁體中文寫一段約三百字的短文，介紹巴黎的歷史、建築與文化，並說明它們之間的關係。"
LONG_PROBE_MAX_TOKENS = 400
LONG_REF = ROOT / "Backup" / "phase_decomp" / "oracle_long_base2_20260919.jsonl"

GATE_TOOL = "scripts/check/m123_oracle_gate.py"


def _long_args():
    return ["--profile", "prod25", "--ref", str(LONG_REF),
            "--probe-prompt", LONG_PROBE_PROMPT,
            "--probe-max-tokens", str(LONG_PROBE_MAX_TOKENS)]


GATES = [
    {
        "id": "G0",
        "title": "instrument determinism",
        "kind": "precondition",
        "status": "met",
        "object_set": "the measurement instruments themselves",
        "gate": "same shape twice -> same reading within tolerance",
        "check": None,
        "note": "met; nothing left to do.",
    },
    {
        "id": "G1",
        "title": "hide the gap",
        "kind": "lever",
        "status": "open-unreachable",
        "object_set": "host-side wait / boundary count in the submit loop",
        "gate": "gap_sum/total <= 5%",
        "check": None,
        "note": "proved unreachable: attainable lower bound 9.0% > 5%. The x1.702 probe "
                "is arithmetically non-order-preserving and is structurally vetoed by G2.",
    },
    {
        "id": "G2",
        "title": "bitwise equivalence",
        "kind": "constraint",
        "status": "open",
        "object_set": "any change that CLAIMS not to alter the numerics -- fusion, victim "
                      "rule, pool geometry, prefetch, barrier placement",
        "gate": "M1 (bit-identical logits) == 1.0 AND M2 (argmax agreement) == 1.0",
        "check": GATE_TOOL,
        "probe": {
            "short": {"n_records": 9, "args": []},
            "long": {"n_records": 884, "args": _long_args()},
        },
        "note": "per-commit gate reads the 9-record probe; any claim of equivalence over "
                "884 steps must be read at the 884-record probe. Different tests.",
    },
    {
        "id": "G3",
        "title": "union compressibility",
        "kind": "precondition",
        "status": "met",
        "object_set": "40-segment strict ordering",
        "gate": "strict order holds",
        "check": None,
        "note": "met.",
    },
    {
        "id": "G4",
        "title": "shrink the union",
        "kind": "lever",
        "status": "closed-not-worth-doing",
        "object_set": "dispatch count per step",
        "gate": "union <= 38*mean_len (union gate, threshold -12.8%)",
        "check": None,
        "note": "2026-09-20: measured price of one dispatch = 0.0736% (ON 10.434 vs OFF "
                "9.620 t/s over delta 115 dispatch/step, both arms sentinel-HEALTHY). "
                "Threshold needs ~174 dispatch/step removed; cluster 1 offers 72 (5.3%).",
    },
    {
        "id": "G5",
        "title": "the union does not change the output",
        "kind": "constraint",
        "status": "open",
        "object_set": "IN: anything that moves WHEN/WHERE work happens but not WHAT is "
                      "computed -- G4-family fusion, victim rule, pool geometry, prefetch, "
                      "barrier placement. OUT (these are NOT governed by G5, they change "
                      "the output by construction): LAYER_CAPS, accept rule, k, sampler "
                      "parity, quantisation, any change to reduction order.",
        "gate": "same rule as G2, but read at the LONG probe (884 records): M1 == 1.0 AND "
                "M2 == 1.0. Long probe required: a scheduling change can be bit-identical "
                "for 9 steps and diverge later.",
        "check": GATE_TOOL,
        "probe": {
            "long": {"n_records": 884, "args": _long_args()},
        },
        "note": "Before 2026-09-20 this gate had no object set, only "
                "`evidence: same rule as G2` -- it could not fail anything.",
    },
    {
        "id": "G6",
        "title": "accept rule",
        "kind": "lever",
        "status": "open",
        "object_set": "MTP draft acceptance policy",
        "gate": "net positive t/s after the change",
        "check": None,
        "note": "k=3 ceiling is 14.9 t/s. Measured currently NET NEGATIVE: +0.375 mean_len "
                "but -0.45 t/s (step time itself grew 14%) -- the step time, not the "
                "threshold, is what needs fixing.",
    },
    {
        "id": "G7",
        "title": "decode >= 25 t/s",
        "kind": "delivery",
        "status": "not-met",
        "object_set": "the delivered decode axis",
        "gate": "decode >= 25 t/s on the delivery shape",
        "check": None,
        "note": "arithmetically unreachable by engine work alone: k=3 => mean_len <= 4 => "
                "ceiling 14.9. 2026-09-20 delivery reading 10.69 t/s (NOMINAL throughout, "
                "sentinel-HEALTHY). The only route to 25 is changing the recipe "
                "(k / quantisation / model).",
    },
]

BY_ID = {g["id"]: g for g in GATES}


def cmd_list():
    print("%-4s %-22s %-14s %s" % ("ID", "TITLE", "KIND", "STATUS"))
    print("-" * 78)
    for g in GATES:
        print("%-4s %-22s %-14s %s" % (g["id"], g["title"][:22], g["kind"], g["status"]))
    print()
    for g in GATES:
        print("[%s] %s -- %s" % (g["id"], g["title"], g["kind"]))
        print("    gate      : %s" % g["gate"])
        print("    objects   : %s" % g["object_set"])
        if g.get("probe"):
            for k, v in g["probe"].items():
                print("    probe %-5s: %d records" % (k, v["n_records"]))
        if g.get("note"):
            print("    note      : %s" % g["note"])
    return 0


def cmd_check(gid, probe=None, tag=""):
    g = BY_ID.get(gid)
    if g is None:
        print("no such gate: %s (have %s)" % (gid, ",".join(BY_ID)), file=sys.stderr)
        return 2
    if not g.get("check"):
        print("%s has no automated check -- it is read by hand. gate: %s"
              % (gid, g["gate"]))
        return 0
    probes = g.get("probe") or {}
    if not probes:
        print("%s has no probe defined" % gid, file=sys.stderr)
        return 2
    if probe is None:
        probe = "long" if "long" in probes else sorted(probes)[0]
    if probe not in probes:
        print("%s has no probe %r (have %s)" % (gid, probe, ",".join(probes)),
              file=sys.stderr)
        return 2
    p = probes[probe]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if not tag:
        tag = "%s_%s_%s" % (gid.lower(), probe, time.strftime("%Y%m%d_%H%M%S"))
    argv = [sys.executable, str(ROOT / g["check"]), "--tag", tag] + list(p["args"])
    print("+ %s" % " ".join(argv[1:]), flush=True)
    t0 = time.time()
    r = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
    out = r.stdout + r.stderr
    (RESULT_DIR / ("check_%s.log" % tag)).write_text(out, errors="replace")
    print(out[-3000:])
    print("--- %.0fs, rc=%d, log=%s" % (time.time() - t0, r.returncode,
                                        RESULT_DIR / ("check_%s.log" % tag)))
    # Record only what the gate itself printed.
    m1 = m2 = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("VERDICT M1"):
            m1 = "PASS" if "PASS" in s else ("FAIL" if "FAIL" in s else None)
        elif s.startswith("VERDICT M2"):
            m2 = "PASS" if "PASS" in s else ("FAIL" if "FAIL" in s else None)
    rec = {"gate": gid, "probe": probe, "n_records_expected": p["n_records"],
           "tag": tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "rc": r.returncode, "m1": m1, "m2": m2}
    history = []
    if STATUS_PATH.exists():
        try:
            history = json.loads(STATUS_PATH.read_text()).get("history", [])
        except Exception:
            history = []
    history.append(rec)
    STATUS_PATH.write_text(json.dumps({"history": history}, ensure_ascii=False, indent=2) + "\n")
    print("recorded: %s" % json.dumps(rec, ensure_ascii=False))
    return 0 if (m1 == "PASS" and m2 == "PASS") else 1


# Fingerprint of the 884-record reference as read on 2026-09-20.
#
# WHY: this ref is NOT in m123_oracle_gate.py's REF_PINS (the gate itself reports
# `ref_pinned: false`), so `--write-ref` can overwrite it silently. A reference
# that can be re-baselined without anyone noticing stops being a comparison:
# it becomes the binary agreeing with itself. Pinning it means editing
# m123_oracle_gate.py, which is another line's uncommitted work -- so instead we
# detect the overwrite here, without touching that file.
LONG_REF_MD5 = "01da069c669710395a6ea6c862699746"
LONG_REF_BYTES = None  # filled on first --verify-ref; size alone catches most clobbers


def cmd_verify_ref():
    import hashlib
    if not LONG_REF.exists():
        print("MISSING: %s" % LONG_REF)
        return 1
    raw = LONG_REF.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    n = sum(1 for _ in raw.splitlines() if _.strip())
    ok = (md5 == LONG_REF_MD5)
    print("ref      : %s" % LONG_REF)
    print("md5      : %s  (expected %s)" % (md5, LONG_REF_MD5))
    print("records  : %d  (expected 884)" % n)
    if ok and n == 884:
        print("VERDICT  : INTACT -- the 884 reference was not re-baselined.")
        return 0
    print("VERDICT  : CHANGED -- do not quote 884/884 until you know why. If this was "
          "a deliberate --write-ref, update LONG_REF_MD5 here in the same commit.")
    return 1


def cmd_selftest():
    ok = 0
    n = 0

    def check(name, cond):
        nonlocal ok, n
        n += 1
        print("  %-58s %s" % (name, "ok" if cond else "FAIL"))
        ok += 1 if cond else 0

    ids = [g["id"] for g in GATES]
    check("G0..G7 all present", ids == ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"])
    check("ids unique", len(ids) == len(set(ids)))
    check("every gate has object_set", all(g.get("object_set") for g in GATES))
    check("every gate has gate criterion", all(g.get("gate") for g in GATES))
    check("every gate has a status", all(g.get("status") for g in GATES))
    # A gate that claims to be checkable must name a probe.
    check("checkable gates name probes",
          all(g.get("probe") for g in GATES if g.get("check")))
    # G5 must be long-probe only: that is the whole point of its object set.
    g5 = BY_ID["G5"]
    check("G5 probe is long-only", list((g5.get("probe") or {})) == ["long"])
    check("G2 has both probes",
          set((BY_ID["G2"].get("probe") or {})) == {"short", "long"})
    check("long ref path is absolute", str(LONG_REF).startswith("/"))
    check("G5 object set has an OUT clause", "OUT" in g5["object_set"])
    check("no gate claims met without a check or a note",
          all(g.get("status") != "met" or g.get("note") for g in GATES))
    print("\nselftest: %d/%d" % (ok, n))
    return 0 if ok == n else 1


def main():
    ap = argparse.ArgumentParser(description="G0..G7 gate registry")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", metavar="ID")
    ap.add_argument("--probe", choices=["short", "long"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--verify-ref", action="store_true",
                    help="check the 884 reference has not been silently re-baselined")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return cmd_selftest()
    if args.verify_ref:
        return cmd_verify_ref()
    if args.list or not args.check:
        if args.check:
            return cmd_check(args.check, args.probe, args.tag)
        return cmd_list()
    return cmd_check(args.check, args.probe, args.tag)


if __name__ == "__main__":
    sys.exit(main())
