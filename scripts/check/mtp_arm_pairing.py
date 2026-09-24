#!/usr/bin/env python3
"""mtp_arm_pairing.py — refuse an MTP on/off pair whose arms differ by more than the mechanism.

WHY THIS EXISTS
---------------
The 2026-09-20 reuse-distance pair (`Backup/phase_decomp/exact_{off,on}.json`) was captured as an
"MTP on vs off" comparison and it was read that way. Diffing the two capture records' own
`server_env_cgc` lists shows it actually differed by SEVEN engine variables, of which two are the
mechanism:

    mechanism (must differ)     CGC_VERIFY_DECODE, CGC_DRAFT_DECODE, CGC_SERVER_MTP, pool size
    confound  (must NOT differ) CGC_MM_BITIDENT, CGC_NO_PREFETCH, CGC_WARM_NPAST,
                                CGC_NO_SEQ_RM_PROBE, CGC_MTP_NO_WARMUP

`CGC_MM_BITIDENT` is the worst of them: it selects a mul_mat KERNEL (ggml-metal-ops.cpp:2470, the
M<=8 M-invariant `mul_mv` path, "bit-identical pillar 1"), and decode's GEMV is M=1 — inside that
range. So one arm of that pair ran a different kernel, and any output difference between the arms
had at least two candidate causes. That is the confound recorded in .workbuddy/memory §EN-193, and
the same one flagged for `plain_match_ab.py`.

`run_server.sh` hoists those knobs (2026-09-19/20) so a caller CAN equalise the arms — hoisting is
permission, not a default change. This script is the other half: it reads the arms' own records and
refuses to let a pair be reported as attributable when it is not.

WHY THE RECORDS AND NOT THE COMMAND LINE
----------------------------------------
The intended env is a claim; the capture record is what the launcher actually resolved (profile
defaults, overrides, `env` last-wins, everything). Only the second can answer "did the two arms run
the same thing".

USAGE
-----
    python3 scripts/check/mtp_arm_pairing.py Backup/phase_decomp/exact_off.json \\
                                            Backup/phase_decomp/exact_on.json
    python3 scripts/check/mtp_arm_pairing.py --selftest

Exit 0 = paired (only the mechanism differs). Exit 2 = REFUSED, with the confounds named.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Variables allowed to differ between an MTP-on and an MTP-off arm, because they ARE the treatment.
MECHANISM = {
    "CGC_SERVER_MTP",     # the switch itself
    "CGC_VERIFY_DECODE",  # verify fast path
    "CGC_DRAFT_DECODE",   # draft fast path
    "CGC_EXPERT_CACHE_BYTES",         # pool size -- allowed only if the caller says so
    "CGC_SERVER_EXPERT_CACHE_BYTES",
}

# Variables that must match. Each one is a behavioural degree of freedom the engine reads outside
# the speculative path, so a difference is a second candidate cause:
#   CGC_MM_BITIDENT      ggml-metal-ops.cpp:2470   -- picks a KERNEL
#   CGC_NO_PREFETCH      llama-context.cpp:2058    -- background slot prefetch
#   CGC_WARM_NPAST       llama-context.cpp:6285    -- decode warm gate
#   CGC_NO_SEQ_RM_PROBE  tools/server/server-context.cpp:1096 -- spec path only, inert at MTP=0
#   CGC_MTP_NO_WARMUP    tools/server/server-context.cpp:1298 -- spec path only, inert at MTP=0
CONFOUNDERS = {
    "CGC_MM_BITIDENT", "CGC_NO_PREFETCH", "CGC_WARM_NPAST",
    "CGC_NO_SEQ_RM_PROBE", "CGC_MTP_NO_WARMUP",
}


def norm(env) -> dict:
    """Capture records store `server_env_cgc` either as a list of KEY=VAL strings or as a dict.

    A bare KEY means the launcher set it without a value (the C++ reads these by presence,
    `getenv(X) != nullptr`), so it is recorded as "1" -- treating it as "" would make a
    presence-only flag look identical to an absent one, which is the bug class this repo has hit
    repeatedly (`CGC_VERIFY_DECODE=0` is still ON: the C++ tests presence, not value).
    """
    if isinstance(env, dict):
        return {k: (str(v) if v is not None else "1") for k, v in env.items()}
    out = {}
    for kv in (env or []):
        k, _, v = str(kv).partition("=")
        out[k] = v or "1"
    return out


def classify(off: dict, on: dict, allow_pool: bool) -> tuple[dict, dict, dict]:
    """Return (mechanism_diff, confound_diff, shared) as {key: (off_value, on_value)}."""
    allowed = set(MECHANISM) if allow_pool else (MECHANISM - {
        "CGC_EXPERT_CACHE_BYTES", "CGC_SERVER_EXPERT_CACHE_BYTES"})
    mech, conf, shared = {}, {}, {}
    for k in sorted(set(off) | set(on)):
        o, n = off.get(k, "<absent>"), on.get(k, "<absent>")
        if o == n:
            shared[k] = o
        elif k in allowed:
            mech[k] = (o, n)
        else:
            conf[k] = (o, n)
    return mech, conf, shared


# The workload identity lives in its own record fields, NOT in server_env_cgc. Comparing only the
# env would let two arms differ by pool size, model or prompt count and still be called a pair --
# and pool size is a first-class variable for this engine (it sets every layer's slot count).
WORKLOAD_FIELDS = ("pool_gb", "temp", "n_prompts", "door")


def workload_diff(ro: dict, rn: dict) -> dict:
    """Fields that must match for the two arms to be the same measurement."""
    d = {}
    for k in WORKLOAD_FIELDS:
        a, b = ro.get(k), rn.get(k)
        if a != b:
            d[k] = (a, b)
    for side, rec in (("off", ro), ("on", rn)):
        m = rec.get("model")
        if isinstance(m, dict):
            m = m.get("name", m.get("realpath"))
        if m is not None:
            d.setdefault("_model", {})[side] = m
    # _model is a side map, not a pair; collapse to a comparison.
    if "_model" in d:
        mm = d.pop("_model")
        if mm.get("off") != mm.get("on"):
            d["model"] = (mm.get("off"), mm.get("on"))
    return d


def check(off_path: Path, on_path: Path, allow_pool: bool = False) -> int:
    # A missing or unreadable record is a REFUSAL with the gate's own exit code, and the message
    # has to say WHICH side is missing. The commonest way to reach this is a capture whose launch
    # was refused by the window guard (the arm never ran, so no record was written) -- and a gate
    # that answers that with a traceback is one a caller learns to run past.
    for side, p in (("off", off_path), ("on", on_path)):
        if not p.exists():
            print(f"REFUSING: no capture record for the {side.upper()} arm at {p}. If that arm's "
                  f"launch was refused (window guard, load failure), nothing was measured and "
                  f"there is no pair to check.")
            return 2
    try:
        ro, rn = json.loads(off_path.read_text())[0], json.loads(on_path.read_text())[0]
    except (json.JSONDecodeError, IndexError, OSError) as e:
        print(f"REFUSING: a capture record could not be read as a one-element JSON list: {e}")
        return 2
    off, on = norm(ro.get("server_env_cgc")), norm(rn.get("server_env_cgc"))
    if not off or not on:
        print("REFUSING: at least one record has no `server_env_cgc`, so the arms' actual "
              "environment cannot be compared. The intended command line is not evidence.")
        return 2
    if str(ro.get("mtp")) == str(rn.get("mtp")):
        print(f"REFUSING: both records say mtp={ro.get('mtp')} -- this is not an on/off pair.")
        return 2

    # The ON arm must actually have the mechanism armed. `CGC_SERVER_MTP=1` is a launcher flag;
    # what makes it the treatment is the verify/draft fast path in the ENGINE. A pair where the ON
    # arm lacks both is "MTP=1 but the spec path was never armed", which is how an earlier A/B read
    # as "no effect" -- and it would pass an env-diff-only check.
    armed = [k for k in ("CGC_VERIFY_DECODE", "CGC_DRAFT_DECODE") if k in on]
    if not armed:
        print(f"REFUSING: the ON arm (mtp={rn.get('mtp')}) carries neither CGC_VERIFY_DECODE nor "
              f"CGC_DRAFT_DECODE, so the verify/draft fast path was never armed on it. There is "
              f"no treatment here to attribute anything to.")
        return 2

    print(f"off: {off_path.name}  (mtp={ro.get('mtp')}, pool {ro.get('pool_gb')} GiB)"
          f"  [{len(off)} engine vars]")
    print(f"on : {on_path.name}  (mtp={rn.get('mtp')}, pool {rn.get('pool_gb')} GiB)"
          f"  [{len(on)} engine vars]")

    mech, conf, shared = classify(off, on, allow_pool)
    wl = workload_diff(ro, rn)
    for k in ("pool_gb",):
        if k in wl and allow_pool:
            wl.pop(k)          # declared deliberate; still printed below
    print(f"\n--- mechanism differences ({len(mech)}) ---")
    for k, (o, n) in mech.items():
        print(f"    {k}: off={o} on={n}")
    print(f"--- shared engine env: {len(shared)} variables identical ---")
    if wl:
        print(f"--- WORKLOAD differences ({len(wl)}) ---")
        for k, (o, n) in wl.items():
            print(f"    {k}: off={o} on={n}")
    if wl:
        print(f"\nREFUSED: {len(wl)} workload field(s) differ ({', '.join(sorted(wl))}). These are "
              f"record fields rather than env vars -- pool size sets every layer's slot count -- so "
              f"the arms are two measurements, not one treatment and one control.")
        print("Pass --allow-pool to declare a pool-size difference deliberate.")
        return 2

    if conf:
        print(f"\nREFUSED: {len(conf)} variable(s) differ that are NOT the mechanism under test.")
        print("Any output or speed difference between these arms has that many extra candidate")
        print("causes, so neither arm's number may be attributed to MTP:")
        for k, (o, n) in conf.items():
            print(f"    {k}: off={o} on={n}")
        print("\nFix by equalising them on the OFF arm (run_server.sh hoists them for this):")
        print("    CGC_SERVER_NO_PREFETCH=1 CGC_MM_BITIDENT=1 CGC_SERVER_WARM_NPAST=0 \\")
        print("    CGC_SERVER_LAYER_CAPS=40-40:256 CGC_SERVER_NO_SEQ_RM_PROBE=1 \\")
        print("    CGC_SERVER_MTP_NO_WARMUP=1")
        print("(and equalise the pool size, or pass --allow-pool to declare it deliberate)")
        return 2

    if not mech:
        print("\nREFUSED: nothing differs at all -- the arms are the same configuration.")
        return 2

    print("\nPAIRED: the only differences are the mechanism under test.")
    return 0


def selftest() -> int:
    bad = 0
    seen = [0]
    tmp = Path("/tmp/_mtp_pairing_off.json")
    tmp2 = Path("/tmp/_mtp_pairing_on.json")

    def write(p, mtp, env, pool=8.0, **extra):
        rec = {"mtp": mtp, "pool_gb": pool, "server_env_cgc": env, "temp": 0.4, "n_prompts": 3,
               "door": "completion", "model": {"name": "m.gguf"}}
        rec.update(extra)
        p.write_text(json.dumps([rec]))

    def run(off_env, on_env, allow_pool=False):
        write(tmp, "0", off_env)
        write(tmp2, "1", on_env)
        return check(tmp, tmp2, allow_pool)

    def chk(name, got, want):
        nonlocal bad
        seen[0] += 1
        ok = got == want
        if not ok:
            bad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r} want {want!r}")

    BASE = ["CGC_MM_BITIDENT=1", "CGC_NO_PREFETCH=1", "CGC_WARM_NPAST=0"]

    # The exact shape the 2026-09-20 pair had, equalised: must PASS.
    chk("equalised arms, only verify/draft differ -> PAIRED",
        run(BASE + ["CGC_SERVER_MTP=0"], BASE + ["CGC_SERVER_MTP=1", "CGC_VERIFY_DECODE=1",
                                                 "CGC_DRAFT_DECODE=1"]), 0)

    # Must-fail: reintroduce the kernel confound. This is the real defect being guarded.
    chk("MM_BITIDENT on one arm only -> REFUSED",
        run(["CGC_NO_PREFETCH=1", "CGC_WARM_NPAST=0"],
            BASE + ["CGC_VERIFY_DECODE=1"]), 2)

    # Must-fail: the warm gate, which is the confound this change newly closed.
    chk("WARM_NPAST absent on the off arm -> REFUSED",
        run(["CGC_MM_BITIDENT=1", "CGC_NO_PREFETCH=1"],
            BASE + ["CGC_WARM_NPAST=0", "CGC_VERIFY_DECODE=1"]), 2)

    # Must-fail: a pool difference is a second variable unless declared. Note the env is IDENTICAL
    # here, so this only passes if the gate reads the record's own pool field rather than the env.
    write(tmp, "0", BASE + ["CGC_SERVER_MTP=0", "CGC_MM_BITIDENT=1"])
    write(tmp2, "1", BASE + ["CGC_SERVER_MTP=1", "CGC_MM_BITIDENT=1", "CGC_VERIFY_DECODE=1"],
          pool=4.0)
    chk("pool differs in the record, env identical -> REFUSED", check(tmp, tmp2), 2)
    chk("pool differs, declared -> PAIRED", check(tmp, tmp2, allow_pool=True), 0)

    # Must-fail: same arm twice is not a pair.
    write(tmp, "1", BASE + ["CGC_VERIFY_DECODE=1"])
    write(tmp2, "1", BASE + ["CGC_VERIFY_DECODE=1"])
    chk("both records say mtp=1 -> REFUSED", check(tmp, tmp2), 2)

    # Must-fail: MTP=1 but the engine-side mechanism was never armed. This is the shape that an
    # env-diff-only check passes and that made an earlier arm read as "no effect".
    chk("ON arm without VERIFY/DRAFT_DECODE -> REFUSED (no treatment present)",
        run(BASE + ["CGC_SERVER_MTP=0"], BASE + ["CGC_SERVER_MTP=1"]), 2)

    # Must-fail: a different model is not a pair.
    write(tmp, "0", BASE + ["CGC_SERVER_MTP=0", "CGC_MM_BITIDENT=1"])
    write(tmp2, "1", BASE + ["CGC_SERVER_MTP=1", "CGC_MM_BITIDENT=1", "CGC_VERIFY_DECODE=1"],
          model={"name": "other.gguf"})
    chk("different model -> REFUSED", check(tmp, tmp2), 2)

    # Must-fail: a record with no env is not evidence.
    tmp.write_text(json.dumps([{"mtp": "0", "pool_gb": 8.0}]))
    tmp2.write_text(json.dumps([{"mtp": "1", "pool_gb": 8.0}]))
    chk("record without server_env_cgc -> REFUSED", check(tmp, tmp2), 2)

    # The presence-vs-value distinction: a bare KEY must not read as equal to an explicit value.
    chk("bare KEY vs KEY=1 is a difference, not a match",
        norm(["CGC_SPAC"]) == {"CGC_SPAC": "1"}, True)

    tmp.unlink()
    tmp2.unlink()
    print(f"\nselftest: {seen[0]} checks, {bad} failed")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="*", help="two capture records: OFF then ON")
    ap.add_argument("--allow-pool", action="store_true",
                    help="declare a pool-size difference deliberate (it is still reported)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if len(a.records) != 2:
        print("REFUSING: give exactly two capture records (off, on) or --selftest")
        return 2
    return check(Path(a.records[0]), Path(a.records[1]), a.allow_pool)


if __name__ == "__main__":
    sys.exit(main())
