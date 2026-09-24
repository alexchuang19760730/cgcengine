#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""io_symmetry.py -- did a paired A/B actually compare the same I/O structure?

The defect this exists to block (2026-09-24, `Backup/seg_batch_s1_pairs/abba_212809.json`):
an ABBA pair was quoted as the single-submit arm being 2.2x the segmented arm. Both arms ran
the same cell, the same build, and neither passed `--spec-type`. But the two arms' own logs say:

    A (41 segments, per-layer hook fires):  miss 4835   file_reads 82293   read 2440 MiB   hit 96.2%
    B (single submit, hook never fires):    miss    0   file_reads     0   read    0 MiB   hit 100.0%

B performed no file reads at all. "No hook -> no demand fill" means the fill path never ran,
so B's per-token cost contains no cache-I/O term, while A's does. A ratio between an arm that
paid for the I/O and an arm that structurally could not is therefore NOT a shape gain: part of
it is the I/O itself. Same family as the `fill_job` missing-bytes counter found on 2026-09-23 --
a quotient whose numerator and denominator are not the same quantity.

So: before a paired ratio is quoted, check that both arms' I/O structure agrees. Refuse
(exit 2, banner) when it does not, and refuse when either arm's counters cannot be read
(fail-closed: an unreadable counter is UNKNOWN, not "equal").

Usage:
    python3 scripts/check/io_symmetry.py --dir Backup/seg_batch_s1_pairs
    python3 scripts/check/io_symmetry.py --arm A=.../stdout_A1.log --arm B=.../stdout_B2.log
    python3 scripts/check/io_symmetry.py --selftest
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

# The counters that carry the claim "these two arms did the same work per token".
# `file_reads`/`reads` and `read_mib` are the decisive pair: zero-vs-nonzero is structural,
# not a dose. `hit_pct` is the cheap cross-check. `fill_wait_us` is the host-side wait.
FIELDS = ("hit_pct", "misses", "file_reads", "read_mib", "fill_wait_us", "prefetch")

# A hit-rate gap this large cannot be drift between two launches of the same configuration.
HIT_PCT_TOL = 2.0

# The engine prints two different lines; both are accepted, neither is preferred.
RE_FINAL = re.compile(
    r"final stats: runtime requests=(?P<req>\d+) hits=(?P<hits>\d+) misses=(?P<misses>\d+)"
    r"\s*\(hit rate (?P<hit_pct>[\d.]+)%\)(?P<mid>.*?)file_reads=(?P<file_reads>\d+)"
    r" pread_usec=(?P<pread_usec>\d+) fill_batch_usec=(?P<fill_batch_usec>\d+)"
    r" fill_wait_us=(?P<fill_wait_us>\d+) prefetch=(?P<pf_hit>\d+)/(?P<pf_try>\d+)")
RE_CACHE = re.compile(
    r"cache: hit (?P<hit_pct>[\d.]+)%\s+misses (?P<misses>\d+)\s+reads (?P<file_reads>\d+)")
RE_READ_MIB = re.compile(r"read_mib=(?P<read_mib>[\d.]+)")


def counters_from(text: str) -> dict:
    """Pull one arm's I/O structure out of a log. Returns {} when nothing parseable is present."""
    m = RE_FINAL.search(text)
    if m:
        g = m.groupdict()
        c = {k: float(g[k]) for k in ("hit_pct", "misses", "file_reads", "fill_wait_us", "pf_try")}
        c["fill_wait_us"] = c["fill_wait_us"] / 1e6            # the engine prints microseconds
        c["prefetch_tried"] = c.pop("pf_try")
        # `read_mib` is not on this line: the engine prints it on the separate `CGC-SHAPE phase=final`
        # line, so scope the search to what follows the final-stats match rather than the whole file
        # (a log holds several CGC-SHAPE lines and only the last one is the run total).
        rm = RE_READ_MIB.search(text[m.start():])
        if rm:
            c["read_mib"] = float(rm.group("read_mib"))
        return c
    m = RE_CACHE.search(text)
    if m:
        g = m.groupdict()
        return {k: float(g[k]) for k in ("hit_pct", "misses", "file_reads")}
    return {}


def _says_io(c: dict) -> bool:
    """Did this arm perform cache file reads at all?"""
    return (c.get("file_reads", 0) or 0) > 0 or (c.get("read_mib", 0) or 0) > 0


def compare(a: dict, b: dict, name_a="A", name_b="B"):
    """Verdict on whether a ratio between these two arms can be quoted as an effect."""
    missing = [f for f in ("file_reads",) if f not in a or f not in b]
    if missing:
        return {"verdict": "unknown", "blocking": True, "differing": [],
                "why": f"counter(s) {missing} unreadable in at least one arm -- "
                       f"an unreadable counter is UNKNOWN, not 'equal'"}
    diff = []
    if _says_io(a) != _says_io(b):
        io_arm, no_arm = (name_a, name_b) if _says_io(a) else (name_b, name_a)
        diff.append(f"{io_arm} performed cache I/O ({int(max(a.get('file_reads', 0), 0))} reads) "
                    f"while {no_arm} performed none -- the fill path never ran in {no_arm}")
    elif abs(a.get("hit_pct", 0) - b.get("hit_pct", 0)) > HIT_PCT_TOL:
        diff.append(f"hit rate {a.get('hit_pct')}% vs {b.get('hit_pct')}% "
                    f"(> {HIT_PCT_TOL}pp)")
    fa, fb = a.get("fill_wait_us", 0) or 0, b.get("fill_wait_us", 0) or 0
    if (fa > 1e3) != (fb > 1e3):
        diff.append(f"fill_wait {fa:.3f}s vs {fb:.3f}s -- one arm waited on fills, the other never did")
    if diff:
        return {"verdict": "asymmetric", "blocking": True, "differing": diff,
                "why": "the two arms did not do the same I/O work per token, so a ratio between "
                       "them is NOT a shape gain: part of it is the I/O itself"}
    return {"verdict": "comparable", "blocking": False, "differing": [],
            "why": "both arms' I/O structure agrees within tolerance"}


def logs_in(d: Path):
    """tag -> log path, for the drivers that write `stdout_<tag>.log` next to the artifact."""
    out = {}
    for p in sorted(Path(d).glob("stdout_*.log")):
        out[p.stem.split("_", 1)[1]] = p
    for p in sorted(Path(d).glob("*.stderr.log")):
        out.setdefault("__combined__" + p.stem, p)
    return out


def audit_dir(d):
    """Audit a finished ABBA directory: per-arm counters, the paired ratio, and the verdict."""
    d = Path(d)
    arts = sorted(d.glob("abba_*.json"))
    if not arts:
        return None
    art_path = arts[-1]
    art = json.loads(art_path.read_text())
    logs = logs_in(d)
    per_arm, tcps = {}, {}
    for rec in art.get("records", []):
        tag, arm = rec.get("tag"), rec.get("arm")
        p = logs.get(tag)
        if p:
            c = counters_from(p.read_text(errors="replace"))
            if c:
                per_arm.setdefault(arm, []).append((tag, c))
        rows = [r for r in rec.get("rows", []) if r.get("n_gen")]
        if rows:
            tcps.setdefault(arm, []).append(rows[0]["avg_ts"])
    arms = sorted(per_arm)
    res = {"artifact": str(art_path), "dir": str(d),
           "tg_median_tps": {a: st.median(v) for a, v in tcps.items()}}
    if len(tcps) == 2:
        # alphabetical arm order ("A" then "B"), so the ratio is always the second arm over the first
        ma, mb = (st.median(tcps[a]) for a in sorted(tcps))
        res["ratio_of_medians"] = mb / ma if ma else None
    if len(per_arm) != 2:
        res.update({"verdict": "unknown", "blocking": True,
                    "why": f"counters found for {len(per_arm)} arm(s), need 2"})
        return res
    name_a, name_b = arms
    ca = _merge([c for _, c in per_arm[name_a]])
    cb = _merge([c for _, c in per_arm[name_b]])
    res["counters"] = {name_a: ca, name_b: cb}
    res.update(compare(ca, cb, name_a, name_b))
    return res


def _merge(cs):
    """One arm across reps: worst-case (max) for every counter, so symmetry is not hidden by mixing."""
    keys = set().union(*[set(c) for c in cs])
    return {k: max(c.get(k, 0) or 0 for c in cs) for k in keys}


def banner(res) -> str:
    if res.get("verdict") == "comparable":
        return f"I/O SYMMETRY: comparable -- {res['why']}"
    return (f"\n{'!' * 78}\n!! I/O STRUCTURE ASYMMETRIC -- DO NOT QUOTE THIS RATIO AS A SHAPE GAIN\n"
            + "".join(f"!!   * {x}\n" for x in res.get("differing", []))
            + f"!! {res['why']}\n"
            + (f"!! ratio {res['ratio_of_medians']:.2f}x spans arms that did different I/O work.\n"
               if res.get("ratio_of_medians") else "")
            + "!" * 78)


# --------------------------------------------------------------------------- selftest

# Verbatim from the two arms' own logs in Backup/seg_batch_s1_pairs/ (2026-09-24).
FIX_FINAL_A = ("llama_expert_cache: final stats: runtime requests=128920 hits=124085 misses=4835 "
               "(hit rate 96.2%)  prewarm req=0 hit=0 miss=0  resident=6197.04 MiB file_reads=82293 "
               "pread_usec=1859711006 fill_batch_usec=24488195 fill_wait_us=3375682 prefetch=0/0\n"
               "CGC-SHAPE v=1 phase=final M=8 M_stat=DEFAULTED width=8 union=64 fits=1 "
               "pool_cap_slots=143 slots_layer=143 n_layer=40 req=128920 hits=124085 misses=4835 "
               "hit_pct=96.25 compulsory=4076 capacity=759 evict=4835 zero_mapped=0 "
               "verify_refused=0 inv_viol=0 read_mib=2440.3 pread_us=1859711006 "
               "fill_wait_us=3375682 final_counters=1")
FIX_FINAL_B = ("llama_expert_cache: final stats: runtime requests=5720 hits=5720 misses=0 "
               "(hit rate 100.0%)  prewarm req=0 hit=0 miss=0  resident=6197.04 MiB file_reads=0 "
               "pread_usec=0 fill_batch_usec=0 fill_wait_us=0 prefetch=0/0\n"
               "CGC-SHAPE v=1 phase=final M=8 width=8 union=64 fits=1 pool_cap_slots=143 "
               "hit_pct=100.00 compulsory=0 capacity=0 evict=0 read_mib=0.0 pread_us=0")
FIX_CACHE_SYM = "  cache: hit 96.1%  misses 4775  reads 82100  us/job 1800  cap% 17.0"


def self_test() -> int:
    bad = 0

    def chk(name, cond):
        nonlocal bad
        if not cond:
            bad += 1
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    ca, cb = counters_from(FIX_FINAL_A), counters_from(FIX_FINAL_B)
    chk("reads both parsed", ca.get("file_reads") == 82293 and cb.get("file_reads") == 0)
    chk("us -> seconds for fill_wait", abs(ca.get("fill_wait_us", 0) - 3.375682) < 1e-4)
    chk("read_mib taken from the phase=final line, not the first CGC-SHAPE one",
        ca.get("read_mib") == 2440.3 and cb.get("read_mib") == 0.0)

    # the real defect must be BLOCKED
    r = compare(ca, cb)
    chk("zero-vs-82293 reads blocks the ratio", r["verdict"] == "asymmetric" and r["blocking"])
    chk("and the reason names the arm that did no I/O", "performed none" in " ".join(r["differing"]))

    # the same configuration must NOT be blocked (or the rule is useless)
    sym = counters_from(FIX_CACHE_SYM)
    chk("same-config arms stay comparable", compare(ca, sym)["verdict"] == "comparable")

    # fail-closed: unreadable counters are UNKNOWN, never 'equal'
    r2 = compare({}, {})
    chk("missing counters -> unknown + blocking", r2["verdict"] == "unknown" and r2["blocking"])
    chk("and it says so, not 'comparable'", "UNKNOWN" in r2["why"])

    # a third case the simple zero-vs-nonzero test misses: both read, one far more
    big = dict(ca, file_reads=200000, read_mib=9000.0, hit_pct=96.2)
    chk("volume-only difference is not silently structural",
        compare(ca, big)["verdict"] == "comparable")
    chk("but a hit-rate gap is", compare(dict(ca, hit_pct=96.2), dict(cb, hit_pct=88.0))["verdict"]
        == "asymmetric")

    # banner must be loud and must not claim a gain
    b = banner(r)
    chk("banner refuses the quote", "DO NOT QUOTE" in b and "2.2x" not in b)
    print(f"\nio_symmetry selftest: {'PASS' if bad == 0 else f'{bad} FAILED'}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="a finished ABBA directory (looks for abba_*.json)")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=LOG")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return self_test()

    if a.dir:
        res = audit_dir(a.dir)
        if res is None:
            print(f"no abba_*.json under {a.dir}")
            return 2
        print(json.dumps(res, indent=1, ensure_ascii=False))
        print(banner(res))
        return 2 if res.get("blocking") else 0

    if len(a.arm) == 2:
        cs = {}
        for spec in a.arm:
            name, _, path = spec.partition("=")
            cs[name] = counters_from(Path(path).read_text(errors="replace"))
        names = sorted(cs)
        res = compare(cs[names[0]], cs[names[1]], names[0], names[1])
        print(json.dumps({"counters": cs, **res}, indent=1, ensure_ascii=False))
        print(banner(res))
        return 2 if res["blocking"] else 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
