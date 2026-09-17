#!/usr/bin/env python3
"""Offline selftest for the PRE-LAUNCH FEASIBILITY GATE in `knifeedge_matrix.py`.

Why a test at all: the gate's whole job is to REFUSE a launch, and a refusal is invisible in the
output -- a cell that is never measured simply has no row, which is indistinguishable from a cell
that was forgotten. So the decision must be pinned by something runnable, not by reading the code.

What it covers:

  * the three verdicts (USABLE / PARTITION-REF / UNUSABLE) through the REAL decision path
    (`feasibility_cell` -> `pool_feasibility.evaluate`), driven by synthetic layer caps so no GGUF
    and no GPU are needed;
  * the P2 floor moving with MTP (`n_rs_seq = 3` -> cap >= 5), because that is the difference
    between "runs at a smaller cap" and "cannot run at all";
  * the override matrix (--allow-partition-ref / --allow-unusable / --no-feasibility-gate);
  * that a MISSING model warns and continues instead of blocking a run for a harness reason;
  * that `run_combo` refuses BEFORE touching the machine (no kill, no launch).

Usage:
    python3 scripts/check/feasibility_gate_selftest.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load_matrix():
    path = os.path.join(HERE, "knifeedge_matrix.py")
    spec = importlib.util.spec_from_file_location("km_selftest", path)
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)
    return km


# A synthetic geometry: only the fields `pool_feasibility.evaluate` reads. 1 MiB per expert per
# layer keeps the slot arithmetic readable, and the layer caps below override the result anyway --
# which is what makes this test independent of any particular GGUF.
FAKE_GEOM = {"max_layer": 41, "per_slot": 1 << 20, "binding_il": 0, "n_nextn": 1,
             "n_decoder": 40, "block_count": 41, "n_expert": 256, "topk": 8, "arch": "fake",
             "per_layer": {0: 1 << 20}, "per_layer_kinds": {}}

# (layer_caps, mtp flag, ordered cap to request, expected verdict, expected cap_max, expected floor)
CASES = [
    # 40 slots -> 39 usable -> cap_max 4, below the MTP floor 5: no cap works at all.
    ("0-40:40", True, None, "UNUSABLE", 4, 5),
    # 56 slots -> 55 usable -> cap_max 6: runs, but the cap has to fall from 8 to 6.
    ("0-40:56", True, None, "PARTITION-REF", 6, 5),
    # 72 slots -> 71 usable -> cap_max 8: the reference cap fits, so a cross-cell diff is pool-only.
    ("0-40:72", True, None, "USABLE", 8, 5),
    # Same cell, smaller cap requested explicitly: runs, but only comparable at that same cap.
    ("0-40:72", True, 6, "USABLE", 8, 5),
    # Requesting cap 9 for a cell whose cap_max is 8 forces the engine to fall back -> partition.
    ("0-40:72", True, 9, "PARTITION-REF", 8, 5),
    # MTP off drops the floor to 1, so the 40-slot cell becomes comparable-with-caveat, not dead.
    ("0-40:40", False, None, "PARTITION-REF", 4, 1),
]


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    return ok


def main():
    km = load_matrix()
    # 2026-09-17（E4 item 2）：knifeedge_matrix.py 拆成套件，pool_geometry 現在住在
    # knifeedge/feasibility.py。`feasibility_cell` 讀的是**那個模組**的全域，所以替身要打在
    # 那裡。打在 km 上只會改到 shim 的綁定，而它沒有任何讀者 —— 這件事本身靜默，
    # 症狀是這個測試掉到真的 GGUF 路徑（ModuleNotFoundError: numpy）。
    km.feasibility.pool_geometry = lambda kind: dict(FAKE_GEOM)   # no GGUF, no GPU
    fails = 0

    print("1. verdicts through the real decision path (synthetic geometry + layer caps)")
    for layer_caps, mtp, cap, verdict, cap_max, floor in CASES:
        extra = [f"CGC_SERVER_LAYER_CAPS={layer_caps}", f"CGC_SERVER_MTP={1 if mtp else 0}"]
        cell = km.feasibility_cell("iq3", 8, cap, extra)
        tag = f"caps={layer_caps} mtp={int(mtp)} cap={cap}"
        fails += not check(tag + " verdict", cell["verdict"], verdict)
        fails += not check(tag + " cap_max", cell["cap_max"], cap_max)
        fails += not check(tag + " floor", cell["floor"], floor)

    print("2. gate: refuse, and keep refusing the right way under overrides")
    cases = [
        ("UNUSABLE refuses", "0-40:40", True, {}, False),
        ("PARTITION-REF refuses", "0-40:56", True, {}, False),
        ("USABLE passes", "0-40:72", True, {}, True),
        ("--allow-partition-ref lets it through", "0-40:56", True,
         {"allow_partition_ref": True}, True),
        ("--allow-unusable lets it through", "0-40:40", True, {"allow_unusable": True}, True),
        ("--no-feasibility-gate skips the check", "0-40:40", True,
         {"no_feasibility_gate": True}, True),
        # The wrong override must not open the wrong door.
        ("--allow-unusable does NOT wave through PARTITION-REF", "0-40:56", True,
         {"allow_unusable": True}, False),
    ]
    for name, layer_caps, _mtp, flags, want_ok in cases:
        args = argparse.Namespace(**{"allow_partition_ref": False, "allow_unusable": False,
                                     "no_feasibility_gate": False, **flags})
        ok, _cell = km.feasibility_gate("iq3", 8, args, None,
                                        [f"CGC_SERVER_LAYER_CAPS={layer_caps}"], where="selftest")
        fails += not check(name, ok, want_ok)

    print("2b. --attach is not judged (that server's cap/pool are fixed outside this process)")
    args = argparse.Namespace(allow_partition_ref=False, allow_unusable=False,
                              no_feasibility_gate=False, attach=True)
    ok, cell = km.feasibility_gate("iq3", 8, args, None,
                                   ["CGC_SERVER_LAYER_CAPS=0-40:40"], where="selftest")
    fails += not check("attach -> ok", ok, True)
    fails += not check("attach -> no cell", cell, None)

    print("3. a missing model warns and continues (a predictor must not block a run)")
    km.feasibility.pool_geometry = lambda kind: (_ for _ in ()).throw(RuntimeError("no such file"))
    args = argparse.Namespace(allow_partition_ref=False, allow_unusable=False,
                              no_feasibility_gate=False)
    ok, cell = km.feasibility_gate("iq3", 8, args, None, [], where="selftest")
    fails += not check("missing geometry -> ok", ok, True)
    fails += not check("missing geometry -> no cell", cell, None)

    print("4. run_combo refuses BEFORE touching the machine")
    km.feasibility.pool_geometry = lambda kind: dict(FAKE_GEOM)

    def boom(*a, **k):
        raise AssertionError("the machine was touched for an infeasible cell")

    km.kill_servers = boom
    km.launch = boom
    args = argparse.Namespace(tag="selftest", force=False, extra_env=["CGC_SERVER_LAYER_CAPS=0-40:40"],
                              pool_cap_resolved={"iq3": None}, pool_cap="auto",
                              allow_partition_ref=False, allow_unusable=False,
                              no_feasibility_gate=False)
    rec = km.run_combo("iq3", 8, args)
    fails += not check("up", rec.get("up"), False)
    fails += not check("skipped", rec.get("skipped"), "infeasible_unusable")
    fails += not check("verdict recorded", (rec.get("feasibility") or {}).get("verdict"),
                       "UNUSABLE")
    fails += not check("no result written (a refusal is not a measurement)",
                       os.path.exists(os.path.join(km.RESULT_DIR,
                                                   "knifeedge_iq3_pool8gb_selftest.json")), False)

    print("5. the cap-invariance path refuses an infeasible cap too (it launches one dump per cap)")
    args = argparse.Namespace(force=True, attach=False,
                              extra_env=["CGC_SERVER_LAYER_CAPS=0-40:40"],
                              allow_partition_ref=False, allow_unusable=False,
                              no_feasibility_gate=False)
    rec = km.cap_invariance_dump("iq3", 8, 5, args)
    fails += not check("up", rec.get("up"), False)
    fails += not check("skipped", rec.get("skipped"), "infeasible_unusable")
    fails += not check("verdict recorded", (rec.get("feasibility") or {}).get("verdict"),
                       "UNUSABLE")

    print("6. cap-probe reconciliation: a measured slot count vs the arithmetic prediction")
    km.feasibility.pool_geometry = lambda kind: dict(FAKE_GEOM)
    env = ["CGC_SERVER_LAYER_CAPS=0-40:72"]          # -> 72 slots, 71 usable, cap_max 8
    args = argparse.Namespace(extra_env=env, allow_geometry_drift=False)

    def rec_with(slots, prov=None):
        r = {"kind": "iq3", "pool_gb": 8, "cap": 8, "union_fit": {"n_slots": slots}}
        if prov is not None:
            r["provenance"] = prov
        return r

    checks = [
        ("measured == predicted -> MATCH", rec_with(72), "cache", "MATCH", True),
        ("measured 71 vs 72, legacy record -> STALE_RECORD", rec_with(71), "cache",
         "STALE_RECORD", False),
        ("measured 71 vs 72, provenance matches this tree -> PREDICTOR_STALE",
         rec_with(71, km._cap_probe_provenance("iq3", args)), "cache", "PREDICTOR_STALE", False),
        ("fresh probe disagreement -> PREDICTOR_STALE", rec_with(71), "probe",
         "PREDICTOR_STALE", False),
        ("no measured slot count -> NO_EVIDENCE", rec_with(None), "cache", "NO_EVIDENCE", False),
    ]
    for name, rec, source, want_verdict, want_ok in checks:
        chk = km.reconcile_cap_probe("iq3", 8, args, rec, source=source)
        fails += not check(name + " [verdict]", chk["verdict"], want_verdict)
        fails += not check(name + " [reusable]", km.cap_probe_reusable(chk), want_ok)

    # Provenance is only used to EXPLAIN a mismatch, so each way it can move has its own case.
    prov = km._cap_probe_provenance("iq3", args)
    moved_geo = json.loads(json.dumps(prov))
    moved_geo["pool_geometry"]["source_digest"] = "deadbeefdeadbeef"
    moved_geo["pool_geometry"]["last_commit"] = "cafe1234 1700000000 change the loader"
    moved_model = json.loads(json.dumps(prov))
    moved_model["model"]["size"] = 123
    moved_caps = json.loads(json.dumps(prov))
    moved_caps["layer_caps"] = "0-40:64"
    for name, p, want in (("geometry code moved -> STALE_RECORD", moved_geo, "STALE_RECORD"),
                         ("weights moved -> STALE_RECORD", moved_model, "STALE_RECORD"),
                         ("launch env moved -> STALE_RECORD", moved_caps, "STALE_RECORD")):
        chk = km.reconcile_cap_probe("iq3", 8, args, rec_with(71, p), source="cache")
        fails += not check(name, chk["verdict"], want)
        if want == "STALE_RECORD":
            fails += not check(name + " [names the cause]",
                               bool(chk.get("cause")) and chk["cause"] != "", True)

    print("7. a FRESH probe that contradicts the arithmetic stops the sweep (before any cell runs)")
    conflict = {"measured_min_layer_slots": 143, "predicted_min_layer_slots": 106,
                "cause": "pool-geometry code changed", "record": "Backup/x/cap_probe.json",
                "verdict": "PREDICTOR_STALE"}
    msg = km.geometry_conflict_error({"geometry_conflict": conflict}, allow=False)
    fails += not check("conflict + no override -> refuses", bool(msg), True)
    fails += not check("message names both numbers",
                       bool(msg) and "143" in msg and "106" in msg, True)
    fails += not check("--allow-geometry-drift -> does not refuse",
                       km.geometry_conflict_error({"geometry_conflict": conflict}, allow=True),
                       None)
    fails += not check("a clean probe -> no refusal",
                       km.geometry_conflict_error({"cap": 8}, allow=False), None)

    print("8. result provenance: when may two rows be laid out next to each other?")
    km.feasibility.pool_geometry = lambda kind: dict(FAKE_GEOM)
    pargs = argparse.Namespace(extra_env=[], allow_geometry_drift=False)
    base = km.record_provenance("iq3", 8, pargs, cap=8)

    def row(gb=8, cap=8, prov=base, **kw):
        r = {"label": f"iq3_pool{gb}gb", "model": "iq3", "pool_gb": gb,
             "pool_cap": cap, "rss_gb": 7.9, "tps_mean": 8.8}
        if prov is not None:
            r["provenance"] = json.loads(json.dumps(prov))
        r.update(kw)
        return r

    def state(recs):
        return km.comparability_verdict(recs)["iq3"]["state"]

    fails += not check("two pools, same provenance -> OK", state([row(4), row(8)]), "OK")
    # The pool size must NOT be part of the key: it is the variable under test.
    fails += not check("pool size is excluded from the geometry key",
                       km.record_geometry_key(row(4)) == km.record_geometry_key(row(8)), True)
    fails += not check("different cap -> REFUSED", state([row(8, 8), row(8, 6)]), "REFUSED")
    fails += not check("different weights -> REFUSED",
                       state([row(8), row(8, prov={**base, "weights": {**base["weights"],
                                                                 "size": 1}})]),
                       "REFUSED")
    moved_geo = json.loads(json.dumps(base))
    moved_geo["engine"]["geometry_digest"] = "0000deadbeef"
    fails += not check("different pool geometry (loader) -> REFUSED",
                       state([row(8), row(8, prov=moved_geo)]), "REFUSED")
    moved_bin = json.loads(json.dumps(base))
    for k in moved_bin["engine"]["binary"]:
        moved_bin["engine"]["binary"][k]["head"] = "0000000000000000"
    fails += not check("different built binary -> REFUSED",
                       state([row(8), row(8, prov=moved_bin)]), "REFUSED")
    fails += not check("runtime numbers do NOT split groups",
                       state([row(8), row(8, rss_gb=1.0, tps_mean=99.0)]), "OK")
    fails += not check("one unstamped row -> UNATTRIBUTED", state([row(8, prov=None)]),
                       "UNATTRIBUTED")
    fails += not check("stamped + unstamped -> REFUSED", state([row(8), row(8, prov=None)]),
                       "REFUSED")

    # A different SUITE shape is a different experiment even when pool/engine/weights match: a
    # 1-question smoke run and a 48-question suite both land in `iq3_pool8gb`.
    suite_a = km.record_provenance("iq3", 8, argparse.Namespace(
        extra_env=[], profiles="qa-zh", per_profile=1, greedy_repeats=2, repeats=2,
        temperature=0.4, max_tokens=64), cap=8)
    suite_b = km.record_provenance("iq3", 8, argparse.Namespace(
        extra_env=[], profiles="qa-zh,math", per_profile=2, greedy_repeats=3, repeats=5,
        temperature=0.4, max_tokens=256), cap=8)
    fails += not check("different suite shape -> REFUSED",
                       state([row(8, prov=suite_a), row(8, prov=suite_b)]), "REFUSED")
    v = km.comparability_verdict([row(8, prov=suite_a), row(8, prov=suite_b)])["iq3"]
    fails += not check("refusal names the suite shape",
                       any("suite shape" in d for d in v["differences"]), True)

    v = km.comparability_verdict([row(8, 8), row(8, 6)])["iq3"]
    fails += not check("refusal names the cap",
                       any("CGC_POOL_MAX_TOKENS" in d for d in v["differences"]), True)
    v = km.comparability_verdict([row(8), row(8, prov=moved_geo)])["iq3"]
    fails += not check("refusal names the loader",
                       any("pool-geometry" in d for d in v["differences"]), True)
    v = km.comparability_verdict([row(8), row(8, prov=None)])["iq3"]
    fails += not check("refusal names the missing provenance",
                       any("provenance" in d.lower() for d in v["differences"]), True)
    # A group with no provenance differs from everything BY CONSTRUCTION. Reporting nine field
    # diffs for it would bury the one fact that matters (and would print the binary tuple).
    fails += not check("an unstamped group is reported as that, not as nine field diffs",
                       v["differences"],
                       ["group(s) ['unattributed'] carry NO provenance, so none of their fields "
                        "can be compared with anything (records written before the stamp "
                        "existed) -- re-measure them if they are meant to sit beside the 1 "
                        "stamped group(s)"])

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
