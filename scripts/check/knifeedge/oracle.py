#!/usr/bin/env python3
"""knifeedge.oracle — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from .anchor import RESULT_DIR, ROOT
from .constants import EXPERT_CACHE_NOHOOK, EXPERT_CACHE_OFF, EXPERT_CACHE_OFF_NOGATHER, EXPERT_CACHE_ON, GROUND_TRUTH_STATES, NUMERIC_SOURCES
from .host import ask_probe
from .identity import _cap_sidecar, _model_guard, _oracle_meta, model_stamp, source_stamp



def expert_cache_state(extra_env=None, pool_bytes=None):
    """Classify how the expert cache participates, from the launch knobs.

    Read from the LAUNCH, not from the label: a dump called `ref_*_NOCACHE*` that was actually
    launched with a budget is worse than no dump at all, because it looks like a control.
    """
    try:
        if pool_bytes is not None and int(pool_bytes) == 0:
            return EXPERT_CACHE_OFF
    except (TypeError, ValueError):
        pass
    for kv in extra_env or []:
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if k == "LLAMA_EXPERT_CACHE_NOGATHER" and v not in ("", "0"):
            return EXPERT_CACHE_OFF_NOGATHER
        if k == "LLAMA_EXPERT_CACHE_NOHOOK" and v not in ("", "0"):
            return EXPERT_CACHE_NOHOOK
    return EXPERT_CACHE_ON



def _write_oracle_meta(path, cap, model_file=None, launch_sig=None, expert_cache=None):
    """Write the provenance sidecar: cap + source stamp + weights identity + LAUNCH KNOBS.

    `launch_sig` records the launch-time environment that shapes the computation (extra env vars,
    MTP on/off). It is not decoration: the cap is not the only knob that changes the numerics, and
    a dump whose label says only model/pool/cap would be silently reused across, say, MTP on and
    MTP off -- the reuse check would see the same cap, the same engine stamp and the same weights
    and conclude it already had the dump it needed.

    `expert_cache` is the 2026-09-15 addition: on / off / off_nogather / on_nohook. It decides
    whether this dump may ever be used as GROUND TRUTH (see _truth_guard). Written as "unknown"
    rather than omitted when the caller does not say, so "never recorded" and "recorded as off"
    stay distinguishable -- an unlabelled dump must not silently pass as a control.
    """
    meta = {"cap": str(cap) if cap is not None else "default",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stamp": source_stamp(),
            "expert_cache": expert_cache if expert_cache is not None else "unknown"}
    if launch_sig is not None:
        meta["launch"] = launch_sig
    m = model_stamp(model_file)
    if m is not None:
        meta["model"] = m
    with open(path + ".cap", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")



def _cap_guard(ref_path, dump_path):
    """Return an error record when the reference and the dump were taken at different caps."""
    cap_ref, cap_new = _cap_sidecar(ref_path), _cap_sidecar(dump_path)
    if cap_ref and cap_new and cap_ref != cap_new:
        return {"ok": False,
                "cap_mismatch": {"ref": cap_ref, "candidate": cap_new},
                "error": (f"CAP MISMATCH: reference was produced at cap={cap_ref}, this dump at "
                          f"cap={cap_new}. The cap clamps n_batch, so these are different "
                          f"computations -- any M1/M2 number would be a shape artifact, not a "
                          f"pool difference. Re-run with --pool-cap {cap_ref}.")}
    return None



def _stamp_guard(ref_path):
    """Refuse to compare when the reference predates a change to the numerics sources.

    Why this exists: on 2026-09-13 a reference captured 21:58 was silently compared against an
    engine fixed at 22:54, and the result (M1 0/53) read as a pool regression for hours. A stale
    reference cannot be detected from the numbers -- it fails loud, looks real, and points at
    the wrong subsystem. So provenance is checked BEFORE the comparison, not after.
    """
    rec = (_oracle_meta(ref_path) or {}).get("stamp")
    cur = source_stamp()
    if not isinstance(rec, dict):
        return {"ok": None, "stale_unknown": True, "current_source_stamp": cur,
                "error": (f"REFERENCE PROVENANCE UNKNOWN: {os.path.basename(ref_path)} has no "
                          f"source stamp, so it is impossible to tell which expert-cache code "
                          f"produced it. Rebuild it (this driver now stamps every reference), "
                          f"or pass --allow-stale-oracle to compare anyway.")}
    same_digest = rec.get("source_digest") == cur["source_digest"]
    same_commit = rec.get("last_commit") == cur["last_commit"]
    if same_digest and same_commit:
        return None
    changed = sorted(r for r in NUMERIC_SOURCES
                     if rec.get("sources", {}).get(r) != cur["sources"].get(r))
    kind = "content_changed" if not same_digest else "commit_changed"
    detail = (f"changed sources: {', '.join(os.path.basename(c) for c in changed) or '(none)'}"
              if not same_digest else
              "file contents are identical; only the newest commit over these paths moved")
    if kind == "content_changed":
        ask = ("Rebuild the reference from the current tree, then re-run the comparison so the "
               "pool-size verdict describes ONE engine.")
    else:
        ask = ("The numerics are unchanged, so a comparison is still meaningful -- pass "
               "--allow-stale-oracle once you have confirmed that, or refresh the reference.")
    return {"ok": False, "stale": True, "stale_kind": kind,
            "ref_source_stamp": rec, "current_source_stamp": cur, "changed_sources": changed,
            "error": (f"STALE REFERENCE ({kind}): {os.path.basename(ref_path)} was produced from "
                      f"different engine code than the tree about to be measured. "
                      f"ref last_commit = {rec.get('last_commit') or '?'} | "
                      f"current last_commit = {cur['last_commit'] or '?'} | {detail}. "
                      f"Any M1/M2 number here describes the code drift, not the pool size. {ask}")}



def _truth_guard(ref_path):
    """Refuse to use an expert-cache-ON dump as ground truth.

    This is the guard that did not exist and whose absence made every "M1 19/19" on this box
    unfalsifiable. A dump produced with the expert cache active can only ever prove that two
    cache-ON runs agree with each other; it carries no information about whether either agrees
    with the model. Measured 2026-09-14/15: the same harness reported M1/M2/M3 = 19/19 for
    4/6/8 GiB while a hand-run A/B against a cache-OFF baseline disagreed at step 0
    (argmax 271 / logit 14.22 vs argmax 198 / logit 27.18).

    Deliberately NOT bypassable by --allow-stale-oracle or --allow-model-mismatch: those two
    flags answer "is this reference old / is it the same file?", which is a different question
    from "is this reference a control?". Conflating them is exactly how the 19/19 happened.
    """
    meta = _oracle_meta(ref_path) or {}
    ec = meta.get("expert_cache")
    if ec is None or ec == "unknown":
        return {"ok": None, "truth_unknown": True, "ref_expert_cache": ec,
                "error": (f"REFERENCE HAS NO EXPERT-CACHE STATE: {os.path.basename(ref_path)} "
                          f"does not record whether the expert cache was active, so it cannot be "
                          f"used as ground truth. Re-dump it with this driver (it now stamps "
                          f"expert_cache=on|off|off_nogather|on_nohook), or use it only for a "
                          f"RELATIVE pool-size comparison (omit --oracle-abs).")}
    if ec not in GROUND_TRUTH_STATES:
        return {"ok": False, "not_ground_truth": True, "ref_expert_cache": ec,
                "accepted_states": list(GROUND_TRUTH_STATES),
                "error": (f"REFERENCE IS NOT GROUND TRUTH: {os.path.basename(ref_path)} was "
                          f"produced with expert_cache={ec}. Comparing a candidate against a "
                          f"cache-ON dump measures agreement between two cache-ON runs, not "
                          f"correctness -- a broken expert cache that is broken the same way in "
                          f"both runs scores 100%. Produce a control with --no-expert-cache "
                          f"(budget 0) or LLAMA_EXPERT_CACHE_NOGATHER=1 and point --oracle-ref "
                          f"at that.")}
    return None



def _oracle_guard(ref_path, dump_path, model_file=None, allow_stale=False,
                  allow_model_mismatch=False, absolute=False):
    """Return an error record when the comparison would be meaningless, else None.

    Four independent preconditions, each of which has already produced a large and entirely
    spurious "the pool broke numerics" verdict on this box: the cap (ubatch shape), the source
    provenance (engine version), the weights identity (which model), and -- when `absolute` --
    whether the reference is a CONTROL at all (expert cache off).
    """
    if absolute:
        bad = _truth_guard(ref_path)
        if bad:
            return bad
    bad = _cap_guard(ref_path, dump_path)
    if bad:
        return bad
    bad = _model_guard(ref_path, model_file)
    if bad and not allow_model_mismatch:
        return bad
    if bad:
        mm = bad.get("model_mismatch")
        if mm:
            print(f"[oracle-guard] WARNING -- model mismatch accepted via "
                  f"--allow-model-mismatch: ref={(mm.get('ref') or {}).get('realpath')} vs "
                  f"now={(mm.get('candidate') or {}).get('realpath')}",
                  file=sys.stderr, flush=True)
        else:
            print("[oracle-guard] WARNING -- unknown model identity accepted via "
                  "--allow-model-mismatch", file=sys.stderr, flush=True)
    bad = _stamp_guard(ref_path)
    if bad and not allow_stale:
        return bad
    if bad:
        print(f"[oracle-guard] WARNING -- stale reference accepted via --allow-stale-oracle: "
              f"{bad.get('stale_kind', bad.get('stale_unknown'))}", file=sys.stderr, flush=True)
    return None



def _run_oracle_compare(a_path, b_path, label):
    """Run the M1/M2/M3 comparison of two dumps; return the report dict, or None on failure.

    Extracted so every caller (pool-size gate, its final re-compare, and the cap-invariance gate)
    goes through ONE comparison implementation. The metrics' definitions -- and the rule that M1
    and M2 are never merged -- live in cgc_logits_oracle_compare.py.
    """
    rep = os.path.join(RESULT_DIR, f"oraclecmp_{label}.json")
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "scripts", "check", "cgc_logits_oracle_compare.py"),
                    "--a", a_path, "--b", b_path, "--report", rep],
                   cwd=ROOT, capture_output=True)
    if not os.path.exists(rep):
        return None
    try:
        return json.load(open(rep, encoding="utf-8"))
    except Exception:  # noqa: BLE001 - harness
        return None



def oracle_gate(port, dump_path, ref_path, label, timeout, allow_stale=False,
                model_file=None, allow_model_mismatch=False, absolute=False):
    """Send the deterministic probe, then compare this combo's logits oracle against a
    reference using the TWO independent metrics (never merged).

    M1 numeric identity  = same row hashes (bit-identical logits)
    M2 decision agreement = same argmax token
    Pool sizes may only differ if M1 says so; 'M2 passed' alone does NOT mean 'no difference'.

    `absolute=True` marks the comparison as CORRECTNESS rather than INVARIANCE: the reference
    must be a control (expert cache off -- see _truth_guard), and the verdict is recorded as
    `comparison: "absolute"` so the two kinds of 19/19 can never be read as the same number
    again.
    """
    if not (dump_path and os.path.exists(dump_path)):
        return {"ok": None, "error": "no oracle dump produced"}
    if not (ref_path and os.path.exists(ref_path)):
        return {"ok": None, "error": f"reference oracle missing: {ref_path}"}
    bad = _oracle_guard(ref_path, dump_path, model_file, allow_stale, allow_model_mismatch,
                        absolute=absolute)
    if bad:
        if absolute:
            bad.setdefault("comparison", "absolute")
        return bad
    ask_probe(port, timeout)
    time.sleep(1.0)
    rep = os.path.join(RESULT_DIR, f"oraclecmp_{label}.json")
    d = _run_oracle_compare(ref_path, dump_path, label)
    if d is None:
        return {"ok": None, "error": "oracle compare produced no report"}
    m1, m2 = d["metrics"]["numeric_identity"], d["metrics"]["decision_agreement"]
    return {"ok": (m1["rate"] == 1.0 and m2["rate"] == 1.0),
            "m1_numeric_identity": f"{m1['equal']}/{m1['n']}",
            "m2_decision_agreement": f"{m2['equal']}/{m2['n']}",
            "cross_tab": d["cross_tab"],
            "comparison": "absolute" if absolute else "relative",
            "report": rep}



def oracle_recompare(dump_path, ref_path, label, allow_stale=False, model_file=None,
                     allow_model_mismatch=False, absolute=False):
    """Re-compare the COMPLETE dump and print the verdict; returns the report dict or None.

    `oracle_gate` deliberately runs first (its comment: the dump's early ubatches must line up
    with the reference run's). But the template and preflight probes that follow keep APPENDING
    to the same JSONL, so the comparison it performs measures only a PREFIX of the file -- on
    2026-09-13 it compared 53 of the eventual 103 step keys. That silently understates the gate
    instead of failing it. The finished file is a strict superset of what was compared, so
    re-comparing here is free and strictly stronger; no extra request is sent.
    """
    if not (dump_path and os.path.exists(dump_path) and ref_path
            and os.path.exists(ref_path)):
        return None
    if _oracle_guard(ref_path, dump_path, model_file, allow_stale, allow_model_mismatch,
                     absolute=absolute):
        return None
    rep = os.path.join(RESULT_DIR, f"oraclecmp_{label}_final.json")
    d = _run_oracle_compare(ref_path, dump_path, f"{label}_final")
    if d is None:
        return None
    m1, m2 = d["metrics"]["numeric_identity"], d["metrics"]["decision_agreement"]
    m3 = d["metrics"]["topk_set_agreement"]
    out = {"ok": (m1["rate"] == 1.0 and m2["rate"] == 1.0),
           "m1_numeric_identity": f"{m1['equal']}/{m1['n']}",
           "m2_decision_agreement": f"{m2['equal']}/{m2['n']}",
           "m3_topk_set_agreement": f"{m3['equal']}/{m3['n']}",
           "n_compared": m1["n"],
           "cross_tab": d["cross_tab"],
           "report": rep}
    print(f"[{label}] gate-oracle-final {'PASS' if out['ok'] else 'CHECK'}  "
          f"M1(bit-identical)={out['m1_numeric_identity']} "
          f"M2(argmax)={out['m2_decision_agreement']} "
          f"M3(topk)={out['m3_topk_set_agreement']} (n={out['n_compared']})", flush=True)
    return out



def _oracle_rows(path):
    """Count usable rows in a dump (a row is a non-empty JSON line)."""
    n = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    return n
