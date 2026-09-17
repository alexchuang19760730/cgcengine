#!/usr/bin/env python3
"""knifeedge.provenance — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import hashlib
import json
import time
from .anchor import ENTRY_FILE
from .constants import GROUND_TRUTH_STATES
from .feasibility import _env_int, effective_launch_env, feasibility_cell
from .host import launch_pool_bytes
from .identity import _as_int, binary_stamp, model_path, model_stamp, pool_geometry_stamp, source_stamp
from .oracle import expert_cache_state



def record_provenance(kind, gb, args, cap=None, facts=None, extra_env=None):
    """The (pool, engine, weights) fingerprint that makes a result comparable -- or not.

    Stored INSIDE every result record. Without it the only way to tell whether two rows describe
    the same computation was to remember, and this project has paid for that twice: a cached cap
    that described an older loader (106 vs 143 slots, silently reused), and an oracle reference
    that described an older engine (M1 0/53 that read as a pool regression for hours).
    """
    ee = list((args.extra_env if extra_env is None else extra_env) or [])
    env = effective_launch_env(kind, ee)
    ec_state = expert_cache_state(ee, launch_pool_bytes(args, gb))
    pool = {"gb": int(gb), "bytes": launch_pool_bytes(args, gb), "cap": cap,
            "expert_cache": ec_state}
    if ec_state in GROUND_TRUTH_STATES:
        # Say it in the record, not just in the filename: a ground-truth row read out of
        # context (a glob over combo_*.json) must still be recognisable as a control.
        pool["ground_truth"] = True
    try:
        cell = feasibility_cell(kind, gb, cap, ee)
        pool.update({"min_layer_slots": cell["min_layer_slots"],
                     "usable_slots": cell["usable_slots"], "cap_max": cell["cap_max"],
                     "floor": cell["floor"], "verdict": cell["verdict"]})
    except Exception as e:  # noqa: BLE001 - stamping must not be able to kill a measurement
        pool["geometry_error"] = str(e)
    if facts:
        # What the engine REPORTED at load. If this disagrees with the prediction above, the row
        # says so -- the prediction is the model, this is the measurement.
        launched_slots = _as_int(facts.get("min_layer_slots"))
        pool.update({"launched_min_layer_slots": launched_slots,
                     "launched_n_slots": _as_int(facts.get("n_slots"))})
        # Per-row geometry mismatch flag: predicted (from pool_feasibility.py arithmetic) vs
        # measured (what the engine actually reported at load). This used to be visible only via
        # a one-shot cap probe; now every row carries it so a stale predictor is caught on the
        # first measurement that disagrees, not on the next probe run.
        predicted = pool.get("min_layer_slots")
        if predicted is not None and launched_slots is not None:
            mismatch = int(predicted) != int(launched_slots)
            pool["geometry_mismatch"] = mismatch
            if mismatch:
                pool["geometry_mismatch_detail"] = (
                    f"predicted {predicted} slots/layer vs launched {launched_slots}")
        else:
            pool["geometry_mismatch"] = None  # cannot tell (no prediction or no measurement)
    ss, gs = source_stamp(), pool_geometry_stamp()
    return {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "pool": pool,
            "engine": {"source_digest": ss["source_digest"], "sources": ss["sources"],
                       "geometry_digest": gs["source_digest"], "geometry_sources": gs["sources"],
                       "head": ss["head"], "last_commit": ss["last_commit"],
                       "dirty": ss["dirty"], "binary": binary_stamp()},
            "weights": model_stamp(model_path(kind)),
            "launch": {"mtp": str(env.get("CGC_SERVER_MTP", "1")).lower()
                               in ("1", "true", "on", "yes"),
                       "spec_n_max": _env_int(env, "CGC_SERVER_MTP_N_MAX", 3),
                       "layer_caps": env.get("CGC_SERVER_LAYER_CAPS") or "default",
                       "mode": "M2_stream" if getattr(args, "m2", False) else "standard",
                       "extra_env": sorted(ee)},
            # The SHAPE of the measurement, not of the engine. Two rows measured with different
            # suites are not the same experiment even when the pool, engine and weights match --
            # a 1-question smoke run and a 48-question suite both land in `iq3_pool8gb`.
            "suite": {"profiles": getattr(args, "profiles", None),
                      "per_profile": getattr(args, "per_profile", None),
                      "greedy_repeats": getattr(args, "greedy_repeats", None),
                      "repeats": getattr(args, "repeats", None),
                      "temperature": getattr(args, "temperature", None),
                      "max_tokens": getattr(args, "max_tokens", None)},
            # Informational only, NEVER part of the comparability key: a harness edit (a print, a
            # new gate) must not retroactively declare every past measurement incomparable.
            "harness": {"script_digest": hashlib.sha256(
                open(ENTRY_FILE, "rb").read()).hexdigest()[:16]}}



def record_geometry_key(rec):
    """Everything that must MATCH for two rows to be comparable, or None when unstamped.

    The POOL SIZE is deliberately absent: it is the variable under test. The cap IS present,
    because it clamps n_batch and therefore decides the prefill step partition (docs/
    CAP_INVARIANCE_LOCALIZATION_2026-09-14.md) -- so two rows at different caps are different
    computations even when everything else matches.
    """
    prov = rec.get("provenance")
    if not prov:
        return None
    eng = prov.get("engine") or {}
    w = prov.get("weights") or {}
    L = prov.get("launch") or {}
    binary = tuple(sorted((k, (v or {}).get("size"), (v or {}).get("head"))
                          for k, v in (eng.get("binary") or {}).items()))
    return {"engine": eng.get("source_digest"),
            "pool_geometry": eng.get("geometry_digest"),
            "binary": binary,
            "weights": (w.get("realpath"), w.get("size"), w.get("digest")),
            "cap": rec.get("pool_cap", (prov.get("pool") or {}).get("cap")),
            "mode": L.get("mode", "standard"),
            "mtp": L.get("mtp"), "spec_n_max": L.get("spec_n_max"),
            "layer_caps": L.get("layer_caps"), "extra_env": tuple(L.get("extra_env") or ()),
            "suite": tuple(sorted(((prov.get("suite") or {}).items()))) }



def _short_val(v, width=52):
    """Keep a refusal readable: the binary stamp alone is a 38-element tuple."""
    s = repr(v)
    if len(s) <= width:
        return s
    if isinstance(v, (tuple, list, dict)):
        return f"{type(v).__name__} of {len(v)}: {s[:width - 24]}…"
    return s[:width - 1] + "…"



def key_fingerprint(key):
    """Short, stable label for one geometry group (so blocks can be named in the output)."""
    if key is None:
        return "unattributed"
    blob = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]
