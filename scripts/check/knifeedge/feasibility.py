#!/usr/bin/env python3
"""knifeedge.feasibility — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import os
import sys
from .anchor import ROOT
from .constants import MODELS
from .identity import model_path



# ---------------------------------------------------------------------------------------------
# PRE-LAUNCH FEASIBILITY GATE (docs/POOL_FEASIBILITY_MATRIX_2026-09-14.html)
#
# A cell whose `cap_max` sits below the reference cap can only be measured at a SMALLER cap -- and
# the cap decides the prefill step partition. Comparing such a cell against a reference-cap cell
# therefore compares two different forward passes, not two pool layouts, while the resulting number
# looks exactly like a pool result. Measured 2026-09-14, one build: cap 8 across pools 4/6/8 GiB is
# bit-identical (M1 117/117) AND cap 6 vs cap 8 in that same batch is M1 3/103.
#
# Refusing at launch is the only place that mistake can be caught before it becomes a table row.
# The arithmetic itself lives in scripts/check/pool_feasibility.py (one definition, also usable
# standalone with `--validate` against the measured cap-probe records); this file only decides
# WHICH cell is being launched, because the cap / MTP / spec-n-max / layer-caps all arrive from
# the environment, and the env-merge order here mirrors `launch()`'s exactly.
# ---------------------------------------------------------------------------------------------
_FEAS_MOD = None

_FEAS_GEOM: dict = {}



def _feasibility():
    """`pool_feasibility` holds the arithmetic; import it lazily.

    It loads THIS file to read MODELS, so a module-level import would be circular. Loading it by
    path under its own name is what its CLI already does.
    """
    global _FEAS_MOD
    if _FEAS_MOD is None:
        import importlib.util
        path = os.path.join(ROOT, "scripts", "check", "pool_feasibility.py")
        spec = importlib.util.spec_from_file_location("cgc_pool_feasibility", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FEAS_MOD = mod
    return _FEAS_MOD



def pool_geometry(kind):
    """GGUF tensor-table geometry for one configured model, cached on (path, size, mtime).

    Tensor table only -- no weights are read and no server is started, so this costs milliseconds
    even for a 16 GB GGUF. Keyed on the file's identity so a re-quantised file at the same path is
    not described by the previous geometry.
    """
    path = model_path(kind)
    try:
        st = os.stat(path)
    except OSError as e:
        raise RuntimeError(f"{path}: {e}") from e
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if _FEAS_GEOM.get("key") != key:
        _FEAS_GEOM.clear()
        _FEAS_GEOM["key"] = key
        _FEAS_GEOM["geom"] = _feasibility().gguf_geometry(path)
    return _FEAS_GEOM["geom"]



def effective_launch_env(kind, extra_env):
    """The knobs run_server.sh will actually see, merged the way `launch()` merges them.

    Only CGC_*/LLAMA_EXPERT_CACHE_* matter here: those are what move the pool geometry and the
    batch shape. `extra_env` wins over the ambient environment because `launch()` sets its own
    values first and then applies `extra_env` on top -- which is how a caller expresses "MTP off"
    (`--extra-env CGC_SERVER_MTP=0`).
    """
    env = {k: v for k, v in os.environ.items()
           if k.startswith(("CGC_", "LLAMA_EXPERT_CACHE_"))}
    env["CGC_SERVER_MTP"] = MODELS[kind]["mtp"]
    for kv in extra_env or []:
        k, _, v = kv.partition("=")
        if k.strip():
            env[k.strip()] = v.strip()
    return env



def _env_int(env, key, default):
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError):
        return default



def feasibility_cell(kind, gb, cap=None, extra_env=None):
    """Pre-launch verdict for one (model, pool[, cap]) cell. Arithmetic, not a measurement.

    Same verdict vocabulary as pool_feasibility, except the reference cap is replaced by the cap
    this cell will actually be launched at:

      USABLE         it runs at the reference cap -> a cross-cell difference is pool-only
      PARTITION-REF  it runs, but the cap has to drop -> the difference includes the prefill STEP
                     PARTITION, so the cell can inform partition sensitivity only
      UNUSABLE       no cap satisfies P1 (capacity) and P2 (survival) at the same time
    """
    feas = _feasibility()
    geom = pool_geometry(kind)
    env = effective_launch_env(kind, extra_env)
    mtp = str(env.get("CGC_SERVER_MTP", "1")).lower() in ("1", "true", "on", "yes")
    spec = _env_int(env, "CGC_SERVER_MTP_N_MAX", feas.DEFAULT_SPEC_N_MAX)
    layer_caps = env.get("CGC_SERVER_LAYER_CAPS") or None
    topk = MODELS[kind].get("topk") or geom["topk"] or 8
    if cap is None and env.get("CGC_POOL_MAX_TOKENS"):
        cap = _env_int(env, "CGC_POOL_MAX_TOKENS", None)

    r = feas.evaluate(geom, int(gb) * 1024 ** 3, topk, mtp, spec, layer_caps)
    ref = feas.REF_CAP
    requested = ref if cap is None else int(cap)
    cap_max, floor = r["cap_max"], r["floor"]

    if cap_max < floor:
        verdict = "UNUSABLE"
        why = (f"cap_max {cap_max} < floor {floor}: P1 and P2 cannot both hold at any cap "
               f"({r['usable_slots']} usable slots, n_rs_seq={r['n_rs_seq']})")
    elif requested > cap_max:
        verdict = "PARTITION-REF"
        why = (f"cap {requested} x topk {topk} = {requested * topk} > usable {r['usable_slots']} "
               f"(cap_max {cap_max}): the engine has to fall back to cap {cap_max}, so this cell's "
               f"forward pass is partitioned differently from one that keeps cap {requested}")
    elif requested < floor:
        verdict = "UNUSABLE"
        why = (f"cap {requested} < floor {floor} (n_rs_seq={r['n_rs_seq']}): "
               f"GGML_ASSERT(n_ubatch > n_keep_tail) aborts")
    elif requested >= ref:
        verdict = "USABLE"
        why = f"runs at cap {requested} (reference cap {ref}); a cross-cell diff is pool-only"
    else:
        verdict = "USABLE"
        why = (f"runs at cap {requested} by request (< reference {ref}); comparable only with "
               f"cells launched at that same cap")

    r.update({"model": kind, "pool_gb": int(gb), "cap_requested": requested,
              "cap_is_default": requested == ref, "cap_source": "cli/env" if cap is not None
              else "engine default",
              "cap_max": cap_max, "floor": floor, "topk": topk, "mtp": mtp,
              "spec_n_max": spec, "verdict": verdict, "why": why,
              "ref_cap_ok": cap_max >= ref, "path": model_path(kind)})
    # Convert numpy scalar types (uint64 etc.) to Python natives so the cell is JSON-serializable.
    for k, v in list(r.items()):
        if hasattr(v, "item") and not isinstance(v, (dict, list)):
            try:
                r[k] = v.item()
            except Exception:  # noqa: BLE001
                pass
    return r



def feasibility_gate(kind, gb, args, cap=None, extra_env=None, where="run"):
    """Refuse a launch that cannot produce a comparable number. Returns (ok, cell).

    `ok=True, cell=None` means "no verdict available" (a predictor must never be the reason a run
    dies, so an unreadable GGUF warns and continues). `ok=False` means the caller must not launch.
    """
    if getattr(args, "no_feasibility_gate", False):
        return True, None
    if getattr(args, "attach", False):
        # --attach measures a server somebody else launched: its cap and its pool are not ours to
        # know, so a verdict about "the cap this cell will launch at" would be a verdict about a
        # launch that is not happening.
        print(f"[{where}] feasibility: skipped (--attach: the running server's cap/pool are fixed "
              f"outside this process)", flush=True)
        return True, None
    try:
        cell = feasibility_cell(kind, gb, cap, extra_env)
    except Exception as e:  # noqa: BLE001 - harness: never fatal
        print(f"[{where}] feasibility: no pre-launch verdict (cannot read the GGUF geometry: "
              f"{e}); launching anyway", file=sys.stderr, flush=True)
        return True, None
    if cell["verdict"] == "USABLE":
        note = "" if cell["cap_is_default"] else " (non-default cap: compare only at the same cap)"
        print(f"[{where}] feasibility USABLE  cap={cell['cap_requested']} "
              f"cap_max={cell['cap_max']} slots={cell['usable_slots']} floor={cell['floor']}{note}",
              flush=True)
        return True, cell

    override = (getattr(args, "allow_partition_ref", False)
                if cell["verdict"] == "PARTITION-REF" else
                getattr(args, "allow_unusable", False))
    head = f"[{where}] feasibility {cell['verdict']}  {kind}@{gb}GiB cap={cell['cap_requested']}"
    if override:
        print(f"{head}\n  {cell['why']}\n  continuing because the override flag was given; the "
              f"record carries verdict={cell['verdict']} and must NOT be read as a pool "
              f"comparison", file=sys.stderr, flush=True)
        return True, cell
    print(f"{head}\n  {cell['why']}\n  REFUSING to launch: this cell cannot produce a number "
          f"comparable with a reference-cap cell.\n  Use --allow-partition-ref (if you want the "
          f"partition sensitivity measured) or --no-feasibility-gate to disable the check; see "
          f"docs/POOL_FEASIBILITY_MATRIX_2026-09-14.html.", file=sys.stderr, flush=True)
    return False, cell



def print_feasibility_matrix(models, pools, args, caps=None):
    """Up-front table: the verdict for every (model, pool) BEFORE anything is launched."""
    caps = caps or {}
    default_cap = None if args.pool_cap in ("auto", "off") else _env_int(
        {"CGC_POOL_MAX_TOKENS": args.pool_cap}, "CGC_POOL_MAX_TOKENS", None)
    print("\nPOOL FEASIBILITY (pre-launch: GGUF tensor table + the engine's own arithmetic)")
    print(f"{'model':<12} {'pool':>5} {'slots':>6} {'cap_max':>8} {'floor':>6} {'cap':>4} "
          f"{'verdict':>14}  why")
    print("-" * 118)
    for kind in models:
        for gb in pools:
            try:
                r = feasibility_cell(kind, gb, caps.get(kind, default_cap), args.extra_env)
            except Exception as e:  # noqa: BLE001 - a table row must not kill the run
                print(f"{kind[:12]:<12} {gb:>4}G {'-':>6} {'-':>8} {'-':>6} {'-':>4} "
                      f"{'UNKNOWN':>14}  {e}")
                continue
            print(f"{kind[:12]:<12} {gb:>4}G {r['usable_slots']:>6} {r['cap_max']:>8} "
                  f"{r['floor']:>6} {str(r['cap_requested']):>4} {r['verdict']:>14}  "
                  f"{r['why'][:70]}")
    print("-" * 118)
    print("USABLE runs at the reference cap (cross-cell diff = pool only) | PARTITION-REF runs "
          "only at a smaller cap (diff includes step partition) | UNUSABLE no cap fits")
    print()



def cap_for(args, kind):
    """Resolved `CGC_POOL_MAX_TOKENS` for this model, shared by EVERY pool (or None = leave it
    to the build's default, which is iq3-independent but not pool-independent)."""
    resolved = getattr(args, "pool_cap_resolved", None)
    if isinstance(resolved, dict):
        return resolved.get(kind)
    if args.pool_cap in ("auto", "off"):
        return None
    try:
        return int(args.pool_cap)
    except (TypeError, ValueError):
        return None
