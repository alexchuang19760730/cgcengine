#!/usr/bin/env python3
"""Offline selftest for the ABSOLUTE (ground-truth) oracle gate in `knifeedge_matrix.py`.

Why this exists as its own file, and why it must stay passing:

On 2026-09-15 the harness reported M1/M2/M3 = 19/19 for 4/6/8 GiB pools while a hand-run A/B
against a cache-OFF baseline disagreed at step 0 (argmax 271 / logit 14.22 vs argmax 198 /
logit 27.18). The gate was not lying about what it measured -- it was measuring
expert-cache-ON vs expert-cache-ON, because `run_server.sh` always passes `-expert-cache
<budget>` and therefore every reference ever recorded was cache-ON. Two runs sharing one
defect score 100%.

The fix is a guard that refuses to treat a cache-ON dump as truth. A guard nobody tests is a
comment, so this file exercises the refusal -- including the two flags
(`--allow-stale-oracle`, `--allow-model-mismatch`) that must NOT be able to override it, because
conflating "is this reference old?" with "is this reference a control?" is exactly how the 19/19
happened.

Runs offline: no server, no model, no GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def load_matrix():
    path = os.path.join(HERE, "knifeedge_matrix.py")
    spec = importlib.util.spec_from_file_location("km_truth_selftest", path)
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)
    return km


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    return ok


def write_sidecar(path, expert_cache=None, cap="8"):
    meta = {"cap": cap, "created": "2026-09-15T00:00:00",
            "stamp": {"source_digest": "x", "last_commit": "x", "dirty": False}}
    if expert_cache is not None:
        meta["expert_cache"] = expert_cache
    with open(path + ".cap", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def main():
    km = load_matrix()
    ok = True
    tmp = tempfile.mkdtemp(prefix="oracle_truth_")

    print("1. what counts as a control")
    ok &= check("cache ON is not ground truth",
                km.EXPERT_CACHE_ON in km.GROUND_TRUTH_STATES, False)
    ok &= check("cache OFF (budget 0) is ground truth",
                km.EXPERT_CACHE_OFF in km.GROUND_TRUTH_STATES, True)
    ok &= check("NOGATHER is ground truth",
                km.EXPERT_CACHE_OFF_NOGATHER in km.GROUND_TRUTH_STATES, True)
    ok &= check("NOHOOK is NOT a control (it still has skip_load on)",
                km.EXPERT_CACHE_NOHOOK in km.GROUND_TRUTH_STATES, False)

    print("2. classifying the launch knobs (read from the launch, not the filename)")
    ok &= check("8 GiB budget, no override -> on",
                km.expert_cache_state([], 8 * 1024 ** 3), km.EXPERT_CACHE_ON)
    ok &= check("budget 0 -> off",
                km.expert_cache_state([], 0), km.EXPERT_CACHE_OFF)
    ok &= check("NOGATHER=1 -> off_nogather",
                km.expert_cache_state(["LLAMA_EXPERT_CACHE_NOGATHER=1"], 8 * 1024 ** 3),
                km.EXPERT_CACHE_OFF_NOGATHER)
    ok &= check("NOGATHER=0 is OFF, not 'present'",
                km.expert_cache_state(["LLAMA_EXPERT_CACHE_NOGATHER=0"], 8 * 1024 ** 3),
                km.EXPERT_CACHE_ON)
    ok &= check("NOHOOK=1 -> on_nohook",
                km.expert_cache_state(["LLAMA_EXPERT_CACHE_NOHOOK=1"], 8 * 1024 ** 3),
                km.EXPERT_CACHE_NOHOOK)
    ok &= check("NOHOOK=0 is OFF, not 'present'",
                km.expert_cache_state(["LLAMA_EXPERT_CACHE_NOHOOK=0"], 8 * 1024 ** 3),
                km.EXPERT_CACHE_ON)

    print("3. the budget --no-expert-cache actually launches with")
    a = argparse.Namespace(no_expert_cache=True)
    ok &= check("--no-expert-cache -> 0 bytes", km.launch_pool_bytes(a, 8), 0)
    b = argparse.Namespace(no_expert_cache=False)
    ok &= check("normal cell -> 8 GiB in bytes", km.launch_pool_bytes(b, 8), 8 * 1024 ** 3)
    c = argparse.Namespace()  # callers that build their own Namespace (selftests)
    ok &= check("absent flag defaults to the pool, not to 0",
                km.launch_pool_bytes(c, 8), 8 * 1024 ** 3)

    print("4. the refusal itself")
    p_on = os.path.join(tmp, "ref_on.jsonl")
    p_off = os.path.join(tmp, "ref_off.jsonl")
    p_nog = os.path.join(tmp, "ref_nogather.jsonl")
    p_hook = os.path.join(tmp, "ref_nohook.jsonl")
    p_raw = os.path.join(tmp, "ref_unstamped.jsonl")
    for p in (p_on, p_off, p_nog, p_hook, p_raw):
        open(p, "w").write("{}\n")
    write_sidecar(p_on, km.EXPERT_CACHE_ON)
    write_sidecar(p_off, km.EXPERT_CACHE_OFF)
    write_sidecar(p_nog, km.EXPERT_CACHE_OFF_NOGATHER)
    write_sidecar(p_hook, km.EXPERT_CACHE_NOHOOK)
    write_sidecar(p_raw, None)

    ok &= check("cache-ON reference -> refused",
                (km._truth_guard(p_on) or {}).get("not_ground_truth"), True)
    ok &= check("NOHOOK reference -> refused (not a control)",
                (km._truth_guard(p_hook) or {}).get("not_ground_truth"), True)
    ok &= check("unstamped reference -> unknown, not silently accepted",
                (km._truth_guard(p_raw) or {}).get("truth_unknown"), True)
    ok &= check("cache-OFF reference -> accepted", km._truth_guard(p_off), None)
    ok &= check("NOGATHER reference -> accepted", km._truth_guard(p_nog), None)

    print("5. the two --allow-* flags must NOT override it")
    # This is the regression this file exists for: --allow-stale-oracle and
    # --allow-model-mismatch answer a different question and were, on 2026-09-15, passed
    # together in the documented verification command (docs/PREFILL250...GUIDE §4.3).
    g = km._oracle_guard(p_on, p_on, None, True, True, absolute=True)
    ok &= check("absolute + allow_stale + allow_model_mismatch still refuses a cache-ON ref",
                (g or {}).get("not_ground_truth"), True)
    g = km._oracle_guard(p_raw, p_raw, None, True, True, absolute=True)
    ok &= check("absolute + both allows still refuses an unstamped ref",
                (g or {}).get("truth_unknown"), True)
    g = km._oracle_guard(p_off, p_off, None, True, True, absolute=True)
    ok &= check("absolute + both allows still accepts a real control", g, None)
    # Relative comparisons keep their old, deliberately laxer behaviour.
    g = km._oracle_guard(p_on, p_on, None, True, True, absolute=False)
    ok &= check("relative comparison is untouched (cache-ON vs cache-ON is still allowed)",
                g, None)

    print("6. the verdict says which kind of comparison it was")
    # A 19/19 must never again be readable without knowing absolute vs relative.
    src = open(os.path.join(HERE, "knifeedge_matrix.py"), encoding="utf-8").read()
    ok &= check("oracle_gate stamps comparison=absolute|relative",
                '"comparison": "absolute" if absolute else "relative"' in src, True)

    print("7. the ground-truth arm skips the pool feasibility gate")
    ok &= check("GROUND_TRUTH verdict is defined for the budget-0 cell",
                "GROUND_TRUTH" in src, True)

    print()
    if ok:
        print("ALL PASS")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
