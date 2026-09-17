#!/usr/bin/env python3
"""knifeedge.cli — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from .anchor import DEFAULT_PORT, RESULT_DIR
from .capinv import _store_capinv, cap_invariance_gate, print_capinv_table
from .caps import comparability_verdict, geometry_conflict_error, preflight, probe_pool_cap
from .constants import CAP_INVARIANCE_DEFAULT_CAPS, MIN_CAP, MODELS
from .feasibility import print_feasibility_matrix
from .host import kill_servers, running_servers
from .identity import gate_model_identity, model_path
from .matrix import run_combo, summary_table



def print_matrix(records):
    print()
    print("=" * 140)
    print("%-26s %5s %7s %8s %8s %7s %7s %8s %5s %6s %6s %6s %14s" % (
        "config", "rss", "free%", "greedy", "seeded", "std", "flips", "tok/s",
        "cap", "tmpl", "nil", "geom", "M1/M2"))
    print("-" * 146)
    for r in records:
        if not r.get("up"):
            # Name the REASON. A cell that never launched because the MACHINE was out of memory
            # is not evidence about the model, and printing it as "did not come up" invites
            # exactly that misreading (and, before the skip-rule fix, a cached stub).
            sk = r.get("skipped") or ""
            fe = r.get("feasibility") or {}
            why = {"low_memory": "machine below --min-free-pct -> NOT measured, nothing cached",
                   "foreign_server": "another process owns the port -> NOT measured"}.get(sk)
            if why is None and sk.startswith("infeasible"):
                # The reason must name the ARITHMETIC, not just "refused". A refused cell is
                # otherwise indistinguishable from a cell nobody got to, and the whole point of
                # the gate is that this number could never have been a pool measurement.
                why = (f"{fe.get('verdict', sk)} at launch: cap_max={fe.get('cap_max')} "
                       f"< cap {fe.get('cap_requested')} (usable slots {fe.get('usable_slots')}, "
                       f"floor {fe.get('floor')}) -> REFUSED, not a pool result; "
                       f"--allow-partition-ref to measure it as partition sensitivity")
            if why is None:
                why = f"server did not come up ({r.get('model')})"
            print("%-26s   -- %s" % (r["label"], why))
            continue
        g = r.get("gates") or {}
        gt = (g.get("template") or {}).get("ok")
        gts = "n/a" if gt is None else ("ok" if gt else "FAIL")
        gn = (g.get("buffer_nil") or {}).get("ok")
        gns = "n/a" if gn is None else ("ok" if gn else "FAIL")
        go = g.get("oracle") or {}
        mpair = f"{go.get('m1_numeric_identity','-')}/{go.get('m2_decision_agreement','-')}"
        # Per-row geometry mismatch: predicted vs launched slots/layer. This is the flag that
        # catches a stale pool_feasibility.py on the FIRST measurement, not on the next probe.
        prov = r.get("provenance") or {}
        gm = (prov.get("pool") or {}).get("geometry_mismatch")
        gms = "n/a" if gm is None else ("MISMATCH" if gm else "ok")
        print("%-26s %5s %7s %8s %8s %7s %7s %8s %5s %6s %6s %6s %14s" % (
            r["label"], r.get("rss_gb"), r.get("free_after_pct"),
            r.get("greedy_all_pass_rate"), r.get("suite_rate_mean"),
            r.get("suite_rate_std"), r.get("seeded_flip_questions"),
            r.get("tps_mean"), r.get("pool_cap", "-"), gts, gns, gms, mpair))
    print("=" * 108)
    print("greedy   = all-%d-greedy-repeats pass rate (determinism side)" % args_greedy_global)
    print("seeded   = mean suite pass rate over fixed seeds (sampling side)")
    print("std      = spread of the per-seed suite rate -> the uncertainty the pool-floor")
    print("           decision has to survive. A pool ranking INSIDE one std is not a ranking.")
    print("cap      = CGC_POOL_MAX_TOKENS actually launched; MUST be identical down the column")
    print("           for a given model, else the pool sizes are not the only variable")
    print("flips    = questions whose verdict changes across seeds")
    print("tok/s    = mean decode tok/s over every request in the session")
    # Same grouping rule as --summary, applied to the sweep that just finished: a table that mixes
    # two geometries is not a comparison, and printing it without saying so is how one gets read
    # as one. (Within one sweep this normally says OK; it exists to stay true when it does not.)
    for model, v in comparability_verdict(records).items():
        if v["state"] == "REFUSED":
            print(f"GEOMETRY: {model} rows span {len(v['groups'])} groups -> NOT comparable: "
                  + "; ".join(v["differences"]))
        elif v["state"] == "OK":
            print(f"GEOMETRY: {model} rows share one geometry group "
                  f"({v['groups'][0]['fingerprint']}) -> comparable")
        else:
            print(f"GEOMETRY: {model} rows are UNATTRIBUTED (no provenance) -> comparability "
                  f"unverified")



def main():
    global args_greedy_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="iq3,iq4")
    ap.add_argument("--pools", default="4,6,8", help="comma list of GB")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--profiles", default="qa-zh,math,coding,reasoning")
    ap.add_argument("--per-profile", type=int, default=2)
    ap.add_argument("--greedy-repeats", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--min-free-pct", type=float, default=40.0,
                    help="refuse to launch a combo when the machine is below this %% free. "
                         "Mirrors run_server.sh's memory guard (full-MTP needs >=40%%); a cell "
                         "skipped for this reason is a MACHINE precondition, not a result")
    ap.add_argument("--health-timeout", type=float, default=240.0)
    # BOTH keys are needed, and one string covers both models:
    #   enable_thinking:false  -> the GGUF embedded template closes its no-think scaffold
    #                             (otherwise it emits an unclosed <think> and the model echoes).
    #   assistant_prefill      -> the custom jinja starts the assistant turn after a short anchor.
    # Without the anchor iq4 emits EOS after ONE token (`completion_tokens: 1`, content ""),
    # identically at 4GB and 8GB -- an anchor problem masquerading as a pool problem. It has to
    # be set HERE because run_server.sh only applies its own anchor when chat-template-kwargs is
    # EMPTY, and this harness always passes some, so it was suppressing the anchor. iq3 ignores
    # keys its template does not use, so one value is safe for both.
    ap.add_argument("--kwargs",
                    default='{"enable_thinking": false, "assistant_prefill": "答："}',
                    help="--chat-template-kwargs value; pass '' to omit")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="extra KEY=VAL for run_server.sh (repeatable)")
    ap.add_argument("--tag", default="", help="suffix for the label (A/B configs)")
    ap.add_argument("--force", action="store_true", help="redo combos that already have results")
    ap.add_argument("--kill-existing", action="store_true",
                    help="kill a pre-existing llama-server instead of refusing")
    ap.add_argument("--attach", action="store_true",
                    help="measure the server already on --port (no launch, no kill); for "
                         "shared checkouts where another agent owns the machine")
    ap.add_argument("--preflight", choices=("abort", "warn", "off"), default="abort",
                    help="gate on a canonical probe before measuring (default abort)")
    ap.add_argument("--preflight-timeout", type=float, default=150.0)
    ap.add_argument("--pilot", action="store_true",
                    help="1 profile x 1 question x 2 greedy x 2 seeds (smoke test)")
    ap.add_argument("--oracle-ref", default=None,
                    help="reference logits oracle JSONL; enables the M1/M2 numeric-invariance "
                         "gate for every combo (a pool size may differ ONLY in speed, so M1 "
                         "numeric identity must be full -- M2 alone is not proof)")
    ap.add_argument("--oracle-abs", action="store_true",
                    help="ABSOLUTE correctness gate: --oracle-ref must be a CONTROL (expert cache "
                         "OFF), and the verdict is recorded as comparison=absolute. Without this "
                         "the gate is only a RELATIVE invariance check -- two cache-ON runs "
                         "agreeing with each other. A cache-ON reference is REFUSED here and "
                         "--allow-stale-oracle / --allow-model-mismatch do NOT override it.")
    ap.add_argument("--no-expert-cache", action="store_true",
                    help="ground-truth arm: launch with -expert-cache 0, so no cache object and "
                         "no hook exist and the forward pass is the stock full-resident one. "
                         "Use with --dump-oracle to produce the reference that --oracle-abs "
                         "demands. Slow (no bounded residency) and NOT a benchmark configuration.")
    ap.add_argument("--dump-oracle", action="store_true",
                    help="write the logits oracle dump WITHOUT comparing it to anything. This is "
                         "how a reference is produced: previously the dump was only written as a "
                         "side effect of --oracle-ref, so making a fresh reference required "
                         "pointing at some other dump first -- a chicken-and-egg that made the "
                         "reference's own provenance accidental.")
    ap.add_argument("--m2", action="store_true",
                    help="M2 prefill-stream mode: sets CGC_PREFILL_STREAM=1 CGC_GATHER_SLAB_CAP=256, "
                         "uses ref_<model>_pool8gb_M2.jsonl as default oracle ref, and records "
                         "mode=M2_stream in provenance. M2 has its own oracle because it changes "
                         "the prefill chunk size (L2 divergence vs standard path).")
    ap.add_argument("--pool-cap", default="auto",
                    help="CGC_POOL_MAX_TOKENS, IDENTICAL for every pool (default 'auto' = probe "
                         "the smallest pool until it reports no `buffer is nil`, then reuse that "
                         "cap everywhere; 'off' leaves the build default). One shared cap is "
                         "required because the cap clamps n_batch and so the ubatch shape.")
    ap.add_argument("--pool-cap-max", type=int, default=8,
                    help="largest cap 'auto' may pick (build default is 8)")
    # PRE-LAUNCH FEASIBILITY GATE (default ON). Rationale + evidence:
    # docs/POOL_FEASIBILITY_MATRIX_2026-09-14.html. The short version: a cell whose cap has to
    # drop is not a pool measurement, and the number it produces is indistinguishable from one.
    ap.add_argument("--no-feasibility-gate", action="store_true",
                    help="disable the pre-launch (model, pool, cap) feasibility check that refuses "
                         "PARTITION-REF / UNUSABLE cells")
    ap.add_argument("--allow-partition-ref", action="store_true",
                    help="launch PARTITION-REF cells anyway (the cap has to drop below the "
                         "reference, so the measurement includes the prefill step partition). "
                         "The verdict is still recorded next to the result")
    ap.add_argument("--allow-unusable", action="store_true",
                    help="launch UNUSABLE cells anyway (no cap satisfies capacity AND survival; "
                         "the server is expected to abort or overflow the union)")
    ap.add_argument("--allow-geometry-drift", action="store_true",
                    help="measure even when the cap probe's MEASURED slot count disagrees with "
                         "pool_feasibility.py's arithmetic. Off by default: the arithmetic is what "
                         "decides whether a cell is comparable at all, so a falsified predictor "
                         "must be fixed before cells are launched (or at least acknowledged)")
    ap.add_argument("--feasibility-only", action="store_true",
                    help="print the pre-launch feasibility table for --models x --pools and exit "
                         "(no launch, no GPU)")
    ap.add_argument("--cap-invariance", action="store_true",
                    help="run ONLY the cap-invariance gate for every (model x pool): dump the logits "
                         "oracle once per cap, realign both dumps on sequence position, and compare "
                         "M1/M2/M3. This is the gate that decides whether a cap change is "
                         "numerically inert -- i.e. whether the pool LAYOUT can be treated as a "
                         "speed-only knob. Measured 2026-09-14 by hand: an 8 GiB pool at cap=5 was "
                         "13-15%% faster than cap=8 and M1 3/82 -- a real lever that no cheaper "
                         "signal (speed, nil, union-fit, template) can distinguish from a clean run.")
    ap.add_argument("--cap-invariance-caps", default=CAP_INVARIANCE_DEFAULT_CAPS,
                    help=f"comma list of caps to compare against each other (default "
                         f"{CAP_INVARIANCE_DEFAULT_CAPS!r}: the harness floor and the build "
                         f"default). Two launches per (model x pool) per cap; dumps are cached "
                         f"and reused while their provenance still matches (see --force).")
    ap.add_argument("--capinv-min-coverage", type=float, default=0.95,
                    help="minimum fraction of position-aligned steps two cap dumps must share "
                         "before the pair may be certified INERT (default 0.95). Gates the PASS "
                         "direction only: a non-full M1 is divergence evidence and is reported as "
                         "FAIL/DRIFT however partial the overlap, because divergent generations "
                         "stop sharing positions BY CONSTRUCTION once a token differs.")
    ap.add_argument("--capinv-min-steps", type=int, default=16,
                    help="absolute floor on the number of POSITION-ALIGNED base-context (DEF) steps "
                         "required to certify a pair INERT (default 16). Coverage alone is not "
                         "enough -- two dumps that both lost the same rows to bad alignment would "
                         "agree perfectly on the little that is left -- but the floor has to sit "
                         "under what the probe actually produces: with MTP on, a 48-token probe "
                         "yields only ~30 base-context rows, because most batches are 4-token "
                         "verify steps rather than 1-token decodes.")
    ap.add_argument("--allow-model-mismatch", action="store_true",
                    help="compare against an oracle reference taken from a DIFFERENT GGUF. Off "
                         "by default and rarely what you want: logits from two models are not "
                         "comparable, so an M1 failure would be guaranteed and meaningless. "
                         "This matrix's `iq3` column once silently resolved to the IQ4_XS file, "
                         "so two columns measured one model -- the model guard exists for that.")
    ap.add_argument("--allow-stale-oracle", action="store_true",
                    help="compare against an oracle reference even when its provenance stamp "
                         "shows it was produced from older expert-cache/context code. Off by "
                         "default: a stale reference fails loudly and looks like a real pool "
                         "regression (measured 2026-09-13: M1 0/53 that looked like a bug for "
                         "hours). Use only when you have confirmed the drift is numerics-neutral.")
    ap.add_argument("--summary", action="store_true",
                    help="print the consolidated (model x pool) gate table from the records on "
                         "disk and exit. Warns when a model was measured with more than one cap.")
    ap.add_argument("--gates-only", action="store_true",
                    help="run only the correctness gates (template / union-fit / buffer-nil / "
                         "oracle M1-M2) and skip the quality suite. Use this to establish that "
                         "pool sizes differ ONLY in speed before spending time on quality.")
    ap.add_argument("--preflight-only", action="store_true",
                    help="probe the server on --port and exit (is its scaffold sane?)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (models/pools/kwargs/facts) and exit")
    args = ap.parse_args()

    # M2 mode: use the M2-specific reference oracle by default. M2 changes the prefill
    # chunk size (L2 divergence vs standard path), so it must NOT be compared against the
    # standard-path reference. The M2 reference lives at ref_<model>_pool8gb_M2.jsonl.
    if args.m2 and not args.oracle_ref and not args.dump_oracle:
        m2_ref = os.path.join(RESULT_DIR, f"ref_{args.models}_pool8gb_M2.jsonl")
        if os.path.exists(m2_ref):
            args.oracle_ref = m2_ref
            print(f"[m2] using M2 reference oracle: {m2_ref}", flush=True)
        else:
            print(f"[m2] WARNING: no M2 reference oracle at {m2_ref}; "
                  f"run with --m2 --dump-oracle on 8GB pool to create it", flush=True)

    if args.summary:
        summary_table()
        return 0

    if args.pilot:
        args.models, args.pools = "iq3,iq4", "8"
        args.profiles, args.per_profile = "qa-zh", 1
        args.greedy_repeats, args.repeats = 2, 2
        args.max_tokens, args.timeout = 64, 120.0

    args_greedy_global = args.greedy_repeats

    # EXCLUSIVITY: earlier rounds were measured with other servers live, which corrupts
    # memory and speed readings. Refuse by default instead of silently contaminating.
    if args.preflight_only:
        pf = preflight(args.port, timeout=args.preflight_timeout)
        print(json.dumps(pf, indent=2, ensure_ascii=False))
        d = os.path.join(RESULT_DIR, "preflight.json")
        os.makedirs(RESULT_DIR, exist_ok=True)
        json.dump(pf, open(d, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"wrote {d}")
        return 0 if pf.get("ok") else 1

    if args.dry_run:
        print_feasibility_matrix([m.strip() for m in args.models.split(",") if m.strip()],
                                 [int(p) for p in args.pools.split(",") if p.strip()], args)
        print(f"models      : {args.models}\npools       : {args.pools}\n"
              f"kwargs      : {args.kwargs!r}\npreflight   : {args.preflight}\n"
              f"attach      : {args.attach}\nextra_env   : {args.extra_env}\n"
              f"result dir  : {RESULT_DIR}")
        for m in (x.strip() for x in args.models.split(",") if x.strip()):
            f = model_path(m) if m in MODELS else "?"
            print(f"  {m:8s} exists={os.path.exists(f)}  {f}")
        print(f"running servers now: {running_servers() or 'none'}")
        if args.pool_cap == "auto":
            print(f"pool-cap    : auto (will probe the smallest pool "
                  f"{min(int(p) for p in args.pools.split(',') if p.strip())}GB per model "
                  f"up to cap-max={args.pool_cap_max})")
        else:
            print(f"pool-cap    : {args.pool_cap}")
        return 0

    existing = running_servers()
    if existing and args.attach:
        print(f"attaching to existing llama-server(s) {existing} (--attach)", flush=True)
    elif existing:
        if args.kill_existing:
            print(f"killing pre-existing llama-server(s): {existing}", flush=True)
            kill_servers()
        else:
            print(f"REFUSING to measure: {len(existing)} llama-server(s) already running "
                  f"(pids {existing}).\nAnother agent may be running its own pool sweep; "
                  f"results would be contaminated.\nRe-run with --kill-existing to clear them.",
                  file=sys.stderr)
            return 2

    os.makedirs(RESULT_DIR, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    pools = [int(p) for p in args.pools.split(",") if p.strip()]
    for m in models:
        if m not in MODELS:
            print(f"unknown model {m!r}; known: {', '.join(MODELS)}", file=sys.stderr)
            return 2
        p = model_path(m)
        if not os.path.exists(p):
            print(f"WARNING: model file missing for {m}: {p}", file=sys.stderr)

    # The whole grid's verdicts, before a single server is started. Cheap (GGUF tensor tables
    # only), and it turns "why was that cell refused?" into something readable up front. Ahead of
    # the model-identity gate on purpose: inspecting one model's grid is legitimate even when two
    # configured columns turn out to be the same file.
    if args.feasibility_only:
        print_feasibility_matrix(models, pools, args)
        return 0

    # Before any cap probing or launching: prove the columns really are different files. A
    # matrix of one duplicated model produces perfectly self-consistent numbers and zero
    # cross-model evidence, so this must be a hard stop, not a warning.
    if not gate_model_identity(models):
        return 2

    print(f"matrix: models={models} pools={pools} kwargs={args.kwargs!r} "
          f"tag={args.tag!r} extra_env={args.extra_env}", flush=True)

    if args.cap_invariance:
        caps = [int(c) for c in args.cap_invariance_caps.split(",") if c.strip()]
        if len(caps) < 2:
            print(f"--cap-invariance needs at least two caps; got {caps}", file=sys.stderr)
            return 2
        if any(c < MIN_CAP for c in caps):
            print(f"WARNING: cap < {MIN_CAP} aborts at llama-batch.cpp:609 "
                  f"(GGML_ASSERT(n_ubatch > n_keep_tail)); caps={caps}", file=sys.stderr)
        print(f"cap invariance: caps={caps} pools={pools} (one launch per cap per pool)",
              flush=True)
        records = []
        for kind in models:
            for gb in pools:
                records.append(cap_invariance_gate(kind, gb, caps, args))
        store = _store_capinv(records)
        print_capinv_table(records)
        print(f"\nwrote {os.path.join(RESULT_DIR, 'cap_invariance.json')} "
              f"({len(store)} stored run(s); --summary prints all of them)")
        return 0 if all(r.get("ok") is True for r in records) else 1

    # Resolve ONE pool cap per model BEFORE any measurement, so that no pool size can fall into
    # the L3-B gather path (the 4GB `buffer is nil` bug) and so that every pool is measured with
    # the same ubatch shape. This is the harness half of "4/6/8GB must differ only in speed";
    # the M1/M2 oracle gate is the verification half.
    args.pool_cap_resolved = {}
    if args.attach:
        for kind in models:
            args.pool_cap_resolved[kind] = None
        print("[cap] --attach: cannot probe, using the server's own cap", flush=True)
    elif args.pool_cap == "off":
        for kind in models:
            args.pool_cap_resolved[kind] = None
        print("[cap] --pool-cap off: leaving the build default (NOT pool-independent)", flush=True)
    elif args.pool_cap != "auto":
        for kind in models:
            args.pool_cap_resolved[kind] = int(args.pool_cap)
        print(f"[cap] fixed cap={args.pool_cap} for every pool", flush=True)
    else:
        smallest = min(pools)
        for kind in models:
            rec = probe_pool_cap(kind, smallest, args)
            # The probe just MEASURED the slots; if they disagree with the arithmetic that every
            # feasibility verdict in this sweep rests on (`cap_max`, hence PARTITION-REF vs
            # UNUSABLE), stop here -- while nothing has been measured yet.
            conflict = geometry_conflict_error(rec, args.allow_geometry_drift)
            if conflict:
                print(f"\n{conflict}", file=sys.stderr)
                return 2
            args.pool_cap_resolved[kind] = rec.get("cap")

    # Every cell's verdict, WITH the caps the sweep resolved above -- this is the table that says
    # which cells the gate below will refuse and why.
    print_feasibility_matrix(models, pools, args, args.pool_cap_resolved)

    records = []
    for kind in models:
        for gb in pools:
            records.append(run_combo(kind, gb, args))

    json.dump(records, open(os.path.join(RESULT_DIR, "matrix.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print_matrix(records)
    print(f"\nwrote {os.path.join(RESULT_DIR, 'matrix.json')}")
    return 0



if __name__ == "__main__":
    sys.exit(main())
