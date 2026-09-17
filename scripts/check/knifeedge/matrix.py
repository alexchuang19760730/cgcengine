#!/usr/bin/env python3
"""knifeedge.matrix — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from .anchor import RESULT_DIR, ROOT
from .caps import apply_template_probe, comparability_verdict, preflight, scan_nil, union_fit_gate
from .constants import KEY_FIELDS, MODELS
from .feasibility import cap_for, feasibility_gate
from .host import ask_probe, kill_servers, launch, launch_pool_bytes, mem_free_pct, read_launch_facts, rss_gb, running_servers, wait_health
from .identity import model_path
from .oracle import _write_oracle_meta, expert_cache_state, oracle_gate, oracle_recompare
from .provenance import record_geometry_key, record_provenance



def combo_label(kind, gb, tag):
    return f"{kind}_pool{gb}gb{('_' + tag) if tag else ''}"



def run_combo(kind, gb, args):
    model = MODELS[kind]
    # A ground-truth dump written over the cache-ON dump of the same (model, pool) would destroy
    # the very evidence it exists to judge -- and the two files have the same default name. So
    # --no-expert-cache forces a distinct label unless the caller named one explicitly.
    tag = args.tag
    if getattr(args, "no_expert_cache", False) and not tag:
        tag = "nocache"
    label = combo_label(kind, gb, tag)
    out_json = os.path.join(RESULT_DIR, f"knifeedge_{label}.json")
    # The cap this cell will launch with, resolved up front because the resume check compares the
    # cached record's geometry -- and the cap is part of that geometry (it clamps n_batch).
    cap = cap_for(args, kind)
    if os.path.exists(out_json) and not args.force:
        # RESUME reads the COMBO record, not the suite output: `knifeedge_*.json` is flip_rate's
        # summary and carries no `up`/`gates`, so gating resume on it meant no full run could ever
        # be skipped -- the check always fell through to REDO. The combo record is the harness's
        # own measurement record, and it now also carries the provenance that decides whether a
        # cached measurement still describes this tree.
        prev_path = os.path.join(RESULT_DIR, f"combo_{label}.json")
        prev = None
        if os.path.exists(prev_path):
            try:
                prev = json.load(open(prev_path, encoding="utf-8"))
            except Exception:  # noqa: BLE001 - harness
                prev = None
        if prev is not None:
            # A combo that never came up is NOT a result. Caching it as one is the same trap as
            # the false-passing cap probe: the next run skips the cell and the table shows a
            # config that was never measured. Measured case (2026-09-11): run_server.sh's memory
            # guard refused to start (`free=15%<40%`), the stub `{"up": false}` was written, and
            # the following run printed SKIP for it.
            why = None
            if not prev.get("up"):
                why = f"it never came up (up={prev.get('up')!r}, skipped={prev.get('skipped')!r})"
            elif not (prev.get("gates") or prev.get("skipped") is None):
                why = "it has no gate evidence"
            elif prev.get("gates_only"):
                why = "it is a --gates-only run, not a quality measurement"
            elif prev.get("skipped"):
                why = f"it was skipped ({prev.get('skipped')!r})"
            else:
                # PROVENANCE: a cached measurement that cannot prove WHICH pool geometry, engine
                # and weights produced it is exactly the 106-vs-143 trap one layer up -- the
                # number is plausible and unattributable.
                prev_key = record_geometry_key(prev)
                want = record_geometry_key({"pool_cap": cap, "provenance": record_provenance(
                    kind, gb, args, cap=cap)})
                if prev_key is None:
                    why = ("it carries no provenance, so nothing ties it to this tree -- re-run "
                           "once with the current harness to make it attributable")
                elif prev_key != want:
                    moved = [human for f, human in KEY_FIELDS
                             if prev_key.get(f) != want.get(f)]
                    why = (f"it was measured under a different geometry ({', '.join(moved)}); the "
                           f"pool-size comparison would mix two computations")
            if why is None:
                print(f"[{label}] SKIP (complete, attributable result exists; --force to redo)",
                      flush=True)
                return prev
            print(f"[{label}] REDO: the cached result is not reusable -- {why}; re-running",
                  flush=True)

    launch_log = os.path.join(RESULT_DIR, f"launch_{label}.log")
    # Oracle dump is env-gated at launch; kept out of the result name so a rerun overwrites it.
    oracle_dump = os.path.join(RESULT_DIR, f"oracle_{label}.jsonl")
    extra = list(args.extra_env)
    # M2 prefill-stream mode: whole-layer slab gather, has its own oracle (L2 divergence vs std)
    # getattr, not args.m2: run_combo is also driven by callers that build their own Namespace
    # (feasibility_gate_selftest.py), and an AttributeError there reads as "the gate crashed"
    # rather than "the flag is off". Off-by-default is the correct meaning of an absent flag.
    if getattr(args, "m2", False):
        extra.append("CGC_PREFILL_STREAM=1")
        extra.append("CGC_GATHER_SLAB_CAP=256")
    # The union-routable gate must not depend on an OPTIONAL diagnostic flag the caller happened
    # to pass: on 2026-09-14 the 2 GiB cell scored M1/M2/M3 117/117 while the gate said FAIL
    # mode=unreachable simply because CGC_UNION_LOG was not set, so the `gather=N/M` evidence it
    # reads had never been written. "Unknown" and "not routable" must not look the same -- the
    # harness therefore always asks the engine for the evidence it is about to judge. (Without
    # this the gate can only ever pass a config whose FIRST version was measured by hand.)
    extra.append("CGC_UNION_LOG=1")
    # One stamp for every record this cell writes. Computed HERE (before the launch) so the same
    # object can go into the failure records too -- a cell that died still has to say which
    # geometry it died under.
    prov = record_provenance(kind, gb, args, cap=cap)
    # PRE-LAUNCH FEASIBILITY: before anything is killed or started. A cell that cannot run at
    # this cap does not produce a pool measurement, so it must not become a table row -- and per
    # the same rule as the low-memory refusal below, a refusal is NOT a result and is not written
    # to out_json (a cached stub would then be SKIPped by later runs).
    if getattr(args, "no_expert_cache", False):
        # Ground-truth arm: there is no pool to be feasible. The gate's whole question
        # ("can a cap satisfy capacity AND survival?") is undefined at budget 0, and asking it
        # would refuse the very cell that has to exist for --oracle-abs to be usable at all.
        g_feas = {"verdict": "GROUND_TRUTH", "why": "expert cache off (budget 0); no pool "
                                                    "geometry to validate", "pool_gb": gb}
        feas_ok = True
    else:
        feas_ok, g_feas = feasibility_gate(kind, gb, args, cap, args.extra_env, where=label)
    if not feas_ok:
        return {"label": label, "model": kind, "pool_gb": gb, "up": False,
                "skipped": "infeasible_" + g_feas["verdict"].lower().replace("-", "_"),
                "feasibility": g_feas}
    if cap is not None:
        # ONE cap for every pool so the ubatch shape and the chunking are identical across pool
        # sizes -- otherwise a pool "difference" is partly a shape difference (see probe_pool_cap).
        extra.append(f"CGC_POOL_MAX_TOKENS={cap}")
    if args.oracle_ref or args.dump_oracle:
        if os.path.exists(oracle_dump):
            os.remove(oracle_dump)
        extra += [f"CGC_LOGITS_ORACLE_DUMP={oracle_dump}", "CGC_LOGITS_ORACLE_TOPN=8"]
        # Record provenance next to the dump so a future comparison can PROVE it is looking at
        # the same computation: the cap (which clamps n_batch) AND the state of the code that
        # decides the numerics. Without this, forgetting --pool-cap silently yields reference
        # and candidate computed under different ubatch shapes, and an expert-cache edit makes
        # an old reference describe an engine that no longer exists -- in both cases the
        # resulting M1 collapse looks exactly like a real pool-induced numeric regression.
        # expert_cache is part of the computation's identity, not a label: without it a
        # cache-ON dump can be handed to --oracle-abs and scored as if it were a control.
        _write_oracle_meta(oracle_dump, cap if cap is not None else args.pool_cap_max,
                           model_file=model_path(kind),
                           launch_sig={"extra_env": sorted(args.extra_env or []),
                                       "mtp": MODELS[kind]["mtp"],
                                       "expert_cache": expert_cache_state(
                                           args.extra_env, launch_pool_bytes(args, gb))},
                           expert_cache=expert_cache_state(args.extra_env,
                                                           launch_pool_bytes(args, gb)))
    if args.attach:
        # Shared checkout: someone else's server may already own the port. Measure it
        # read-only (no launch, no kill) instead of fighting over the machine.
        print(f"[{label}] ATTACH to already-running server on port {args.port}", flush=True)
        free_before = free_after = mem_free_pct()
        took, facts = 0.0, {"attached": True}
    else:
        print(f"[{label}] launching {model['file']} pool={gb}GB mtp={model['mtp']}", flush=True)
        kill_servers()
        if running_servers():
            print(f"[{label}] REFUSING: another agent's server will not clear "
                  f"({running_servers()}); its /health would be measured instead of ours.",
                  file=sys.stderr, flush=True)
            return {"label": label, "model": kind, "pool_gb": gb, "up": False,
                    "skipped": "foreign_server"}
        free_before = mem_free_pct()
        # MEMORY PRECONDITION. run_server.sh refuses to start full-MTP below 40% free, so
        # launching under that is guaranteed to produce nothing but a launch log full of guard
        # errors -- and (before the skip-rule fix) a cached stub that later runs would SKIP.
        # Check it here so the failure is loud, named, and not written as a measurement.
        if free_before < args.min_free_pct:
            print(f"[{label}] REFUSING to launch: machine is {free_before}% free, below the "
                  f"{args.min_free_pct}% precondition (run_server.sh's memory guard would "
                  f"reject it: 'startup blocked by memory guard').\n"
                  f"  This is a MACHINE precondition, not a result. Free memory (a reboot clears "
                  f"the compressor + swap that repeated 13-18 GB model loads leave behind) and "
                  f"re-run; nothing was measured for this cell.", file=sys.stderr, flush=True)
            return {"label": label, "model": kind, "pool_gb": gb, "up": False,
                    "skipped": "low_memory", "free_before_pct": free_before}
        pre = tuple(running_servers())
        launch(model_path(kind), model["mtp"], launch_pool_bytes(args, gb), args.port, args.kwargs,
               extra, launch_log)
        took = wait_health(args.port, model_path(kind), args.health_timeout, pre_pids=pre)

    if took is None:
        print(f"[{label}] FAILED to become healthy (server died, timed out, or a foreign "
              f"server owns port {args.port})", flush=True)
        tail = ""
        try:
            tail = open(launch_log, "rb").read().decode("utf-8", "replace")[-800:]
        except Exception:  # noqa: BLE001
            pass
        rec = {"label": label, "model": kind, "pool_gb": gb, "up": False,
               "feasibility": g_feas, "provenance": prov,
               "free_before_pct": free_before, "launch_tail": tail}
        json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        kill_servers()
        return rec

    if not args.attach:
        facts = read_launch_facts(launch_log)
    # Re-stamp with the facts the engine reported at load: the record then carries both the
    # PREDICTED slot geometry and the MEASURED one, and a disagreement is visible per row.
    prov = record_provenance(kind, gb, args, cap=cap, facts=facts)
    rss = rss_gb()
    free_after = mem_free_pct()
    print(f"[{label}] healthy in {took:.0f}s  rss={rss}GB free={free_before}%->{free_after}%  "
          f"kind={facts.get('model_kind')} template={facts.get('template_file', '(embedded)')}",
          flush=True)

    # GATE: numeric invariance vs the reference oracle. Run this BEFORE anything else talks to
    # the server so the dump's early ubatches line up with the reference run's.
    g_oracle = None
    g_oracle_final = None
    if args.oracle_ref:
        g_oracle = oracle_gate(args.port, oracle_dump, args.oracle_ref, label, args.timeout,
                               allow_stale=args.allow_stale_oracle,
                               model_file=model_path(kind),
                               allow_model_mismatch=args.allow_model_mismatch,
                               absolute=getattr(args, "oracle_abs", False))
        print(f"[{label}] gate-oracle  {'PASS' if g_oracle.get('ok') else 'CHECK'}  "
              f"M1(bit-identical)={g_oracle.get('m1_numeric_identity')} "
              f"M2(argmax)={g_oracle.get('m2_decision_agreement')} "
              f"{g_oracle.get('error', '')}", flush=True)
    elif args.dump_oracle:
        # Producing a reference must exercise the SAME probe sequence a comparison cell will.
        # `ask_probe()` lives inside oracle_gate and is what appends those steps to the dump.
        # Without this call the reference is a strict PREFIX of every candidate, so a
        # prefix-matching compare reports a perfect 60/60 while the remaining keys are never
        # tested -- the same silent understatement as §6 of POOLSIZE_INVARIANCE_2026-09-13.md,
        # just moved into the reference instead of the comparison.
        ask_probe(args.port, args.timeout)
        time.sleep(1.0)
        print(f"[{label}] dump-oracle: probe sent, so the reference covers the same steps as "
              f"a comparison cell", flush=True)

    # GATE: the rendered generation prompt must be a CLOSED scaffold. A degenerating model can
    # still stumble onto the right answer, so the answer check alone is not sufficient.
    g_template = apply_template_probe(args.port)
    print(f"[{label}] gate-template {'PASS' if g_template.get('ok') else 'FAIL' if g_template.get('ok') is False else 'n/a'}  "
          f"unclosed_think={g_template.get('unclosed_think')} "
          f"tail={g_template.get('tail', '')[-70:]!r}", flush=True)

    # GATE: `buffer is nil` = the pool path silently fell through to the L3-B gather path on a
    # GPU-resident layer. Invalidates the combo AND explains any numeric drift.
    g_nil = scan_nil([launch_log, facts.get("log_path")])
    print(f"[{label}] gate-buffer-nil {'PASS' if g_nil['ok'] else 'FAIL'}  {g_nil['hits']}", flush=True)

    # GATE: is the broken path unreachable BY CONSTRUCTION? `union_max <= usable_slots` is the
    # arithmetic guarantee, independent of which experts this particular prompt happens to route
    # to -- so it holds for every model and every pool size, not just the probes here.
    g_union = union_fit_gate(facts, cap if cap is not None else args.pool_cap_max,
                             MODELS[kind].get("topk", 8),
                             paths=[launch_log, facts.get("log_path")])
    _ev = g_union.get("gather") or {}
    print(f"[{label}] gate-union-routable "
          f"{'PASS' if g_union['ok'] else 'FAIL' if g_union['ok'] is False else 'n/a'}  "
          f"mode={g_union.get('mode')} cap={g_union['cap']} x topk={g_union['topk']} = "
          f"{g_union['union_max']} vs usable={g_union.get('usable_slots')} "
          f"(n_slots={g_union.get('n_slots')}); wide route: layers={_ev.get('wide_layers')} "
          f"steps={_ev.get('gather_steps')}/{_ev.get('total_steps')} "
          f"{g_union.get('error', '')}", flush=True)

    # PREFLIGHT: refuse to produce numbers from a broken prompt scaffold. This is the gate
    # that was missing when the earlier pool curve / flip_rate runs were silently measuring
    # a template defect instead of the expert cache.
    if args.preflight != "off":
        pf = preflight(args.port, timeout=args.preflight_timeout)
        rec_pf = {k: pf.get(k) for k in ("ok", "answers_42", "echoes_prompt",
                                         "scaffold_leak", "empty", "finish", "error")}
        print(f"[{label}] preflight {'PASS' if pf.get('ok') else 'FAIL'}  "
              f"42={pf.get('answers_42')} echo={pf.get('echoes_prompt')} "
              f"leak={pf.get('scaffold_leak')} empty={pf.get('empty')} "
              f"finish={pf.get('finish')}  content={pf.get('content','')[:90]!r}", flush=True)
        # The probes above appended to the same dump the oracle gate compared, so that verdict
        # only covered a prefix. Re-compare the finished file before anything is recorded.
        g_oracle_final = oracle_recompare(oracle_dump, args.oracle_ref, label,
                                         allow_stale=args.allow_stale_oracle,
                                         model_file=model_path(kind),
                                         allow_model_mismatch=args.allow_model_mismatch,
                                         absolute=getattr(args, "oracle_abs", False))
        if not pf.get("ok"):
            msg = (f"[{label}] preflight FAILED -- the prompt scaffold is broken, so any "
                   f"quality number from this config would be measuring the template, "
                   f"not the pool. Fix the scaffold first (see "
                   f"docs/CHAT_SCAFFOLD_ROOTCAUSE_2026-09-11.md).")
            if args.preflight == "abort":
                print(msg, file=sys.stderr, flush=True)
                # Keep the gates in the record even on abort. The oracle/template/nil verdicts
                # were already measured and they are exactly what you need when preflight fails:
                # dropping them here once threw away the M1/M2 comparison precisely in the case
                # where the config is already suspect.
                rec = {"label": label, "model": kind, "pool_gb": gb, "up": True,
                       "preflight": rec_pf,
                       "gates": {"template": g_template, "buffer_nil": g_nil,
                                 "union_fit": g_union, "oracle": g_oracle,
                                 "oracle_final": g_oracle_final},
                       "pool_cap": cap, "feasibility": g_feas, "provenance": prov,
                       "free_before_pct": free_before,
                       "free_after_pct": free_after, "rss_gb": rss,
                       "launch_facts": facts, "extra_env": args.extra_env,
                       "skipped": "preflight_failed"}
                json.dump(rec, open(os.path.join(RESULT_DIR, f"combo_{label}.json"), "w",
                                    encoding="utf-8"), indent=2, ensure_ascii=False)
                if not args.attach:
                    kill_servers()
                return rec
            print(msg, file=sys.stderr, flush=True)
    else:
        rec_pf = None
        # No preflight traffic, but the template probe still grew the dump; re-compare it.
        g_oracle_final = oracle_recompare(oracle_dump, args.oracle_ref, label,
                                         allow_stale=args.allow_stale_oracle,
                                         model_file=model_path(kind),
                                         allow_model_mismatch=args.allow_model_mismatch,
                                         absolute=getattr(args, "oracle_abs", False))

    if args.gates_only:
        print(f"[{label}] --gates-only: skipping the quality suite", flush=True)
        if not args.attach:
            kill_servers()
        rec = {"label": label, "model": kind, "pool_gb": gb, "up": True,
               "preflight": rec_pf,
               "gates": {"template": g_template, "buffer_nil": g_nil, "union_fit": g_union,
                         "oracle": g_oracle,
                         "oracle_final": g_oracle_final},
               "pool_cap": cap, "gates_only": True, "feasibility": g_feas,
               "provenance": prov,
               "gates_ok": (bool(g_nil["ok"]) and g_template.get("ok") is not False
                            and g_union.get("ok") is not False),
               "free_before_pct": free_before, "free_after_pct": free_after,
               "rss_gb": rss, "launch_facts": facts, "extra_env": args.extra_env,
               "attached": bool(args.attach)}
        json.dump(rec, open(os.path.join(RESULT_DIR, f"combo_{label}.json"), "w",
                            encoding="utf-8"), indent=2, ensure_ascii=False)
        return rec

    cmd = [sys.executable, os.path.join(ROOT, "scripts", "check", "flip_rate.py"),
           "--base-url", f"http://127.0.0.1:{args.port}/v1", "--model", "local",
           "--label", label, "--profiles", args.profiles,
           "--per-profile", str(args.per_profile),
           "--greedy-repeats", str(args.greedy_repeats),
           "--repeats", str(args.repeats),
           "--temperature", str(args.temperature),
           "--max-tokens", str(args.max_tokens),
           "--timeout", str(args.timeout),
           "--output", out_json]
    subprocess.run(cmd, cwd=ROOT)

    # Re-scan for `buffer is nil` AFTER the traffic above. The early scan runs before any request
    # has been served, so an empty result there only means "no decode built a union yet" -- it
    # cannot clear a combo. This second scan is the one that can actually observe the failure.
    g_nil_post = scan_nil([launch_log, facts.get("log_path")])
    if not g_nil_post["ok"]:
        if g_nil["ok"]:
            print(f"[{label}] gate-buffer-nil FAIL (post-traffic) {g_nil_post['hits']}",
                  flush=True)
        g_nil = g_nil_post

    if not args.attach:
        kill_servers()
    rec = {"label": label, "model": kind, "pool_gb": gb, "up": True,
           "preflight": rec_pf,
           "gates": {"template": g_template, "buffer_nil": g_nil, "union_fit": g_union,
                     "oracle": g_oracle,
                     "oracle_final": g_oracle_final},
           "pool_cap": cap, "feasibility": g_feas, "provenance": prov,
           "gates_ok": (bool(g_nil["ok"]) and g_template.get("ok") is not False
                        and g_union.get("ok") is not False),
           "free_before_pct": free_before, "free_after_pct": free_after,
           "rss_gb": rss, "launch_facts": facts, "extra_env": args.extra_env,
           "attached": bool(args.attach)}
    if os.path.exists(out_json):
        sess = json.load(open(out_json, encoding="utf-8"))
        rec.update({k: sess.get(k) for k in
                    ("greedy_all_pass_rate", "greedy_unstable_questions",
                     "seeded_flip_questions", "suite_rate_per_seed", "suite_rate_mean",
                     "suite_rate_std", "suite_rate_min", "suite_rate_max",
                     "tps_mean", "tps_std", "tps_max", "n_questions")})
        # Stamp the SUITE's own output too. `knifeedge_*.json` is the artifact read months later,
        # and until now it was the one file in the tree that described a measurement without
        # saying which pool geometry, engine and weights produced it -- so two of them could be
        # lined up side by side with no way to tell whether that was legitimate.
        try:
            sess["provenance"] = prov
            sess["model"] = kind
            sess["pool_gb"] = gb
            sess["pool_cap"] = cap
            json.dump(sess, open(out_json, "w", encoding="utf-8"), indent=2,
                      ensure_ascii=False)
        except Exception as e:  # noqa: BLE001 - harness
            print(f"[{label}] WARNING: could not stamp {out_json}: {e}", file=sys.stderr,
                  flush=True)
    json.dump(rec, open(os.path.join(RESULT_DIR, f"combo_{label}.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    return rec



def summary_table():
    """One row per (model x pool) with all four gates, assembled from the result records.

    Read the GEOMETRY GROUPS first, then the columns together. Rows are only laid out next to
    each other when the provenance says they were measured under the same engine, weights, launch
    env and cap; otherwise they are printed in separate blocks and the comparison is REFUSED,
    naming which of those moved. That is the same rule the cap probe and the oracle guard already
    enforce at launch, applied to the table decisions actually get made from.

    Read the columns together, not one at a time:
      cap must be IDENTICAL down a model's rows, or the pools were not the only variable;
      union-fit is the arithmetic reason the gather path is unreachable;
      nil is that prediction OBSERVED (nil PASS with union-fit n/a proves nothing);
      M1/M2 must stay apart -- M2 agreement on its own is not "no difference".
    """
    import glob
    recs, seen = [], {}
    # Read BOTH artifacts. `knifeedge_*.json` is the suite's own output (the file people actually
    # open months later) and `combo_*.json` is the harness record that carries the gates; they
    # describe the same cell, so they are merged by label rather than listed twice.
    for pat in ("combo_*.json", "knifeedge_*.json"):
        for p in sorted(glob.glob(os.path.join(RESULT_DIR, pat))):
            try:
                r = json.load(open(p, encoding="utf-8"))
            except Exception:  # noqa: BLE001 - harness
                continue
            base = os.path.basename(p)
            label = r.get("label") or base.split("_", 1)[1][:-len(".json")]
            r.setdefault("label", label)
            if label in seen:
                for k, v in r.items():
                    seen[label].setdefault(k, v)
                continue
            seen[label] = r
            recs.append(r)
    if not recs:
        print("no result records yet")
        return
    order = {"iq3": 0, "iq4": 1}
    recs.sort(key=lambda r: (order.get(r.get("model"), 9), r.get("pool_gb", 0)))
    verdict = comparability_verdict(recs)
    print("=" * 134)
    print("%-16s %4s %6s %6s %6s %9s %6s %6s %10s %12s %7s" % (
        "config", "cap", "rss", "free%", "tmpl", "union-fit", "nil", "geom", "M1 bit-id",
        "M2 argmax", "preflt"))
    print("-" * 134)

    def mark(v):
        return {True: "PASS", False: "FAIL", None: "n/a"}.get(v, "-")

    def pct(v):
        """M1/M2 come back either as a ratio or as '7/39' depending on the compare script."""
        if v is None:
            return "n/a"
        if isinstance(v, str):
            mm = re.match(r"^(\d+)\s*/\s*(\d+)$", v.strip())
            if mm and int(mm.group(2)):
                return f"{v} ({int(mm.group(1)) / int(mm.group(2)):.1%})"
            return v
        try:
            return f"{float(v):.1%}"
        except (TypeError, ValueError):
            return str(v)

    def row_line(r):
        g = r.get("gates") or {}
        t = (g.get("template") or {}).get("ok")
        n = (g.get("buffer_nil") or {}).get("ok")
        u = (g.get("union_fit") or {})
        o = (g.get("oracle") or {})
        pf = (r.get("preflight") or {}).get("ok")
        uf = mark(u.get("ok"))
        if u.get("headroom") is not None:
            uf += f" ({u['union_max']}<={u.get('usable_slots')}, {u['headroom']:+d})"
        # Per-row geometry mismatch: predicted vs launched slots/layer. Catches a stale
        # pool_feasibility.py on the first measurement, not on the next probe run.
        prov = r.get("provenance") or {}
        gm = (prov.get("pool") or {}).get("geometry_mismatch")
        gms = "n/a" if gm is None else ("MISMATCH" if gm else "ok")
        return "%-16s %4s %6s %6s %6s %-9s %6s %6s %10s %12s %7s" % (
            r.get("label"), r.get("pool_cap", "-"), r.get("rss_gb"),
            r.get("free_after_pct"), mark(t), uf, mark(n), gms,
            pct(o.get("m1_numeric_identity")), pct(o.get("m2_decision_agreement")), mark(pf))

    def geom_line(key, rows):
        """Name what produced a group, so the grouping itself is auditable rather than assumed."""
        if key is None:
            return ("  (no provenance: recorded before the stamp existed -> comparability "
                    "unverified)")
        eng = ((rows[0].get("provenance") or {}).get("engine") or {})
        w = key["weights"] or (None, None, None)
        return (f"  engine={str(key['engine'])[:8]} geometry={str(key['pool_geometry'])[:8]} "
                f"head={eng.get('head') or '?'}{' (dirty)' if eng.get('dirty') else ''} "
                f"weights={os.path.basename(str(w[0] or '?'))}@{w[1]} "
                f"cap={key['cap']} mtp={key['mtp']} spec_n_max={key['spec_n_max']} "
                f"layer_caps={key['layer_caps']} extra_env={list(key['extra_env']) or '-'}")

    for model in [m for m in verdict]:
        v = verdict[model]
        if v["state"] == "REFUSED":
            print(f"{model}: REFUSED -- {len(v['groups'])} geometry group(s); rows from different "
                  f"groups are NOT comparable and are printed separately below, with no "
                  f"cross-group claim made:")
            for d in v["differences"]:
                print(f"    - {d}")
        elif v["state"] == "UNATTRIBUTED":
            print(f"{model}: UNATTRIBUTED -- one group, but it carries no provenance, so its rows "
                  f"cannot be certified comparable (re-measure with the current harness).")
        for g in v["groups"]:
            print(f"[{model} group {g['fingerprint']}]  {len(g['rows'])} row(s)")
            print(geom_line(g["key"], g["rows"]))
            for r in sorted(g["rows"], key=lambda r: r.get("pool_gb", 0)):
                print(row_line(r))
    print("-" * 126)
    for model, v in verdict.items():
        if v["state"] == "OK":
            print(f"{model}: one geometry group and a single cap in it -> pool size is the only "
                  f"variable. OK")
        elif v["state"] == "UNATTRIBUTED":
            print(f"{model}: comparability UNVERIFIED (no provenance) -- do not read a pool "
                  f"ranking off these rows as though it were one experiment.")
        else:
            print(f"{model}: comparison REFUSED across {len(v['groups'])} geometry groups -- "
                  f"nothing in this table may be read as a pool ranking until the rows in "
                  f"different groups are re-measured under one tree.")
    # Cap invariance rides along in the same summary: it is the gate that says whether the `cap`
    # column above may be read as a speed-only knob, so leaving it in a separate mode would hide
    # the one verdict that qualifies every other row in this table.
    ci_path = os.path.join(RESULT_DIR, "cap_invariance.json")
    if os.path.exists(ci_path):
        try:
            ci = json.load(open(ci_path, encoding="utf-8"))
        except Exception:  # noqa: BLE001 - harness
            ci = []
        print()
        print("CAP INVARIANCE (same pool, different CGC_POOL_MAX_TOKENS)")
        print("-" * 140)
        print("%-30s %6s %6s %6s %8s %10s %12s %9s %7s" % (
            "config", "pool", "capA", "capB", "verdict", "M1 bit-id", "M2 argmax", "cover",
            "partn"))
        for r in ci:
            if not r.get("pairs"):
                print("%-30s   %s" % (r.get("label"), r.get("error") or "no pair compared"))
                continue
            for p in r["pairs"]:
                part = p.get("partition") or {}
                print("%-30s %5sG %6s %6s %8s %10s %12s %9s %7s" % (
                    r.get("label"), r.get("pool_gb"), p.get("cap_a"), p.get("cap_b"),
                    p.get("verdict"), p.get("m1_numeric_identity", "-"),
                    p.get("m2_decision_agreement", "-"),
                    (f"{p['coverage']:.0%}" if isinstance(p.get("coverage"), float) else "-"),
                    ("same" if part.get("same") else
                     ("diff" if part.get("a") is not None else "n/a"))))
        print("-" * 140)
        # The cap moves TWO things at once: it re-labels pool slots AND, via the n_batch clamp, it
        # changes the prefill step partition. Only the second one was measured to move the logits
        # (fixed cap across pools 2/4/6/8/10 GiB = bit-identical; different partitions = different
        # completions). `partn` says which of the two a row is actually about.
        print("partn = prefill step partition, same/diff. diff => the two caps ran different forward "
              "passes, so the row is NOT a statement about pool layout.")
        print("PASS = logits did not move (only meaningful when partn=same) | "
              "DRIFT = same decisions, different bits (NOT a pass) | FAIL = both differ")
    print("tmpl = scaffold closed | union-fit = cap*topk <= min_slots-1 (arithmetic) | "
          "nil = observed) | preflt = bare-prompt answer")
    print("=" * 126)
