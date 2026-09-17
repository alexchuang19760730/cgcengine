#!/usr/bin/env python3
"""knifeedge.capinv — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import json
import os
import time
from .anchor import RESULT_DIR
from .caps import scan_nil
from .constants import MODELS
from .feasibility import feasibility_gate
from .host import ask_probe, kill_servers, launch, launch_pool_bytes, mem_free_pct, read_launch_facts, rss_gb, running_servers, wait_health
from .identity import _cap_sidecar, _model_guard, _oracle_meta, model_path
from .oracle import _oracle_rows, _run_oracle_compare, _stamp_guard, _write_oracle_meta, expert_cache_state



def _realign_decode(dump, out_path, only_ctx=None):
    """Rewrite a dump so rows are keyed by the SEQUENCE POSITION of the token they describe.

    Why the raw keys cannot be used across two caps, in three parts -- each one measured while
    building this:

      1. `step` is a per-process counter of dump CALLS, i.e. of ubatches, and the cap clamps
         n_batch. Two caps therefore split one prompt into a different number of ubatches and
         number the same position differently.
      2. Dropping multi-token rows ("only compare decode steps") does not work either: with MTP on
         the base context runs VERIFY batches, so the bulk of its rows have n_tokens == 1+n_max.
         Measured on the existing 8 GiB cap=8 dump: 117 rows -> 21 "decode" rows, of which only 2
         were the base model's. The comparison would have been of the draft head, not the model.
      3. Positions repeat across REQUESTS (dump_seq never resets), so position alone collides.

    The fix: in a causal batch the rows cover contiguous positions ending at `pmax` (= the max
    sequence position after the ubatch), so row `token_idx` of an n-token dump sits at
    `pmax - (n-1) + token_idx`. That is a per-token position, valid for prefill chunks, single
    decode steps and verify batches alike.

    The second key slot is the OCCURRENCE index of that position (`token_idx` in the comparison
    script's key), NOT a request number. Two things make a position repeat inside one dump, both
    of them normal: a second request restarts at position 0, and an MTP draft rejection ROLLS
    BACK the memory so the same position is recomputed. An earlier version tried to detect
    request boundaries from a decreasing position -- that is unsound here, because how many
    rollbacks happen depends on the draft accept pattern, i.e. on the very numerics under test,
    so the two dumps would have been split into different numbers of "requests" (measured: 6 vs
    8) and only part of each would have aligned. Counting occurrences instead is canonical: for
    two dumps running the same requests in the same order, the k-th computation of position p
    aligns with the k-th computation of position p, whatever the chunking.

    Returns a stats dict, not a bare count, because how many rows were KEPT is part of the claim:
    a gate that silently compared three steps would pass vacuously.
    """
    stats = {"kept": 0, "bad_json": 0, "rows": 0, "ctx": {}, "repeat_rows": 0,
             "dropped_ctx": 0, "first_pos": None, "last_pos": None}
    occ = {}
    try:
        fin = open(dump, encoding="utf-8")
    except OSError:
        return stats
    with fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            if obj.get("pmax") is None:
                continue
            ctx = obj.get("ctx_type", "DEF")
            if only_ctx is not None and ctx != only_ctx:
                stats["dropped_ctx"] += 1
                continue
            n_tok = max(1, int(obj.get("n_tokens", 1)))
            idx = int(obj.get("token_idx", 0))
            pos = int(obj["pmax"]) - (n_tok - 1) + idx
            k = occ.get((ctx, pos), 0)
            occ[(ctx, pos)] = k + 1
            if k > 0:
                stats["repeat_rows"] += 1
            obj["step"] = pos
            obj["token_idx"] = k
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["ctx"][ctx] = stats["ctx"].get(ctx, 0) + 1
            if stats["first_pos"] is None or pos < stats["first_pos"]:
                stats["first_pos"] = pos
            stats["last_pos"] = pos
    return stats



def _first_divergence(a_path, b_path):
    """Smallest aligned position where two realigned dumps differ at the M1 (bits) level.

    This is the actionable half of a cap verdict: 'the logits differ' says something is wrong,
    'they first differ at position 12 of request 0, argmax 1234 -> 1235' says where to look.
    """
    def load(path):
        out = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    out[(int(o["step"]), int(o.get("token_idx", 0)), o.get("ctx_type", "DEF"))] = o
        except (OSError, json.JSONDecodeError):
            pass
        return out
    a, b = load(a_path), load(b_path)
    for k in sorted(set(a) & set(b)):
        if a[k].get("row_fnv1a64") != b[k].get("row_fnv1a64"):
            return {"pos": k[0], "occurrence": k[1], "ctx": k[2],
                    "argmax_a": a[k].get("argmax_token"), "argmax_b": b[k].get("argmax_token"),
                    "sum_a": a[k].get("sum"), "sum_b": b[k].get("sum")}
    return None



def step_partition(log_path, il=0):
    """Per-forward token counts actually used by the prefill, from the CGC-IDS hook.

    This exists because a cap-invariance FAIL is otherwise ambiguous. `CGC_POOL_MAX_TOKENS` is
    clamped onto `n_batch`, and `n_ubatch` follows from it, so the cap does not merely relabel
    pool slots -- it changes HOW MANY TOKENS EACH FORWARD PASS PROCESSES. Measured 2026-09-14
    (docs/CAP_INVARIANCE_LOCALIZATION_2026-09-14.md): three arms that differ only in that
    partition (cap5 -> 5,5,5...; cap8 -> 8,8,8...; cap8 with `-ub 5` -> 5,3,5,3...) produce three
    DIFFERENT completions, while the same cap across pools 2/4/6/8/10 GiB is bit-identical.
    So the partition is the variable that moves the numerics, and recording it turns a bare FAIL
    into a statement about which knob actually changed.

    Only the prefill matters here: the decode tail is single-token in every arm by construction.
    Returns the run-length-compressed sequence plus its length.
    """
    import re
    pat = re.compile(r"CGC-IDS: ctx=(\S+) pmax=(\d+) il=(\d+) ntok=(\d+) ")
    seq = []
    try:
        with open(log_path, errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if not m or int(m.group(3)) != il:
                    continue
                step = (int(m.group(2)), int(m.group(4)))
                if not seq or seq[-1] != step:
                    seq.append(step)
    except OSError:
        return {"available": False, "reason": "log not readable"}
    if not seq:
        return {"available": False, "reason": "no CGC-IDS rows (CGC_UNION_LOG off?)"}
    pre = []
    for _pmax, ntok in seq:
        if ntok == 1:
            break
        pre.append(ntok)
    runs = []
    for n in pre:
        if runs and runs[-1][0] == n:
            runs[-1][1] += 1
        else:
            runs.append([n, 1])
    return {"available": True, "prefill_ntok": pre, "prefill_steps": len(pre),
            "compressed": ",".join(f"{n}x{c}" for n, c in runs),
            "largest_step": (max(pre) if pre else None)}



def _capinv_guard(a_path, b_path, model_file=None):
    """Preconditions for comparing two dumps taken at DIFFERENT caps.

    Deliberately NOT `_oracle_guard`: that one refuses a cap mismatch, which is exactly the
    variable under test here. What must still hold is everything that would make the verdict
    describe something other than the cap -- the same engine code (stamp) and the same weights
    (model), checked on BOTH dumps, because one stale side poisons the pair as thoroughly as two.
    """
    for path in (a_path, b_path):
        bad = _stamp_guard(path)
        if bad:
            return {"ok": None, "stale": True, "which": os.path.basename(path), "error": bad["error"]}
        bad = _model_guard(path, model_file)
        if bad:
            return {"ok": None, "model_mismatch": True, "which": os.path.basename(path),
                    "error": bad["error"]}
    return None



def cap_invariance_dump(kind, gb, cap, args, occ=0):
    """Produce (or legitimately reuse) one oracle dump at exactly one cap, one pool size.

    `occ` is the occurrence of this cap within the requested pair list: non-zero only when the
    caller asks for the same cap twice (`--cap-invariance-caps 8,8`), which is the gate's own
    DETERMINISM control -- two independent runs at one cap must agree bit-for-bit, otherwise an
    M1 failure between two different caps cannot be attributed to the cap at all.

    Reuse is allowed only when the dump's OWN sidecar proves it is still this computation: same
    cap, same engine stamp, same weights. A cached dump is a claim about the past, so it is
    trusted only as far as its provenance can be checked -- otherwise the gate would report last
    night's verdict about last night's binary.
    """
    label = f"capinv_{kind}_pool{gb}gb_cap{cap}" + (f"_r{occ}" if occ else "")
    dump = os.path.join(RESULT_DIR, f"oracle_{label}.jsonl")
    launch_log = os.path.join(RESULT_DIR, f"launch_{label}.log")
    model_file = model_path(kind)
    rec = {"label": label, "model": kind, "pool_gb": gb, "cap": cap, "dump": dump}
    extra = list(args.extra_env) + ["CGC_UNION_LOG=1", f"CGC_POOL_MAX_TOKENS={cap}"]
    # The MTP flag reaches run_server.sh as CGC_SERVER_MTP, which launch() sets and then lets
    # extra_env override -- so an MTP-off run is expressed as `--extra-env CGC_SERVER_MTP=0`. It
    # belongs in the signature because the MTP context adds its own dumps and changes the base
    # context's batch shape (verify batches instead of single-token decodes).
    launch_sig = {"extra_env": sorted(extra), "mtp": MODELS[kind]["mtp"]}

    if os.path.exists(dump) and _oracle_rows(dump) > 0 and not args.force:
        stale = _stamp_guard(dump)
        bad_model = _model_guard(dump, model_file)
        same_cap = _cap_sidecar(dump) == str(cap)
        prev_sig = (_oracle_meta(dump) or {}).get("launch")
        same_launch = prev_sig == launch_sig
        if same_cap and same_launch and stale is None and bad_model is None:
            rows = _oracle_rows(dump)
            print(f"[capinv] {kind} pool{gb}GB cap={cap}: reusing existing dump ({rows} rows; "
                  f"cap/stamp/weights verified)", flush=True)
            rec.update({"up": True, "reused": True, "rows": rows})
            # The partition is a property of the LAUNCH, so it survives from the cached run's log.
            if os.path.exists(launch_log):
                rec["partition"] = step_partition(launch_log)
            return rec
        why = ([f"cached cap={_cap_sidecar(dump)}"] if not same_cap else [])
        if not same_launch:
            why.append(f"cached launch={prev_sig} vs {launch_sig}")
        if stale is not None:
            why.append(stale.get("stale_kind", "stale"))
        if bad_model is not None:
            why.append("different weights")
        print(f"[capinv] {kind} pool{gb}GB cap={cap}: re-dumping ({'; '.join(why)})", flush=True)

    if args.attach:
        rec.update({"up": False, "skipped": "attach_cannot_set_cap",
                    "error": "--attach cannot set CGC_POOL_MAX_TOKENS: the cap is decided at "
                             "launch, so a cap-invariance pair needs two launches."})
        return rec

    if os.path.exists(dump):
        os.remove(dump)
    os.path.exists(dump + ".cap") and os.remove(dump + ".cap")
    extra += [f"CGC_LOGITS_ORACLE_DUMP={dump}", "CGC_LOGITS_ORACLE_TOPN=8"]

    # PRE-LAUNCH FEASIBILITY: an infeasible cap is exactly the case this gate exists for. Without
    # it, asking for cap 5 on a pool whose cap_max is 4 would launch, abort or fall back, and the
    # resulting dump would be compared as if it described cap 5.
    feas_ok, g_feas = feasibility_gate(kind, gb, args, cap, extra, where="capinv")
    if not feas_ok:
        rec.update({"up": False,
                    "skipped": "infeasible_" + g_feas["verdict"].lower().replace("-", "_"),
                    "feasibility": g_feas,
                    "error": g_feas["why"]})
        return rec

    kill_servers()
    if running_servers():
        rec.update({"up": False, "skipped": "foreign_server",
                    "error": f"another process owns the port: {running_servers()}"})
        return rec
    free_before = mem_free_pct()
    if free_before < args.min_free_pct:
        rec.update({"up": False, "skipped": "low_memory", "free_before_pct": free_before,
                    "error": f"machine {free_before}% free < --min-free-pct {args.min_free_pct}%"})
        return rec

    pre = tuple(running_servers())
    print(f"[capinv] {kind} pool{gb}GB cap={cap}: launching {MODELS[kind]['file']} mtp="
          f"{MODELS[kind]['mtp']} free={free_before}%", flush=True)
    launch(model_file, MODELS[kind]["mtp"], launch_pool_bytes(args, gb), args.port, args.kwargs,
           extra, launch_log)
    took = wait_health(args.port, model_file, args.health_timeout, pre_pids=pre)
    if took is None:
        tail = ""
        try:
            tail = open(launch_log, "rb").read().decode("utf-8", "replace")[-800:]
        except Exception:  # noqa: BLE001
            pass
        kill_servers()
        rec.update({"up": False, "launch_tail": tail,
                    "error": "server died, timed out, or a foreign server owns the port"})
        return rec

    facts = read_launch_facts(launch_log)
    # Real traffic: the dump only contains rows for ubatches that actually ran.
    ask_probe(args.port, timeout=min(args.timeout, 240.0))
    time.sleep(1.0)
    _write_oracle_meta(dump, cap, model_file=model_file, launch_sig=launch_sig,
                       expert_cache=expert_cache_state(extra, launch_pool_bytes(args, gb)))
    rows = _oracle_rows(dump)
    nil = scan_nil([launch_log, facts.get("log_path")])
    # Record the PREFILL STEP PARTITION, not just the requested cap: the cap's real effect on the
    # numerics is via n_batch/n_ubatch (see step_partition), so the runs that differ only in
    # pool size but share a partition are the ones whose comparison is meaningful.
    part = step_partition(facts.get("log_path") or launch_log)
    rec.update({"up": True, "reused": False, "rows": rows, "health_s": round(took, 1),
                "rss_gb": rss_gb(), "nil": nil, "launch_facts": facts, "partition": part})
    print(f"[capinv] {kind} pool{gb}GB cap={cap}: healthy in {took:.0f}s  rows={rows}  "
          f"rss={rec['rss_gb']}GB  nil={'ok' if nil.get('ok') else 'FAIL'}  "
          f"partition={part.get('compressed', 'n/a')}", flush=True)
    kill_servers()
    return rec



def cap_invariance_gate(kind, gb, caps, args):
    """Same pool, every pair of caps: dump each, realign on position, compare M1/M2/M3.

    The verdict never merges M1 and M2 (see cgc_logits_oracle_compare.py): 'M2 agreed' is not 'no
    difference', and this gate exists precisely because greedy decoding is chaotic enough that a
    layout-induced drift can sit at 100% decision agreement right up to the token where it does
    not.
    """
    label = f"capinv_{kind}_pool{gb}gb"
    out_json = os.path.join(RESULT_DIR, f"{label}.json")
    rec = {"label": label, "model": kind, "pool_gb": gb, "caps": list(caps),
           "dumps": [], "pairs": []}

    seen_caps = {}
    plan = []
    for cap in caps:
        plan.append((cap, seen_caps.get(cap, 0)))
        seen_caps[cap] = seen_caps.get(cap, 0) + 1
    dumps = [cap_invariance_dump(kind, gb, cap, args, occ) for cap, occ in plan]
    rec["dumps"] = dumps
    for d in dumps:
        if not d.get("up"):
            rec.update({"ok": None, "verdict": "INCOMPLETE",
                        "error": f"cap={d['cap']} produced no dump ({d.get('error')})"})
            json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            print(f"[{label}] INCOMPLETE: cap={d['cap']} produced no dump - {d.get('error')}",
                  flush=True)
            return rec

    model_file = model_path(kind)
    for i in range(len(dumps)):
        for j in range(i + 1, len(dumps)):
            a, b = dumps[i], dumps[j]
            pair = {"cap_a": a["cap"], "cap_b": b["cap"], "a": a["dump"], "b": b["dump"]}
            guard = _capinv_guard(a["dump"], b["dump"], model_file)
            if guard:
                pair.update({"ok": None, "verdict": "REFUSED", "error": guard["error"]})
                rec["pairs"].append(pair)
                continue
            plabel = f"{label}_cap{a['cap']}v{b['cap']}"
            ra = os.path.join(RESULT_DIR, f"realign_{a['label']}.jsonl")
            rb = os.path.join(RESULT_DIR, f"realign_{b['label']}.jsonl")
            rad = os.path.join(RESULT_DIR, f"realign_{a['label']}_def.jsonl")
            rbd = os.path.join(RESULT_DIR, f"realign_{b['label']}_def.jsonl")
            pair["realign"] = {"a": _realign_decode(a["dump"], ra),
                               "b": _realign_decode(b["dump"], rb),
                               "a_def": _realign_decode(a["dump"], rad, only_ctx="DEF"),
                               "b_def": _realign_decode(b["dump"], rbd, only_ctx="DEF")}
            pair["realigned"] = {"a": ra, "b": rb, "a_def": rad, "b_def": rbd}
            # TWO comparisons, and the verdict comes from the DEF-only one:
            #   DEF = the target model's own context. This is what the cap is supposed to be
            #         inert to, so it is the primary claim.
            #   all = every context together, draft head (MTP) included. Reported alongside because
            #         the draft head has its own layout sensitivity, but it must not be allowed to
            #         speak for the model -- a cap change that only moves the draft head is a
            #         different finding from one that moves the target.
            d_def = _run_oracle_compare(rad, rbd, f"{plabel}_def")
            d_all = _run_oracle_compare(ra, rb, f"{plabel}_all")
            if d_def is None:
                pair.update({"ok": None, "verdict": "ERROR", "error": "compare produced no report"})
                rec["pairs"].append(pair)
                continue
            m1 = d_def["metrics"]["numeric_identity"]
            m2 = d_def["metrics"]["decision_agreement"]
            m3 = d_def["metrics"]["topk_set_agreement"]
            n_common, n_a, n_b = d_def["n_common"], d_def["n_a"], d_def["n_b"]
            denom = min(n_a, n_b) if min(n_a, n_b) else 0
            coverage = (n_common / denom) if denom else 0.0
            m1_full, m2_full = m1["rate"] == 1.0, m2["rate"] == 1.0
            # Verdict, in the order the evidence deserves to be weighed:
            #
            #  * DIVERGENCE stands on its own. A non-full M1 is positive evidence that the cap
            #    moved the numerics, and it is not weakened by imperfect alignment -- imperfect
            #    alignment is in fact its CONSEQUENCE: once a token differs, the two generations
            #    diverge and their later positions stop overlapping. (An earlier version required
            #    >=95% coverage for any verdict, which made the gate silent in exactly the case it
            #    exists for: measured 94% coverage on a pair whose M1 was 2/29.)
            #  * INERTNESS is the claim that needs coverage and sample size: 'the cap changes
            #    nothing' is only supported if the full aligned sequence was actually compared.
            #    So the thresholds gate the PASS direction only, where the degenerate failure mode
            #    is a vacuous pass on almost nothing.
            if not m1_full:
                verdict, ok = ("FAIL", False) if not m2_full else ("DRIFT", False)
            elif n_common < args.capinv_min_steps or coverage < args.capinv_min_coverage:
                verdict, ok = "INCONCLUSIVE", None
            else:
                verdict, ok = "PASS", True
            pair["first_divergence"] = _first_divergence(rad, rbd)
            # WHICH KNOB MOVED? A FAIL here is only informative once this is answered, because a
            # cap change does two things at once: it re-labels pool slots AND (via the n_batch
            # clamp) it changes the prefill step partition. Measured 2026-09-14: at a FIXED cap the
            # result is bit-identical across pools 2/4/6/8/10 GiB; across caps the partition
            # changes and so does the output. So when the partitions differ, this pair is
            # comparing two different forward passes rather than two pool layouts.
            part_a, part_b = a.get("partition") or {}, b.get("partition") or {}
            both_part = bool(part_a.get("available")) and bool(part_b.get("available"))
            same_part = both_part and part_a.get("compressed") == part_b.get("compressed")
            if not both_part:
                attribution = "unknown: prefill step partition not recorded for both arms"
            elif same_part:
                attribution = ("pool layout / slab geometry: prefill step partition is IDENTICAL "
                               f"({part_a.get('compressed')}), so the divergence is NOT the partition")
            else:
                attribution = ("prefill step partition: "
                               f"cap{pair['cap_a']}={part_a.get('compressed')} vs "
                               f"cap{pair['cap_b']}={part_b.get('compressed')} -- the two caps "
                               "chunk the prompt differently, so this compares two forward passes")
            pair["partition"] = {"a": part_a.get("compressed"), "b": part_b.get("compressed"),
                                 "a_steps": part_a.get("prefill_steps"),
                                 "b_steps": part_b.get("prefill_steps"), "same": same_part}
            pair["attribution"] = attribution
            pair.update({"ok": ok, "verdict": verdict, "ctx": "DEF", "n_common": n_common,
                         "coverage": round(coverage, 4),
                         "repeats": {"a": pair["realign"]["a_def"].get("repeat_rows"),
                                     "b": pair["realign"]["b_def"].get("repeat_rows")},
                         "pos_range": {"a": [pair["realign"]["a_def"].get("first_pos"),
                                             pair["realign"]["a_def"].get("last_pos")],
                                       "b": [pair["realign"]["b_def"].get("first_pos"),
                                             pair["realign"]["b_def"].get("last_pos")]},
                         "rows": {"a": pair["realign"]["a_def"].get("kept"),
                                  "b": pair["realign"]["b_def"].get("kept")},
                         "m1_numeric_identity": f"{m1['equal']}/{m1['n']}",
                         "m2_decision_agreement": f"{m2['equal']}/{m2['n']}",
                         "m3_topk_set_agreement": f"{m3['equal']}/{m3['n']}",
                         "cross_tab": d_def["cross_tab"],
                         "report": os.path.join(RESULT_DIR, f"oraclecmp_{plabel}_def.json"),
                         "diff_examples": d_def.get("diff_examples", [])[:3]})
            if d_all is not None:
                am1 = d_all["metrics"]["numeric_identity"]
                am2 = d_all["metrics"]["decision_agreement"]
                am3 = d_all["metrics"]["topk_set_agreement"]
                pair["all_ctx"] = {"n_common": d_all["n_common"],
                                  "m1_numeric_identity": f"{am1['equal']}/{am1['n']}",
                                  "m2_decision_agreement": f"{am2['equal']}/{am2['n']}",
                                  "m3_topk_set_agreement": f"{am3['equal']}/{am3['n']}",
                                  "m1_full": am1["rate"] == 1.0, "m2_full": am2["rate"] == 1.0}
            rec["pairs"].append(pair)

    verdicts = [p.get("verdict") for p in rec["pairs"]]
    rec["ok"] = all(p.get("ok") is True for p in rec["pairs"]) if rec["pairs"] else None
    rec["verdict"] = ("PASS" if rec["ok"] else
                      ("INCONCLUSIVE" if all(v in ("INCONCLUSIVE", "REFUSED", "ERROR")
                                             for v in verdicts) else
                       ("DRIFT" if "DRIFT" in verdicts else "FAIL")))
    json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    for p in rec["pairs"]:
        if p.get("verdict") in ("PASS", "DRIFT", "FAIL"):
            ac = p.get("all_ctx") or {}
            fd = p.get("first_divergence") or {}
            where = (f"  first divergence at pos={fd['pos']} occ={fd['occurrence']} "
                     f"({fd['ctx']}) argmax {fd['argmax_a']} -> {fd['argmax_b']}" if fd else "")
            print(f"[{label}] cap{p['cap_a']} vs cap{p['cap_b']}: {p['verdict']}  "
                  f"DEF M1(bit-identical)={p['m1_numeric_identity']} "
                  f"M2(argmax)={p['m2_decision_agreement']} "
                  f"M3(topk)={p['m3_topk_set_agreement']} "
                  f"n={p['n_common']} coverage={p['coverage']:.1%}  | "
                  f"ALL(ctx) M1={ac.get('m1_numeric_identity', '-')} "
                  f"M2={ac.get('m2_decision_agreement', '-')} n={ac.get('n_common', '-')}"
                  f"{where}", flush=True)
            if p.get("verdict") in ("FAIL", "DRIFT"):
                print(f"[{label}]   attribution: {p['attribution']}", flush=True)
        elif p.get("verdict") == "INCONCLUSIVE" and p.get("m1_numeric_identity"):
            # Report the numbers even when the pair cannot be judged: "too few steps" is a
            # statement about the ALIGNMENT, not about the logits, and hiding the observed M1/M2
            # would make an unmatched pair indistinguishable from a clean one.
            print(f"[{label}] cap{p['cap_a']} vs cap{p['cap_b']}: INCONCLUSIVE "
                  f"(aligned DEF steps n={p['n_common']} of {p.get('rows', {}).get('a')}/"
                  f"{p.get('rows', {}).get('b')} realigned rows; coverage "
                  f"{p['coverage']:.0%} < {args.capinv_min_coverage:.0%} or n < "
                  f"{args.capinv_min_steps}) -- INDICATIVE ONLY, not a verdict: "
                  f"M1={p['m1_numeric_identity']} M2={p['m2_decision_agreement']}. The two dumps "
                  f"did not run the same requests in the same order, so their positions overlap "
                  f"only in part; check pos_range/repeats in the record.",
                  flush=True)
        else:
            print(f"[{label}] cap{p['cap_a']} vs cap{p['cap_b']}: {p.get('verdict')} - "
                  f"{p.get('error', '')}", flush=True)
    print(f"[{label}] VERDICT {rec['verdict']}  (wrote {out_json})", flush=True)
    return rec



def _store_capinv(records):
    """Merge this run's cap-invariance records into the on-disk store, keyed by (label, caps).

    Accumulating rather than overwriting is not a convenience: the natural workflow is to re-run
    `--cap-invariance-caps 8,8` as a determinism control right after a 6-vs-8 run, and with a
    single-record file that control silently deletes the FAIL it was supposed to qualify.
    Re-running the SAME (label, caps) still replaces that entry, so a run stays re-runnable.
    """
    path = os.path.join(RESULT_DIR, "cap_invariance.json")
    store = []
    if os.path.exists(path):
        try:
            prev = json.load(open(path, encoding="utf-8"))
            store = prev if isinstance(prev, list) else (prev.get("runs") or [])
        except Exception:  # noqa: BLE001 - harness
            store = []
    keep = []
    for r in store:
        replaced = any(r.get("label") == n.get("label")
                       and tuple(r.get("caps") or ()) == tuple(n.get("caps") or ())
                       for n in records)
        if not replaced:
            keep.append(r)
    out = keep + list(records)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return out



def print_capinv_table(records):
    """Consolidated cap-invariance verdicts, one row per (model x pool x cap pair)."""
    print()
    print("=" * 118)
    print("CAP INVARIANCE -- same pool, different CGC_POOL_MAX_TOKENS: a cap change must be "
          "numerically inert")
    print("-" * 118)
    print("%-26s %5s %5s %5s %8s %11s %11s %11s %9s %11s" % (
        "config", "pool", "capA", "capB", "verdict", "M1 DEF", "M2 DEF", "M3 DEF",
        "cover", "M1 all-ctx"))
    print("-" * 118)
    for r in records:
        if not r.get("pairs"):
            print("%-26s   %s" % (r.get("label"), r.get("error") or "no pair compared"))
            continue
        for p in r["pairs"]:
            ac = p.get("all_ctx") or {}
            print("%-26s %4sG %5s %5s %8s %11s %11s %11s %9s %11s" % (
                r.get("label"), r.get("pool_gb"), p.get("cap_a"), p.get("cap_b"),
                p.get("verdict"), p.get("m1_numeric_identity", "-"),
                p.get("m2_decision_agreement", "-"), p.get("m3_topk_set_agreement", "-"),
                (f"{p['coverage']:.0%}" if isinstance(p.get("coverage"), float) else "-"),
                ac.get("m1_numeric_identity", "-")))
    print("-" * 118)
    print("DEF = the target model's own context (the primary claim); all-ctx includes the MTP")
    print("      draft head, whose layout sensitivity is a separate finding.")
    print("PASS         = M1 full: the cap changed the pool LAYOUT and the logits did not move.")
    print("DRIFT        = M1 not full but M2 full: same decisions, different bits. NOT a pass --")
    print("               greedy decoding is chaotic, so this can flip at any later token.")
    print("FAIL         = M1 and M2 both differ.")
    print("INCONCLUSIVE = too few position-aligned decode steps to judge (see 'cover').")
    print("cover        = aligned positions / smaller dump. Alignment is by sequence position,")
    print("               because the dump's own step counter counts UBATCHES and the cap decides")
    print("               the chunking -- keying on it across two caps compares unrelated rows.")
    print("=" * 118)
