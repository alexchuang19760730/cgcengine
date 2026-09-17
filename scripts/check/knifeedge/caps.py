#!/usr/bin/env python3
"""knifeedge.caps — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from .anchor import RESULT_DIR
from .constants import GATHER_RE, KEY_FIELDS, MIN_CAP, MODELS, NIL_RE, PROBE_PROMPT
from .feasibility import _env_int, effective_launch_env, feasibility_cell
from .host import ask_probe, kill_servers, launch, launch_pool_bytes, read_launch_facts, running_servers, wait_health
from .identity import model_path, model_stamp, pool_geometry_stamp
from .provenance import _short_val, key_fingerprint, record_geometry_key



def preflight(port, timeout=150.0):
    payload = {"model": "local",
               "messages": [{"role": "user", "content": PROBE_PROMPT}],
               "temperature": 0.0, "max_tokens": 64}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001 - harness
        return {"ok": False, "error": str(e)[:200], "content": ""}
    ch = (body.get("choices") or [{}])[0]
    content = (ch.get("message", {}).get("content") or "")
    finish = ch.get("finish_reason", "?")
    norm = re.sub(r"\s+", "", content)
    echoes = norm.count("15+27") > 1 or re.sub(r"\s+", "", PROBE_PROMPT) in norm
    leak = ("<think" in content) or ("think>" in content) or ("</think" in content)
    empty = content.strip() == ""
    answers_42 = "42" in content
    return {"ok": bool(answers_42 and not leak), "content": content[:240], "finish": finish,
            "answers_42": answers_42, "echoes_prompt": echoes, "scaffold_leak": leak,
            "empty": empty}



# ---------------------------------------------------------------------------
# GATES. A combo only counts as "same mechanism, just slower" if it passes ALL of these.
# Each gate exists because the corresponding failure was silent before.
# ---------------------------------------------------------------------------


def apply_template_probe(port, prompt=None, timeout=30.0):
    """Render the generation prompt the server would really use and check the scaffold is CLOSED.

    Checking the *answer* is not enough: a degenerating model can still stumble onto "42".
    This asks the server to render the template itself, so an unclosed `<think>` (the
    enable_thinking gate never satisfied) is caught even when the output happens to look fine.
    """
    body = {"messages": [{"role": "user", "content": prompt or PROBE_PROMPT}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/apply-template",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return {"available": False, "ok": None, "error": f"http {e.code}"}
    except Exception as e:  # noqa: BLE001 - harness
        return {"available": False, "ok": None, "error": str(e)[:120]}
    text = d.get("prompt", "") or ""
    last_open, last_close = text.rfind("<think"), text.rfind("</think>")
    unclosed = last_open != -1 and last_open > last_close
    return {"available": True, "ok": not unclosed, "unclosed_think": unclosed,
            "tail": text[-140:]}



def scan_nil(paths):
    """Grep for `buffer is nil` -- the signature of the L3-B gather path repointing a
    Metal-resident FFN tensor at a HOST std::vector. Any hit means the combo's numbers are
    invalid AND that the pool path silently switched. Must never be ignored."""
    hits = []
    for p in paths:
        if not p:
            continue
        try:
            txt = open(p, "rb").read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - harness
            continue
        n = len(NIL_RE.findall(txt))
        if n:
            hits.append({"file": os.path.basename(p), "count": n,
                         "n_blocks": len(set(re.findall(r"blk\.\d+", txt)))})
    return {"ok": not hits, "hits": hits}



def scan_gather_evidence(paths):
    """Did the WIDE-union route actually run, and how many steps went through it?

    Why this exists: `union > slots` used to be a hard FAIL because the only route out was the
    L3-B gather path, which repointed a Metal-resident FFN tensor at HOST memory and produced
    `buffer is nil` (measured 2 GiB, 2026-09-13: 468 nil, M1 2/42, first divergence at step 2,
    that step's sum off by 55%). That is no longer the case -- the gather path now fills a Metal
    slab (docs/M1_POOL_GRAPH_DECOUPLE_PLAN_2026-09-14.md 3.1) -- so the criterion becomes
    "routable": EITHER the pool is arithmetically guaranteed to hold the union, OR the wide-union
    route ran and produced no nil. Arithmetic alone would mark a correct 2 GiB cell as broken;
    nil alone would pass a cell that never exercised the path, which is the failure mode that
    hid this for two days.
    """
    wide_layers, steps, total = 0, 0, 0
    for p in paths:
        if not p:
            continue
        try:
            txt = open(p, "rb").read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - harness
            continue
        for _layer, hit, tot in GATHER_RE.findall(txt):
            total += int(tot)
            steps += int(hit)
            if int(hit) > 0:
                wide_layers += 1
    return {"wide_layers": wide_layers, "gather_steps": steps, "total_steps": total,
            "exercised": wide_layers > 0}



def union_fit_gate(facts, cap, topk, paths=None):
    """Is the pool EITHER guaranteed to hold the worst-case union, OR proven routable?

    The decode hook takes the pool path only when `union <= slots_per_layer`; otherwise it falls
    through to the wide-union route. A ubatch of `cap` tokens selects at most `cap * topk`
    distinct experts, so `cap * topk <= n_slots - 1` (slot 0 is the reserved ZERO slot) is the
    exact condition under which that fallback is unreachable -- for ANY model and ANY pool size.

    Reported as ok=None when the launch log did not expose n_slots (e.g. --attach to a server
    someone else started): unknown is not the same as passing.
    """
    # The binding number is the MINIMUM per-layer cap: the hook compares the union against the
    # cap of the layer it is currently in, so n_slots (or the average) would overstate the room
    # and could approve a cap that still overflows some layer. Fall back to n_slots only when
    # the log predates the min= field, and say which was used.
    src = "min_layer_slots" if facts.get("min_layer_slots") else "n_slots"
    try:
        n_slots = int(facts.get(src))
    except (TypeError, ValueError):
        return {"ok": None, "cap": cap, "topk": topk, "n_slots": None, "source": None,
                "union_max": cap * topk,
                "error": "neither min per-layer slots nor n_slots in the launch log"}
    union_max = cap * topk
    usable = n_slots - 1
    fit = bool(union_max <= usable)
    ev = scan_gather_evidence(paths or [])
    # mode is reported so a PASS cannot be mistaken for the other reason: `pool` means the
    # fallback was unreachable by arithmetic, `gather` means it was reached and stayed clean.
    if fit:
        mode = "pool"
    elif ev["exercised"]:
        mode = "gather"
    else:
        # Not routable by arithmetic AND no evidence the wide route ran. Two causes, and they
        # need different fixes: either the cell genuinely never reaches the wide route (then the
        # arithmetic is what matters and this is a real FAIL), or the evidence was never
        # recorded (CGC_UNION_LOG off -> `layers=0 steps=0/0`). The harness now sets that flag
        # unconditionally, so seeing this mode means the fallback really was not exercised.
        mode = "unreachable"
    return {"ok": bool(fit or ev["exercised"]), "mode": mode, "fit": fit,
            "cap": cap, "topk": topk, "n_slots": n_slots,
            "source": src, "usable_slots": usable, "union_max": union_max,
            "headroom": usable - union_max, "gather": ev}



def comparability_verdict(recs):
    """Group rows per model by geometry and say whether they may be compared at all.

    states:
      OK            one geometry group -> pool size is the only variable
      UNATTRIBUTED  one group, but it carries no provenance (pre-2026-09-14 records) -> nothing can
                    be certified; re-measure with the current harness to make it comparable
      REFUSED       two or more groups -> the rows are NOT comparable, and the caller must say so
                    instead of printing them as one table
    """
    by_model = {}
    for r in recs:
        by_model.setdefault(r.get("model") or "?", []).append(r)
    out = {}
    for model, rows in by_model.items():
        groups = {}
        for r in rows:
            key = record_geometry_key(r)
            fp = key_fingerprint(key)
            g = groups.setdefault(fp, {"key": key, "fingerprint": fp, "rows": []})
            g["rows"].append(r)
        if len(groups) == 1:
            only = next(iter(groups.values()))
            state = "UNATTRIBUTED" if only["key"] is None else "OK"
        else:
            state = "REFUSED"
        # Name every field whose value differs between groups, with the rows on each side.
        # A group with NO provenance differs from everything by construction, so it is reported as
        # that fact instead of as a list of nine fields whose real cause is one: nobody recorded
        # them. (Otherwise the refusal drowns in its own noise, starting with the binary tuple.)
        differences = []
        if state == "REFUSED":
            glist = list(groups.values())
            stamped = [g for g in glist if g["key"] is not None]
            unstamped = [g["fingerprint"] for g in glist if g["key"] is None]
            if unstamped:
                differences.append(
                    f"group(s) {unstamped} carry NO provenance, so none of their fields can be "
                    f"compared with anything (records written before the stamp existed)"
                    + (f" -- re-measure them if they are meant to sit beside the "
                       f"{len(stamped)} stamped group(s)" if stamped else ""))
            for field, human in KEY_FIELDS:
                seen = {g["fingerprint"]: (g["key"] or {}).get(field) for g in stamped}
                if len(seen) > 1 and len({repr(v) for v in seen.values()}) > 1:
                    parts = ", ".join(f"{fp}: {_short_val(val)}" for fp, val in seen.items())
                    differences.append(f"{human} differs -> {parts}")
        out[model] = {"state": state, "groups": list(groups.values()),
                      "differences": differences}
    return out



def _cap_probe_provenance(kind, args):
    """Everything that must still hold for a cached cap-probe record to describe THIS tree."""
    env = effective_launch_env(kind, args.extra_env)
    return {"pool_geometry": pool_geometry_stamp(),
            "model": model_stamp(model_path(kind)),
            "layer_caps": env.get("CGC_SERVER_LAYER_CAPS") or "default",
            "mtp": str(env.get("CGC_SERVER_MTP", "1")).lower() in ("1", "true", "on", "yes"),
            "spec_n_max": _env_int(env, "CGC_SERVER_MTP_N_MAX", 3),
            "extra_env": sorted(args.extra_env or [])}



def _recorded_slots(rec):
    """The MEASURED per-layer slot count in a cap-probe record, or None."""
    uf = rec.get("union_fit") or {}
    if uf.get("n_slots"):
        return uf["n_slots"]
    for t in reversed(rec.get("tried") or []):
        tuf = t.get("union_fit") or {}
        if tuf.get("n_slots"):
            return tuf["n_slots"]
    return None



def reconcile_cap_probe(kind, gb, args, rec, source="cache"):
    """Cross-check a MEASURED cap-probe slot count against the arithmetic prediction.

    Why this is necessary and not paranoia: `cap_probe_iq3_pool8gb.json` (recorded 2026-09-11) says
    **106** slots while the prediction -- and every 2026-09-14 launch log -- says **143**. It was
    being reused verbatim, so the whole sweep's cap was derived from a geometry this engine no
    longer has, and the number looked perfectly plausible in the record. Worse, the same stale
    number is what `--cap-invariance` would then compare across pools.

    The rule is: the MEASUREMENT wins (it is the engine); the mismatch is what has to be explained,
    and the two explanations need opposite actions:

      MATCH          -> reuse is sound
      STALE_RECORD   -> the record's code / weights / launch env moved since it was written
                        (or it carries no provenance at all). Discard and re-probe.
      PREDICTOR_STALE-> the record (or the launch that just happened) matches this tree exactly, so
                        the ARITHMETIC in pool_feasibility.py is what no longer describes the
                        loader. Every feasibility verdict rests on it -> stop.
      NO_EVIDENCE    -> the record holds no measured slot count at all.
    """
    out = {"source": source, "ok": None, "verdict": "UNKNOWN"}
    try:
        cell = feasibility_cell(kind, gb, rec.get("cap"), list(args.extra_env or []))
    except Exception as e:  # noqa: BLE001 - a cross-check must never be why a run dies
        out.update({"error": f"no prediction available ({e})"})
        return out
    measured = _recorded_slots(rec)
    out.update({"predicted_min_layer_slots": cell["min_layer_slots"],
                "predicted_usable_slots": cell["usable_slots"],
                "predicted_cap_max": cell["cap_max"],
                "measured_min_layer_slots": measured,
                "record": os.path.join(RESULT_DIR, f"cap_probe_{kind}_pool{gb}gb.json")})
    if measured is None:
        out.update({"ok": False, "verdict": "NO_EVIDENCE",
                    "cause": "the record contains no measured slot count (its probe never reached a "
                             "loaded server)"})
        out["error"] = (f"CAP-PROBE UNVERIFIABLE: {out['record']} has no measured slot count, so "
                        f"its cap={rec.get('cap')} cannot be checked against the arithmetic "
                        f"(predicted {cell['min_layer_slots']} slots). Re-probing.")
        return out
    if int(measured) == int(cell["min_layer_slots"]):
        out.update({"ok": True, "verdict": "MATCH"})
        return out

    prov = rec.get("provenance")
    cur = _cap_probe_provenance(kind, args)
    if source == "probe" or not prov:
        verdict = "PREDICTOR_STALE" if source == "probe" else "STALE_RECORD"
        cause = ("the launch that produced this number just happened from this tree, so "
                 "pool_feasibility.py no longer describes the loader"
                 if source == "probe" else
                 "the record carries NO provenance, so nothing ties it to this tree "
                 "(records written before 2026-09-14 have none)")
    else:
        diffs = []
        pg = prov.get("pool_geometry") or {}
        if pg.get("source_digest") != cur["pool_geometry"]["source_digest"]:
            diffs.append(f"pool-geometry code changed (record digest {pg.get('source_digest') or '?'}"
                         f", now {cur['pool_geometry']['source_digest']}; newest commit over those "
                         f"paths then: {pg.get('last_commit') or '?'})")
        pm, cm = prov.get("model") or {}, cur["model"] or {}
        if (pm.get("size"), pm.get("digest")) != (cm.get("size"), cm.get("digest")):
            diffs.append(f"weights differ (record {pm.get('realpath') or '?'} "
                         f"size={pm.get('size')}, now {cm.get('realpath') or '?'} "
                         f"size={cm.get('size')})")
        for k in ("layer_caps", "mtp", "spec_n_max", "extra_env"):
            if prov.get(k) != cur.get(k):
                diffs.append(f"{k} changed (record {prov.get(k)!r} vs now {cur.get(k)!r})")
        if diffs:
            verdict, cause = "STALE_RECORD", "; ".join(diffs)
        else:
            verdict = "PREDICTOR_STALE"
            cause = ("the record's provenance matches this tree exactly, so the engine code did "
                     "not move -- the arithmetic in pool_feasibility.py is what is wrong")
    out.update({"ok": False, "verdict": verdict, "cause": cause,
                "provenance_recorded": prov is not None})
    out["error"] = (f"CAP-PROBE MISMATCH ({verdict}): measured {measured} slots vs predicted "
                    f"{cell['min_layer_slots']} for {kind} pool{gb}GB. Cause: {cause}. "
                    f"Both numbers and the record path are in the artifact; the MEASURED one wins "
                    f"(it is the engine).")
    return out



def cap_probe_reusable(check):
    """May a cached cap-probe record be reused? Only when it reconciles (see reconcile_cap_probe)."""
    return check.get("ok") is True



def geometry_conflict_error(rec, allow=False):
    """The refusal message for a FRESH probe that contradicts the arithmetic, or None.

    Kept separate from main() so the fatal path is testable without launching anything: this is
    the branch that decides "do not measure at all", and it has to be pinned like any other
    decision (the whole reason this gate exists is that a wrong number otherwise looks fine).
    """
    gc = rec.get("geometry_conflict")
    if not gc or allow:
        return None
    return (f"REFUSING to start the sweep: the cap probe measured "
            f"{gc.get('measured_min_layer_slots')} slots where pool_feasibility.py predicts "
            f"{gc.get('predicted_min_layer_slots')}.\n"
            f"  {gc.get('cause')}\n"
            f"  Record: {gc.get('record')}\n"
            f"  Fix the arithmetic (or the loader) and re-run; --allow-geometry-drift records "
            f"the disagreement and measures anyway, with the conflicting geometry stamped "
            f"into every result.")



def probe_pool_cap(kind, gb, args):
    """Pick ONE `CGC_POOL_MAX_TOKENS` that keeps the SMALLEST pool on the pool path.

    Generic across models: it only uses the observed `buffer is nil` signal, so a new GGUF with
    a different expert count or expert width needs no new hand-tuned constant.

    A single cap for every pool is mandatory, not a convenience: the cap clamps n_batch and
    therefore decides the ubatch shape and the prefill chunking. Letting each pool pick its own
    cap would compare different computations, not different pool sizes. Measured here: 8GB with
    pmax=8 vs 8GB with pmax=6 agreed on argmax only 15.4% of steps -- the cap IS a quality knob.
    """
    cache_path = os.path.join(RESULT_DIR, f"cap_probe_{kind}_pool{gb}gb.json")
    if os.path.exists(cache_path) and not args.force:
        rec = json.load(open(cache_path, encoding="utf-8"))
        check = reconcile_cap_probe(kind, gb, args, rec, source="cache")
        if check.get("ok") is None:
            # No prediction available (GGUF unreadable / module missing): the cache can still be
            # used, but say that it was NOT verified rather than implying it was.
            print(f"[cap-probe] {kind} pool{gb}GB cached -> cap={rec.get('cap')} "
                  f"(nil={rec.get('nil')}); NOT cross-checked ({check.get('error')})", flush=True)
            return rec
        if cap_probe_reusable(check):
            strong = bool(rec.get("provenance"))
            print(f"[cap-probe] {kind} pool{gb}GB cached -> cap={rec.get('cap')} "
                  f"(nil={rec.get('nil')}); geometry verified: "
                  f"{check['measured_min_layer_slots']} slots = predicted "
                  f"{check['predicted_min_layer_slots']}"
                  + ("" if strong else "  [legacy record: no provenance, so the AGREEMENT is what "
                                       "it stands on]"), flush=True)
            rec["verified_slots"] = {
                "measured": check["measured_min_layer_slots"],
                "predicted": check["predicted_min_layer_slots"],
                "basis": ("recorded provenance matches this tree" if strong else
                          "legacy record: no provenance, agrees with the arithmetic"),
                "pool_geometry_digest_now": pool_geometry_stamp()["source_digest"]}
            return rec
        # NOT reusable. Do not raise here: a stale cache is repaired by re-probing, which is
        # ground truth. What must never happen again is using it SILENTLY.
        print(f"[cap-probe] {kind} pool{gb}GB cache REFUSED\n  {check.get('error')}\n"
              f"  Re-probing instead of reusing it: the cap must come from a geometry this "
              f"engine actually has. (--force always re-probes; this refusal is the same action, "
              f"taken because the cached number is provably wrong or unattributable.)",
              file=sys.stderr, flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    topk = MODELS[kind].get("topk", 8)
    rec = {"kind": kind, "pool_gb": gb, "topk": topk, "cap": None, "tried": [],
           "min_cap": MIN_CAP}
    cap = min(8, args.pool_cap_max)
    for _ in range(3):
        log_path = os.path.join(RESULT_DIR, f"capprobe_{kind}_pool{gb}gb_cap{cap}.log")
        print(f"[cap-probe] {kind} pool{gb}GB trying cap={cap} (union <= {cap * topk})",
              flush=True)
        pre = tuple(running_servers())
        kill_servers()
        if running_servers():
            rec["tried"].append({"cap": cap, "error": "foreign server would not clear"})
            break
        launch(model_path(kind), MODELS[kind]["mtp"], launch_pool_bytes(args, gb), args.port,
               args.kwargs, list(args.extra_env) + [f"CGC_POOL_MAX_TOKENS={cap}"], log_path)
        took = wait_health(args.port, model_path(kind), args.health_timeout,
                           pre_pids=pre)
        if took is None:
            rec["tried"].append({"cap": cap, "error": "server down, crashed, or a foreign "
                                                     "server owns the port"})
            kill_servers()
            break
        # Send real traffic BEFORE scanning: `buffer is nil` only appears once a decode actually
        # builds an expert union. Without this the too-large cap is rejected on arithmetic alone
        # and the log looks clean -- observation is the stronger evidence, arithmetic is what
        # makes the result generalisable to a prompt that happens not to overflow.
        ask_probe(args.port, timeout=min(args.health_timeout, 240.0))
        facts = read_launch_facts(log_path)
        nil = scan_nil([log_path, facts.get("log_path")])
        fit = union_fit_gate(facts, cap, topk)
        rec["tried"].append({"cap": cap, "nil": nil["hits"], "union_fit": fit,
                             "health_s": round(took, 1)})
        kill_servers()
        # A cap only counts as safe when the LAUNCH LOG proves the invariant. `ok is not False`
        # was too weak: a server that never loaded reports no nil AND no n_slots, which read as
        # a pass. Require the arithmetic proof (`ok is True`), not the absence of evidence.
        if nil["ok"] and fit.get("ok") is True:
            rec["cap"] = cap
            rec["nil"] = nil["hits"]
            rec["union_fit"] = fit
            break
        if fit.get("ok") is False:
            # The MEASURED pool cannot hold cap*topk experts, so the gather path is reachable.
            # Jump straight to the largest cap that provably fits (not merely cap-1).
            nxt = min(cap - 1, fit["usable_slots"] // topk)
            if nxt < MIN_CAP:
                rec["error"] = (f"pool {gb}GB holds {fit['usable_slots']} usable slots: cap "
                                f"{nxt}*{topk} would fit but cap < {MIN_CAP} aborts at "
                                f"llama-batch.cpp:609 (GGML_ASSERT(n_ubatch > n_keep_tail)) "
                                f"during warmup. This pool is too small for topk={topk}.")
                break
            cap = nxt
            continue
        cap -= 1
        if cap < MIN_CAP:
            rec["error"] = f"no tried cap >= {MIN_CAP} produced a clean pool path"
            break
    # Stamp the record and reconcile it with the arithmetic, in the same write. Both halves are
    # needed: without the stamp the next run cannot tell whether a mismatch means "this file is
    # old" or "the loader changed"; without the reconciliation a wrong number is only visible if
    # someone happens to compare it by hand -- which is exactly how 106 vs 143 survived.
    rec["provenance"] = _cap_probe_provenance(kind, args)
    check = reconcile_cap_probe(kind, gb, args, rec, source="probe")
    rec["reconciled"] = check
    if check.get("ok") is False and cap_probe_reusable(check) is False:
        # The MEASURED geometry and the arithmetic model disagree, and this launch came from the
        # current tree -- so the arithmetic every feasibility verdict rests on is wrong. Record it
        # as a conflict and let main() refuse the sweep: launching cells under a geometry model we
        # have just falsified is how "the pool broke numerics" verdicts get manufactured.
        # The measured cap itself is kept (it IS a measurement); what is refused is acting on it.
        rec["geometry_conflict"] = check
    json.dump(rec, open(cache_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if check.get("ok") is True:
        print(f"[cap-probe] geometry reconciled: measured "
              f"{check['measured_min_layer_slots']} slots = predicted "
              f"{check['predicted_min_layer_slots']}")
    elif check.get("ok") is False:
        print(f"[cap-probe] GEOMETRY CONFLICT\n  {check.get('error')}", file=sys.stderr, flush=True)
    if rec["cap"] is None and not rec.get("geometry_conflict"):
        print(f"[cap-probe] WARNING: no cap kept pool{gb}GB on the pool path; "
              f"the matrix will still run but the nil gate will fail.", file=sys.stderr,
              flush=True)
    else:
        print(f"[cap-probe] {kind} -> cap={rec['cap']} for ALL pools "
              f"(union {rec['cap'] * topk} <= usable {rec.get('union_fit', {}).get('usable_slots')})",
              flush=True)
    return rec
