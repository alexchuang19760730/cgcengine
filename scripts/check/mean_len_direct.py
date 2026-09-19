#!/usr/bin/env python3
"""Per-request direct count: how many tokens does one decode step actually deliver?

Why this exists
---------------
`mean_len` (tokens per step) is the denominator of every throughput claim in
`docs/PREFILL250_DECODE25_STRATEGY_*.html`, and it has been wrong twice on this line:

  * ``2.67`` — derived here as ``draft_slots - accepted`` rounds, which is not a step count.
  * ``1.75`` — an MTP-on token count divided by an **MTP-off** step time. Mixing the
    numerator from one arm with the denominator from another is a category error, not a
    measurement conflict.

Both were the same failure: a quotient assembled from two different regimes. So this tool
assembles nothing. It reads what the engine prints per request and cross-checks the step
count along independent routes:

  A. ``steps = emitted_tokens / mean_len``           (the engine's own mean_len print)
  B. ``steps = generated_drafts / k``                (an independent counter, only if
                                                      speculation ran and the width held)
  C. ``steps ~= graphs_reused``                      (loose, ±1–2, reported not enforced)

The engine's definition (src/llama.cpp/tools/server/server-context.cpp:674) is

    mean_acc_len = 1.0 + n_draft_accepted / n_draft_verif_steps

which inverts to ``steps = accepted / (mean_len - 1)``. That is **algebra, not evidence** —
it is the same print re-read, so it is reported as such and never counted as a check. The
same goes for ``mean_len / step_ms``: it is identically equal to the eval line's own
``tokens per second`` (``emitted/eval_ms``), so reproducing it proves nothing.

Outputs per request: emitted tokens, steps, mean_len, and the **same-arm** step cost
(``mean_len / step_ms``) — the only pairing that means anything, because a verify step at
width k is not the same object as an unspeculated step.

Usage
-----
    python3 scripts/check/mean_len_direct.py selftest
    python3 scripts/check/mean_len_direct.py scan Backup/cgc_logs/llama_server_2026*.log --k 3
    python3 scripts/check/mean_len_direct.py scan <log> --json /tmp/out.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path


def _load_ruler():
    """The ruler owns the definition AND the parse, so there is only ever one of each.

    A second copy of these regexes is how the next wrong number gets in: it can drift from the
    engine's format without anything failing loudly.
    """
    spec = importlib.util.spec_from_file_location(
        "mtp_ruler", Path(__file__).resolve().parent / "mtp_ruler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RULER = _load_ruler()

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

# Canonical patterns, imported from mtp_ruler.py (strict about the unit strings "ms"/"tokens"
# so a format change shows up as a coverage miss, not a silent wrong number).
RE_TASK = _RULER.RE_TASK
RE_PROMPT = _RULER.RE_PROMPT
RE_EVAL = _RULER.RE_EVAL
RE_TOTAL = _RULER.RE_TOTAL
RE_GRAPHS = _RULER.RE_GRAPHS
RE_ACCEPT = _RULER.RE_ACCEPT
RE_RELEASE = _RULER.RE_RELEASE
RE_DECODED = _RULER.RE_DECODED


def parse_log(text: str, k: int) -> tuple[list[dict], dict]:
    """Return (requests, coverage).

    A request is assembled from the `slot print_timing` block of one task id. Only requests
    carrying BOTH an eval line and an acceptance line are usable: without the acceptance
    line there is no mean_len, and without the eval line there is no emitted-token count.
    """
    reqs: dict[str, dict] = {}
    order: list[str] = []
    coverage = {"print_timing_lines": 0, "prompt_lines": 0, "eval_lines": 0,
                "accept_lines": 0, "graph_lines": 0, "release_lines": 0,
                "decoded_lines": 0, "usable": 0, "dropped_no_accept": 0,
                "dropped_no_eval": 0}

    for line in text.splitlines():
        if "print_timing" not in line:
            continue
        m_task = RE_TASK.search(line)
        if not m_task:
            continue
        coverage["print_timing_lines"] += 1
        key = m_task.group(1)
        if key not in reqs:
            reqs[key] = {"task": key}
            order.append(key)
        r = reqs[key]

        m = RE_PROMPT.search(line)
        if m:
            coverage["prompt_lines"] += 1
            r["prompt_ms"], r["prompt_tokens"] = float(m.group(1)), int(m.group(2))
            continue
        m = RE_EVAL.search(line)
        if m:
            coverage["eval_lines"] += 1
            r["eval_ms"], r["emitted"] = float(m.group(1)), int(m.group(2))
            r["ms_per_token_reported"], r["tps_reported"] = float(m.group(3)), float(m.group(4))
            continue
        m = RE_TOTAL.search(line)
        if m:
            r["total_ms"], r["total_tokens"] = float(m.group(1)), int(m.group(2))
            continue
        m = RE_GRAPHS.search(line)
        if m:
            coverage["graph_lines"] += 1
            r["graphs_reused"] = int(m.group(1))
            continue
        m = RE_ACCEPT.search(line)
        if m:
            coverage["accept_lines"] += 1
            r["accept"] = float(m.group(1))
            r["accepted"], r["drafts"] = int(m.group(2)), int(m.group(3))
            r["mean_len_engine"] = float(m.group(4))
            continue
        m = RE_RELEASE.search(line)
        if m:
            coverage["release_lines"] += 1
            r["n_tokens_release"] = int(m.group(1))
            continue
        m = RE_DECODED.search(line)
        if m:
            coverage["decoded_lines"] += 1
            r["n_decoded"], r["tg_reported"] = int(m.group(1)), float(m.group(2))

    out = []
    for key in order:
        r = reqs[key]
        if "mean_len_engine" not in r:
            coverage["dropped_no_accept"] += 1
            continue
        if "emitted" not in r:
            coverage["dropped_no_eval"] += 1
            continue
        out.append(derive(r, k))
        coverage["usable"] += 1
    return out, coverage


def derive(r: dict, k: int) -> dict:
    """Resolve one request. Routes A/B are data; the definition route is algebra."""
    emitted = r["emitted"]
    mean_len = r["mean_len_engine"]
    drafts = r["drafts"]

    steps_a = emitted / mean_len
    # Route B is only defined when speculation actually ran and the width held at k.
    steps_b = (drafts / k) if (k > 0 and drafts > 0) else float("nan")
    # Algebra, not evidence: it re-reads the same print through the definition.
    steps_def = (r["accepted"] / (mean_len - 1.0)) if mean_len > 1.0 else float("nan")

    step_ms = r["eval_ms"] / steps_a
    agree_pct = (abs(steps_a - steps_b) / steps_a * 100.0
                 if steps_b == steps_b else None)  # NaN-safe

    return {
        "task": r["task"],
        "emitted": emitted,
        "mean_len_engine": round(mean_len, 3),
        "mean_len_model_1_plus_k_a": round(1.0 + k * r["accept"], 3),
        "mean_len_gap_pct": round(abs(mean_len - (1.0 + k * r["accept"])) / mean_len * 100.0, 2),
        "steps_from_mean_len": round(steps_a, 2),
        "steps_from_drafts_over_k": (round(steps_b, 2) if steps_b == steps_b else None),
        "steps_from_definition_algebra": (round(steps_def, 2) if steps_def == steps_def else None),
        "steps_routes_agree_pct": (round(agree_pct, 2) if agree_pct is not None else None),
        "steps_abs_gap": (round(abs(steps_a - steps_b), 2) if steps_b == steps_b else None),
        "applicable": bool(steps_a >= MIN_STEPS_FOR_CHECKS),
        "graphs_reused": r.get("graphs_reused"),
        "graphs_vs_steps_pct": (round(abs(steps_a - r["graphs_reused"]) / steps_a * 100.0, 2)
                                if r.get("graphs_reused") else None),
        "eval_ms": r["eval_ms"],
        "step_ms": round(step_ms, 2),
        "tps_same_arm": round(mean_len / step_ms * 1000.0, 2),
        "tps_reported": r.get("tps_reported"),
        "accept": r["accept"],
        "accepted": r["accepted"],
        "drafts": drafts,
        "prompt_tokens": r.get("prompt_tokens"),
    }


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

ROUTE_TOL_PCT = 5.0
MODEL_TOL_PCT = 5.0

# Applicability, derived rather than tuned. The last round of a request cannot draft a full
# width k (only 1..k-1 tokens remain), so `generated_drafts/k` UNDERCOUNTS steps by ~1/steps,
# and `1 + k*accept` assumes drafts == k*steps exactly. Both checks are therefore only
# meaningful once the truncation is small relative to the request. Measured on the real logs:
#   steps  6.4 -> route disagreement 16.7%   (1/6.4 = 15.6% predicted by the 1/steps law)
#   steps 32.3 -> 2.0-3.1%
#   steps 53.3 -> 0.6%      (= 1/53.3 = 1.9% worst case)
# A request shorter than this is reported with applicable=False and excluded from the verdict,
# not counted as passing and not counted as failing. Short requests are also where an
# off-by-one in `emitted = steps + accepted` shows up, so they cannot adjudicate k.
MIN_STEPS_FOR_CHECKS = 20.0


def verdict(rows: list[dict]) -> list[tuple[str, str]]:
    v: list[tuple[str, str]] = []
    if not rows:
        v.append(("bad", "no usable request: a request needs BOTH an eval line and a "
                         "draft-acceptance line; without the acceptance line there is no "
                         "mean_len to report"))
        return v

    short = [r for r in rows if not r["applicable"]]
    if short:
        v.append(("info", f"{len(short)} request(s) excluded from the route/model checks as too "
                          f"short (<{MIN_STEPS_FOR_CHECKS:.0f} steps; worst task "
                          f"{short[0]['task']}: {short[0]['steps_from_mean_len']:.1f} steps) — "
                          f"the final partial round biases both routes by ~1/steps. Excluded, "
                          f"not passed and not failed"))

    # Route A vs B — the only genuine independent check here.
    spec = [r for r in rows if r["applicable"] and r["steps_routes_agree_pct"] is not None]
    nospec = [r for r in rows if r["applicable"] and r["steps_routes_agree_pct"] is None]
    if not spec:
        v.append(("warn", f"no LONG-ENOUGH request had speculation (drafts>0), so route B is "
                          f"inapplicable and k stays unverified"))
    else:
        bad_routes = [r for r in spec if r["steps_abs_gap"] > 1.5 and
                      r["steps_routes_agree_pct"] > ROUTE_TOL_PCT]
        if bad_routes:
            ex = max(bad_routes, key=lambda r: r["steps_routes_agree_pct"])
            v.append(("bad", f"{len(bad_routes)}/{len(spec)} request(s) where the two step "
                             f"routes disagree by >{ROUTE_TOL_PCT:.0f}% (worst task "
                             f"{ex['task']}: {ex['steps_from_mean_len']} vs "
                             f"{ex['steps_from_drafts_over_k']}) => the draft width did not "
                             f"hold at k, so accept-derived numbers from those requests are "
                             f"not comparable"))
        else:
            v.append(("ok", f"both step routes agree within {ROUTE_TOL_PCT:.0f}% on all "
                            f"{len(spec)} speculative request(s) => the width held at k and "
                            f"the engine's mean_len is a real per-request step count"))
        if nospec:
            v.append(("info", f"{len(nospec)} request(s) had no drafts (route B "
                              f"inapplicable) and are excluded from the route check, not "
                              f"counted as passing it"))

    # The simple accept model, which the whole accept→throughput argument rests on.
    bad_model = [r for r in rows if r["applicable"] and r["mean_len_gap_pct"] > MODEL_TOL_PCT]
    if bad_model:
        ex = max(bad_model, key=lambda r: r["mean_len_gap_pct"])
        v.append(("bad", f"{len(bad_model)}/{len(rows)} request(s) where mean_len != 1 + k*a "
                         f"by >{MODEL_TOL_PCT:.0f}% (worst task {ex['task']}: engine "
                         f"{ex['mean_len_engine']} vs model {ex['mean_len_model_1_plus_k_a']})"
                         f" => the accept model does not describe this run"))
    else:
        n_model = len([r for r in rows if r["applicable"]])
        if n_model:
            v.append(("ok", f"mean_len == 1 + k*a within {MODEL_TOL_PCT:.0f}% on all "
                            f"{n_model} long-enough request(s) => validated on this run "
                            f"rather than assumed"))
        else:
            v.append(("warn", "no request was long enough to validate the accept model"))

    # Loose third route.
    g = [r for r in rows if r["graphs_vs_steps_pct"] is not None]
    if g:
        med = sorted(r["graphs_vs_steps_pct"] for r in g)[len(g) // 2]
        v.append(("info", f"graphs reused tracks steps to a median {med:.1f}% "
                          f"(reported, not enforced: graph reuse is a cache counter, not a "
                          f"step counter)"))

    ms = [r["step_ms"] for r in rows]
    ml = [r["mean_len_engine"] for r in rows]
    v.append(("info", f"same-arm step cost: median {sorted(ms)[len(ms)//2]:.0f} ms/step at "
                      f"median mean_len {sorted(ml)[len(ml)//2]:.2f} "
                      f"=> {sorted(ml)[len(ml)//2] / (sorted(ms)[len(ms)//2]/1000.0):.1f} t/s"))

    v.append(("note", "Identity, NOT evidence: mean_len/step_ms equals the eval line's own "
                      "tokens-per-second (emitted/eval_ms), and steps = accepted/(mean_len-1) "
                      "is the definition re-read. Neither can fail, so neither is a check."))
    v.append(("note", "step_ms is the VERIFY step when speculation is on. Pairing it with an "
                      "unspeculated token count (or the reverse) is the exact error that "
                      "produced both of this project's wrong mean_lens."))
    return v


# ---------------------------------------------------------------------------
# selftest — the fixture lines are verbatim from
# Backup/cgc_logs/llama_server_20260919_190212.log:28048-28053
# ---------------------------------------------------------------------------

_REAL_T8 = (
    "0.50.259.672 I slot print_timing: id  0 | task {task} | prompt eval time =    {pms} ms "
    "/    {ptok} tokens (  335.72 ms per token,     2.98 tokens per second)\n"
    "0.50.259.672 I slot print_timing: id  0 | task {task} |        eval time =   {ems} ms "
    "/   {em} tokens (  102.07 ms per token,     9.80 tokens per second)\n"
    "0.50.259.673 I slot print_timing: id  0 | task {task} |       total time =   17764.67 ms "
    "/  {tt} tokens\n"
    "0.50.259.677 I slot print_timing: id  0 | task {task} |    graphs reused =        {gr}\n"
    "0.50.259.914 I slot print_timing: id  0 | task {task} | draft acceptance = {acc} "
    "(   {accepted} accepted /   {drafts} generated), mean len =  {ml}\n"
)
_REAL_T8_DEFAULTS = dict(pms="4700.05", ptok=14, ems="13064.61", em=128, tt=142, gr=56,
                         acc="0.46541", accepted=74, drafts=159, ml="2.40")


def _real(task=8, **over):
    d = dict(_REAL_T8_DEFAULTS)
    d.update(over)
    d["task"] = task
    return _REAL_T8.format(**d)


def _selftest() -> int:
    fails = []

    def check(name, cond, extra=""):
        if not cond:
            fails.append(f"{name}  {extra}")

    # (1) verbatim real block
    rows, cov = parse_log(_real(), k=3)
    check("real: 1 request parsed", len(rows) == 1, f"got {len(rows)} cov={cov}")
    r = rows[0]
    check("real: emitted", r["emitted"] == 128, r["emitted"])
    check("real: mean_len", r["mean_len_engine"] == 2.40, r["mean_len_engine"])
    check("real: route A = 128/2.40 = 53.3", abs(r["steps_from_mean_len"] - 53.33) < 0.05,
          r["steps_from_mean_len"])
    check("real: route B = 159/3 = 53.0", r["steps_from_drafts_over_k"] == 53.0,
          r["steps_from_drafts_over_k"])
    check("real: routes agree <5%", r["steps_routes_agree_pct"] < 5.0,
          r["steps_routes_agree_pct"])
    check("real: algebra route = 74/1.40 = 52.9",
          abs(r["steps_from_definition_algebra"] - 52.86) < 0.05,
          r["steps_from_definition_algebra"])
    check("real: step_ms = 13064.61/53.33 = 245.0", abs(r["step_ms"] - 245.0) < 0.5,
          r["step_ms"])
    check("real: graphs 56 within 10%", (r["graphs_vs_steps_pct"] or 99) < 10.0,
          r["graphs_vs_steps_pct"])
    check("real: model 1+3*0.46541 = 2.396 agrees", r["mean_len_gap_pct"] < 1.0,
          r["mean_len_gap_pct"])
    check("real: no FAIL verdict", not any(k == "bad" for k, _ in verdict(rows)),
          [m for k, m in verdict(rows) if k == "bad"])
    check("real: identity is labelled a note, not a check",
          any(k == "note" and "NOT evidence" in m for k, m in verdict(rows)))

    # (2) width does NOT hold at k: same mean_len, 212 drafts -> 70.7 vs 53.3. Must go red.
    rows_b, _ = parse_log(_real(task=9, drafts=212), k=3)
    check("width-broken: route mismatch seen",
          rows_b[0]["steps_routes_agree_pct"] > 5.0, rows_b[0]["steps_routes_agree_pct"])
    check("width-broken: verdict FAIL", any(k == "bad" for k, _ in verdict(rows_b)),
          verdict(rows_b))

    # (3) accept model broken: mean_len 1.30 vs model 2.396 must be flagged.
    rows_c, _ = parse_log(_real(task=10, ml="1.30"), k=3)
    check("model-broken: flagged", any(k == "bad" for k, _ in verdict(rows_c)), verdict(rows_c))

    # (4) no speculation: drafts=0 => route B inapplicable, NOT a mismatch.
    rows_d, _ = parse_log(_real(task=11, acc="0.00000", accepted=0, drafts=0, ml="1.00"), k=3)
    check("no-spec: route B is None", rows_d and rows_d[0]["steps_from_drafts_over_k"] is None,
          rows_d[0]["steps_from_drafts_over_k"] if rows_d else None)
    check("no-spec: verdict not FAIL", not any(k == "bad" for k, _ in verdict(rows_d)),
          verdict(rows_d))
    check("no-spec: verdict says k unverified",
          any("unverified" in m or "inapplicable" in m for _, m in verdict(rows_d)),
          verdict(rows_d))

    # (5) eval line but no acceptance line => dropped, and the verdict explains why.
    no_acc = ("0.1.0.0 I slot print_timing: id  0 | task 12 |        eval time =    1000.00 ms"
              " /   64 tokens (   15.62 ms per token,    64.00 tokens per second)\n")
    rows_e, cov_e = parse_log(no_acc, k=3)
    check("no-accept: dropped", len(rows_e) == 0 and cov_e["dropped_no_accept"] == 1, cov_e)
    check("no-accept: verdict explains", any("BOTH" in m for _, m in verdict(rows_e)),
          verdict(rows_e))

    # (6) empty input must not crash and must not silently pass.
    rows_f, _ = parse_log("", k=3)
    check("empty: verdict FAIL", verdict(rows_f) and verdict(rows_f)[0][0] == "bad")

    # (7) k=0 must not divide by zero.
    rows_g, _ = parse_log(_real(), k=0)
    check("k=0: route B is None", rows_g and rows_g[0]["steps_from_drafts_over_k"] is None,
          rows_g[0]["steps_from_drafts_over_k"] if rows_g else None)

    # (8) two requests stay separate and ordered.
    rows_h, _ = parse_log(_real(task=8) + _real(task=9, drafts=212), k=3)
    check("multi: 2 requests", len(rows_h) == 2, len(rows_h))
    check("multi: order kept", [r["task"] for r in rows_h] == ["8", "9"],
          [r["task"] for r in rows_h])

    # (9) A SHORT request, verbatim from the same log (task 0: 16 tokens, 9/16 accepted).
    #     Its routes "disagree" by 16.7% purely because the last round cannot draft a full k.
    #     It must be excluded, NOT reported as a failure of k.
    short = _real(task=0, ems="2467.8", em=16, tt=30, gr=4,
                  acc="0.56250", accepted=9, drafts=16, ml="2.50")
    rows_s, _ = parse_log(short, k=3)
    rs = rows_s[0]
    check("short: routes look inconsistent", rs["steps_routes_agree_pct"] > 5.0,
          rs["steps_routes_agree_pct"])
    check("short: marked inapplicable", rs["applicable"] is False, rs["applicable"])
    check("short: NOT a FAIL", not any(k == "bad" for k, _ in verdict(rows_s)), verdict(rows_s))
    check("short: verdict names the exclusion",
          any(k == "info" and "too short" in m for k, m in verdict(rows_s)), verdict(rows_s))

    # (10) applicability is what separates (2) from (9): same k, same mean_len, only length.
    check("applicability: long is checked, short is not",
          rows[0]["applicable"] is True and rs["applicable"] is False,
          (rows[0]["applicable"], rs["applicable"]))

    # (11) the two-route check must actually be able to go red on real data:
    #     if route B is computed from the SAME print as route A it could never disagree.
    #     Prove it uses the drafts counter by changing only `accepted` (used by the algebra
    #     route) and confirming route B is unchanged while the algebra route moves.
    a = parse_log(_real(task=20, accepted=74), k=3)[0][0]
    b = parse_log(_real(task=20, accepted=37), k=3)[0][0]
    check("independence: route B unaffected by `accepted`",
          a["steps_from_drafts_over_k"] == b["steps_from_drafts_over_k"],
          (a["steps_from_drafts_over_k"], b["steps_from_drafts_over_k"]))
    check("independence: algebra route moved with `accepted`",
          a["steps_from_definition_algebra"] != b["steps_from_definition_algebra"],
          (a["steps_from_definition_algebra"], b["steps_from_definition_algebra"]))

    if fails:
        print("SELFTEST FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("SELFTEST OK — 11 scenarios, all branches pass")
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def _fmt_table(rows: list[dict]) -> str:
    hdr = (f"{'task':>5} {'emitted':>8} {'mean_len':>9} {'1+k*a':>7} {'steps_A':>8} "
           f"{'steps_B':>8} {'agree%':>7} {'graphs':>7} {'eval_ms':>9} {'step_ms':>8} "
           f"{'t/s':>6} {'accept':>7}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        b = r["steps_from_drafts_over_k"]
        ag = r["steps_routes_agree_pct"]
        gr = r["graphs_reused"]
        out.append(
            f"{r['task']:>5} {r['emitted']:>8} {r['mean_len_engine']:>9.2f} "
            f"{r['mean_len_model_1_plus_k_a']:>7.2f} {r['steps_from_mean_len']:>8.1f} "
            f"{(f'{b:.1f}' if b is not None else 'n/a'):>8} "
            f"{(f'{ag:.1f}' if ag is not None else 'n/a'):>7} "
            f"{(gr if gr is not None else 'n/a'):>7} "
            f"{r['eval_ms']:>9.1f} {r['step_ms']:>8.1f} {r['tps_same_arm']:>6.2f} "
            f"{r['accept']:>7.5f}")
    return "\n".join(out)


ICON = {"ok": "PASS", "bad": "FAIL", "warn": "CAVEAT", "info": "INFO", "note": "NOTE"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sc = sub.add_parser("scan")
    sc.add_argument("logs", nargs="+")
    sc.add_argument("--k", type=int, default=3,
                    help="draft width from --spec-draft-n-max (default 3)")
    sc.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return _selftest()

    all_rows, per_log = [], []
    for pat in args.logs:
        for path in sorted(glob.glob(pat)):
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError as e:
                print(f"skip {path}: {e}", file=sys.stderr)
                continue
            rows, cov = parse_log(text, args.k)
            for r in rows:
                r["log"] = path.rsplit("/", 1)[-1]
            all_rows += rows
            per_log.append((path.rsplit("/", 1)[-1], cov, len(rows)))

    for name, cov, n in per_log:
        print(f"{name}: usable={n} eval={cov['eval_lines']} accept={cov['accept_lines']} "
              f"graphs={cov['graph_lines']} dropped(no accept)={cov['dropped_no_accept']} "
              f"dropped(no eval)={cov['dropped_no_eval']}")
    print()
    print(_fmt_table(all_rows))
    print()
    for kind, msg in verdict(all_rows):
        print(f"[{ICON[kind]}] {msg}")

    # The mandated block. This is the ruler's whole point: mean_len is never quoted on its own,
    # and the step cost beside it is from the SAME arm (pair() would refuse anything else).
    if all_rows:
        resolved = [_RULER.resolve(r, args.k) for r in all_rows]
        print()
        # arm is "mtp-on" because every request here comes from a log that printed an acceptance
        # line, i.e. speculation ran. The step time is from the SAME request block, and block()
        # re-checks that with pair() rather than trusting this comment.
        print(_RULER.block("mtp-on", args.k, resolved, step_arm="mtp-on",
                           extra="same-arm step = eval_ms/steps on each request"))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"k": args.k, "rows": all_rows,
                       "verdict": [{"kind": k, "msg": m} for k, m in verdict(all_rows)]},
                      fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
