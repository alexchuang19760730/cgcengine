#!/usr/bin/env python3
"""The cross-arm ruler: mean_len and step time are one number, not two.

Why this is a module and not a convention
----------------------------------------
`mean_len` (tokens per decode step) has been reported wrong twice on this line, and both
times the arithmetic looked fine in isolation:

  * ``2.67`` — ``draft_slots - accepted`` was called a round count. It is not a step count.
  * ``1.75`` — an **MTP-on** token count divided by an **MTP-off** step time.

Both are quotients assembled from two different regimes. A prose convention ("remember to
quote the step time too") does not prevent that, because the failure happens in whoever
assembles the quotient, not in whoever reads it. So the pairing is enforced here, at the one
place the numbers are produced.

The engine's own definition (src/llama.cpp/tools/server/server-context.cpp:674):

    mean_acc_len = 1.0 + n_draft_accepted / n_draft_verif_steps

What this ruler will and will not accept
----------------------------------------
* ``mean_len`` must come from the engine's print (``mean_len_engine``) — not from
  ``1 + k*a``, which is a *model* of it and is checked against it, not substituted for it.
* ``verify_steps`` is taken from an **independent counter** (``generated / k``) when
  speculation ran and the width held. ``accepted / (mean_len - 1)`` is the definition
  re-read: algebra, never counted as a check.
* ``step_ms`` must be ``eval_ms / verify_steps`` **of the same arm**. There is no legal way
  to combine an arm's mean_len with another arm's step time; `pair()` raises.
* A quoted pair must carry equal arm labels, so ``arm`` is required, not optional.

A third failure shape, found by the first audit run rather than by reasoning
-----------------------------------------------------------------------
``mtp_accept_ab*.json`` carries ``mean_len=61.0`` next to ``accept=0.5825``. There, ``mean_len``
is the **generated output length in tokens**, not tokens per decode step — the same key name
meaning two different quantities in two families of products. That is how the next wrong quotient
gets assembled: a reader takes 61.0 for a step ratio and divides something by it. The audit
flags it; the products themselves should rename it (``gen_len``), and until then no number from
those files can be paired with anything.

Usage
-----
    python3 scripts/check/mtp_ruler.py selftest
    python3 scripts/check/mtp_ruler.py audit Backup/cgc_logs/*.json Backup/phase_decomp/*.json
    python3 scripts/check/mtp_ruler.py scan Backup/cgc_logs/llama_server_2026*.log --k 3
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- the definition
DEFINITION_SRC = "src/llama.cpp/tools/server/server-context.cpp:674"
DEFINITION = "mean_acc_len = 1.0 + n_draft_accepted / n_draft_verif_steps"
# tolerance for "the engine's mean_len matches the 1+k*a model"
MODEL_TOL_PCT = 5.0
# route A vs route B (independent step counters) must agree within this
ROUTE_TOL_PCT = 5.0
# below this many steps, the last round's truncation bias (about 1/steps) is not bounded
MIN_STEPS_FOR_CHECKS = 20

# --------------------------------------------------------------------------- canonical parsing
# These live here so there is exactly ONE parse of these log lines in the tree; consumers
# import them instead of re-declaring (a second copy is how the next wrong number gets in).
# Deliberately strict about unit strings: a format change must show up as a coverage miss.
RE_TASK = re.compile(r"\|\s*task\s+(\d+)\s*\|")
RE_PROMPT = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
RE_EVAL = re.compile(
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*"
    r"\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)"
)
RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
RE_GRAPHS = re.compile(r"graphs reused\s*=\s*(\d+)")
RE_ACCEPT = re.compile(
    r"draft acceptance\s*=\s*([\d.]+)\s*"
    r"\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\s*\),\s*mean len\s*=\s*([\d.]+)"
)
RE_RELEASE = re.compile(r"stop processing:\s*n_tokens\s*=\s*(\d+)")
RE_DECODED = re.compile(r"n_decoded\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+)\s*t/s")

# keys that can serve as a same-arm step time (per decode step), and keys that name the arm
STEP_TIME_KEYS = ("step_ms", "verify_step_ms", "decode_step_ms", "ms_per_step")
ARM_KEYS = ("arm", "mtp", "mtp_on", "spec", "spec_on", "config", "carrier")
MEAN_LEN_KEYS = ("mean_len", "mean_len_engine", "mean_acc_len", "draft_mean_len")
ACCEPT_KEYS = ("accept", "draft_accept", "acceptance", "draft_acceptance")


class CrossArmError(ValueError):
    """Two numbers from different arms were about to be combined."""


# --------------------------------------------------------------------------- the pair rule
def pair(mean_len: float, mean_len_arm: str, step_ms: float, step_ms_arm: str,
         *, where: str = "") -> tuple[float, float]:
    """Return (mean_len, step_ms) only if both carry the same arm label.

    This is the rule the whole module exists for: an MTP-on token count over an MTP-off
    step time is a category error, and it is silent — every arithmetic check passes.
    """
    if not mean_len_arm or not step_ms_arm:
        raise CrossArmError(
            f"{where or 'pair'}: mean_len and step_ms must each carry an arm label; "
            f"got mean_len<{mean_len_arm!r}> step_ms<{step_ms_arm!r}>")
    if mean_len_arm != step_ms_arm:
        raise CrossArmError(
            f"{where or 'pair'}: mean_len from arm {mean_len_arm!r} with step_ms from arm "
            f"{step_ms_arm!r} — that quotient describes neither arm (the historical 1.75)")
    if mean_len <= 1.0:
        raise CrossArmError(f"{where or 'pair'}: mean_len={mean_len} <= 1 implies no accepted "
                            f"draft token; it cannot be paired with a speculation step time")
    if step_ms <= 0:
        raise CrossArmError(f"{where or 'pair'}: step_ms={step_ms} is not a duration")
    return mean_len, step_ms


def resolve(r: dict, k: int) -> dict:
    """One request's numbers, with the step count resolved from two independent routes.

    `r` carries the engine's per-request print: emitted, eval_ms, accept, accepted, drafts,
    mean_len_engine. Nothing here is assembled from a foreign regime: every field is from the
    same request block.
    """
    emitted, mean_len = float(r["emitted"]), float(r["mean_len_engine"])
    drafts, accepted = float(r["drafts"]), float(r["accepted"])

    steps_a = emitted / mean_len                      # the engine's own mean_len
    steps_b = (drafts / k) if (k > 0 and drafts > 0) else None   # independent counter
    steps_def = (accepted / (mean_len - 1.0)) if mean_len > 1.0 else None  # algebra

    step_ms = float(r["eval_ms"]) / steps_a
    agree_pct = (abs(steps_a - steps_b) / steps_a * 100.0
                 if steps_b else None)
    model = 1.0 + k * float(r["accept"])
    return {
        "emitted": emitted, "mean_len": mean_len, "accept": float(r["accept"]),
        "accepted": accepted, "drafts": drafts, "k": k,
        "steps_from_mean_len": steps_a,
        "steps_from_drafts_over_k": steps_b,
        "steps_from_definition_algebra": steps_def,
        "routes_agree_pct": agree_pct,
        "model_1_plus_k_a": model,
        "model_gap_pct": abs(mean_len - model) / mean_len * 100.0,
        "eval_ms": float(r["eval_ms"]), "step_ms": step_ms,
        "tps_same_arm": mean_len / step_ms * 1000.0,
        "applicable": steps_a >= MIN_STEPS_FOR_CHECKS,
    }


def complaints(res: dict) -> list[str]:
    """Everything that makes this request's mean_len unfit to quote. Empty means quotable."""
    out = []
    if not res["applicable"]:
        out.append(f"only {res['steps_from_mean_len']:.1f} steps (<{MIN_STEPS_FOR_CHECKS}): the "
                   f"last round's draft truncation bias (~1/steps) is not bounded, so this "
                   f"request is excluded — not passed and not failed")
    if res["routes_agree_pct"] is not None and res["routes_agree_pct"] > ROUTE_TOL_PCT:
        out.append(f"step-count routes disagree by {res['routes_agree_pct']:.2f}% "
                   f"({res['steps_from_mean_len']:.1f} vs {res['steps_from_drafts_over_k']:.1f})")
    if res["model_gap_pct"] > MODEL_TOL_PCT:
        out.append(f"engine mean_len {res['mean_len']:.3f} != 1+k*a model "
                   f"{res['model_1_plus_k_a']:.3f} by {res['model_gap_pct']:.2f}%: the numerator "
                   f"and the accept pair may come from different regimes")
    return out


def summarise(resolved: list[dict]) -> dict:
    """The quotable form of a run: median mean_len + the SAME-ARM step cost."""
    good = [r for r in resolved if not complaints(r)]
    if not good:
        return {"quotable": False, "reason": "no request passed the ruler", "n": len(resolved)}
    ml = sorted(r["mean_len"] for r in good)
    sm = sorted(r["step_ms"] for r in good)
    med = lambda xs: xs[len(xs) // 2]
    return {
        "quotable": True, "n": len(good), "n_excluded": len(resolved) - len(good),
        "mean_len": med(ml), "mean_len_p10": ml[max(0, len(ml) // 10)],
        "mean_len_p90": ml[min(len(ml) - 1, (len(ml) * 9) // 10)],
        "step_ms": med(sm),
        "tps_same_arm": med(ml) / med(sm) * 1000.0,
        "accept_median": med(sorted(r["accept"] for r in good)),
    }


def block(arm: str, k: int, resolved: list[dict], *, engine: str = "", extra: str = "",
          step_arm: str | None = None) -> str:
    """The two lines every MTP report must carry. Both numbers, one arm, one place.

    `step_arm` lets a caller declare which arm its step time came from. The (SAME arm) claim
    printed below is then MACHINE-CHECKED by pair(), so a caller holding an MTP-on mean_len and
    an MTP-off step time cannot print a compliant-looking block -- the block itself refuses.
    """
    s = summarise(resolved)
    if not s["quotable"]:
        return f"MTP RULER arm={arm} k={k}: NOT QUOTABLE — {s['reason']}"
    pair(s["mean_len"], arm, s["step_ms"], step_arm or arm, where="block()")
    head = f"MTP RULER  arm={arm} k={k}" + (f"  engine={engine}" if engine else "")
    return "\n".join([
        head,
        f"  mean_len={s['mean_len']:.2f} (p10 {s['mean_len_p10']:.2f} / p90 {s['mean_len_p90']:.2f}, "
        f"n={s['n']} req, {s['n_excluded']} excluded)   "
        f"accept={s['accept_median']:.4f}",
        f"  step={s['step_ms']:.1f} ms (SAME arm)   ->  {s['tps_same_arm']:.2f} t/s"
        + (f"   {extra}" if extra else ""),
        f"  def: {DEFINITION}   [{DEFINITION_SRC}]",
    ])


# --------------------------------------------------------------------------- the audit
def _arm_of(row: dict):
    for key in ARM_KEYS:
        if key in row and row[key] not in (None, ""):
            return key, str(row[key])
    return None, None


def _step_ms_of(row: dict):
    """(value, provenance) or (None, why-not). Per-step, or computable from the same row."""
    for key in STEP_TIME_KEYS:
        if isinstance(row.get(key), (int, float)) and row[key]:
            return float(row[key]), key
    if isinstance(row.get("eval_ms"), (int, float)) and isinstance(row.get("mean_len"), (int, float)):
        if row["mean_len"] > 1:
            return float(row["eval_ms"]) / (row.get("emitted", 0) / row["mean_len"]), "eval_ms/mean_len"
    if isinstance(row.get("predicted_ms"), (int, float)) and isinstance(row.get("predicted_n"), int):
        if row["predicted_n"]:
            return float(row["predicted_ms"]) / row["predicted_n"], "predicted_ms/predicted_n"
    return None, "no per-step time in the row"


def audit_row(row: dict) -> tuple[str, str]:
    """('ok'|'orphan'|'n/a'|'unlabelled', detail) for one report row."""
    ml = next((row[k] for k in MEAN_LEN_KEYS if isinstance(row.get(k), (int, float))), None)
    ac = next((row[k] for k in ACCEPT_KEYS if isinstance(row.get(k), (int, float))), None)
    if ml is None and ac is None:
        return "n/a", ""
    if ml is None or ac is None:
        return "unlabelled", f"has {'mean_len' if ml else 'accept'} but not the other"
    key, arm = _arm_of(row)
    if not arm:
        return "unlabelled", "mean_len/accept without an arm label (which MTP setting?)"
    val, prov = _step_ms_of(row)
    if val is None:
        return "orphan", f"mean_len={ml} accept={ac} with no same-arm step time ({prov})"
    if val <= 0:
        return "orphan", f"step time {prov}={val} is not a duration"
    return "ok", f"mean_len={ml} accept={ac} step={val:.2f} ms ({prov}, arm={arm})"


def audit(paths: list[str], *, verbose: bool = False) -> dict:
    """Scan report products for the two historical failures.

    Honest boundary: key matching is heuristic. It finds rows that QUOTE one of the numbers
    without a same-arm step time; it cannot know what a differently-named field means.
    """
    rep = {"files": 0, "rows": 0, "ok": 0, "orphan": 0, "unlabelled": 0, "n/a": 0, "hits": []}

    def walk(obj, path, out):
        if isinstance(obj, dict):
            out.append((path, obj))
            for kk, vv in obj.items():
                walk(vv, f"{path}.{kk}", out)
        elif isinstance(obj, list):
            for i, vv in enumerate(obj):
                walk(vv, f"{path}[{i}]", out)

    for pat in paths:
        for f in sorted(glob.glob(pat)):
            try:
                data = json.loads(Path(f).read_text())
            except Exception:
                continue
            rep["files"] += 1
            rows: list = []
            walk(data, Path(f).name, rows)
            for path, row in rows:
                state, detail = audit_row(row)
                rep[state] = rep.get(state, 0) + (1 if state != "n/a" else 0)
                rep["rows"] += 1
                if state in ("orphan", "unlabelled"):
                    rep["hits"].append({"where": path, "state": state, "detail": detail})
                elif verbose and state == "ok":
                    rep["hits"].append({"where": path, "state": state, "detail": detail})
    return rep


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails: list[str] = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            fails.append(f"{name}: got {got!r} want {want!r}")

    def expect_raises(name, fn):
        try:
            fn()
        except CrossArmError as exc:
            print(f"  ok   {name}: refused ({str(exc)[:60]}...)")
            return
        print(f"  FAIL {name}: accepted a cross-arm pair")
        fails.append(name)

    print("the two historical mistakes must be REFUSED")
    # 1.75: MTP-on mean_len over an MTP-off step time
    expect_raises("1.75 = on-mean_len / off-step", lambda: pair(1.75, "on", 143.4, "off"))
    # and it is the arms, not the arithmetic: the same quotient with one label is legal
    expect("same-arm pair is legal", pair(1.75, "on", 143.4, "on"), (1.75, 143.4))
    expect_raises("missing arm label", lambda: pair(2.40, "", 226.8, "on"))
    expect_raises("mean_len<=1 cannot be a verify step", lambda: pair(1.0, "on", 226.8, "on"))
    expect_raises("non-duration step", lambda: pair(2.40, "on", 0.0, "on"))

    print("\nthe 2.67 mistake (a bad divisor) must be caught by the route check")
    # A REAL record, verbatim from llama_server_20260919_190212.log task 8:
    #   draft acceptance = 0.46541 ( 74 accepted / 159 generated), mean len = 2.40
    #   eval line: 128 tokens emitted. Routes: 128/2.40 = 53.3 vs 159/3 = 53.0 (0.6% apart).
    # NOTE: an earlier draft of this fixture paired the whole-run totals (180/309) with a single
    # request's emitted/mean_len -- the exact cross-regime quotient this ruler exists to reject,
    # caught by the route check on the first run.
    real = {"emitted": 128, "eval_ms": 12096.0, "accept": 0.46541, "accepted": 74,
            "drafts": 159, "mean_len_engine": 2.40}
    res = resolve(real, 3)
    expect("routes agree on the real record", res["routes_agree_pct"] is not None
           and res["routes_agree_pct"] < ROUTE_TOL_PCT, True)
    expect("real record has no complaints", complaints(res), [])
    expect("same-arm step cost lands where the log says (~227 ms)",
           round(res["step_ms"]) , 227)
    # 2.67 came from calling (slots - accepted) = 108 a round count
    expect("the 108-round divisor is rejected against the engine's 53.3",
           abs(108 - res["steps_from_mean_len"]) / res["steps_from_mean_len"] * 100 > ROUTE_TOL_PCT,
           True)
    bad = dict(real, drafts=53)                  # drafts collapsed to one per round
    expect("collapsed draft counter is caught",
           any("routes disagree" in c for c in complaints(resolve(bad, 3))), True)
    wrong_k = resolve(real, 1)                   # k wrong -> route B moves
    expect("wrong k moves route B beyond tolerance",
           any("routes disagree" in c for c in complaints(wrong_k)), True)
    # the whole-run total is NOT a per-request number: pairing them is the 1.75/2.67 shape
    expect("whole-run totals mixed into a request are caught",
           any("routes disagree" in c for c in
               complaints(resolve(dict(real, accepted=180, drafts=309), 3))), True)

    print("\nthe model check must bite when mean_len is not 1+k*a")
    off_model = dict(real, mean_len_engine=3.40)
    expect("mean_len inconsistent with accept is caught",
           any("!= 1+k*a model" in c for c in complaints(resolve(off_model, 3))), True)
    consistent = dict(real, mean_len_engine=1 + 3 * 0.46541)
    expect("consistent mean_len passes every check", complaints(resolve(consistent, 3)), [])

    print("\nshort requests are excluded, never passed and never failed")
    short = dict(real, emitted=16, eval_ms=1500.0)
    expect("short request is not applicable", resolve(short, 3)["applicable"], False)
    expect("short request is a complaint",
           any("not passed and not failed" in c for c in complaints(resolve(short, 3))), True)

    print("\nthe summariser must not silently accept an all-excluded run")
    expect("no quotable rows -> not quotable", summarise([resolve(short, 3)])["quotable"], False)
    s = summarise([resolve(real, 3)])
    expect("quotable run reports both numbers",
           ("mean_len" in s and "step_ms" in s and "tps_same_arm" in s), True)
    expect("tps is the same-arm quotient",
           round(s["tps_same_arm"], 2), round(2.40 / res["step_ms"] * 1000.0, 2))
    expect("cross-arm tps would have been 1.5x higher (the 1.75 shape)",
           round(2.40 / 143.4 * 1000.0, 2) > round(s["tps_same_arm"], 2), True)

    print("\nblock() must carry both numbers in every quotable line")
    b = block("on", 3, [resolve(real, 3)], engine="aab787412e")
    expect("block names the arm", "arm=on" in b, True)
    expect("block carries mean_len", "mean_len=" in b, True)
    expect("block carries the same-arm step", "(SAME arm)" in b, True)
    expect("block cites the definition source", DEFINITION_SRC in b, True)
    expect("the (SAME arm) claim is machine-checked: one arm label is legal",
           "(SAME arm)" in block("on", 3, [resolve(real, 3)], step_arm="on"), True)
    expect_raises("a block claiming an off step beside an on mean_len is refused",
                  lambda: block("on", 3, [resolve(real, 3)], step_arm="off"))
    expect_raises("a block with no declared step arm is refused",
                  lambda: block("", 3, [resolve(real, 3)]))

    print("\naudit must flag an orphan mean_len and pass a paired row")
    expect("orphan row", audit_row({"arm": "on", "mean_len": 2.40, "accept": 0.58})[0], "orphan")
    expect("paired row", audit_row({"arm": "on", "mean_len": 2.40, "accept": 0.58,
                                    "step_ms": 226.8})[0], "ok")
    expect("unlabelled row", audit_row({"mean_len": 2.40, "accept": 0.58,
                                        "step_ms": 226.8})[0], "unlabelled")
    expect("row without either number is n/a", audit_row({"mtp": "1", "tps": 10.0})[0], "n/a")
    expect("accept without mean_len is unlabelled",
           audit_row({"arm": "on", "accept": 0.58, "step_ms": 226.8})[0], "unlabelled")

    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("SELFTEST PASS — the ruler refuses both historical quotients and both failure shapes")
    return 0


# --------------------------------------------------------------------------- log scan
def scan(logs: list[str], k: int) -> int:
    """Read the engine's own lines and print the ruler block per log file."""
    any_ok = False
    for pat in logs:
        for f in sorted(glob.glob(pat)):
            text = Path(f).read_text(errors="replace")
            reqs: dict[str, dict] = {}
            arms = set()
            for m in re.finditer(RE_ACCEPT, text):
                arms.add("mtp-on")
            cur = None
            for line in text.splitlines():
                if "print_timing" not in line:
                    continue
                mt = RE_TASK.search(line)
                if not mt:
                    continue
                r = reqs.setdefault(mt.group(1), {"task": mt.group(1)})
                if (m := RE_EVAL.search(line)):
                    r["eval_ms"], r["emitted"] = float(m.group(1)), int(m.group(2))
                elif (m := RE_ACCEPT.search(line)):
                    r["accept"], r["accepted"] = float(m.group(1)), int(m.group(2))
                    r["drafts"], r["mean_len_engine"] = int(m.group(3)), float(m.group(4))
            resolved = [resolve(r, k) for r in reqs.values()
                        if all(x in r for x in ("eval_ms", "emitted", "accept", "accepted",
                                               "drafts", "mean_len_engine"))]
            if not resolved:
                print(f"  {Path(f).name}: no request carries both an eval line and an acceptance "
                      f"line — nothing to measure here")
                continue
            arm = "on" if arms else "off"
            print(f"  {Path(f).name}")
            print("  " + block(arm, k, resolved).replace("\n", "\n  "))
            any_ok = True
    if not any_ok:
        print("\n(no log carried a usable pair; nothing was quoted)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sc = sub.add_parser("scan")
    sc.add_argument("logs", nargs="+")
    sc.add_argument("--k", type=int, default=3)
    au = sub.add_parser("audit")
    au.add_argument("paths", nargs="+")
    au.add_argument("--verbose", action="store_true")
    au.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "scan":
        return scan(args.logs, args.k)
    if args.cmd == "audit":
        rep = audit(args.paths, verbose=args.verbose)
        print(f"audited {rep['files']} file(s), {rep['rows']} object(s)")
        print(f"  compliant rows with a same-arm step time : {rep['ok']}")
        print(f"  ORPHAN (mean_len/accept, no step time)   : {rep['orphan']}")
        print(f"  UNLABELLED (no arm)                      : {rep['unlabelled']}")
        for h in rep["hits"]:
            print(f"    [{h['state']}] {h['where']}: {h['detail']}")
        if args.json:
            Path(args.json).write_text(json.dumps(rep, indent=1))
            print(f"json -> {args.json}")
        return 0 if rep["orphan"] == 0 and rep["unlabelled"] == 0 else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
