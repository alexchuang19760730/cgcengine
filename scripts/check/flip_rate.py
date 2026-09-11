#!/usr/bin/env python3
"""Knife-edge-resistant quality evaluation (fixed seeds + repeats + verdict-flip statistics).

Why this exists: single greedy questions on this stack are NOT a measurement. Two runs of the
same question under greedy decoding can land in different trajectories (answer vs echo/loop)
because a ~0.02 logit offset — which comes from which MoE path/layout a step took — is enough
to flip a near-tie about ten tokens in. Ranking pool sizes on one greedy question therefore
measures luck, not quality.

This harness makes the uncertainty explicit:

  * --greedy-repeats N : send the SAME greedy request N times (temp 0, no seed) -> measures
                         within-session determinism (should be 100% identical).
  * --repeats N        : send each question with N fixed seeds at --temperature -> measures the
                         sampling-side fragility of the prompt (deterministic per seed).
  * per question       : pass rate over seeds + a "flip" flag when the verdict varies.
  * room level         : suite pass rate per seed -> mean / std / min / max (the variance the
                         pool-floor decision must survive).
  * --compare A B ...  : verdict-by-verdict diff between sessions/configs (flip count).

Scoring is the same all-pass rule set as the 37/48 baseline (rules imported from bare_48.py),
so numbers stay comparable with earlier runs.

  python3 scripts/check/flip_rate.py --label 8gb-run1 --profiles qa-zh,math,coding,reasoning \\
      --per-profile 2 --repeats 5 --temperature 0.2 --greedy-repeats 2 \\
      --output /tmp/flip_8gb_run1.json
  python3 scripts/check/flip_rate.py --compare /tmp/flip_8gb_run1.json /tmp/flip_4gb_run1.json
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from bare_48 import load_reference, score_one  # noqa: E402


def ask(base_url, model, prompt, temperature, seed, timeout, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"content": "", "finish_reason": f"http_{e.code}", "elapsed": time.time() - t0,
                "error": str(e)}
    except Exception as e:  # noqa: BLE001 - harness
        return {"content": "", "finish_reason": "error", "elapsed": time.time() - t0,
                "error": str(e)}
    ch = (body.get("choices") or [{}])[0]
    elapsed = time.time() - t0
    # Speed matters for the pool-floor decision (a small pool can be "correct but
    # unusably slow"), so capture completion tokens + tok/s when the server reports
    # usage. Falls back to None rather than guessing from word counts.
    usage = body.get("usage") or {}
    ctok = usage.get("completion_tokens")
    tps = None
    if ctok and elapsed > 0:
        tps = round(ctok / elapsed, 2)
    return {"content": (ch.get("message", {}).get("content") or ""),
            "finish_reason": ch.get("finish_reason", "unknown"),
            "elapsed": round(elapsed, 1),
            "completion_tokens": ctok,
            "tps": tps}


def build_questions(ref, profiles, per_profile):
    out = []
    for p in profiles:
        rules = ref[p]
        for i, prompt in enumerate(rules["prompts"][:per_profile]):
            out.append({"profile": p, "index": i, "prompt": prompt})
    return out


def run_session(args):
    ref = load_reference()
    profiles = args.profiles.split(",")
    questions = build_questions(ref, profiles, args.per_profile)
    seeds = [args.seed_base + i for i in range(args.repeats)]

    print(f"session {args.label}: {len(questions)} questions x "
          f"({args.greedy_repeats} greedy + {len(seeds)} seeded) = "
          f"{len(questions) * (args.greedy_repeats + len(seeds))} requests", flush=True)

    rows = []
    for q in questions:
        rules = ref[q["profile"]]
        greedy = []
        for _ in range(args.greedy_repeats):
            r = ask(args.base_url, args.model, q["prompt"], 0.0, None, args.timeout, args.max_tokens)
            ok, checks = score_one(q["profile"], rules, r["content"], r["finish_reason"])
            greedy.append({"pass": ok, "checks": checks, "content": r["content"],
                           "finish": r["finish_reason"], "elapsed": r["elapsed"],
                           "completion_tokens": r["completion_tokens"], "tps": r["tps"]})
        seeded = []
        for s in seeds:
            r = ask(args.base_url, args.model, q["prompt"], args.temperature, s,
                    args.timeout, args.max_tokens)
            ok, checks = score_one(q["profile"], rules, r["content"], r["finish_reason"])
            seeded.append({"seed": s, "pass": ok, "checks": checks, "content": r["content"],
                           "finish": r["finish_reason"], "elapsed": r["elapsed"],
                           "completion_tokens": r["completion_tokens"], "tps": r["tps"]})
        g_pass = [x["pass"] for x in greedy]
        s_pass = [x["pass"] for x in seeded]
        row = {
            "profile": q["profile"], "index": q["index"], "prompt": q["prompt"],
            "greedy_verdicts": g_pass,
            "greedy_stable": len(set(g_pass)) == 1,
            "seeded_pass_rate": sum(s_pass) / len(s_pass) if s_pass else 0.0,
            "seeded_verdicts": s_pass,
            "seeded_flip": len(set(s_pass)) > 1,
            "greedy_detail": greedy, "seeded_detail": seeded,
        }
        rows.append(row)
        print(f"  [{q['profile']}#{q['index']}] greedy={g_pass} seeded={s_pass} "
              f"rate={row['seeded_pass_rate']:.2f}"
              + ("  FLIP" if row["seeded_flip"] or not row["greedy_stable"] else ""), flush=True)

    # suite-level pass rate per seed (the variance the pool-floor decision has to survive)
    per_seed = []
    for k in range(len(seeds)):
        per_seed.append(round(sum(1 for r in rows if r["seeded_verdicts"][k]) / len(rows), 4))
    greedy_rate = sum(1 for r in rows if all(r["greedy_verdicts"])) / len(rows) if rows else 0.0

    summary = {
        "label": args.label,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "seeds": seeds,
        "n_questions": len(rows),
        "greedy_repeats": args.greedy_repeats,
        "greedy_all_pass_rate": round(greedy_rate, 4),
        "greedy_unstable_questions": sum(1 for r in rows if not r["greedy_stable"]),
        "seeded_flip_questions": sum(1 for r in rows if r["seeded_flip"]),
        "suite_rate_per_seed": per_seed,
        "suite_rate_mean": round(statistics.mean(per_seed), 4) if per_seed else None,
        "suite_rate_std": round(statistics.pstdev(per_seed), 4) if len(per_seed) > 1 else 0.0,
        "suite_rate_min": min(per_seed) if per_seed else None,
        "suite_rate_max": max(per_seed) if per_seed else None,
        "questions": rows,
    }

    # Speed across ALL requests in this session (greedy + seeded). Reported alongside the
    # pass rate because the pool floor is a quality x speed trade, not a quality alone.
    all_tps = [d["tps"] for r in rows for d in (r["greedy_detail"] + r["seeded_detail"])
               if d.get("tps")]
    summary["tps_mean"] = round(statistics.mean(all_tps), 2) if all_tps else None
    summary["tps_std"] = round(statistics.pstdev(all_tps), 2) if len(all_tps) > 1 else 0.0
    summary["tps_max"] = round(max(all_tps), 2) if all_tps else None
    summary["tps_n"] = len(all_tps)
    if args.output:
        json.dump(summary, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\nwrote {args.output}", flush=True)

    print(f"\n=== {args.label} ===")
    print(f"greedy all-pass rate      : {summary['greedy_all_pass_rate']:.3f} "
          f"(unstable questions: {summary['greedy_unstable_questions']})")
    print(f"seeded suite rate per seed: {per_seed}")
    print(f"seeded mean / std / range : {summary['suite_rate_mean']} / "
          f"{summary['suite_rate_std']} / [{summary['suite_rate_min']}, {summary['suite_rate_max']}]")
    print(f"questions that flip       : {summary['seeded_flip_questions']}/{len(rows)}")
    print(f"speed (tok/s) mean/std/max: {summary['tps_mean']} / {summary['tps_std']} / "
          f"{summary['tps_max']}  (n={summary['tps_n']})")
    return summary


def compare(files):
    sessions = [json.load(open(f, encoding="utf-8")) for f in files]
    print(f"{'question':<34}" + "".join(f"{s['label'][:12]:>14}" for s in sessions))
    flips = 0
    bit_identical = 0
    for i, q0 in enumerate(sessions[0]["questions"]):
        key = f"{q0['profile']}#{q0['index']}"
        verdicts = []
        contents = []
        for s in sessions:
            q = next((x for x in s["questions"]
                      if x["profile"] == q0["profile"] and x["index"] == q0["index"]), None)
            verdicts.append(q["seeded_pass_rate"] if q else float("nan"))
            if q and q["greedy_detail"]:
                contents.append(q["greedy_detail"][0]["content"])
            else:
                contents.append("")
        mark = "  <-- FLIP" if len(set(round(v, 3) for v in verdicts)) > 1 else ""
        flips += 1 if mark else 0
        if len(set(contents)) == 1 and contents[0] != "":
            bit_identical += 1
        print(f"{key:<34}" + "".join(f"{v:>14.2f}" for v in verdicts) + mark)
    print(f"\nsuite rate (seeded mean): "
          + ", ".join(f"{s['label']}={s['suite_rate_mean']}" for s in sessions))
    print(f"suite rate (greedy)     : "
          + ", ".join(f"{s['label']}={s['greedy_all_pass_rate']}" for s in sessions))
    print(f"questions with a differing pass rate across sessions: {flips}/{len(sessions[0]['questions'])}")
    print(f"bit-identical questions (greedy[0] content identical): {bit_identical}/{len(sessions[0]['questions'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", default="local")
    ap.add_argument("--profiles", default="qa-zh,math,coding,reasoning")
    ap.add_argument("--per-profile", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--greedy-repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return
    run_session(args)


if __name__ == "__main__":
    main()
