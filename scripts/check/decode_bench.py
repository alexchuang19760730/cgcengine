#!/usr/bin/env python3
"""Controlled decode benchmark: short prompt, long generation, temperature=0, N rounds.

Why this exists: every decode number quoted in the docs so far came from an *uncontrolled*
request mix (some prompts were 4600 tokens, others 4 tokens) and the KV-cache size alone
moves decode tok/s by ~5x. So "2.1 t/s" and "22 t/s" were never measurements of the same
thing. This harness fixes the shape so A/B comparisons mean something:

  - fixed short prompt (~30 tokens) -> small KV, isolates expert fetch + decode compute
  - fixed n_predict, temperature 0 -> deterministic token count and path
  - N rounds, report median / min / max -> reject single-sample noise
  - optionally a warmup round that is excluded from stats

Server-side knifeedge stats are NOT used here on purpose: they need the process to exit.
Read them from the log after killing the server, or use --no-server management.

Usage:
    python3 scripts/check/decode_bench.py --rounds 7 --n-predict 160
    python3 scripts/check/decode_bench.py --rounds 5 --tag prefetch-on --json /tmp/dp.json
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"

# In-band thermal reading. IMPORTED, not re-implemented: one parser, and one place where the
# "unreadable is not zero" rule lives (thermal_pressure.py documents the notifyutil trap).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thermal_pressure as tp  # noqa: E402

# ~30-token Chinese prompt. Short on purpose: the point is to measure decode under a small
# KV cache, which is the regime the 25 tok/s target was written for.
SHORT_PROMPT = "請用繁體中文簡短說明巴黎為什麼是法國的首都。"


def ask(url, prompt, n_predict, timeout=600):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": 0,
    }, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.time()
    with opener.open(req, timeout=timeout) as r:
        raw = r.read()
    wall = time.time() - t0
    return json.loads(raw), wall


def extract(d):
    """Pull the server's own timing counters. Falls back to wall clock if absent."""
    t = d.get("timings") or {}
    out = {}
    for k_src, k_dst in (("prompt_n", "prompt_n"), ("prompt_ms", "prompt_ms"),
                         ("predicted_n", "predicted_n"), ("predicted_ms", "predicted_ms")):
        if k_src in t:
            out[k_dst] = t[k_src]
    n = out.get("predicted_n", 0)
    ms = out.get("predicted_ms", 0.0)
    out["decode_tps"] = (n / (ms / 1000.0)) if ms > 0 and n > 0 else 0.0
    pn = out.get("prompt_n", 0)
    pms = out.get("prompt_ms", 0.0)
    out["prefill_tps"] = (pn / (pms / 1000.0)) if pms > 0 and pn > 0 else 0.0
    if "choices" in d:
        out["text"] = d["choices"][0].get("message", {}).get("content", "")
    if "error" in d:
        out["error"] = str(d["error"])[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--n-predict", type=int, default=160)
    ap.add_argument("--tag", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--prompt", default=SHORT_PROMPT)
    args = ap.parse_args()

    rows = []
    for i in range(args.warmup + args.rounds):
        # Read the level on BOTH sides of the request. `before` is the one that carries the
        # claim -- it is the state the round was launched into; `after` shows whether the
        # round itself pushed the box past it (a 3-round arm can heat its own later rounds).
        th_before = tp.stamp()
        try:
            d, wall = ask(args.url, args.prompt, args.n_predict)
        except Exception as e:  # noqa: BLE001
            print(f"  round {i}: REQUEST FAILED {e}", file=sys.stderr)
            continue
        th_after = tp.stamp()
        m = extract(d)
        m["wall_s"] = round(wall, 2)
        m["round"] = i
        m["thermal_before"] = th_before
        m["thermal_after"] = th_after
        rows.append(m)
        kind = "warmup" if i < args.warmup else "round"
        if "error" in m:
            print(f"  [{kind} {i}] ERROR {m['error']}")
        else:
            print(f"  [{kind} {i}] decode {m['decode_tps']:6.2f} t/s  "
                  f"prefill {m['prefill_tps']:7.2f} t/s  "
                  f"(n={m.get('predicted_n')}, {m.get('predicted_ms', 0):.0f} ms)  "
                  f"thermal {th_before['label']}->{th_after['label']}")

    scored = rows[args.warmup:]
    tps = [r["decode_tps"] for r in scored if r.get("decode_tps")]
    pps = [r["prefill_tps"] for r in scored if r.get("prefill_tps")]
    texts = [r.get("text", "") for r in scored]
    result = {
        "tag": args.tag,
        "n_rounds": len(tps),
        "decode_tps_median": round(statistics.median(tps), 2) if tps else 0.0,
        "decode_tps_min": round(min(tps), 2) if tps else 0.0,
        "decode_tps_max": round(max(tps), 2) if tps else 0.0,
        "prefill_tps_median": round(statistics.median(pps), 2) if pps else 0.0,
        "n_tokens_sample": scored[0].get("predicted_n") if scored else 0,
        "answer_stable": len(set(texts)) == 1,
        "answer_md5_set": sorted({__import__("hashlib").md5(t.encode()).hexdigest()[:8]
                                  for t in texts}),
        "sample": texts[0][:160] if texts else "",
        # Per-round detail, because the aggregate hides the thing worth knowing. Measured
        # 2026-09-16: one arm under a uniformly NOMINAL reading still spanned 8.48 .. 20.32
        # t/s across three rounds -- so "median 20.32" and "min 8.48" are two facts about
        # the same server session, and only the per-round pairs say which round was which.
        "rounds": [{"round": r.get("round"),
                    "decode_tps": r.get("decode_tps"),
                    "prefill_tps": r.get("prefill_tps"),
                    "predicted_n": r.get("predicted_n"),
                    "predicted_ms": r.get("predicted_ms"),
                    "wall_s": r.get("wall_s"),
                    "thermal_before": r.get("thermal_before"),
                    "thermal_after": r.get("thermal_after")} for r in rows],
        # The level the run was LAUNCHED into, the worst seen anywhere in it, and the shape.
        # A median level would hide "3 rounds cold then 3 hot" behind "6 rounds warm", and
        # those are not the same experiment (lesson eng-mh-0036).
        "thermal_launch": rows[0].get("thermal_before") if rows else None,
        "thermal_worst": tp.worst(
            [r.get("thermal_before") for r in rows] + [r.get("thermal_after") for r in rows]),
        "thermal_hist": tp.histogram(
            [r.get("thermal_before") for r in rows] + [r.get("thermal_after") for r in rows]),
    }
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        prev = []
        if os.path.exists(args.json):
            try:
                prev = json.load(open(args.json))
            except Exception:  # noqa: BLE001
                prev = []
        prev.append(result)
        json.dump(prev, open(args.json, "w"), ensure_ascii=False, indent=2)
        print(f"\nappended -> {args.json}")


if __name__ == "__main__":
    main()
