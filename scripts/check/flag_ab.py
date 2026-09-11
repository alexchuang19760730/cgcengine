#!/usr/bin/env python3
"""A/B any two CGC server flags against a fixed probe set -- "should we keep this on?".

WHY THIS EXISTS
---------------
CGC_STRIP_SCAFFOLD and CGC_LOOP_GUARD are both ON, and both look reasonable in code, but a
flag is only worth keeping if it (a) actually fires on the failure it targets and (b) does not
damage output that is already correct. Neither can be judged by reading the diff, and a single
question cannot judge either, because a mechanism that never fires and a mechanism that fixes
everything both look like "the output changed" if you only look at one prompt.

So this runs the SAME probe set under each flag combo and reports, per probe:
  * markers  - did scaffold text (<think>, </think>, " thinking") survive into content?
  * loop     - does the content contain a >=6-char unit repeated >=3x consecutively?
  * echo     - did the model repeat the prompt instead of answering?
  * expect   - did the expected token appear (this is the regression check: a stripper that
               eats legitimate answers shows up as expect=N here, not as markers=0)
  * finish   - stop vs length

The probe set is deliberately split between a failure to fix and output that must NOT change:

  math_zh      the bare-prompt echo/loop trigger
  en_short     a normally-answering English question (must not regress)
  code_marker  asks for code that legitimately CONTAINS "<think>" and "</think>" literals.
               This is the trap for Pass 1's "remove everything between <think> and </think>":
               a stripper that cannot tell scaffold from content will mangle a correct answer.

Usage
-----
    # does the loop guard matter? (embedded template path, which is the one that loops)
    python3 scripts/check/flag_ab.py \
        --base-env CGC_SERVER_PROFILE=qa-zh,CGC_SERVER_CHAT_AB=off \
        --combo CGC_LOOP_GUARD=0,CGC_STRIP_SCAFFOLD=1 \
        --combo CGC_LOOP_GUARD=1,CGC_STRIP_SCAFFOLD=1

    # does scaffold stripping matter, and does it break the code_marker probe?
    python3 scripts/check/flag_ab.py \
        --base-env CGC_SERVER_PROFILE=qa-zh,CGC_SERVER_CHAT_AB=off \
        --combo CGC_LOOP_GUARD=1,CGC_STRIP_SCAFFOLD=0 \
        --combo CGC_LOOP_GUARD=1,CGC_STRIP_SCAFFOLD=1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import knifeedge_matrix as km  # noqa: E402  (launch/wait/health live there already)

RESULT_DIR = os.path.join(km.ROOT, "Backup", "flag_ab")

MARKER_RE = re.compile(r"<think>|</think>| thinking| response|&lt;think&gt;")

# (name, prompt, expected substring that proves the answer was NOT damaged)
PROBES = [
    ("math_zh", "15+27 等於多少？請只輸出答案", "42"),
    ("en_short", "What is 15+27? Answer with the number only.", "42"),
    ("code_marker",
     "用 Python 寫一個函式 strip_think(s)，移除字串中的 <think> 與 </think> 標籤，"
     "回傳清理後的字串。只輸出程式碼。", "strip_think"),
]


def detect_loop(text, window=256, min_unit=6, min_repeat=3, max_unit=300):
    """Same rule as server-context.cpp cgc_detect_phrase_loop, applied to the whole text."""
    def find(t):
        n = len(t)
        if n < min_unit * min_repeat:
            return False
        max_len = min(max_unit, n // min_repeat)
        for p in range(min_unit, max_len + 1):
            for i in range(0, n - p * min_repeat + 1):
                cnt = 1
                j = i + p
                while j + p <= n and t[j:j + p] == t[i:i + p]:
                    cnt += 1
                    j += p
                    if cnt >= min_repeat:
                        return True
        return False

    start = max(0, len(text) - window)
    if find(text[start:]):
        return True
    stripped = re.sub(r"[ \t\r\n]", "", text[start:])
    return find(stripped)


def ask(port, prompt, max_tokens, timeout):
    payload = {"model": "local", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": max_tokens}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                data=json.dumps(payload).encode("utf-8"),
                                headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001 - harness
        return {"content": "", "finish": f"error:{str(e)[:60]}", "elapsed": round(time.time() - t0, 1)}
    ch = (body.get("choices") or [{}])[0]
    return {"content": (ch.get("message", {}).get("content") or ""),
            "finish": ch.get("finish_reason", "?"),
            "elapsed": round(time.time() - t0, 1),
            "completion_tokens": (body.get("usage") or {}).get("completion_tokens")}


def analyse(content, prompt, expect):
    norm = re.sub(r"\s+", "", content)
    return {
        "markers": bool(MARKER_RE.search(content)),
        "loop": detect_loop(content) if content else False,
        "echo": bool(norm) and (re.sub(r"\s+", "", prompt) in norm or norm.count(prompt[:8]) > 1),
        "expect": expect in content,
        "len": len(content),
    }


def run_combo(combo, args):
    label = ",".join(f"{k.replace('CGC_', '')}={v}" for k, v in combo.items()) or "base"
    print(f"\n===== [{label}] launching =====", flush=True)
    km.kill_servers()
    log = os.path.join(RESULT_DIR, f"launch_{label.replace(',', '_')}.log")
    km.launch(km.MODELS[args.model]["file"], km.MODELS[args.model]["mtp"],
              args.pool * 1024 ** 3, args.port, args.kwargs, args.base_env + [f"{k}={v}" for k, v in combo.items()],
              log)
    took = km.wait_health(args.port, args.health_timeout)
    if took is None:
        print(f"[{label}] FAILED to come up", flush=True)
        km.kill_servers()
        return {"label": label, "combo": combo, "up": False}
    facts = km.read_launch_facts(log)
    print(f"[{label}] up in {took:.0f}s  rss={km.rss_gb()}GB  facts={facts}", flush=True)

    rows = []
    for name, prompt, expect in PROBES:
        r = ask(args.port, prompt, args.max_tokens, args.timeout)
        a = analyse(r["content"], prompt, expect)
        rows.append({"probe": name, "expect_str": expect, **r, **a})
        print(f"  {name:12s} markers={int(a['markers'])} loop={int(a['loop'])} "
              f"echo={int(a['echo'])} expect={int(a['expect'])} finish={r['finish']:7s} "
              f"len={a['len']:<4d} {r['content'][:70]!r}", flush=True)
    km.kill_servers()
    rec = {"label": label, "combo": combo, "up": True, "facts": facts,
           "rss_gb": km.rss_gb(), "rows": rows}
    json.dump(rec, open(os.path.join(RESULT_DIR, f"combo_{label.replace(',', '_')}.json"),
                        "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return rec


def print_table(records):
    print("\n" + "=" * 100)
    print("%-46s %-11s %7s %5s %5s %7s %-7s" %
          ("combo", "probe", "markers", "loop", "echo", "expect", "finish"))
    print("-" * 100)
    for rec in records:
        if not rec.get("up"):
            print("%-46s  -- server did not come up" % rec["label"])
            continue
        for r in rec["rows"]:
            print("%-46s %-11s %7d %5d %5d %7d %-7s" %
                  (rec["label"][:46], r["probe"], r["markers"], r["loop"], r["echo"],
                   r["expect"], r["finish"]))
    print("=" * 100)
    print("expect=1 means the answer survived (regression check). A flag is only worth keeping")
    print("if it turns a markers/loop/echo 1 into 0 WITHOUT turning any expect 1 into 0.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", action="append", required=True,
                    help="KEY=VAL[,KEY=VAL] (repeatable). Keys are full env names, e.g. "
                         "CGC_LOOP_GUARD=1,CGC_STRIP_SCAFFOLD=0")
    ap.add_argument("--base-env", default="",
                    help="comma list applied to every combo, e.g. "
                         "CGC_SERVER_PROFILE=qa-zh,CGC_SERVER_CHAT_AB=off")
    ap.add_argument("--model", default="iq3", choices=sorted(km.MODELS))
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--health-timeout", type=float, default=300.0)
    args = ap.parse_args()

    args.base_env = [kv for kv in (s.strip() for s in args.base_env.split(",")) if kv]

    servers = km.running_servers()
    if servers:
        print(f"killing {len(servers)} pre-existing llama-server(s): {servers}", flush=True)
        km.kill_servers()

    os.makedirs(RESULT_DIR, exist_ok=True)
    combos = []
    for spec in args.combo:
        combos.append({kv.split("=", 1)[0].strip(): kv.split("=", 1)[1].strip()
                       for kv in spec.split(",") if kv.strip()})
    print(f"base-env: {args.base_env}\ncombos  : {combos}\nprobes  : {[p[0] for p in PROBES]}")

    records = [run_combo(c, args) for c in combos]
    json.dump(records, open(os.path.join(RESULT_DIR, "ab.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print_table(records)
    print(f"\nwrote {os.path.join(RESULT_DIR, 'ab.json')}")


if __name__ == "__main__":
    main()
