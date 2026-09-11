#!/usr/bin/env python3
"""Deterministic 15+27 probe: N shots against the local OpenAI-compatible server.

Used to check small-pool quality (the 8GB-pool baseline answers 42 10/10; the 4GB pool used
to echo the prompt instead). Reports per-shot content + decode t/s, and a summary.

  python3 scripts/check/probe_skip0.py [--url http://127.0.0.1:8080] [--shots 10] [--prompt-en]
"""
import argparse
import json
import time
import urllib.request

PROMPTS = {
    "zh": "15+27 等於多少？",
    "en": "What is 15+27?",
}


def ask(url: str, prompt: str, timeout: float = 300.0, max_tokens: int = 64):
    body = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    dt = time.time() - t0
    ch = data["choices"][0]
    msg = ch.get("message", {})
    content = (msg.get("content") or "").strip()
    usage = data.get("usage", {})
    ct = usage.get("completion_tokens", 0) or 0
    return content, ct, dt, ch.get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--prompt-en", action="store_true")
    args = ap.parse_args()

    prompt = PROMPTS["en" if args.prompt_en else "zh"]
    print(f"prompt: {prompt!r}  shots={args.shots}", flush=True)
    outs, speeds = [], []
    for i in range(args.shots):
        try:
            content, ct, dt, finish = ask(args.url, prompt)
        except Exception as exc:  # noqa: BLE001 - probe tool
            print(f"  #{i + 1}: ERROR {exc}", flush=True)
            continue
        tps = ct / dt if dt > 0 else 0.0
        outs.append(content)
        speeds.append(tps)
        print(f"  #{i + 1}: {content!r} tokens={ct} {tps:.1f} t/s finish={finish}", flush=True)

    uniq = sorted(set(outs))
    ok = sum(1 for o in outs if "42" in o)
    print(f"\ncases={len(outs)} correct(42)={ok} unique_outputs={len(uniq)}")
    print(f"decode t/s: mean={sum(speeds) / len(speeds):.1f}" if speeds else "no samples")
    for u in uniq:
        print(f"   unique: {u!r}")


if __name__ == "__main__":
    main()
