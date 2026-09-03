#!/usr/bin/env python3
import argparse
import json
import sys
from urllib import request as urlrequest


LONGFORM_STOP = ["<|end|>", "<|output|>", "<|user|>"]


def http_json(url, payload, timeout):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_payload(profile, model, max_tokens):
    if profile == "qa-zh":
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "巴黎是哪個國家的首都？請只用一句中文回答。",
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens or 24,
            "stop": ["。", "<|end|>", "<|output|>", "<|user|>"],
        }

    if profile == "longform-zh":
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "為什麼巴黎會成為法國的政治與文化中心？請用一段中文說明。",
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens or 220,
            # Keep explicit stops on the request path so replay stays aligned with the
            # launcher hint and does not leak template delimiters back into content.
            "stop": LONGFORM_STOP,
        }

    raise ValueError(f"unsupported profile: {profile}")


def parse_args():
    parser = argparse.ArgumentParser(description="Replay fixed profile payloads against llama-server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--profile", choices=["qa-zh", "longform-zh"], required=True)
    parser.add_argument("--model", default="test")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--print-payload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_payload(args.profile, args.model, args.max_tokens)
    if args.print_payload:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    url = args.base_url.rstrip("/") + "/chat/completions"
    resp = http_json(url, payload, args.timeout)

    content = resp["choices"][0]["message"]["content"]
    finish_reason = resp["choices"][0]["finish_reason"]
    timings = resp.get("timings", {})
    predicted_n = timings.get("predicted_n")
    decode_tps = timings.get("predicted_per_second")
    prompt_tps = timings.get("prompt_per_second")
    draft_n = timings.get("draft_n")
    draft_n_accepted = timings.get("draft_n_accepted")
    accept_pct = None
    if isinstance(draft_n, (int, float)) and draft_n:
        accept_pct = (float(draft_n_accepted or 0) / float(draft_n)) * 100.0

    print(json.dumps(
        {
            "profile": args.profile,
            "finish_reason": finish_reason,
            "decode_tps": decode_tps,
            "prompt_tps": prompt_tps,
            "completion_tokens": predicted_n,
            "draft_accept_pct": accept_pct,
            "content": content,
            "timings": {
                "prompt_ms": timings.get("prompt_ms"),
                "predicted_ms": timings.get("predicted_ms"),
                "predicted_per_second": decode_tps,
                "prompt_per_second": prompt_tps,
                "predicted_n": predicted_n,
                "draft_n": draft_n,
                "draft_n_accepted": draft_n_accepted,
            },
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
