#!/usr/bin/env python3
"""Live NIXL end-to-end byte-for-byte check (M1v2).

POSTs the SAME prompt to the cloud (emit) and the edge (resume) so their
(rank, step) handoff keys line up, then compares the generated text.
Correctness = edge output byte-identical to cloud output.
"""
import json
import sys
import urllib.request

CLOUD = "http://47.95.250.55:30001/v1/completions"
EDGE = "http://39.106.118.206:30000/v1/completions"

PROMPTS = [
    "France is",
    "Germany is",
]


def post(url, prompt, timeout=180):
    body = json.dumps({
        "model": "/data/models/DeepSeek-V4-Flash-UD-IQ2",
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    all_ok = True
    for p in PROMPTS:
        print(f"\n===== PROMPT: {p!r} =====")
        try:
            c = post(CLOUD, p)
            cloud_text = c["choices"][0]["text"]
        except Exception as e:
            print(f"[CLOUD] ERROR: {e!r}")
            all_ok = False
            continue
        print(f"[CLOUD] {cloud_text!r}")
        try:
            e = post(EDGE, p)
            edge_text = e["choices"][0]["text"]
        except Exception as ex:
            print(f"[EDGE]  ERROR: {ex!r}")
            all_ok = False
            continue
        print(f"[EDGE]  {edge_text!r}")
        match = (cloud_text == edge_text)
        print(f"[RESULT] {'BYTE-MATCH ✅' if match else 'MISMATCH ❌'}")
        if not match:
            all_ok = False
    print("\n===== SUMMARY =====")
    print("ALL BYTE-MATCH ✅" if all_ok else "SOME MISMATCH ❌")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
