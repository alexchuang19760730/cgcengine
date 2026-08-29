#!/usr/bin/env python3
# Copyright (c) 2025 SandAI. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see pd_server.py header).

"""
CGC 端雲 E2E 聯調測試（2026-08-29）— 真機最後一哩驗證

流程（dual 模式 = 端雲聯調本體）：
  1. GET 兩台機器的 /v1/cgc/profile → DeviceProfile + PDNode
  2. ComputeRouter.select(model, prompt_tokens, output_tokens) → 路由決策
  3. POST 被選中節點的 /v1/cgc/resume → SSE token 流
  4. 印出決策 + 實測延遲 + binary 回報的 decode t/s 對照（估算 vs 實測）
  5. --ab：兩台都跑一次做 A/B（驗證「送誰誰快」）

用法：
  solo（單機冒煙，Mac 上先驗 server 本身）：
    python3 pd_e2e_test.py --solo --url http://127.0.0.1:1234 \
        --model qwen36_35b -n 6 -p "The capital of France is"

  dual（真端雲聯調：local=Windows, remote=Mac）：
    python3 pd_e2e_test.py --local http://192.168.x.x:1234 \
        --remote http://192.168.y.y:1234 --model qwen25_7b -n 32 \
        --prompt-tokens 200 -p "..."
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse

sys.path.insert(0, __file__.rsplit("/", 1)[0] or ".")
from discovery import DeviceProfile, PDNode, NodeStatus  # noqa: E402
from router import ComputeRouter, MODEL_PRESETS  # noqa: E402


def http_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_node(url, tag):
    """GET /v1/cgc/profile → PDNode（HEALTHY，心跳時間=現在）。"""
    host = urlparse(url).hostname or "127.0.0.1"
    port = urlparse(url).port or 1234
    prof_raw = http_json(url + "/v1/cgc/profile")["profile"]
    prof = DeviceProfile(**{k: v for k, v in prof_raw.items()
                            if k in DeviceProfile.__dataclass_fields__})
    node = PDNode(node_id=f"{tag}-{host}", host=host, port=port,
                  status=NodeStatus.HEALTHY, profile=prof)
    return node


def stream_resume(url, prompt, max_tokens, seed=None):
    """POST /v1/cgc/resume → 逐 SSE event。回 (text_after_echo, summary)。"""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "seed": seed}).encode()
    req = urllib.request.Request(url + "/v1/cgc/resume", data=body,
                                 headers={"Content-Type": "application/json"})
    text, summary, t0 = [], None, time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            obj = json.loads(line[5:])
            ev = obj.pop("event", "")
            if ev == "token":
                text.append(obj.get("t", ""))
            elif ev == "summary":
                summary = obj
    # stdout = prompt 回顯 + 生成；把回顯剝掉（client 知道 prompt 內容）
    full = "".join(text)
    gen = full[len(prompt):] if full.startswith(prompt[:32]) else full
    return gen, summary, time.time() - t0


def est_tokens(prompt, chars_per_tok=3.5):
    return max(8, int(len(prompt) / chars_per_tok))


def main():
    ap = argparse.ArgumentParser(description="CGC 端雲 E2E 聯調")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--solo", action="store_true", help="單機冒煙（只測 server）")
    g.add_argument("--local", help="本地端 server URL（如 Windows）")
    ap.add_argument("--remote", help="遠端 server URL（如 Mac）")
    ap.add_argument("--url", help="solo 模式的 server URL")
    ap.add_argument("--model", default="qwen36_35b",
                    help=f"Router 模型名，可選: {sorted(MODEL_PRESETS)}")
    ap.add_argument("-p", "--prompt", default="The capital of France is")
    ap.add_argument("-n", "--max-tokens", type=int, default=16)
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="Router 用的 prompt token 數（預設按字元估算）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ab", action="store_true", help="兩台都跑做 A/B 對照")
    args = ap.parse_args()

    # ── solo：單機冒煙 ──
    if args.solo:
        url = args.url or "http://127.0.0.1:1234"
        h = http_json(url + "/v1/cgc/health")
        print(f"[solo] health: {h}")
        n = fetch_node(url, "solo")
        print(f"[solo] profile: {n.profile.__dict__}")
        gen, summ, wall = stream_resume(url, args.prompt, args.max_tokens, args.seed)
        print(f"[solo] gen ({args.max_tokens} tok, wall {wall:.1f}s): {gen[:120]!r}")
        print(f"[solo] binary perf: {summ}")
        return 0 if summ and summ.get("rc") == 0 else 1

    # ── dual：端雲聯調 ──
    if not (args.local and args.remote):
        print("dual 模式需要 --local 與 --remote", file=sys.stderr)
        return 2

    local_n = fetch_node(args.local, "local")
    remote_n = fetch_node(args.remote, "remote")
    print(f"[nodes] local  = {local_n.node_id}  score={0 if not local_n.profile else ''}")
    print(f"[nodes] remote = {remote_n.node_id}")

    router = ComputeRouter(local_node=local_n)
    router.register(remote_n)
    pt = args.prompt_tokens or est_tokens(args.prompt)
    d = router.select(prompt_tokens=pt, output_tokens=args.max_tokens, model=args.model)
    print("\n[decision]")
    for k, v in d.to_dict().items():
        print(f"  {k:14s}: {v}")
    print(f"  reason        : {d.reason}")

    # 被選中的 decode 節點 = 執行節點（文本橋：整段推理在 decode_node）
    chosen_url = args.local if d.decode_node == local_n else args.remote
    print(f"\n[exec] decode_node={d.decode_node.node_id} → {chosen_url}")
    gen, summ, wall = stream_resume(chosen_url, args.prompt, args.max_tokens, args.seed)
    print(f"[exec] gen: {gen[:120]!r}")
    print(f"[exec] wall {wall:.1f}s  binary perf: "
          f"decode_tps={summ.get('decode_tps')} rc={summ.get('rc')}")
    print(f"[exec] Router 估 decode {d.decode_latency_ms:.0f}ms vs 實測 {wall*1000:.0f}ms"
          f"（含模型重載 {summ.get('load_ms', '?')}ms）")

    if args.ab:
        other = args.remote if chosen_url == args.local else args.local
        print(f"\n[A/B] other node {other}")
        gen2, summ2, wall2 = stream_resume(other, args.prompt, args.max_tokens, args.seed)
        print(f"[A/B] gen: {gen2[:120]!r}")
        print(f"[A/B] wall {wall2:.1f}s  decode_tps={summ2.get('decode_tps')}")
        win = "chosen" if wall <= wall2 else "other"
        print(f"[A/B] verdict: {win} wins ({wall:.1f}s vs {wall2:.1f}s)")
    return 0 if summ and summ.get("rc") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
