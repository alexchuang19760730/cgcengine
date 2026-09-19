#!/usr/bin/env python3
"""
MTP 長序列 M1/M2/M3 驗證腳本

在完整的 MTP 生成過程中（100+ tokens）捕獲 oracle dump，
然後對比不同配置下的 logits/token 一致性。

用法:
  python3 scripts/check/mtp_long_sequence_verify.py --port 8080 --max-tokens 100 --output /tmp/oracle_long.jsonl
  python3 scripts/check/mtp_long_sequence_verify.py --compare ref_a.jsonl ref_b.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


# 長測試 prompt（確保生成足夠多的 tokens）
LONG_PROMPT = """請用繁體中文寫一篇關於人工智慧發展歷史的文章，至少 200 字。
內容要包含：
1. 人工智慧的起源
2. 機器學習的興起
3. 深度學習的突破
4. 大語言模型的革命
5. 未來的發展方向

請直接開始寫作，不要重複題目。"""


def send_chat_request(port, prompt, max_tokens=100, temperature=0.0, timeout=300.0):
    """發送 chat completion 請求"""
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
        return body
    except Exception as e:
        return {"error": str(e)}


def wait_for_server(port, timeout=120):
    """等待 server 就緒"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok" or data.get("ok"):
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def analyze_oracle_dump(dump_path):
    """分析 oracle dump 的內容"""
    if not os.path.exists(dump_path):
        return {"error": f"oracle dump not found: {dump_path}"}

    rows = []
    with open(dump_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {"error": "oracle dump is empty"}

    # 統計
    ctx_types = {}
    steps = set()
    n_tokens_dist = {}
    for r in rows:
        ct = r.get("ctx_type", "unknown")
        ctx_types[ct] = ctx_types.get(ct, 0) + 1
        steps.add(r.get("step", -1))
        nt = r.get("n_tokens", 0)
        n_tokens_dist[nt] = n_tokens_dist.get(nt, 0) + 1

    return {
        "total_rows": len(rows),
        "ctx_types": ctx_types,
        "step_range": [min(steps), max(steps)] if steps else [],
        "n_steps": len(steps),
        "n_tokens_distribution": n_tokens_dist,
        "first_5_rows": rows[:5],
        "last_5_rows": rows[-5:],
    }


def compare_oracle_dumps(path_a, path_b):
    """對比兩個 oracle dump 的 M1/M2/M3 一致性"""
    def load_rows(path):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    rows_a = load_rows(path_a)
    rows_b = load_rows(path_b)

    # 建立索引：(step, ctx_type, token_idx) -> row
    def index_rows(rows):
        idx = {}
        for r in rows:
            key = (r.get("step"), r.get("ctx_type"), r.get("token_idx"))
            idx[key] = r
        return idx

    idx_a = index_rows(rows_a)
    idx_b = index_rows(rows_b)

    common_keys = set(idx_a.keys()) & set(idx_b.keys())
    only_a = set(idx_a.keys()) - set(idx_b.keys())
    only_b = set(idx_b.keys()) - set(idx_a.keys())

    m1_equal = 0  # row_fnv1a64 相同
    m2_equal = 0  # argmax_token 相同
    m3_equal = 0  # top token set 相同
    total = 0

    diff_examples = []

    for key in sorted(common_keys):
        ra = idx_a[key]
        rb = idx_b[key]
        total += 1

        # M1: row_fnv1a64
        if ra.get("row_fnv1a64") == rb.get("row_fnv1a64"):
            m1_equal += 1

        # M2: argmax_token
        if ra.get("argmax_token") == rb.get("argmax_token"):
            m2_equal += 1
        else:
            if len(diff_examples) < 10:
                diff_examples.append({
                    "key": key,
                    "a_argmax": ra.get("argmax_token"),
                    "b_argmax": rb.get("argmax_token"),
                    "a_row_hash": ra.get("row_fnv1a64"),
                    "b_row_hash": rb.get("row_fnv1a64"),
                })

        # M3: top token set
        top_a = set(t.get("t") for t in ra.get("top", []))
        top_b = set(t.get("t") for t in rb.get("top", []))
        if top_a == top_b:
            m3_equal += 1

    return {
        "path_a": path_a,
        "path_b": path_b,
        "rows_a": len(rows_a),
        "rows_b": len(rows_b),
        "common_keys": len(common_keys),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "M1_numeric_identity": {
            "equal": m1_equal,
            "total": total,
            "rate": m1_equal / total if total > 0 else 0,
        },
        "M2_decision_agreement": {
            "equal": m2_equal,
            "total": total,
            "rate": m2_equal / total if total > 0 else 0,
        },
        "M3_topk_set_agreement": {
            "equal": m3_equal,
            "total": total,
            "rate": m3_equal / total if total > 0 else 0,
        },
        "diff_examples": diff_examples,
    }


def main():
    parser = argparse.ArgumentParser(description="MTP 長序列 M1/M2/M3 驗證")
    parser.add_argument("--port", type=int, default=8080, help="server port")
    parser.add_argument("--max-tokens", type=int, default=100, help="生成的最大 tokens 數")
    parser.add_argument("--output", type=str, default="/tmp/oracle_long_seq.jsonl", help="oracle dump 輸出路徑")
    parser.add_argument("--prompt", type=str, default=LONG_PROMPT, help="測試 prompt")
    parser.add_argument("--analyze", type=str, help="分析指定的 oracle dump 文件")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="對比兩個 oracle dump 文件")
    parser.add_argument("--launch", action="store_true", help="自動啟動 server（需要 run_server.sh）")
    parser.add_argument("--shutdown", action="store_true", help="測試完成後關閉 server")
    args = parser.parse_args()

    # 分析模式
    if args.analyze:
        result = analyze_oracle_dump(args.analyze)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # 對比模式
    if args.compare:
        result = compare_oracle_dumps(args.compare[0], args.compare[1])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # 啟動 server（可選）
    if args.launch:
        print(f"[launch] 啟動 server on port {args.port}...")
        env = os.environ.copy()
        env["CGC_LOGITS_ORACLE_DUMP"] = args.output
        env["CGC_LOGITS_ORACLE_TOPN"] = "8"
        env["CGC_LOGITS_ORACLE_FIRST_N"] = "0"  # unlimited

        # The shared box probe (docs/SERVER_WINDOW_LEDGER_2026-09-19.md): refuses to load a model
        # plus a pool into a busy box, because a number taken then describes the neighbour.
        # Self-contained import so this file needs no sys.path setup.
        import importlib.util as _ilu, os as _os
        _spec = _ilu.spec_from_file_location(
            "server_window", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                           "server_window.py"))
        _sw = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_sw)
        _sw.require_first(port=int(env.get("CGC_SERVER_PORT", "8080")),
                          need_mb=float(_os.environ.get("CGC_WINDOW_NEED_MB", "8000")),
                          where="mtp_long_sequence_verify")

        # start_new_session: survive the caller's process group (see run_server.sh's own --detach
        # and the cross-kill notes in docs/PROFILE_DUO_2026-09-18.md §5).
        proc = subprocess.Popen(
            ["./scripts/run_server.sh"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"[launch] server PID: {proc.pid}")

    # 等待 server 就緒
    print(f"[wait] 等待 server on port {args.port}...")
    if not wait_for_server(args.port):
        print("[error] server 未就緒，超時")
        sys.exit(1)
    print("[wait] server 已就緒")

    # 刪除舊的 oracle dump
    if os.path.exists(args.output):
        os.remove(args.output)
        print(f"[clean] 已刪除舊的 oracle dump: {args.output}")

    # 發送長請求
    print(f"[request] 發送長請求（max_tokens={args.max_tokens}）...")
    start_time = time.time()
    result = send_chat_request(args.port, args.prompt, max_tokens=args.max_tokens)
    elapsed = time.time() - start_time

    if "error" in result:
        print(f"[error] 請求失敗: {result['error']}")
        sys.exit(1)

    # 輸出生成結果
    content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = result.get("usage", {})
    timings = result.get("timings", {})

    print(f"\n[result] 生成完成（耗時 {elapsed:.1f}s）")
    print(f"  prompt_tokens: {usage.get('prompt_tokens', 0)}")
    print(f"  completion_tokens: {usage.get('completion_tokens', 0)}")
    print(f"  total_tokens: {usage.get('total_tokens', 0)}")
    print(f"  predicted_per_second: {timings.get('predicted_per_second', 0):.2f} t/s")
    print(f"  prompt_per_second: {timings.get('prompt_per_second', 0):.2f} t/s")
    print(f"  draft_n: {timings.get('draft_n', 0)}")
    print(f"  draft_n_accepted: {timings.get('draft_n_accepted', 0)}")
    print(f"\n  生成內容（前 200 字）:\n{content[:200]}")

    # 分析 oracle dump
    print(f"\n[oracle] 分析 oracle dump: {args.output}")
    analysis = analyze_oracle_dump(args.output)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))

    # 關閉 server（可選）
    if args.shutdown:
        print("[shutdown] 關閉 server...")
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{args.port}/shutdown", method="POST")
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass
        print("[shutdown] 已發送關閉信號")

    print("\n[done] 完成")


if __name__ == "__main__":
    main()
