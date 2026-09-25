#!/usr/bin/env python3
"""3a (CGC_ZERO_MISS) 正確性收益 —— 用 llama-completion 當載體的四臂文本比對。

為什麼換載體：llama-server 那條路實測 `CGC-ZEROMISS: applied=0 skipped=40`（vmask=0），
也就是 **server 的圖建構根本沒進 `cgc_slot_table_gpu` 分支，vmask 是 nullptr** ⇒ 3a 從未生效，
在 server 上跑四臂只能量到 SEG_BATCH 的 garbage，量不到 3a。llama-bench 上同一份 binary 是
applied=39 skipped=1，所以換 llama-completion（跟 bench 同一條圖建構路徑，而且**會印文字**）。

比對單位：沒有 token id 陣列（completion 不回傳），所以用**字元級**重合率 + 前綴剖面。
這比 id 陣列弱，但「garbage  degeneration」在字元級上一眼可辨，足以回答 3a 有沒有把輸出
從「垃圾權重」推向「已填專家的正確部分和」。

臂：
  A0  乾淨基線（分段提交 + 有 fill）                ⇒ 參考輸出
  B0  CGC_SEG_BATCH + B_SCHEME + SLOT_TABLE_GPU      ⇒ 無 fill，garbage
  B1  B0 + CGC_MISS_MASK + CGC_ZERO_MISS             ⇒ 無 fill，缺失項被置零
  A1  A0 + SLOT_TABLE_GPU + MISS_MASK + ZERO_MISS    ⇒ 有 fill 下的 3a（對照）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "src" / "llama.cpp" / "build" / "bin" / "llama-completion"
MODEL = "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"

BASE_ENV = {
    "LLAMA_EXPERT_CACHE_ALLOW_NGL": "1",
    "CGC_EXPERT_SKIP_READRAW": "1",
    "LLAMA_EXPERT_CACHE_WORKERS": "8",
    "CGC_N_CB": "8",
    "CGC_OA_ASYNC": "1",
    "CGC_GATHER_SLAB_CAP": "256",
    "CGC_DBUF": "1",
    "CGC_SPAC": "1",
    "CGC_SPAC_ALPHA": "0.75",
    "CGC_GLU_FUSED_DOWN": "1",
    "CGC_MM_BITIDENT": "1",
}

S1 = {"CGC_SLOT_TABLE_GPU": "1"}
SEG = {"CGC_SEG_BATCH": "1", "CGC_B_SCHEME": "1", "CGC_SLOT_TABLE_GPU": "1"}
MASK = {"CGC_MISS_MASK": "1"}
ZERO = {"CGC_ZERO_MISS": "1"}

ARMS = {
    "A0": {},
    "A1": {**S1, **MASK, **ZERO},
    "B0": dict(SEG),
    "B1": {**SEG, **MASK, **ZERO},
}

PROMPT = ("用條列方式說明快取置換策略的取捨，並比較 LRU 與成本感知淘汰在長序列推論下的差異。"
          "請具體舉例，不要只列名詞。")


def argv(n_predict: int, batch: int = 8) -> list[str]:
    # [2026-09-25] -b/-ub **必須是 8**，不能沿用交付 cell 的 5632：L4 pool 會把 n_batch cap 到
    #   CGC-PHASE-SPLIT: L4 pool capacity=143 -> n_batch 5632 capped to 8
    # 而 llama-completion 的 common_prompt_batch_decode 用 cparams.n_batch 當「一次餵多少」，
    # 整個 prompt 塞進一個 batch ⇒ llama-context.cpp:2511 GGML_ASSERT(n_tokens_all <= n_batch)
    # 直接 abort（實測：三臂 stdout 全 0 bytes）。llama-server 能跑是因為它自己分塊。
    # 給 -b 8 ⇒ common 按 8 分塊餵 prompt，assert 通過；decode 仍是 n_tokens=1，形狀不變。
    return [BIN.as_posix(), "-m", MODEL, "-ngl", "99", "--load-mode", "none",
            "-t", "8", "-c", "8192", "-expert-cache", "8589934592",
            "-b", str(batch), "-ub", str(batch), "-n", str(n_predict), "-s", "0",
            "--temp", "0", "--no-display-prompt", "-p", PROMPT]


def char_overlap(a: str, b: str) -> dict:
    if not a or not b:
        return {"rate": None, "n": 0}
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i] == b[i])
    return {"rate": round(same / n, 4), "n": n}


def prefix_profile(a: str, b: str, cuts=(8, 32, 128, 512)) -> dict:
    out = {}
    for c in cuts:
        n = min(c, len(a), len(b))
        if n:
            out[str(n)] = round(sum(1 for i in range(n) if a[i] == b[i]) / n, 4)
    return out


def run_arm(arm: str, n_predict: int, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(ARMS[arm])
    t0 = time.time()
    p = subprocess.run(argv(n_predict), env=env, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=3600)
    err = p.stderr
    (workdir / f"{arm}.stderr.log").write_text(err)
    (workdir / f"{arm}.stdout.txt").write_text(p.stdout)
    # [2026-09-25] 格式是 `CGC-ZEROMISS[<n>]: applied=%d skipped=%d`（每滿 40 層印一次）。
    # 取**最後一筆**：第一筆可能是 reserve/dry-run（vmask 未建）的假 0。
    zm = re.findall(r"CGC-ZEROMISS\[\d+\]: applied=(\d+) skipped=(\d+)", err)
    sk = re.findall(r"CGC-ZEROMISS: skipped il=\d+ \(vmask=(\d+) n_tokens=(\d+) "
                    r"weights_ne=\[(\d+),(\d+),(\d+)\] k=(\d+)\)", err)
    return {
        "arm": arm, "env": ARMS[arm], "rc": p.returncode, "secs": round(time.time() - t0, 1),
        "text": p.stdout.strip(), "n_chars": len(p.stdout.strip()),
        "zero_miss_applied": zm[-1] if zm else None,
        "zero_miss_all_dumps": zm,
        "zero_miss_skip_detail": sk[:2] if sk else None,
        "prewarm": (re.findall(r"prewarm req=(\d+) hit=(\d+) miss=(\d+)", err) or [None])[-1],
        "pool_hits": (re.findall(r"decode/pool .*?hits=(\d+)/(\d+)", err) or [None])[-1],
    }


def self_test() -> int:
    ok = 0

    def chk(name, cond):
        nonlocal ok
        ok += 1 if cond else 0
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    chk("四臂定義", set(ARMS) == {"A0", "A1", "B0", "B1"})
    chk("A0 乾淨", ARMS["A0"] == {})
    chk("B1 = B0 + mask + zero",
        all(ARMS["B1"].get(k) == v for k, v in ARMS["B0"].items())
        and ARMS["B1"].get("CGC_ZERO_MISS") == "1")
    chk("A1 帶 SLOT_TABLE_GPU", ARMS["A1"].get("CGC_SLOT_TABLE_GPU") == "1")
    chk("ALLOW_NGL 在 BASE_ENV", BASE_ENV.get("LLAMA_EXPERT_CACHE_ALLOW_NGL") == "1")
    chk("P0 SKIP_READRAW 在 BASE_ENV", BASE_ENV.get("CGC_EXPERT_SKIP_READRAW") == "1")
    chk("temp 0", "--temp" in argv(8) and argv(8)[argv(8).index("--temp") + 1] == "0")
    chk("char_overlap 全同", char_overlap("abc", "abc")["rate"] == 1.0)
    chk("char_overlap 空字串 None", char_overlap("", "abc")["rate"] is None)
    chk("prefix_profile 前綴", prefix_profile("abcdef", "abcxyz", cuts=(3, 6))["3"] == 1.0)
    chk("正則抓得到 CGC-ZEROMISS[<n>]",
        re.findall(r"CGC-ZEROMISS\[\d+\]: applied=(\d+) skipped=(\d+)",
                   "CGC-ZEROMISS[2]: applied=39 skipped=41 -- x") == [("39", "41")])
    chk("多筆時取最後一筆（第一筆是 reserve 的假 0）",
        re.findall(r"CGC-ZEROMISS\[\d+\]: applied=(\d+) skipped=(\d+)",
                   "CGC-ZEROMISS[1]: applied=0 skipped=40 -- a\n"
                   "CGC-ZEROMISS[2]: applied=39 skipped=41 -- b")[-1] == ("39", "41"))
    chk("binary 存在", BIN.exists())
    print(f"selftest {ok}/13")
    return 0 if ok == 13 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A0,B0,B1")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--workdir", default="Backup/zeromiss_completion")
    ap.add_argument("--json", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    wd = Path(args.workdir)
    res = {"n_predict": args.n_predict}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        print(f"[{arm}] running ...", flush=True)
        res[arm] = run_arm(arm, args.n_predict, wd)
        r = res[arm]
        print(f"[{arm}] rc={r['rc']} {r['secs']}s chars={r['n_chars']} "
              f"applied={r['zero_miss_applied']}", flush=True)
    a0 = res.get("A0", {}).get("text", "")
    for arm in ("A1", "B0", "B1"):
        if arm not in res:
            continue
        t = res[arm]["text"]
        res[arm]["overlap_vs_A0"] = char_overlap(a0, t) if a0 else None
        res[arm]["prefix_vs_A0"] = prefix_profile(a0, t) if a0 else None
    out = Path(args.json) if args.json else wd / "zeromiss_completion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            for k, v in prev.items():
                if k not in res:
                    res[k] = v
        except Exception as e:
            print(f"[warn] 合併失敗 {e}", file=sys.stderr)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n=== VERDICT ===")
    print(f"A0 chars={len(a0)}")
    for arm in ("A1", "B0", "B1"):
        if arm in res:
            print(f"{arm}: applied={res[arm]['zero_miss_applied']} "
                  f"overlap={res[arm].get('overlap_vs_A0')} prefix={res[arm].get('prefix_vs_A0')}")
            print(f"    text[:70]={res[arm]['text'][:70]!r}")
    print(f"\njson: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
