#!/usr/bin/env python3
"""[CGC 2026-10-06] 輸出健康度 = 自動化的「正確性信號」。

動機：detok 人眼看到 W2 真文字會退化成正規**重複迴圈**（`在中在中在中…`、`，，，，…`）。
那是**正確性**問題，不只是效率：退化會**抬高接受率 ⇒ 抬高 emit**，
所以「快」可能是壞掉的產物。打平判詞必須拒收這種臂。

來源：`LLAMA_BENCH_SPEC_DUMP=1` ⇒ llama-bench 印一行
      `CGC-SPEC-DUMP phase=timed n=N ids=a,b,c,…`
（只有 spec 路徑會 dump；off 臂沒有 ⇒ 這是已知的見證缺口，見檔尾。）

三個指標全在 **token id 序列**上算 —— 不需要 vocab/detok ⇒ 可離線、決定性、無依賴：
  run_max  最長「連續同一個 id」的長度                        （token 迴圈的直觀簽名）
  selfsim  在 n=1..W 內最大的 mean_i[ ids[i] == ids[i-n] ]     （短週期自相似）
  rep_k    (1 - distinct k-gram / k-gram 數)                  （片語重複；**只當診斷，不當閘**）

判定：
  DEGENERATE ⇔ run_max >= RUN_MAX_MAX 或 selfsim >= SELFSIM_MAX
  OK         ⇔ 兩者都沒過
  NO_DATA    ⇔ 沒有 dump ⇒ **fail-closed**（沒有正確性見證就不能判）

**逐 dump 取最壞（2026-10-06 修）**：一趟 arm 每個 rep 印一條 dump（5 reps ⇒ 5 條）。
原本只讀**最後一條**，於是 rep 0 的 101 連發（`sp_mtpon_on_s05` 實測）會被最後一條
（run_max=10）蓋掉一半、更健康的 rep 更能整個蓋掉。現在 `health_from_stderr` 對**每一條**
各判一次、回傳最壞的那一判 ⇒ 少一條健康 dump 不會翻案，多一條退化 dump 一定擋下。

門檻標定見 `scripts/check/output_health_calibrate.py`（2026-10-06 的 19 臂）。
"""
from __future__ import annotations

import re

# ── 門檻（由 calibrate 腳本標定；改這裡＝改判準）────────────────────────────────
RUN_MAX_MAX = 5      # ≥5 個連續相同 token ⇒ 退化
SELFSIM_MAX = 0.25   # 週期自相似 ≥0.25 ⇒ 退化
SELFSIM_W   = 8      # selfsim 掃的週期上限
DUP_K       = 8      # rep_k 的 k（診斷用）

_RE_DUMP = re.compile(r"CGC-SPEC-DUMP\s+phase=(\S+)\s+n=(\d+)\s+ids=([0-9,\-]+)")


# ── 解析 ─────────────────────────────────────────────────────────────────────────
def parse_dump_all(text: str, phase: str = "timed") -> list[list[int]]:
    """回傳該 phase 的**每一條** dump（依檔案順序 = 逐 rep）。找不到 ⇒ 空 list（不猜）。"""
    out = []
    for m in _RE_DUMP.finditer(text):
        if m.group(1) != phase:
            continue
        out.append([int(x) for x in m.group(3).split(",")])
    return out


def parse_dump(text: str, phase: str = "timed"):
    """最後一條 dump 的 id 序列（單趟語意；逐 rep 判讀請用 parse_dump_all）。
    找不到回 None（不猜、不回空 list）。"""
    best = None
    for ids in parse_dump_all(text, phase):
        best = ids
    return best


# ── 指標（純函式）────────────────────────────────────────────────────────────────
def run_max(ids) -> int:
    """最長連續相同 id 的長度（空序列 ⇒ 0）。"""
    if not ids:
        return 0
    best = cur = 1
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i - 1] else 1
        if cur > best:
            best = cur
    return best


def selfsim(ids, W: int = SELFSIM_W):
    """回傳 (最大值, 達到該值的週期 n)。n 從 1 掃到 W。"""
    if len(ids) < 2:
        return 0.0, 0
    best, bn = 0.0, 0
    for n in range(1, min(W, len(ids) - 1) + 1):
        v = sum(1 for i in range(n, len(ids)) if ids[i] == ids[i - n]) / (len(ids) - n)
        if v > best:
            best, bn = v, n
    return best, bn


def rep_frac(ids, k: int = DUP_K) -> float:
    """k-gram 的型例比倒數（1 - distinct/total）。診斷用，不當閘。"""
    if len(ids) < k + 1:
        return 0.0
    g = [tuple(ids[i:i + k]) for i in range(len(ids) - k + 1)]
    return 1.0 - len(set(g)) / len(g)


def metrics(ids) -> dict:
    s, sn = selfsim(ids)
    return {"n": len(ids), "run_max": run_max(ids),
            "selfsim": round(s, 4), "selfsim_n": sn,
            "rep_k": round(rep_frac(ids), 4), "k": DUP_K,
            "run_max_max": RUN_MAX_MAX, "selfsim_max": SELFSIM_MAX}


# ── 判定 ─────────────────────────────────────────────────────────────────────────
def health(ids) -> dict:
    """ids=None ⇒ NO_DATA（fail-closed）。否則 OK / DEGENERATE ＋ 理由。"""
    if ids is None:
        return {"state": "NO_DATA", "reasons": ["沒有 CGC-SPEC-DUMP ⇒ 沒有正確性見證"],
                "metrics": None}
    m = metrics(ids)
    reasons = []
    if m["run_max"] >= RUN_MAX_MAX:
        reasons.append(f"run_max={m['run_max']} ≥ {RUN_MAX_MAX}（連續相同 token）")
    if m["selfsim"] >= SELFSIM_MAX:
        reasons.append(f"selfsim={m['selfsim']:.3f} (n={m['selfsim_n']}) ≥ {SELFSIM_MAX}"
                       f"（短週期重複迴圈）")
    return {"state": "DEGENERATE" if reasons else "OK", "reasons": reasons, "metrics": m}


def health_all(dumps: list) -> dict:
    """逐 dump 判、回傳**最壞**的那一判（代表整趟 arm）。空 ⇒ NO_DATA。
    `metrics["per_dump"]` = 每個 rep 的 {run, state, metrics}，供判詞逐 rep 引用。"""
    if not dumps:
        return health(None)
    per = []
    worst = None
    worst_key = -1.0
    for i, ids in enumerate(dumps):
        h = health(ids)
        per.append({"run": i, "state": h["state"], "metrics": h["metrics"]})
        m = h["metrics"]
        # 嚴重度：兩支指標各自對門檻的倍率取大者（退化中「最壞」的那一條才是判詞要引的）
        key = max(m["run_max"] / m["run_max_max"], m["selfsim"] / m["selfsim_max"])
        if h["state"] == "DEGENERATE" and (worst is None or key > worst_key):
            worst = h
            worst_key = key
            worst_run = i
    if worst is None:
        worst = health(dumps[-1])       # 全 OK：判詞引最後一條即可
    out = dict(worst)
    if out["state"] == "DEGENERATE":
        # 理由要指明是哪一個 rep（否則「取最壞」會讓理由與最後一條對不上）
        out["reasons"] = [f"rep {worst_run}: {r}" for r in out["reasons"]] + \
                         [f"（逐 dump 取最壞；本趟 {len(dumps)} 條 dump："
                          + ",".join(p["state"] for p in per) + "）"]
    out["metrics"] = dict(out["metrics"])
    out["metrics"]["per_dump"] = per
    out["metrics"]["n_dumps"] = len(dumps)
    return out


def health_from_stderr(text: str, phase: str = "timed"):
    """從一份 stderr 全文取判：**逐條 dump 取最壞**。沒有 dump ⇒ NO_DATA（fail-closed）。"""
    return health_all(parse_dump_all(text, phase))


def summary_line(h: dict) -> str:
    """判詞用的一行。"""
    m = h.get("metrics")
    if h["state"] == "NO_DATA":
        return "輸出健康度 NO_DATA（沒有 dump ⇒ 無正確性見證）"
    head = {"OK": "✅ OK", "DEGENERATE": "⛔ DEGENERATE"}[h["state"]]
    runs = f" [{m['n_dumps']} dumps: " + ",".join(p["state"] for p in m["per_dump"]) + "]" \
        if m.get("per_dump") else ""
    return (f"{head} n={m['n']} run_max={m['run_max']}/{m['run_max_max']} "
            f"selfsim={m['selfsim']:.3f}/{m['selfsim_max']} (n={m['selfsim_n']}) "
            f"rep{m['k']}={m['rep_k']:.3f}{runs}"
            + ("  ← " + "; ".join(h["reasons"]) if h["reasons"] else ""))


# ── 已知見證缺口（寫在這裡，避免下一個人以為它被蓋住了）─────────────────────────
# ⛔ off 臂（純解碼）**不會** dump ⇒ 只有 on 臂（spec 路徑）有正確性見證。
#    2026-10-06 更正：「在 test_gen 補一條同樣格式的 dump」**做不到**，因為 llama-bench 的
#    純解碼迴圈（`llama-bench.cpp` `test_gen`）**根本不取樣** —— 它用 `std::rand() % n_vocab`
#    造下一個 token 餵進去，logits 從頭到尾沒被讀。在那裡 dump 只會存到**輸入流**（亂數），
#    不是輸出 ⇒ 不是見證。真正的兩條路：(a) 對純解碼臂加一個 env（例：`CGC_BENCH_SAMPLE=1`）
#    讓它像 spec 路徑一樣取樣後 dump（＝另一種 cell，量到的是「有取樣的 off 臂」）；
#    (b) 用 server 路徑跑 off 臂並在 server 端 dump。在那之前，本閘只認證 on 臂那一半。
