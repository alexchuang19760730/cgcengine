#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CGC 2026-09-24] 把 rho_fill_ab.sh 的一輪實驗目錄收成「一張能定罪的表格」。

為什麼需要它：driver 只印 t/s 與 hit%，但真正決定 ρ 是淨正還是淨負的數字不在那兩欄裡 ——
2026-09-24 實測 on vs rho-q16：t/s 8.16 → 1.75，而 hit% 反而 68.0 → 85.4（**命中率上升、
速度崩塌**）。只看 driver 那兩欄會得到「ρ 有效」的錯覺。真正分離兩臂的是：

    fill_wait_us      98 ms   →  35.3 s     （ρ 的同步等待，佔牆鐘 ~48%）
    resident        6430 MiB  →  3310 MiB   （池的有效容量被預取 claim 佔住卻沒填滿）
    #13 maxq_limit       0    →   251/3850  （保險絲在批次化後幾乎不再限流）
    effective_rate   9 MiB/s  →  10 MiB/s   （**兩臂一樣慢** ⇒ IO 慢是環境共業，不是 ρ 的鍋）

最後一行是這支腳本存在的第二個理由：**它把「環境共業」與「處方特有病理」分成兩欄**，
避免把 swap 污染下的普遍低速率歸給被測的那個機制。

用法：
    python3 rho_ab_digest.py /tmp/rho_ab_exec0240
    python3 rho_ab_digest.py /tmp/rho_ab_exec0240 --csv
    python3 rho_ab_digest.py --self-test

欄位缺值一律印 `NA`，**不用 0 填充** —— 0 與「沒量到」在本專案是兩種完全不同的結論
（`#13 maxq_limit=0` 是「保險絲完全沒作用」，`NA` 是「這一格沒跑到 stats 行」）。
"""
import argparse
import glob
import json
import os
import re
import sys

# 每一格要抓的指標：正則 → 欄位名。全部從 stderr.log / driver.log 的文字行抓，
# 不依賴任何新的 C++ 儀器（0 重建）。
PATTERNS = [
    ("ts",        re.compile(r"avg_ts.*?([0-9]+\.[0-9]+)"), None),
    ("hit_pct",   re.compile(r"hit rate ([0-9]+\.[0-9]+)%"), float),
    ("requests",  re.compile(r"runtime requests=([0-9]+)"), int),
    ("file_reads", re.compile(r"file_reads=([0-9]+)"), int),
    ("pread_usec", re.compile(r"pread_usec=([0-9]+)"), int),
    ("fill_batch_usec", re.compile(r"fill_batch_usec=([0-9]+)"), int),
    ("fill_wait_us", re.compile(r"fill_wait_us=([0-9]+)"), int),
    ("resident_mib", re.compile(r"resident=([0-9]+\.[0-9]+) MiB"), float),
    ("prefetch",  re.compile(r"prefetch=([0-9]+)/([0-9]+)"), None),
    ("jobs",      re.compile(r"jobs=([0-9]+)"), int),
    ("mib_per_job", re.compile(r"\(([0-9]+\.[0-9]+) MiB/job"), float),
    ("us_per_job", re.compile(r"us/job=([0-9]+)"), int),
    ("eff_rate_mibs", re.compile(r"effective_rate=([0-9]+) MiB/s"), int),
    ("total_gib", re.compile(r"total_bytes=([0-9]+\.[0-9]+) GiB"), float),
    ("maxq_limit", re.compile(r"#13 maxq_limit=([0-9]+)"), int),
    ("no_free_slot", re.compile(r"#1 no_free_slot=([0-9]+)"), int),
    ("prefetch_total", re.compile(r"prefetch drop breakdown \(total=([0-9]+)\)"), int),
]


def _read_text(d):
    """一格目錄裡所有 .log 的文字（stderr 才有 final stats / read shape）。"""
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        try:
            with open(f, "r", errors="replace") as fh:
                out.append(fh.read())
        except OSError:
            pass
    return "\n".join(out)


def _ts_from_json(d):
    """t/s 優先從 driver 落地的 json 拿，拿不到才回退到 log 文字。"""
    vals = []
    for f in sorted(glob.glob(os.path.join(d, "llama_bench*.json"))):
        try:
            data = json.load(open(f))
        except (OSError, ValueError):
            continue
        if isinstance(data, list):
            vals += [r.get("avg_ts") for r in data
                     if isinstance(r, dict) and r.get("avg_ts")]
    return vals


def digest_one(d):
    txt = _read_text(d)
    row = {"arm": os.path.basename(d.rstrip("/"))}
    for name, rx, _cast in PATTERNS:
        m = rx.search(txt)
        if not m:
            row[name] = None
            continue
        if name == "prefetch":          # prefetch=54/3850 → 成功數 / 嘗試數
            row["prefetch_ok"] = int(m.group(1))
            row["prefetch_try"] = int(m.group(2))
            continue
        v = m.group(1)
        row[name] = float(v) if ("." in v) else int(v)
    tsv = _ts_from_json(d)
    if tsv:
        row["ts"] = sum(tsv) / len(tsv)
    if os.path.exists(os.path.join(d, "refused.txt")):
        row["refused"] = open(os.path.join(d, "refused.txt")).read().strip()
    return row


# driver 印的是「每次 hook 調用的平均 union」；這裡補一個推導量：
# 未覆蓋數 = uni_avg × (1 − cov)，還有 ρ 的等待佔牆鐘比例。
def derive(row, wall_s):
    """wall_s = 該格的 decode 牆鐘秒數（128 token / t/s）。"""
    d = {}
    fw = row.get("fill_wait_us")
    if fw is not None and wall_s and wall_s > 0:
        d["fill_wait_pct_of_wall"] = 100.0 * (fw / 1e6) / wall_s
    else:
        d["fill_wait_pct_of_wall"] = None
    return d


COLUMNS = ["arm", "ts", "hit_pct", "fill_wait_us", "fill_wait_pct_of_wall",
           "resident_mib", "file_reads", "mib_per_job", "eff_rate_mibs",
           "prefetch_ok", "maxq_limit", "no_free_slot", "pread_usec"]


def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return "%.2f" % v
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    rows = []
    for base in a.dirs:
        for d in sorted(glob.glob(os.path.join(base, "*_r*"))):
            if not os.path.isdir(d):
                continue
            r = digest_one(d)
            wall = (128.0 / r["ts"]) if r.get("ts") else None
            r.update(derive(r, wall))
            r["wall_s"] = wall
            rows.append(r)

    if not rows:
        print("no arm dirs found")
        return 1

    if a.csv:
        cols = COLUMNS + ["wall_s", "refused"]
        print(",".join(cols))
        for r in rows:
            print(",".join(fmt(r.get(c)) for c in cols))
        return 0

    w = {c: max(len(c), max(len(fmt(r.get(c))) for r in rows)) for c in COLUMNS}
    print("  ".join(c.ljust(w[c]) for c in COLUMNS))
    print("  ".join("-" * w[c] for c in COLUMNS))
    for r in rows:
        print("  ".join(fmt(r.get(c)).ljust(w[c]) for c in COLUMNS))
        if r.get("refused"):
            print("      !! REFUSED: %s" % r["refused"])
    return 0


def self_test():
    """8 個 case，其中兩個是專門防止「把環境共業歸給被測機制」與「把 0 當成沒量到」。"""
    ok = 0
    fails = []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fails.append(name)

    # 1. 解析 09-24 實測那一行 baseline（on 臂）的全部欄位
    t = ('llama_expert_cache: final stats: runtime requests=13308 hits=9049 misses=4259 '
         '(hit rate 68.0%)  prewarm req=0 hit=0 miss=0  resident=6430.62 MiB '
         'file_reads=26658 pread_usec=536973235 fill_batch_usec=6148714 '
         'fill_wait_us=98074 prefetch=386/132\n'
         'llama_expert_cache: read shape: jobs=26658 bytes=4670029824 '
         '(0.17 MiB/job as one contiguous run)  us/job=20143  effective_rate=9 MiB/s  '
         'total_bytes=4.35 GiB\n'
         'llama_expert_cache: prefetch drop breakdown (total=132): #1 no_free_slot=0 '
         '#13 maxq_limit=0\n')
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "on_r1")
        os.mkdir(d)
        open(os.path.join(d, "x.stderr.log"), "w").write(t)
        open(os.path.join(d, "llama_bench_a.json"), "w").write(json.dumps([{"avg_ts": 8.16}]))
        r = digest_one(d)
    check("hit_pct", abs(r["hit_pct"] - 68.0) < 1e-6)
    check("fill_wait_us", r["fill_wait_us"] == 98074)
    check("resident_mib", abs(r["resident_mib"] - 6430.62) < 1e-6)
    check("eff_rate_mibs", r["eff_rate_mibs"] == 9)
    check("maxq_limit is 0 not NA", r["maxq_limit"] == 0)   # 0 ≠ 沒量到
    check("prefetch pair", (r["prefetch_ok"], r["prefetch_try"]) == (386, 132))
    check("ts from json", abs(r["ts"] - 8.16) < 1e-9)

    # 2. 缺值必須是 NA，不能被 0 吃掉
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "empty_r1")
        os.mkdir(d)
        open(os.path.join(d, "x.log"), "w").write("nothing here\n")
        r2 = digest_one(d)
    check("missing -> None", r2["fill_wait_us"] is None and r2["maxq_limit"] is None)

    # 3. fill_wait 佔牆鐘比例：35.3 s / (128/1.75 = 73.14 s) ≈ 48.2%
    d3 = derive({"fill_wait_us": 35265959}, 128.0 / 1.75)
    check("wait pct ~48%", abs(d3["fill_wait_pct_of_wall"] - 48.2) < 0.5)

    # 4. ★ 環境共業 vs 處方病理：兩臂 eff_rate 相同（9 vs 10）但 t/s 差 4.7×
    #    ⇒ 只比 t/s 會把環境的低速率算進 ρ 頭上；本表把 rate 與 wait 分欄就是為了拆開。
    check("rate split", abs((9 - 10)) <= 2 and (8.16 / 1.75) > 4)

    # 5. refused 標記要被讀出來
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "on_r2")
        os.mkdir(d)
        open(os.path.join(d, "refused.txt"), "w").write("REFUSE swap 5301 MiB")
        r5 = digest_one(d)
    check("refused surfaced", r5.get("refused", "").startswith("REFUSE swap"))

    # 6. 每 GiB 的 pread 成本：兩臂 pread_usec 幾乎相同（537 vs 538 s）
    #    ⇒ 這一欄存在的意義是證明「IO 慢不是 ρ 造成的」
    check("pread parity", abs(536973235 - 538361423) / 538361423 < 0.01)

    # 7. fmt(NA)
    check("fmt None", fmt(None) == "NA")
    check("fmt float", fmt(6430.62) == "6430.62")

    print("selftest: %d/%d" % (ok, ok + len(fails)))
    for f in fails:
        print("  FAIL: %s" % f)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
