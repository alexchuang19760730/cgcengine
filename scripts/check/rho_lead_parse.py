#!/usr/bin/env python3
"""rho_lead_parse.py -- 從 CGC-NSCB 行量「影子 router 比真實路由早多少」（lead）。

為什麼要這個檔案
───────────────
ρ 的準度（`cov_uni`，`rho_probe_parse.py`）與 ρ 的**提前量**是兩個獨立的量。準度在上界公式裡
決定「能藏掉幾次 fill」，提前量決定「那些 fill 有多少時間可以藏」。2026-09-23 14:58 那一輪只
量到前者：影子 node 建在 attn(L) 之後 ⇒ 提前量 ≈ 0 ⇒ 從那支 log 讀不出任何 lead。
這一版用 `CGC_GPU_NODES_START=1` 印出來的 per-command-buffer **絕對 GPU 時間戳**把後者補上。

讀的是什麼
──────────
`CGC-NSCB step=<s> a=<first node> b=<last node> start_ns=<s> end_ns=<e> nm=<names>`
  * 一個 segment（≈ 一層）會連續印 n_cb+1 條；**主 buffer（a=0）印在該 group 的最後**
    （`ggml_metal_cgc_gpu_take_cb` 的 slot `i == n_cb` 才是 main thread 那顆）。
    ⇒ 分 group 的規則：遇到 `a=0` 就收尾。
  * 一個 buffer 覆蓋 [a, b) 這些 node，`nm=` 是它們全部的名字（>8 個會被截斷並以 `,...` 結尾）。
  * `start_ns/end_ns` 是 `GPUStartTime/GPUEndTime` × 1e9，**絕對**時間 ⇒ 同一次 run 內可相減。

lead 的定義（給區間，不給單點）
──────────────────────────────
真實路由 = `ffn_moe_argsort-<L>`（真正在算 expert id 的那顆；`ffn_moe_topk-*` 是 VIEW，
見 ggml-backend.cpp 的註解）。影子 = `cgc_rho_logits-<L>`。
buffer 有 5 個 node 寬（可用 n_cb 上限 ≈ 16），所以我們只知道「該 node 落在這個 buffer 裡」：

    lead_lower = argsort_buf.start - rho_buf.end      （最悲觀：影子到最後才好、路由最早就好）
    lead_upper = argsort_buf.end   - rho_buf.start    （最樂觀）

⇒ 報區間 + 中位數。若 lead_lower <= 0 而 lead_upper > 0，結論是「量不出正的提前量」。

用法
────
    python3 scripts/check/rho_lead_parse.py <log> [--tail-frac 0.5] [--verbose]
    python3 scripts/check/rho_lead_parse.py --self-test

產生該 log 的那一趟（**不可引用 t/s**：圖裡多一個 matmul，CGC_TD_CB 序列化 pipeline，
CGC_N_CB/CGC_CB_N_MAIN 又改了 command buffer 形狀）：
    CGC_SERVER_N_CB=16 CGC_CB_N_MAIN=1 CGC_GPU_NODES=1 CGC_GPU_NODES_START=1 \
        ./scripts/check/prebind_probe_run.sh                      # early
    CGC_RHO_PROBE_LATE=1 ...(同上)                                # late 對照臂
"""
import argparse
import re
import statistics
import sys

NSCB = re.compile(
    r"^CGC-NSCB step=(\d+) a=(\d+) b=(\d+) start_ns=(\d+) end_ns=(\d+) nm=(.*)$")

RHO = "cgc_rho_logits"
ARG = "ffn_moe_argsort"


# ── 解析 ────────────────────────────────────────────────────────────────────
def parse_segments(lines):
    """把 NSCB 行切成一個個 segment。回傳 list[dict]。"""
    segs = []
    cur = []
    for ln in lines:
        m = NSCB.match(ln.strip())
        if not m:
            continue
        step, a, b, s, e, names = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                   int(m.group(4)), int(m.group(5)), m.group(6))
        cur.append({"step": step, "a": a, "b": b, "start": s, "end": e,
                    "names": [x for x in names.split(",") if x]})
        if a == 0:                      # 主 buffer 是該 group 的最後一條
            segs.append(cur)
            cur = []
    if cur:                             # 尾巴沒收尾的（跑到一半的 log）
        segs.append(cur)
    return segs


def _find(bufs, key, layer):
    """在這個 segment 裡找 `key-layer` 這顆 node 屬於哪個 buffer。"""
    want = "%s-%d" % (key, layer)
    for i, bf in enumerate(bufs):
        names = bf["names"]
        if bf["b"] - bf["a"] > len(names):
            continue                    # 名字被截斷 ⇒ 不能保證它在不在這裡
        for j, nm in enumerate(names):
            if nm == want:
                return i, bf["a"] + j
    return None, None


def layers_in(bufs, key):
    """這個 segment 裡出現過哪些 layer 的 `key-<L>`。"""
    out = set()
    for bf in bufs:
        for nm in bf["names"]:
            if nm.startswith(key + "-"):
                try:
                    out.add(int(nm.rsplit("-", 1)[1]))
                except ValueError:
                    pass
    return out


def lead_of(bufs, layer):
    """回傳 dict 或 None（該層在這個 segment 裡不完整）。"""
    ri, rix = _find(bufs, RHO, layer)
    ai, aix = _find(bufs, ARG, layer)
    if ri is None or ai is None:
        return None
    rb, ab = bufs[ri], bufs[ai]
    n_nodes = max(bf["b"] for bf in bufs)
    return {
        "layer": layer,
        "rho_idx": rix, "arg_idx": aix, "n_nodes": n_nodes,
        "rho_pct": 100.0 * rix / n_nodes if n_nodes else 0.0,
        "arg_pct": 100.0 * aix / n_nodes if n_nodes else 0.0,
        "lead_lo_ms": (ab["start"] - rb["end"]) / 1e6,
        "lead_hi_ms": (ab["end"] - rb["start"]) / 1e6,
        "seg_span_ms": (max(bf["end"] for bf in bufs) - min(bf["start"] for bf in bufs)) / 1e6,
    }


def analyze(lines, tail_frac=1.0, nodes=None):
    """nodes=N ⇒ 只留 node 數為 N 的 segment。

    為什麼需要：一次 run 裡混著不同形狀的 segment（prefill / ntok=1 verify / ntok=4 draft …
    實測 node 數有三個族群：102、100、92）。各族群的 lead 與 wall span 都不同
    （2026-09-23：102 → [1.47,2.27] ms；92 → [1.01,1.81] ms），混著取中位數會得到一個
    不屬於任何形狀的數字。⇒ 對外引用時要指定族群，並在文件裡寫明是哪一個。
    """
    segs = parse_segments(lines)
    if tail_frac < 1.0:
        segs = segs[int(len(segs) * (1.0 - tail_frac)):]
    if nodes is not None:
        segs = [s for s in segs if max(bf["b"] for bf in s) == nodes]
    rows = []
    n_seg = 0
    for bufs in segs:
        n_seg += 1
        for L in sorted(layers_in(bufs, RHO) & layers_in(bufs, ARG)):
            r = lead_of(bufs, L)
            if r:
                rows.append(r)
    return {"n_segments": n_seg, "n_pairs": len(rows), "rows": rows}


def summarize(res):
    rows = res["rows"]
    out = {"n_segments": res["n_segments"], "n_pairs": res["n_pairs"]}
    if not rows:
        out["verdict"] = "INCOMPLETE: no (rho, argsort) pair in the same segment"
        return out
    lo = [r["lead_lo_ms"] for r in rows]
    hi = [r["lead_hi_ms"] for r in rows]
    rp = [r["rho_pct"] for r in rows]
    ap = [r["arg_pct"] for r in rows]
    med = statistics.median
    out.update({
        "lead_lo_ms": med(lo), "lead_hi_ms": med(hi),
        "lead_lo_p10": sorted(lo)[max(0, int(0.10 * len(lo)))],
        "lead_hi_p90": sorted(hi)[min(len(hi) - 1, int(0.90 * len(hi)))],
        "rho_pos_pct": med(rp), "arg_pos_pct": med(ap),
        "seg_span_ms": med([r["seg_span_ms"] for r in rows]),
    })
    if out["lead_lo_ms"] > 0:
        out["verdict"] = "POSITIVE lead (even the pessimistic bound is > 0)"
    elif out["lead_hi_ms"] > 0:
        out["verdict"] = "INDETERMINATE: bound straddles 0 (buffer too coarse to resolve)"
    else:
        out["verdict"] = "NO lead (even the optimistic bound is <= 0)"
    return out


def report(path, res, s):
    print("=" * 70)
    print("rho LEAD — 影子 router 比真實路由早多少")
    print("=" * 70)
    print("  log: %s" % path)
    print("  segments=%d   (layer, rho+argsort) 配對=%d" % (res["n_segments"], res["n_pairs"]))
    if "verdict" in s and "lead_lo_ms" not in s:
        print("  %s" % s["verdict"])
        return
    print()
    print("  中位數 lead（悲觀下界）: %+.3f ms" % s["lead_lo_ms"])
    print("  中位數 lead（樂觀上界）: %+.3f ms" % s["lead_hi_ms"])
    print("  p10 下界 / p90 上界   : %+.3f / %+.3f ms" % (s["lead_lo_p10"], s["lead_hi_p90"]))
    print("  segment 總跨度（中位）: %.3f ms" % s["seg_span_ms"])
    print()
    print("  影子 node 在 segment 的位置: %.1f%%" % s["rho_pos_pct"])
    print("  真實 argsort 的位置        : %.1f%%" % s["arg_pos_pct"])
    print()
    print("  判決: %s" % s["verdict"])
    print()
    print("  ⚠ buffer 有 ~5 個 node 寬 ⇒ lead 只能給區間；區間跨 0 就必須說「量不出來」，")
    print("    不能挑一邊報。這一趟的 t/s 不可引用。")


# ── selftest ────────────────────────────────────────────────────────────────
def _seg(step, pairs, names_for=None):
    """造一個 segment 的 NSCB 行：pairs = [(a,b,start,end)], 最後補一條 a=0。"""
    out = []
    for (a, b, s, e) in pairs:
        # ⚠ `names_for` 回傳的已經是「逗號串好」的字串；再 join 一次會把它拆成逐字元
        #   （",".join("a,b") -> "a,,,b"），於是一個名字都對不上 ⇒ 配對數 0。
        nm = names_for(a) if names_for else ",".join("n%d" % k for k in range(a, b))
        out.append("CGC-NSCB step=%d a=%d b=%d start_ns=%d end_ns=%d nm=%s"
                   % (step, a, b, s, e, nm))
    out.append("CGC-NSCB step=%d a=0 b=1 start_ns=%d end_ns=%d nm=norm-0"
               % (step, pairs[-1][3] + 1000, pairs[-1][3] + 2000))
    return out


def self_test():
    ok = 0
    fails = []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fails.append(name)

    # 1. 基本分組：兩個 segment，各 3 條 worker + 1 條 main
    lines = _seg(0, [(1, 6, 1000, 2000), (6, 11, 2000, 3000), (11, 16, 3000, 4000)])
    lines += _seg(0, [(1, 6, 5000, 6000), (6, 11, 6000, 7000)])
    segs = parse_segments(lines)
    check("grouping: 2 segments", len(segs) == 2)
    check("grouping: main buffer last", segs[0][-1]["a"] == 0)
    check("grouping: sizes 4 and 3", (len(segs[0]), len(segs[1])) == (4, 3))

    # 2. lead：影子在 buffer0（start 1000,end 2000），argsort 在 buffer2（start 3000,end 4000）
    names = {1: "node_1,norm-3,cgc_rho_logits-3,x,y",
             6: "a,b,c,d,e",
             11: "a,b,ffn_moe_argsort-3,d,e"}
    lines = _seg(7, [(1, 6, 1000, 2000), (6, 11, 2000, 3000), (11, 16, 3000, 4000)],
                 names_for=lambda a: names[a])
    r = analyze(lines)
    s = summarize(r)
    check("pair found", r["n_pairs"] == 1)
    # 悲觀 = argsort.start - rho.end = 3000-2000 = 1000 ns = 0.001 ms
    check("lead_lo", abs(s["lead_lo_ms"] - 0.001) < 1e-9)
    # 樂觀 = argsort.end - rho.start = 4000-1000 = 3000 ns = 0.003 ms
    check("lead_hi", abs(s["lead_hi_ms"] - 0.003) < 1e-9)
    check("positive verdict", s["verdict"].startswith("POSITIVE"))

    # 3. 影子排在 argsort 之後 ⇒ lead 為負 ⇒ NO lead
    names2 = {1: "a,b,c,d,e",
              6: "a,b,ffn_moe_argsort-3,d,e",
              11: "node_1,norm-3,cgc_rho_logits-3,x,y"}
    lines = _seg(8, [(1, 6, 1000, 2000), (6, 11, 2000, 3000), (11, 16, 3000, 4000)],
                 names_for=lambda a: names2[a])
    s = summarize(analyze(lines))
    check("late placement -> no lead", s["verdict"].startswith("NO lead"))

    # 4. 區間跨 0（buffer 太粗）⇒ INDETERMINATE，不能挑一邊報
    names3 = {1: "node_1,norm-3,cgc_rho_logits-3,x,y",
              6: "ffn_moe_argsort-3,b,c,d,e"}
    lines = _seg(9, [(1, 6, 1000, 5000), (6, 11, 3000, 4000)],
                 names_for=lambda a: names3[a])
    s = summarize(analyze(lines))
    # lo = 3000-5000 <0 ; hi = 4000-1000 >0
    check("straddling -> indeterminate", s["verdict"].startswith("INDETERMINATE"))

    # 5. 缺一半（只有 rho 沒有 argsort）⇒ 不配對
    names4 = {1: "node_1,norm-3,cgc_rho_logits-3,x,y", 6: "a,b,c,d,e"}
    lines = _seg(10, [(1, 6, 1000, 2000), (6, 11, 2000, 3000)],
                 names_for=lambda a: names4[a])
    r = analyze(lines)
    check("no pair when half missing", r["n_pairs"] == 0 and r["n_segments"] == 1)

    # 6. 名字被截斷（buffer 寬 > 名字數）⇒ 不認領（保守）
    lines = ["CGC-NSCB step=1 a=1 b=9 start_ns=0 end_ns=10 nm=a,b,c,...",
             "CGC-NSCB step=1 a=0 b=1 start_ns=11 end_ns=12 nm=norm-0"]
    r = analyze(lines)
    check("truncated names skipped", r["n_pairs"] == 0)

    # 7. 位置百分比：影子在 index 2 / 16 nodes
    names5 = {1: "n1,n2,cgc_rho_logits-5,n4,n5",
              6: "n6,n7,n8,ffn_moe_argsort-5,n10",
              11: "n11,n12,n13,n14,n15"}
    lines = _seg(11, [(1, 6, 0, 10), (6, 11, 10, 20), (11, 16, 20, 30)],
                 names_for=lambda a: names5[a])
    s = summarize(analyze(lines))
    # 位置是 **node index**（a + 在 names 裡的下標），不是 buffer 序號：
    #   rho  在 a=1 的 buffer 的第 3 個名字 ⇒ idx 1+2 = 3
    #   argsort 在 a=6 的 buffer 的第 4 個名字 ⇒ idx 6+3 = 9
    check("rho position pct", abs(s["rho_pos_pct"] - 100.0 * 3 / 16) < 1e-6)
    check("arg position pct", abs(s["arg_pos_pct"] - 100.0 * 9 / 16) < 1e-6)

    # 8. tail_frac 只留後半
    lines = _seg(0, [(1, 6, 0, 1)]) + _seg(1, [(1, 6, 0, 1)]) + \
            _seg(2, [(1, 6, 0, 1)]) + _seg(3, [(1, 6, 0, 1)])
    check("tail_frac keeps half", analyze(lines, tail_frac=0.5)["n_segments"] == 2)

    # 9. --nodes 只留指定形狀的族群（三個族群混著取中位數會得到不屬於任何形狀的數字）
    nm = {1: "n1,n2,cgc_rho_logits-5,n4,n5", 6: "n6,n7,n8,ffn_moe_argsort-5,n10",
          11: "n11,n12,n13,n14,n15"}
    big = _seg(20, [(1, 6, 0, 10), (6, 11, 10, 20), (11, 16, 20, 30)], names_for=lambda a: nm[a])
    small = _seg(21, [(1, 6, 0, 10), (6, 11, 10, 20)], names_for=lambda a: nm[a])
    check("nodes filter keeps one shape",
          analyze(big + small, nodes=16)["n_segments"] == 1 and
          analyze(big + small, nodes=11)["n_segments"] == 1)
    check("nodes filter excludes others", analyze(big + small, nodes=99)["n_segments"] == 0)

    print("%d/%d passed" % (ok, ok + len(fails)))
    for f in fails:
        print("  FAILED: %s" % f)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="stderr log containing CGC-NSCB lines")
    ap.add_argument("--tail-frac", type=float, default=1.0,
                    help="只取最後這個比例的 segment（去掉 warmup/prefill）")
    ap.add_argument("--nodes", type=int, default=None,
                    help="只留 node 數為 N 的 segment（形狀族群；對外引用必填）")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.log:
        ap.error("log 或 --self-test 要給一個")
    with open(a.log, errors="replace") as f:
        lines = f.readlines()
    res = analyze(lines, tail_frac=a.tail_frac, nodes=a.nodes)
    report(a.log, res, summarize(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
