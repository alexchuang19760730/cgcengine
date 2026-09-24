#!/usr/bin/env python3
"""MTP verify 的剩餘空間定價器（離線，0 GPU）。

為什麼需要這支
--------------
既有的 `DEVICE_BUSY_ATTRIBUTED_2026-09-23.md` §3 用「T=2 → T=4 的均值差」算出
邊際 cb = +31.71 ms/token。但那份 log（`Backup/phase_decomp/node_attr/nsm_shard_k3_20260923.log`）
的結構是 **ntok 分段**：ntok=8 是一整段 64 步的獨立 test case，而 ntok=4／2 多為孤立單步。
同一支 log 內 **ntok=4 的 cb 本身就有 13.23 ~ 225.68 ms（17.1×）的散佈**
⇒ 跨段求均值再相減得到的「邊際」，其誤差比訊號大 ⇒ **那個數不可引用**（見 §負結果）。

本工具因此**不用邊際差**，只用兩個來源不同、可交叉驗證的量：

  1. **交付 anchor**（`prod25-stream`, reps=3, NOMINAL, `--warm-skip 64`）：
     step 247.98 ms、12.57 t/s ⇒ 每 step token 數 = t/s × step。
  2. **expert bank 家族的 shape-probe 地板**（`MOE_GATHER_BOUND_2026-09-22.md` §2，
     真幾何 256 experts × top-8 × 40 層）：8.09 / 15.98 / 23.79 / 31.81 ms/步（T=1..4），
     `extra_over_first = 0.977`（第 2 個 token 付第 1 個的 97.7%，近線性）。

用法
----
    mtp_verify_headroom.py --self-test
    mtp_verify_headroom.py                     # 預設交付 anchor
    mtp_verify_headroom.py --from-perf 12.57 247.98
    mtp_verify_headroom.py --k-sweep           # k=2..8 的 ASL 與收益

輸出一律標明「假設」與「不可引用的部分」，不允許把模型的輸出當實測。
"""

import argparse
import math
import sys

# ── 交付 anchor（唯一可引用的來源）──────────────────────────────────────
DELIV = dict(ts=12.57, step_ms=247.98, k=3)
# expert bank 家族 shape-probe 地板（ms/步，T=1..4）


def bank_floor():
    return [8.09, 15.98, 23.79, 31.81]


def tok_per_step(ts=DELIV["ts"], step_ms=DELIV["step_ms"]):
    """每 step 產出的 token 數（= ASL）。"""
    return ts * step_ms / 1000.0


def solve_p(tokens_per_step, k):
    """從 1 + p + p^2 + ... + p^k = tokens_per_step 反解每位置接受率 p。

    chain draft（不是 tree）：第 i 個 draft token 只有在前面全部被接受時才算。
    二分法求解；回傳 None 表示無解（超出該 k 的理論上界 1+k）。
    """
    lo, hi = 0.0, 1.0
    f = lambda p: 1.0 + sum(p ** i for i in range(1, k + 1))
    if tokens_per_step > f(hi) + 1e-9:
        return None
    if tokens_per_step < f(lo) - 1e-9:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) < tokens_per_step:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def asl(p, k):
    return 1.0 + sum(p ** i for i in range(1, k + 1))


def extra_over_first(T=4, floor=None):
    """從 shape-probe 地板算『第 2 個 token 付第 1 個的幾成』。"""
    fl = floor or bank_floor()
    if T > len(fl):
        return None
    first = fl[0]
    marginal = (fl[T - 1] - fl[0]) / (T - 1)
    return marginal / first


def bank_total_if_share(T, share, fl=None):
    """若相鄰 verify token 的專家集可共用（第 2 個起只付 share 成），家族總成本。"""
    fl = fl or bank_floor()
    return fl[0] * (1.0 + share * (T - 1))


def step_from_two(s1, s4):
    """用 step(T=1) 與 step(T=4) 反解線性模型 step(T) = c + m*T 的 (c, m)。"""
    m = (s4 - s1) / 3.0
    return s1 - m, m


def breakeven_m(p, k, c):
    """從 k 加深到 k+1 恰好打平所需的邊際成本 m（固定成本 c 已知）。

    推導：A=ASL(k), B=ASL(k+1)
        A/(c+(k+2)m) == B/(c+(k+1)m)
      ⇒ A(c+(k+2)m) = B(c+(k+1)m)
      ⇒ (A(k+2) - B(k+1))·m = (B-A)·c
      ⇒ m = (B-A)·c / (A(k+2) - B(k+1))
    m 高於這個值 ⇒ 加深是淨虧；低於 ⇒ 淨賺。
    """
    A = asl(p, k)
    B = asl(p, k + 1)
    den = A * (k + 2) - B * (k + 1)
    if den <= 0:
        return None
    return (B - A) * c / den


def cmdhdr(s):
    print("\n" + "=" * 66)
    print(s)
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser(description="MTP verify 剩餘空間定價器")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--from-perf", nargs=2, type=float, metavar=("TS", "STEP_MS"))
    ap.add_argument("-k", "--draft-k", type=int, default=DELIV["k"])
    ap.add_argument("--k-sweep", action="store_true")
    ap.add_argument("--share-sweep", action="store_true")
    # step(T) = c + m*T 的兩點反解：給 T=1 與 T=4 的實測 step(ms)
    ap.add_argument("--step-model", nargs=2, type=float, metavar=("STEP_T1", "STEP_T4"))
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    ts, step = (a.from_perf if a.from_perf else (DELIV["ts"], DELIV["step_ms"]))
    k = a.draft_k
    tps = tok_per_step(ts, step)
    p = solve_p(tps, k)

    cmdhdr("0. 輸入")
    print("交付/輸入    %.2f t/s @ step %.2f ms   k=%d" % (ts, step, k))
    print("⇒ 每 step token 數（ASL）= %.3f   （理論上界 1+k = %d）" % (tps, 1 + k))
    print("⇒ 效率 = ASL / (1+k) = %.1f%%" % (100 * tps / (1 + k)))
    if p is None:
        print("\n⚠ 無法反解 p：ASL 超出 k=%d 的上界 ⇒ k 設定與 anchor 不自洽，"
              "改用較大的 --draft-k。" % k)
        return 2
    print("⇒ 反解每位置接受率 p = %.4f" % p)

    cmdhdr("1. 空間 A：提高接受率 p（verify 寬度不變，純賺）")
    print("假設 step 時間不隨 ASL 改變（verify 仍然跑同樣的圖 ⇒ 這個假設對 attn/派送成立，")
    print("對 cb 不成立 —— expert 訪問數隨接受的 token 數走。故此欄是**上界**。）")
    print()
    print("%-8s %10s %12s" % ("p", "ASL", "t/s(+%)"))
    base_asl = tps
    for p2 in [p, 0.85, 0.88, 0.90, 0.92, 0.95]:
        if p2 < p - 1e-9:
            continue
        a2 = asl(p2, k)
        print("%-8.3f %10.3f %12s" % (p2, a2, "%+.1f%%" % (100 * (a2 / base_asl - 1))))

    if a.k_sweep:
        cmdhdr("2. 空間 B：加深 draft（k 變大）")
        print("%-6s %10s %12s %s" % ("k", "ASL", "vs 今天", "註"))
        for kk in range(2, 9):
            a_kk = asl(p, kk)
            print("%-6d %10.3f %12s %s" % (
                kk, a_kk, "%+.1f%%" % (100 * (a_kk / base_asl - 1)),
                "← 今天" if kk == k else ""))
        print("\n⚠ 這欄只算分子。k 增大 ⇒ verify 要同時跑更多 token ⇒ step 時間也漲。")
        print("  真正的判據是 `d(ASL)/dk ÷ d(step)/dk`，而 step 的邊際目前**沒有可信實測**")
        print("  （見檔案頭負結果）⇒ 只當上界讀，不可當收益。")

    if a.step_model:
        cmdhdr("2b. 加深 draft 的淨判決（需要 2 個實測 step，不是猜）")
        s1, s4 = a.step_model
        # step(T) = c + m*T
        m_step = (s4 - s1) / 3.0
        c_step = s1 - m_step
        print("輸入：step(T=1)=%.2f ms，step(T=4)=%.2f ms" % (s1, s4))
        print("⇒ 線性模型 step(T) = %.2f + %.2f×T" % (c_step, m_step))
        print("  （固定成本 %.2f ms；每個 verify token 的邊際 %.2f ms）" % (c_step, m_step))
        print()
        base_ts = asl(p, k) / (c_step + m_step * (k + 1)) * 1000.0
        print("%-6s %8s %10s %10s %10s" % ("k", "verify T", "ASL", "step(ms)", "t/s"))
        for kk in range(2, 9):
            T = kk + 1
            if m_step * T + c_step <= 0:
                continue
            st = c_step + m_step * T
            tts = asl(p, kk) / st * 1000.0
            print("%-6d %8d %10.3f %10.2f %10.2f %s" % (
                kk, T, asl(p, kk), st, tts,
                "  ← 今天" if kk == k else ("  ← 優於今天" if tts > base_ts else "")))
        print("\n⇒ 這個判決翻轉了上面的『只算分子』表：只要 verify 的邊際成本不接近零，")
        print("  加深 draft 是淨虧。決定勝負的就是 m（%.2f ms/token）這一個數。" % m_step)

        cmdhdr("2c. ★ 不打折扣的判決門檻（不需要信任任何 step 估計值）")
        print("上式依賴輸入的 step(T=1)。下面改成問：「m 要小到多少，加深才划算？」")
        print("這樣 c 的大小無關（比值），而 c 目前可由 step(T=1) 直接量出來。")
        print()
        print("%-14s %14s %14s" % ("加深", "打平 m/c", "實測 m/c"))
        ratio = m_step / c_step if c_step else float("nan")
        for kk in range(2, 8):
            bm = breakeven_m(p, kk, 1.0)
            if bm is None:
                continue
            print("%-14s %14.4f %14s" % (
                "k=%d → %d" % (kk, kk + 1), bm,
                "← 需低於此才划算" if ratio > bm else "（實測已低於 ⇒ 划算）"))
        print("\n實測 m/c = %.4f（由上面的輸入反解；⚠ T=1 那點若口徑不同，此值不可引用）" % ratio)
        print("意義：verify 的固定成本越低、或邊際成本越高，就越不該加深 draft。")
        print("⇒ 想讓加深變划算，只有兩條：(1) 降 m（讓每個 verify token 更便宜）")
        print("   (2) 降 c 但這會同時讓所有 k 都變好，不改變勝負方向。")

    if a.share_sweep:
        cmdhdr("3. 空間 C：讓相鄰 verify token 共享專家（MoE 的唯一攤薄機制）")
        fl = bank_floor()
        share_now = extra_over_first(4, fl)
        print("shape-probe 地板（expert bank 家族，ms/步）：%s" % fl)
        print("⇒ 實測 extra_over_first = %.3f（第 2 個 token 付第 1 個的 %.1f%%）"
              % (share_now, 100 * share_now))
        print("   成因：top-8 of 256 ⇒ 兩 token 的專家集期望重疊僅 8²/256 = %.2f 個"
              % (64 / 256))
        print()
        print("%-10s %12s %12s" % ("share", "T=4 成本", "vs 今天"))
        cur = fl[3]
        for s in [share_now, 0.75, 0.5, 0.25, 0.0]:
            v = bank_total_if_share(4, s, fl)
            print("%-10.3f %12.2f %12s" % (s, v, "%+.1f%%" % (100 * (v / cur - 1))))
        print("\n讀法：這個家族只佔 device 的一部份（MoE 區 95.4 ms/step 中的一部份），")
        print("所以上表是『家族內』的降幅，不是 step 的降幅。")

    cmdhdr("負結果：為什麼這裡沒有『cb 邊際』那一欄")
    print("既有的 cb 邊際 = +31.71 ms/token 來自 `DEVICE_BUSY_ATTRIBUTED` §3，")
    print("其算法是 T=2(n=16) 與 T=4(n=48) 兩個**不同 test case** 的均值相減。")
    print("同一支 log 內 ntok=4 的 cb 散佈為 13.23~225.68 ms（17.1×），")
    print("ntok=2 為 7.84~43.43 ms（5.5×）⇒ 段內散佈 ≫ 跨段差值 ⇒ 該邊際不可引用。")
    print("同一份 log 若按 ntok 分段還會得到『cb 邊際 40.04 → 5.42』這種差 7.4 倍的結論，")
    print("兩個方向相反的答案出自同一份資料 ⇒ 這是結構 artifact，不是物理。")
    return 0


def self_test():
    n_ok = [0]
    n_fail = [0]

    def chk(name, cond, got=""):
        if cond:
            n_ok[0] += 1
            print("ok   %-46s" % name)
        else:
            n_fail[0] += 1
            print("FAIL %-46s %s" % (name, got))

    # 1. ASL 反解的基本性質
    chk("asl(p=0,k) == 1", abs(asl(0.0, 3) - 1.0) < 1e-12)
    chk("asl(p=1,k=3) == 4", abs(asl(1.0, 3) - 4.0) < 1e-12)
    # 2. solve_p 與 asl 互逆
    for pw in (0.3, 0.5, 0.7, 0.83, 0.95):
        got = solve_p(asl(pw, 3), 3)
        chk("solve_p∘asl(p=%.2f) == p" % pw, abs(got - pw) < 1e-6, "%.6f" % got)
    # 3. 超上界要 None，不是亂給一個數
    chk("solve_p(5.0, k=3) -> None", solve_p(5.0, 3) is None)
    # 4. tok_per_step 單位正確
    chk("tok_per_step(12.57, 247.98) ≈ 3.117",
        abs(tok_per_step(12.57, 247.98) - 3.117) < 0.002, "%.4f" % tok_per_step(12.57, 247.98))
    # 5. extra_over_first 從地板反算 ≈ 0.977
    e = extra_over_first(4, bank_floor())
    chk("extra_over_first(T=4) ≈ 0.977", abs(e - 0.977) < 0.01, "%.4f" % e)
    # 6. share=0 時家族成本 == 第一個 token 的成本（完全共享）
    chk("share=0 -> cost == first token",
        abs(bank_total_if_share(4, 0.0) - bank_floor()[0]) < 1e-9)
    # 7. share=1 時 == 線性
    chk("share=1 -> cost == T*first",
        abs(bank_total_if_share(4, 1.0) - 4 * bank_floor()[0]) < 1e-9)
    # 8. 交付 anchor 反解的 p 落在合理區間
    pp = solve_p(tok_per_step(), 3)
    chk("交付 anchor 反解 p ∈ (0.7, 0.95)", pp is not None and 0.7 < pp < 0.95, str(pp))
    # 9. k 增大單調增
    seq = [asl(pp, kk) for kk in range(2, 9)]
    chk("asl 對 k 單調不減", all(seq[i] <= seq[i + 1] + 1e-12 for i in range(len(seq) - 1)))
    # 10. k→∞ 收斂到 1/(1-p)
    chk("k=60 逼近 1/(1-p)", abs(asl(pp, 60) - 1 / (1 - pp)) < 0.01)
    # 11. 線性 step 模型的兩點反解要自洽（用構造出來的點驗）
    c0, m0 = 53.0, 48.0
    chk("step 線性模型在 T=1 還原", abs((c0 + m0 * 1) - 101.0) < 1e-9)
    chk("step 線性模型在 T=4 還原", abs((c0 + m0 * 4) - 245.0) < 1e-9)
    # 12. 邊際為正時，k 增大不會讓 t/s 無限上升（幾何收斂 vs 線性成本）
    got = [asl(pp, kk) / (c0 + m0 * (kk + 1)) for kk in range(2, 30)]
    chk("t/s 對 k 最終單調下降（凸性有上界）",
        got[20] < max(got), "tail=%.3f peak=%.3f" % (got[20], max(got)))
    # 13. breakeven_m 是打平點：代入該 m 時 k 與 k+1 的 t/s 必須相等
    for kk in (2, 3, 5):
        bm = breakeven_m(pp, kk, 1.0)
        if bm is None:
            continue
        t_now = asl(pp, kk) / (1.0 + (kk + 1) * bm)
        t_next = asl(pp, kk + 1) / (1.0 + (kk + 2) * bm)
        chk("breakeven_m(k=%d) 兩邊 t/s 相等" % kk,
            abs(t_now - t_next) < 1e-9, "%.9f vs %.9f" % (t_now, t_next))
    # 14. 打平門檻隨 k 單調下降（加深越來越難划算）
    seq = [breakeven_m(pp, kk, 1.0) for kk in range(2, 8)]
    seq = [s for s in seq if s is not None]
    chk("breakeven 對 k 單調下降",
        all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)))
    # 15. step 兩點反解自洽
    cc, mm = step_from_two(101.0, 245.0)
    chk("step_from_two 還原 T=1", abs(cc + mm - 101.0) < 1e-9)
    chk("step_from_two 還原 T=4", abs(cc + 4 * mm - 245.0) < 1e-9)

    print("\n%d/%d passed" % (n_ok[0], n_ok[0] + n_fail[0]))
    return 0 if n_fail[0] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
