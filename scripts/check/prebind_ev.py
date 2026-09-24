#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prebind_ev.py -- 預指派 slot（speculative slot binding）的期望收益定價。

零 GPU。回答一個問題：**命中率／寬度要到哪裡，預指派才值得做？**

上游事實（`docs/PREBIND_SLOT_DESIGN_2026-09-23.md`、`docs/STEP_SERIALIZATION_2026-09-23.md`）：

  * 穩態 verify step（ntok=4）: total 183.5 / union 120.1 / gap 58.8 / cb 39.3 ms
  * `gap = 1.01 * (cb + submit) + 8.0`（斜率 1.01、截距 +8.0，submit ~11 ms，r=0.980）
    => CPU 在段間多花 1 ms，GPU 就多空轉 1 ms（全額轉嫁）
  * 每層 union U ~ 15.5 個專家；41 段（40 個可路由層）
  * 基線冷缺口 15%（`llama-context.cpp:5380` "15% of selected experts are non-resident"）

------------------------------------------------------------------------------
★ 兩個結論（本工具存在的理由）

**(1) 門檻事件是「整層 union 零 miss」，不是「8 個全中」。**
    所以是 `r^U`，U=15.5 —— 指數從 8 變 15.5，門檻陡很多。

**(2) 但 r 不是「預測覆蓋率 q」，而是 `r = p_res0 + (1 - p_res0) * q`。**
    基線已有 85% 的被選專家常駐；預測只是救那 15% 冷的。
    ⚠ 我第一版把 q 直接當 r，**把預指派的好處低估了一個數量級**（q=0.87 從 +0.8%
    變成 +17%）。這個錯誤的方向是保守的，但結論反了，必須記下來。

=> **A 的真正槓桿是「寬度」不是「準度」**：

  寬度  8 ids/層（現況，只取 draft j=0）  -> q 結構上限 0.52 -> Δt/s  +5.5%
  寬度 16 ids/層                          -> q 上限 0.76     -> Δt/s +12.6%
  寬度 24 ids/層（draft 三行全取，樹上現成）-> q 上限 0.91    -> Δt/s +20.1%
  寬度 32 ids/層（+ prev token）           -> q 上限 1.00     -> Δt/s +25.8%

  損益兩平：step -10% 需要 q >= 0.734（p_res0=0.85 時）

------------------------------------------------------------------------------
模型（明確聲明是線性代理，不是 F1 的係數模型）

  r(q)      = p_res0 + (1 - p_res0) * q        # 該專家 hook 時已 resident 的機率
  f_clean   = E[r_z ** U]                       # 整層零 miss 的層比例
  cb(f)     = cb_min + (cb_base - cb_min) * (1 - f) / (1 - f0),  f0 = p_res0 ** U
  gap(cb)   = 1.01 * (cb + submit) + 8.0
  step      = union + other + gap

  之所以不用 F1 的 `cb = 0.46*L + 0.695*M`：**那兩個係數與 ntok=4 的實測 cb=39.3 ms
  不相容**（代入 15% 冷缺口 x 15.5 union x 41 層 => 93 次 fill x 0.695 = 65 ms，
  單 fill 項就超過實測 cb）。見 docs/PREBIND_TRADEOFF_2026-09-23.md §8。

相關性（「專家選擇不是獨立的」）用一個旋鈕量化：
  r_i = logistic(mu + sigma * z)，z ~ N(0,1) 是**整層共用**的隨機效應。
  sigma = 0  => 獨立（r^U，下界）
  sigma > 0  => 層內正相關；因為 r^U 對 r 是凸的，Jensen => E[r^U] 上升
  => 相關性**確實**幫忙，但買不到一個數量級（見 --sweep）。

用法:
    python3 prebind_ev.py --self-test
    python3 prebind_ev.py --sweep
    python3 prebind_ev.py --q 0.87 [--sigma 1.0] [--p-res0 0.95]
    python3 prebind_ev.py --breakeven 0.10
"""

from __future__ import annotations

import argparse
import math
import sys

# ---------------------------------------------------------------- 實測常數
CB_BASE = 39.3       # 穩態 verify step 的 cb（ntok=4，ms）
CB_MIN = 2.0         # 全快路徑的 cb（純查表，設計文件估 ~2 ms，未實測）
SUBMIT = 11.0        # submit（ms）
GAP_SLOPE = 1.01     # gap ~ (cb + submit) 的斜率（ntok=4）
GAP_INTERCEPT = 8.0  # 截距（ms）
UNION_MS = 120.1     # GPU busy（ms）
OTHER_MS = 4.6       # union+gap 之外的固定殘差（ms）；設計文件 total 183.5 vs
                     # union 120.1 + gap 58.8 = 178.9 的差。常數，不受預指派影響。
U_DEFAULT = 15.5     # 每層 union 大小（ntok=4，專家數）
P_RES0 = 0.85        # 基線「已 resident」比例（15% 冷缺口）
WAIT_WINDOW = 129.8  # ntok=4 的 wait 總和（ms）= CPU 可以躲進去的窗口
IDS_PER_TOKEN = 8.0  # top-8 路由
NTOK = 4             # verify batch token 數


# ---------------------------------------------------------------- 數值工具
def _logistic(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _quad(func, lo: float = -8.0, hi: float = 8.0, n: int = 1201):
    """E_{z~N(0,1)}[func(z)]，Simpson 複合積分。"""
    if n % 2 == 0:
        n += 1
    h = (hi - lo) / (n - 1)
    inv = 1.0 / math.sqrt(2.0 * math.pi)
    total = 0.0
    for i in range(n):
        z = lo + i * h
        w = 1.0 if i in (0, n - 1) else (4.0 if i % 2 == 1 else 2.0)
        total += w * inv * math.exp(-0.5 * z * z) * func(z)
    return total * h / 3.0


def _mu_for_r(r: float, sigma: float) -> float:
    """解 mu 使 E[logistic(mu + sigma*z)] == r。"""
    if sigma == 0.0:
        return _logit(r)

    def marginal(mu: float) -> float:
        return _quad(lambda z: _logistic(mu + sigma * z))

    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if marginal(mid) < r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def union_of_k_tokens(k: int, u: float = U_DEFAULT, s: float = 0.6) -> float:
    """k 個 token 的 top-8 聯集大小。

    已知兩點：U(1) = 8（單 token 的 8 個 ids 互不重複）、U(4) = 15.5（實測）。
    中間用幾何飽和 `U(k) = A - (A - 8) * s^(k-1)` 插值到漸近線 A。
    ⚠ s 是**假設**不是實測；階段 0 要印真實 uni_size 來換掉它。
    """
    if k <= 0:
        return 0.0
    if k == 1:
        return IDS_PER_TOKEN
    # 解 A 使 U(NTOK) == u
    s_pow = s ** (NTOK - 1)
    a = (u - IDS_PER_TOKEN * s_pow) / (1.0 - s_pow)
    return a - (a - IDS_PER_TOKEN) * (s ** (k - 1))


def q_ceiling(n_tokens_predicted: int, u: float = U_DEFAULT,
              s: float = 0.6) -> float:
    """預測 n 個 token 的 ids 時，per-expert 覆蓋率 q 的結構上限。"""
    if n_tokens_predicted >= NTOK:
        return 1.0
    return min(1.0, union_of_k_tokens(n_tokens_predicted, u, s) / u)


# ---------------------------------------------------------------- 核心
def resident_prob(q: float, p_res0: float = P_RES0) -> float:
    """該專家在 hook 時已 resident 的機率。

    ⚠ 這是本模型最容易搞錯的一行：基線已有 p_res0 常駐，預測只救那 (1-p_res0) 冷的。
    """
    return p_res0 + (1.0 - p_res0) * q


def f_clean(q: float, u: float = U_DEFAULT, sigma: float = 0.0,
            p_res0: float = P_RES0) -> float:
    """整層 union 零 miss 的層比例。"""
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    r = resident_prob(q, p_res0)
    if not (0.0 <= r <= 1.0):
        raise ValueError("resident prob out of range")
    if u <= 0:
        raise ValueError("u must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if r == 0.0:
        return 0.0
    if r >= 1.0:
        return 1.0
    if sigma == 0.0:
        return r ** u
    mu = _mu_for_r(r, sigma)
    return min(1.0, _quad(lambda z: _logistic(mu + sigma * z) ** u))


def moments_of_r(r: float, sigma: float) -> tuple:
    """r_z = logistic(mu + sigma*z) 的一、二階矩（mu 使 E[r_z] == r）。

    回傳 (E[r_z], Var[r_z])。sigma = 0 退化成點質量 => Var = 0。
    """
    if sigma <= 0.0:
        return r, 0.0
    mu = _mu_for_r(r, sigma)
    m1 = _quad(lambda z: _logistic(mu + sigma * z))
    m2 = _quad(lambda z: _logistic(mu + sigma * z) ** 2)
    return m1, max(0.0, m2 - m1 * m1)


def sigma_from_res_spread(res_var: float, mean_r: float, u: float) -> float:
    """由「逐層常駐比例」的變異數反解 sigma —— **不是**由 clean_pct 反解。

    ★ 為什麼要有這個（2026-09-23）：sigma 若用 clean_pct 反解就是循環論證 ——
      clean_pct 正是模型要預測的那個量 E[r_z^U]，拿它配適出來的 sigma 再去降門檻，
      等於「用來放寬的那個數字」和「放寬後要預測的數字」是同一個。
      改用**另一個統計量**：逐層常駐比例的變異數（二階矩）。這樣 clean_pct（U 階矩）
      就不再被拿去配適，而是變成檢定「logistic-normal 隨機效應」這個函數形式的驗證點。

    觀測模型：res_layer = (1/U) * Σ Bern(r_z)
      => Var[res] = Var[r_z] + E[r_z (1 - r_z)] / U
    二項修正項本身也依賴 sigma，所以用不動點迭代（3 輪足夠）。
    """
    if u <= 0 or res_var <= 0.0 or not (0.0 < mean_r < 1.0):
        return 0.0
    var_r = res_var - mean_r * (1.0 - mean_r) / u
    if var_r <= 0.0:
        return 0.0
    s = 0.0
    for _ in range(3):
        lo, hi = 0.0, 12.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if moments_of_r(mean_r, mid)[1] < var_r:
                lo = mid
            else:
                hi = mid
        s = 0.5 * (lo + hi)
        _, vr = moments_of_r(mean_r, s)
        e_r2 = vr + mean_r * mean_r
        var_r = res_var - (mean_r - e_r2) / u
        if var_r <= 0.0:
            return 0.0
    return s


def _f0(u: float = U_DEFAULT, p_res0: float = P_RES0) -> float:
    return p_res0 ** u


def cb_of(f: float, u: float = U_DEFAULT, p_res0: float = P_RES0) -> float:
    """cb 作為「整層零 miss 層比例」的函數（線性代理）。"""
    if f >= 1.0:
        return CB_MIN
    return CB_MIN + (CB_BASE - CB_MIN) * (1.0 - f) / (1.0 - _f0(u, p_res0))


def gap_of(cb: float) -> float:
    return GAP_SLOPE * (cb + SUBMIT) + GAP_INTERCEPT


def evaluate(q: float, u: float = U_DEFAULT, sigma: float = 0.0,
             p_res0: float = P_RES0, rejected_frac: float = 0.0) -> dict:
    """給定 per-expert 覆蓋率 q，回傳整條鏈的收益。

    rejected_frac: draft token 被拒（=> 其 ids 是垃圾）的比例；被拒的 round 掉回慢路徑。
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    f = f_clean(q, u, sigma, p_res0)
    cb = cb_of(f, u, p_res0)
    gap = gap_of(cb)
    step = UNION_MS + OTHER_MS + gap
    gap_base = gap_of(CB_BASE)
    step_eff = step * (1.0 - rejected_frac) + (UNION_MS + OTHER_MS + gap_base) * rejected_frac
    gap_eff = gap * (1.0 - rejected_frac) + gap_base * rejected_frac
    return {
        "q": q, "u": u, "sigma": sigma, "p_res0": p_res0,
        "r": resident_prob(q, p_res0),
        "f_clean": f, "cb": cb, "gap": gap_eff, "step": step_eff,
        "d_gap_pct": 100.0 * (gap_eff / gap_base - 1.0),
        "d_step_pct": 100.0 * (step_eff / (UNION_MS + OTHER_MS + gap_base) - 1.0),
        "d_tps_pct": 100.0 * ((UNION_MS + OTHER_MS + gap_base) / step_eff - 1.0),
    }


def breakeven_q(target_step_gain: float = 0.10, u: float = U_DEFAULT,
                sigma: float = 0.0, p_res0: float = P_RES0) -> float:
    """達到 step 降幅 target_step_gain（如 0.10 = -10%）所需的 q。"""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if -evaluate(mid, u, sigma, p_res0)["d_step_pct"] / 100.0 < target_step_gain:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def shadow_cost(width_per_layer: int, cold_rate: float = 0.05,
                fill_ms: float = 0.695, layers: int = 41) -> tuple:
    """影子 ensure 的 CPU 成本，以及它塞不塞得進 GPU 窗口。

    width_per_layer: 預指派幾個專家/層（現況 draft 只取 8；三行全取 = 24）
    """
    fills = width_per_layer * layers * cold_rate
    shadow_ms = fills * fill_ms + layers * 0.10  # 0.10 ms/層 純查表
    fits = shadow_ms <= WAIT_WINDOW
    return shadow_ms, WAIT_WINDOW, fits, (WAIT_WINDOW / shadow_ms if shadow_ms else float("inf"))


# ---------------------------------------------------------------- 輸出
def _row(r: dict) -> str:
    return ("q={q:<5.2f} r={r:<5.3f} s={sigma:<4.1f} | f_clean={f_clean:6.1%} "
            "cb={cb:5.1f} gap={gap:5.1f} step={step:6.1f} "
            "| Dstep={d_step_pct:+6.1f}% Dt/s={d_tps_pct:+6.1f}%").format(**r)


def do_sweep() -> int:
    base = UNION_MS + OTHER_MS + gap_of(CB_BASE)
    print("基線 step = union {} + other {} + gap {:.1f} = {:.1f} ms".format(
        UNION_MS, OTHER_MS, gap_of(CB_BASE), base))
    print()
    print("== per-expert 覆蓋率 q 的收益曲線（sigma=0 獨立下界, U=15.5, p_res0=0.85）==")
    for q in (0.00, 0.30, 0.52, 0.70, 0.77, 0.87, 0.91, 1.00):
        print(_row(evaluate(q)))
    print()
    print("== 相關性旋鈕 sigma（q=0.60 固定）==")
    for s in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        print(_row(evaluate(0.60, sigma=s)))
    print()
    print("== ★ 寬度才是槓桿：q 的結構上限（假設預測完全正確）==")
    print("   draft ctx 一次算好 3 個 token x 8 ids = 24 個；")
    print("   現有程式碼（llama-context.cpp:5390-5391）只取 j=0 一行 = 8 個")
    for k in (1, 2, 3, 4):
        qc = q_ceiling(k)
        r = evaluate(qc)
        print("   預測 {:d} 個 token ({:>2d} ids/層): q 上限 {:5.1%} -> {}".format(
            k, int(k * IDS_PER_TOKEN), qc, _row(r)))
    print()
    print("== 基線常駐率 p_res0 的敏感度（這個數字目前有矛盾，見 §8）==")
    for pr in (0.85, 0.90, 0.95):
        r = evaluate(0.52, p_res0=pr)
        print("   p_res0={:.2f} 時，寬度 8（q=0.52）: {}".format(pr, _row(r)))
        print("   p_res0={:.2f} 時，損益兩平 -10% 需要 q >= {:.3f}".format(
            pr, breakeven_q(0.10, p_res0=pr)))
    print()
    print("== 損益兩平（p_res0=0.85）==")
    for tgt in (0.05, 0.10, 0.15, 0.20):
        print("  step -{:.0f}% 需要 q >= {:.4f}".format(100 * tgt, breakeven_q(tgt)))
    print()
    print("== 影子 ensure 塞得進 GPU 窗口嗎（窗口 {:.1f} ms）==".format(WAIT_WINDOW))
    for w in (8, 16, 24, 32):
        ms, win, fits, head = shadow_cost(w)
        print("  寬度 {:>2d}: shadow {:5.1f} ms, 餘量 {:4.2f}x, {}".format(
            w, ms, head, "OK" if fits else "超出窗口"))
    return 0


# ---------------------------------------------------------------- selftest
def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def self_test() -> int:
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # 1. sigma=0 退化成獨立：f = r^U
    check("independence r^U",
          _approx(f_clean(0.9), 0.9 * 0.0 + resident_prob(0.9) ** U_DEFAULT, 1e-9))

    # 2. U=1 => f = r
    check("U=1", _approx(f_clean(0.73, 1.0), resident_prob(0.73), 1e-12))

    # 3. ★ q=0 時 f_clean 必須等於基線 p_res0^U（預指派沒做，行為不變）
    check("q=0 -> baseline f0", _approx(f_clean(0.0), P_RES0 ** U_DEFAULT, 1e-12))

    # 4. q=1 時全部層乾淨
    check("q=1", f_clean(1.0) == 1.0)

    # 5. ★ 單調：q 上升 => r 上升 => f 上升
    fs = [f_clean(i / 20.0) for i in range(21)]
    check("monotone in q", all(b >= a - 1e-12 for a, b in zip(fs, fs[1:])))

    # 6. 相關性幫忙（Jensen：r^U 凸 => 方差上升 => E 上升）
    a0, a1, a2 = f_clean(0.6, sigma=0.0), f_clean(0.6, sigma=1.0), f_clean(0.6, sigma=2.0)
    check("correlation helps", a1 > a0 and a2 > a1)

    # 7. 相關性買不到一個數量級
    check("correlation bounded", f_clean(0.6, sigma=3.0) < 5.0 * a0)

    # 7b. ★ sigma 可由「逐層常駐比例的變異數」反解（不是由 clean_pct 反解）
    m0, v0 = moments_of_r(0.902, 0.0)
    check("moments sigma=0 -> point mass", _approx(m0, 0.902) and v0 == 0.0)
    v1 = moments_of_r(0.902, 1.2)[1]
    v2 = moments_of_r(0.902, 2.4)[1]
    check("var rises with sigma", 0.0 < v1 < v2)

    # 7c. 往返：造一組 (sigma_true, U) 的觀測變異數，反解要回到 sigma_true。
    #     這是唯一能證明「反解不是湊的」的檢查。
    for st in (0.5, 1.2, 2.0):
        _, vr = moments_of_r(0.902, st)
        e_r2 = vr + 0.902 ** 2
        obs = vr + (0.902 - e_r2) / 20.23          # Var[res] = Var[r_z] + E[r(1-r)]/U
        check("sigma round-trip {:.1f}".format(st),
              abs(sigma_from_res_spread(obs, 0.902, 20.23) - st) < 0.05)
    # 觀測變異數為 0（或低於二項噪音）=> sigma = 0，不能是負的也不能亂跳
    check("zero spread -> sigma 0", sigma_from_res_spread(0.0, 0.902, 20.23) == 0.0)
    check("sub-binomial spread -> sigma 0",
          sigma_from_res_spread(0.902 * 0.098 / 20.23 * 0.5, 0.902, 20.23) == 0.0)

    # 8. gap 模型重現基線 58.8
    check("gap baseline", _approx(gap_of(CB_BASE), 58.8, 2e-3))

    # 9. cb 在 q=0 時重現實測 39.3
    check("cb baseline", _approx(cb_of(_f0()), CB_BASE, 1e-9))

    # 10. cb 在全快路徑 = cb_min
    check("cb min", _approx(cb_of(1.0), CB_MIN, 1e-12))

    # 11. 全快路徑 step 降幅落在設計文件的 -20% 附近
    check("full fast path ~ -20%", -25.0 <= evaluate(1.0)["d_step_pct"] <= -15.0)

    # 12. 損益兩平自洽
    qb = breakeven_q(0.10)
    check("breakeven in range", 0.0 < qb < 1.0)
    check("breakeven self-consistent",
          _approx(-evaluate(qb)["d_step_pct"] / 100.0, 0.10, 1e-3))

    # 13. ★ 寬度 8（現況）已有正收益，但不是設計文件說的 -20%
    r8 = evaluate(q_ceiling(1))
    check("width 8 positive but small", 0.0 < r8["d_tps_pct"] < 10.0)

    # 14. 寬度單調：預測越多 token，q 上限越高
    cs = [q_ceiling(k) for k in (1, 2, 3, 4)]
    check("width monotone", all(b >= a - 1e-12 for a, b in zip(cs, cs[1:])))
    check("width 4 saturates", _approx(cs[3], 1.0))

    # 15. union_of_k_tokens 的兩個已知點
    check("U(1)=8", _approx(union_of_k_tokens(1), IDS_PER_TOKEN))
    check("U(4)=15.5", _approx(union_of_k_tokens(NTOK), U_DEFAULT, 1e-9))

    # 16. 被拒比例方向正確
    check("rejection hurts",
          evaluate(0.9, rejected_frac=0.08)["step"] > evaluate(0.9)["step"])

    # 17. p_res0 越高，基線 f0 越高，同樣 q 的絕對 cb 越低
    check("higher p_res0 -> higher f0", _f0(p_res0=0.95) > _f0(p_res0=0.85))

    # 18. 錯值拋出
    for bad in (lambda: f_clean(-0.1), lambda: f_clean(1.1),
                lambda: f_clean(0.5, 0.0), lambda: f_clean(0.5, 15.5, -1.0)):
        try:
            bad()
            check("raises", False)
        except ValueError:
            check("raises", True)

    # 19. 影子 ensure 在寬度 24 時仍塞得進窗口
    ms, win, fits, head = shadow_cost(24)
    check("shadow fits at width 24", fits and head > 1.5)

    # 20. resident_prob 的兩個端點
    check("r(q=0)=p_res0", _approx(resident_prob(0.0), P_RES0))
    check("r(q=1)=1", _approx(resident_prob(1.0), 1.0))

    print("selftest: {}/{} passed".format(ok, ok + len(fail)))
    for f in fail:
        print("  FAIL: {}".format(f))
    return 0 if not fail else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="跑內建自測")
    p.add_argument("--sweep", action="store_true", help="印出完整敏感度表")
    p.add_argument("--q", type=float, help="per-expert 覆蓋率（0..1）")
    p.add_argument("--u", type=float, default=U_DEFAULT, help="每層 union 大小")
    p.add_argument("--sigma", type=float, default=0.0, help="層內相關性旋鈕（0=獨立）")
    p.add_argument("--p-res0", type=float, default=P_RES0, help="基線常駐率")
    p.add_argument("--rejected-frac", type=float, default=0.0,
                   help="draft token 被拒比例（那幾輪掉回慢路徑）")
    p.add_argument("--breakeven", type=float, metavar="GAIN",
                   help="求達到 step 降幅 GAIN（如 0.10）所需的 q")
    a = p.parse_args(argv)

    if a.self_test:
        return self_test()
    if a.sweep:
        return do_sweep()
    if a.breakeven is not None:
        qb = breakeven_q(a.breakeven, a.u, a.sigma, a.p_res0)
        print("step -{:.1f}% 需要 q >= {:.4f}  (U={}, sigma={}, p_res0={})".format(
            100 * a.breakeven, qb, a.u, a.sigma, a.p_res0))
        print(_row(evaluate(qb, a.u, a.sigma, a.p_res0)))
        return 0
    if a.q is not None:
        print(_row(evaluate(a.q, a.u, a.sigma, a.p_res0, a.rejected_frac)))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
