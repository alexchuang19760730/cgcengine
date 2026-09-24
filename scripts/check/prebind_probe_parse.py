#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prebind_probe_parse.py -- 解析 `CGC-PREBIND-PROBE` / `CGC-PREBIND-SUM` 的 log。

零 GPU。把 `docs/PREBIND_STAGE0_SPEC_2026-09-23.md` 規格的儀器輸出轉成 gate 判決。

吃兩種行：

    CGC-PREBIND-PROBE: il=12 ntok=4 uni=15 rows=3 pred=1 cov8=0.533 cov16=0.800
                       cov24=0.933 cov32=0.947 res=0.867 clean=0
                       hist=3 w8=8 w16=14 w24=19 w32=19
    CGC-PREBIND-SUM:   steps=221 layers=9061 uni_avg=15.40 pred_pct=100.0%
                       q8=0.521 q16=0.764 q24=0.912 q32=0.930 p_res0=0.871
                       clean_pct=42.3% w24=18.7 q24w=0.915

輸出：按 ntok 分組的 q8/q16/q24/q32、uni_avg、p_res0、clean_pct，以及 gate 判決。

★ 聚合方式：**q = Σhit / Σuni**（加權平均），不是逐層平均的平均。
  因為 q 的定義是「隨機挑一個 union 成員，它被覆蓋的機率」⇒ 必須以 uni 為權。

★★ `pred=` 是最重要的欄位（2026-09-23 第一支實測 log 加上它的原因）：
  `pred=0` = 這一層**沒有預測源可用**。它的 cov 全是 0，但那**不代表「預測不準」**，
  而是「沒量到」。
  ⇒ q 的分母只能算 `pred=1` 的層；`pred=0` 的層仍然貢獻 uni / p_res0 / clean
    （那三個數字根本不需要預測）。
  ⇒ 若整輪 `pred=1` 的層數 = 0，本工具直接宣告「q 不可得」，**不**判成 FAIL。
    「預測源沒收到」和「預測不準」是兩種失敗，解法完全不同，混在一起會害人去改預測器。

★ `pred=1` 的意義隨預測源而變（2026-09-23 第二次改版）：
  第一版用 draft context 當預測源；實測 `pred_pct = 0.0%`（MTP draft graph 只跑 nextn
  block，collect 只在 `il == n_layer` 觸發，verify 用的 il 是 0..39，永遠對不上）。
  第二版改用**歷史 token**（最近 1/2/3 個 decode 步的 per-layer ids 聯集）= 寬度 8/16/24。
  ⇒ 現在 `pred=1` = 「這一層有至少一個歷史步的 ids」，與 draft 無關。

★ `w*` = **真實寬度**，不是名義寬度。q24 的「24」是名義值：相鄰 token 的 top-8 會重疊，
  實測聯集可能只有 19 ⇒ 它不是「24 寬的預測器」，而是「至多 24 寬」。
  （這正是「rows<3 時 q24 不是真的寬度 24」那條質疑在換源之後的對應量 —— 換成歷史源
  後 rows 不再是寬度的證據，聯集大小才是。rows 仍印，但只當 draft 有沒有跑的診斷。）

★ `hist=` = 非空歷史緩衝個數（0..3）。hist<3 = 冷啟動（前 3 個 decode 步）。
  `q24w` = 只算 hist==3（滿歷史）那一桶的 q24 —— 冷啟動不該混進穩態平均值。

用法:
    python3 prebind_probe_parse.py <log>              # 主判決（用 ntok=4 組）
    python3 prebind_probe_parse.py <log> --ntok 2     # 指定組
    python3 prebind_probe_parse.py <log> --gate 0.73  # 自訂門檻
    python3 prebind_probe_parse.py <log> --gate 0.543 --expect-u 20.23 --expect-p 0.902
    python3 prebind_probe_parse.py --self-test

退出碼: 0 = PASS / 2 = FAIL / 3 = 門檻失效（實測 U、p_res0 與導出門檻時用的輸入不一致）
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# 定價模型（f_clean / breakeven / sigma 反解）由 prebind_ev.py **單獨擁有**：
# 這裡不複寫第二份公式，兩份一定會飄。取不到就退化成「不估 sigma」，不靜默給假數字。
_EV = None
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import prebind_ev as _EV
except Exception:      # pragma: no cover -- 只在 prebind_ev.py 不在同目錄時發生
    _EV = None

PROBE_RE = re.compile(
    r"CGC-PREBIND-PROBE:\s+il=(-?\d+)\s+ntok=(\d+)\s+uni=(\d+)\s+rows=(\d+)"
    r"(?:\s+pred=(\d+))?\s+"
    r"cov8=([0-9.]+)\s+cov16=([0-9.]+)\s+cov24=([0-9.]+)"
    r"(?:\s+cov32=([0-9.]+))?\s+res=([0-9.]+)\s+clean=(\d+)"
    # 第二版欄位（選配）：舊 binary 沒有它們，解析器仍要能吃
    r"(?:\s+hist=(\d+))?"
    r"(?:\s+w8=(\d+)\s+w16=(\d+)\s+w24=(\d+)\s+w32=(\d+))?")

SUM_RE = re.compile(
    r"CGC-PREBIND-SUM:\s+steps=(\d+)\s+layers=(\d+)\s+uni_avg=([0-9.]+)"
    r"(?:\s+pred_pct=([0-9.]+)%)?\s+"
    r"q8=([0-9.]+)\s+q16=([0-9.]+)\s+q24=([0-9.]+)"
    r"(?:\s+q32=([0-9.]+))?\s+p_res0=([0-9.]+)\s+clean_pct=([0-9.]+)%"
    r"(?:\s+w24=([0-9.]+)\s+q24w=([0-9.]+))?")

DEFAULT_GATE = 0.73
DELIVERY_NTOK = 4
# 導出 DEFAULT_GATE 時用的模型輸入。實測值若偏離，門檻本身就失效（見 inputs_stale）。
DEFAULT_EXPECT_U = 15.5
DEFAULT_EXPECT_P = 0.85


class Agg:
    """以 uni 為權的累加器。

    q 的分母是 `uni_pred`（只有 pred=1 的層），其餘量（uni_avg / p_res0 / clean_pct）
    的母體是全部層 —— 它們不需要預測。
    """

    def __init__(self, ntok: int):
        self.ntok = ntok
        self.n_layers = 0
        self.uni = 0            # 全部層
        self.uni_pred = 0       # 只有 pred=1 的層（q 的分母）
        self.n_pred = 0
        self.h8 = self.h16 = self.h24 = 0.0
        self.h32 = None         # None = 這輪 log 沒有 cov32 欄位
        self.res = 0.0
        self.clean = 0
        self.rows_seen = {}
        # ★ 逐層 res 的**未加權**二階矩：每一層是隨機效應 z 的一次抽樣，所以估變異數時
        #   不能以 uni 加權（加權會把「大 union 的層」放大，拿到的是別的東西）。
        self.res_n = 0
        self.res_sum = 0.0
        self.res_sq = 0.0
        # 第二版：真實寬度 + 歷史年齡
        self.w8 = self.w16 = self.w24 = self.w32 = 0
        self.w_n = 0
        self.hist_seen = {}
        self.h24w = 0.0         # 只累積 hist==3 的層
        self.uni_w = 0
        self.n_w = 0

    def add(self, uni: int, cov8: float, cov16: float, cov24: float,
            res: float, clean: int, rows: int, pred=True, cov32=None,
            hist=None, w8=None, w16=None, w24=None, w32=None):
        self.n_layers += 1
        self.uni += uni
        if pred:
            self.n_pred += 1
            self.uni_pred += uni
            self.h8 += cov8 * uni
            self.h16 += cov16 * uni
            self.h24 += cov24 * uni
            if cov32 is not None:
                self.h32 = (self.h32 or 0.0) + cov32 * uni
            if hist is not None:
                self.hist_seen[hist] = self.hist_seen.get(hist, 0) + 1
                if hist == 3:                      # 滿歷史桶（穩態）
                    self.h24w += cov24 * uni
                    self.uni_w += uni
                    self.n_w += 1
            if w8 is not None:
                self.w8 += w8
                self.w16 += w16
                self.w24 += w24
                self.w32 += w32
                self.w_n += 1
        self.res += res * uni
        self.clean += 1 if clean else 0
        self.rows_seen[rows] = self.rows_seen.get(rows, 0) + 1
        self.res_n += 1
        self.res_sum += res
        self.res_sq += res * res

    def _avg(self, num, den):
        return (float(num) / float(den)) if den else None

    def res_moments(self) -> tuple:
        """逐層 res 的 (mean, sample_var) —— 未加權。層數 < 2 回傳 (None, None)。"""
        if self.res_n < 2:
            return None, None
        m = self.res_sum / float(self.res_n)
        v = (self.res_sq - float(self.res_n) * m * m) / float(self.res_n - 1)
        return m, max(0.0, v)

    def as_dict(self) -> dict:
        u = float(self.uni) if self.uni else 1.0
        up = float(self.uni_pred)
        rm, rv = self.res_moments()
        u_avg = (float(self.uni) / self.n_layers) if self.n_layers else 0.0
        # ★ sigma 由「逐層 res 的變異數」估，**不是**由 clean_pct 反解。
        #   用 clean_pct 反解是循環論證：clean_pct 正是模型要預測的量 E[r_z^U]，
        #   拿它配適出的 sigma 再去降門檻，等於用來放寬的數字 == 放寬後要預測的數字。
        sig = None
        # 注意 `rv is not None` 不能寫成 `rv`：rv == 0.0（res 全等）是合法輸入，
        # 它的答案就是 sigma = 0，寫成 truthy 檢查會把 0 誤當成「沒量到」。
        if _EV is not None and rm is not None and rv is not None and u_avg > 0:
            sig = _EV.sigma_from_res_spread(rv, rm, u_avg)
        return {
            "ntok": self.ntok,
            "layers": self.n_layers,
            "pred_pct": 100.0 * self.n_pred / self.n_layers if self.n_layers else 0.0,
            "uni_avg": (float(self.uni) / self.n_layers) if self.n_layers else 0.0,
            # q 只在真的有預測時才有意義；沒有就用 None，讓報表顯示 n/a 而不是 0.0000
            "q8": (self.h8 / up) if up else None,
            "q16": (self.h16 / up) if up else None,
            "q24": (self.h24 / up) if up else None,
            "q32": (self.h32 / up) if (up and self.h32 is not None) else None,
            "p_res0": self.res / u,
            "clean_pct": 100.0 * self.clean / self.n_layers if self.n_layers else 0.0,
            "rows_hist": dict(sorted(self.rows_seen.items())),
            # 第二版
            "w8_avg": self._avg(self.w8, self.w_n),
            "w16_avg": self._avg(self.w16, self.w_n),
            "w24_avg": self._avg(self.w24, self.w_n),
            "w32_avg": self._avg(self.w32, self.w_n),
            "q24w": (self.h24w / float(self.uni_w)) if self.uni_w else None,
            "n_warm": self.n_w,
            "hist_hist": dict(sorted(self.hist_seen.items())),
            "res_mean": rm,
            "res_sd": (rv ** 0.5) if rv else (0.0 if rm is not None else None),
            "sigma_hat": sig,
        }


def parse(lines) -> tuple:
    """回傳 ({ntok: Agg}, [sum dict])。認不出來的行靜默略過（log 裡混著一堆別的東西）。"""
    groups = {}
    sums = []
    for ln in lines:
        m = PROBE_RE.search(ln)
        if m:
            il, ntok, uni, rows = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            pred = True if m.group(5) is None else (int(m.group(5)) != 0)
            cov8, cov16, cov24 = float(m.group(6)), float(m.group(7)), float(m.group(8))
            cov32 = float(m.group(9)) if m.group(9) is not None else None
            res, clean = float(m.group(10)), int(m.group(11))
            hist = int(m.group(12)) if m.group(12) is not None else None
            if m.group(13) is not None:
                w8, w16 = int(m.group(13)), int(m.group(14))
                w24, w32 = int(m.group(15)), int(m.group(16))
            else:
                w8 = w16 = w24 = w32 = None
            g = groups.setdefault(ntok, Agg(ntok))
            g.add(uni, cov8, cov16, cov24, res, clean, rows, pred, cov32,
                  hist, w8, w16, w24, w32)
            continue
        m = SUM_RE.search(ln)
        if m:
            sums.append({
                "steps": int(m.group(1)), "layers": int(m.group(2)),
                "uni_avg": float(m.group(3)),
                "pred_pct": float(m.group(4)) if m.group(4) is not None else None,
                "q8": float(m.group(5)), "q16": float(m.group(6)), "q24": float(m.group(7)),
                "q32": float(m.group(8)) if m.group(8) is not None else None,
                "p_res0": float(m.group(9)), "clean_pct": float(m.group(10)),
                "w24": float(m.group(11)) if m.group(11) is not None else None,
                "q24w": float(m.group(12)) if m.group(12) is not None else None,
            })
    return {k: groups[k] for k in sorted(groups)}, sums


def f_clean_model(q: float, u: float, p_res0: float) -> float:
    """定價模型的 f_clean（獨立下界），用來驗算實測 clean_pct。"""
    r = p_res0 + (1.0 - p_res0) * q
    return 100.0 * (r ** u)


def inputs_stale(d: dict, expect_u: float, expect_p: float) -> bool:
    """門檻是不是用別的一組模型輸入推出來的？

    q24 的門檻是 `prebind_ev.py --u U --p-res0 p` 的輸出 —— 它是 U 與 p_res0 的函數。
    實測 U 一旦偏離導出門檻時用的那個 U，門檻本身就不再是「對應這次量測的門檻」，
    此時的 PASS/FAIL 沒有意義（2026-09-23：假設 15.5，實測 20.23 ⇒ 門檻 0.731 → 0.543）。
    """
    return abs(d["uni_avg"] - expect_u) > 3.0 or abs(d["p_res0"] - expect_p) > 0.05


def stale_msg(d: dict, expect_u: float, expect_p: float) -> str:
    return ("門檻失效：本輪實測 U={:.2f} / p_res0={:.4f}，與導出門檻時用的 U={:.2f} / "
            "p_res0={:.2f} 不一致 => 先用 prebind_ev.py --u {:.2f} --p-res0 {:.4f} 重算門檻，"
            "再用 --gate <新值> --expect-u {:.2f} --expect-p {:.4f} 重跑。"
            "此輪的 PASS/FAIL 不作數。".format(
                d["uni_avg"], d["p_res0"], expect_u, expect_p,
                d["uni_avg"], d["p_res0"], d["uni_avg"], d["p_res0"]))


def verdict(d: dict, gate: float, expect_u: float = DEFAULT_EXPECT_U,
            expect_p: float = DEFAULT_EXPECT_P) -> tuple:
    """回傳 (pass_bool, [訊息])。判準見規格 §6。"""
    msgs = []
    ok = True

    if d["layers"] == 0:
        return False, ["無效：layers == 0（宣告本輪無效，不解讀其他數字 —— F2 的坑）"]

    # ★ 門檻 0（2026-09-23）：q 到底有沒有被量到。
    # 「預測源沒收集到」與「預測不準」是兩種失敗：前者要修收集，後者才要修預測器/寬度。
    # 把它們混成一個 FAIL 會直接把人導向錯的那一個。
    if d["pred_pct"] <= 0.0:
        return False, [
            "q 不可得：pred=1 的層數 = 0 / {} 層 => 預測源完全沒收集到，gate **無法判定**。".format(
                d["layers"]),
            "  -> 這不是「q 太低」。先查為什麼歷史 ids 沒進來（mirror collect 條件、"
            "decode graph 判定、層索引）。",
            "  -> 這一輪仍然可用的數字：uni_avg = {:.2f}、p_res0 = {:.4f}、"
            "clean_pct = {:.1f}%（三個都不需要預測）。".format(
                d["uni_avg"], d["p_res0"], d["clean_pct"]),
        ]

    if d["pred_pct"] < 100.0:
        msgs.append("注意：只有 {:.1f}% 的層有預測 => q 是那批層上的條件值，"
                    "不是全層的平均值（U／p_res0／clean 仍是全層）".format(d["pred_pct"]))

    q24 = d["q24"]
    msgs.append("主判準 q24 = {:.4f} vs 門檻 {:.2f}  => {}".format(
        q24, gate, "PASS" if q24 >= gate else "FAIL"))
    if q24 < gate:
        ok = False
        msgs.append("  -> 預指派不值得：停在階段 0，改走 pread_usec 路線")

    # 滿歷史桶（穩態）：冷啟動那幾步不該混進來
    q24w = d.get("q24w")
    if q24w is not None:
        delta = q24w - q24
        msgs.append("滿歷史桶 q24w = {:.4f}（hist==3，{} 層；與 q24 差 {:+.4f}）=> {}".format(
            q24w, d["n_warm"], delta,
            "冷啟動幾乎不影響，q24 可當穩態值" if abs(delta) < 0.02 else
            "冷啟動有偏，決策要用 q24w（穩態）而不是 q24"))

    # ★ 真實寬度（取代舊的 rows<3 檢查）：q24 的「24」只是名義值
    w24 = d.get("w24_avg")
    if w24 is not None:
        msgs.append("真實寬度：|ps8|={} |ps16|={} |ps24|={} |ps32|={}（平均值）".format(
            _fmt_w(d["w8_avg"]), _fmt_w(d["w16_avg"]),
            _fmt_w(d["w24_avg"]), _fmt_w(d["w32_avg"])))
        if w24 < 24.0 - 0.5:
            msgs.append("  -> 注意：q24 不是真的 24 寬 —— 相鄰 token 的 top-8 重疊，"
                        "實測聯集只有 {:.1f}。它是「至多 24 寬的預測器」，"
                        "定價時寬度要照 {:.1f} 算。".format(w24, w24))

    # 門檻失效（模型輸入與導出門檻時不一致）=> PASS/FAIL 不作數
    if inputs_stale(d, expect_u, expect_p):
        msgs.append(stale_msg(d, expect_u, expect_p))

    # ★ 模型檢定（2026-09-23 改版）：拿 **q=0** 那一點檢定，不是 q24 那一點。
    #   probe 是純量測、不改行為 => 實測 clean_pct 永遠是「沒有預指派」的基線（q=0）。
    #   舊版拿它去比 `f_clean(q24)`，是拿基線比介入後 —— 兩個不同的點，差值沒有意義，
    #   而且符號固定為負 => 每一輪都會被判成「模型不符」，把 gate 的判決吃掉。
    #   正確做法：sigma 由**另一個統計量**（逐層 res 的變異數 = 二階矩）估出來，再用它
    #   預測 q=0 的 clean。這時 clean_pct（U 階矩）是驗證點，不是配適點 ——
    #   「二階矩預測 U 階矩」才是一個可以為偽的檢定。
    sig = d.get("sigma_hat")
    if sig is None:
        msgs.append("模型檢定：缺 sigma（要 VERBOSE 的逐層 res，或 prebind_ev.py 不在同目錄）"
                    " => 不檢定，門檻沿用 --gate")
    else:
        msgs.append("sigma_hat = {:.3f}（由逐層 res 變異數估：mean={:.4f} sd={:.4f}，n={}；"
                    "**不是**由 clean_pct 反解）".format(
                        sig, d["res_mean"], d["res_sd"], d["layers"]))
        pred0 = _EV.f_clean(0.0, d["uni_avg"], sig, d["p_res0"]) * 100.0
        diff = d["clean_pct"] - pred0
        msgs.append("模型檢定（q=0 點）clean 實測 {:.1f}% vs 模型 {:.1f}% (差 {:+.1f} pp) => {}".format(
            d["clean_pct"], pred0, diff, "OK" if abs(diff) <= 10.0 else "模型不符"))
        if abs(diff) > 10.0:
            ok = False
            msgs.append("  -> 二階矩預測不了 U 階矩：logistic-normal 隨機效應這個函數形式"
                        "不適用，先修模型，不進階段 1")
        # 三區門檻（決策規則事先寫死：step -10% 損益兩平；數字隨實測輸入變）
        g0 = _EV.breakeven_q(0.10, d["uni_avg"], 0.0, d["p_res0"])
        gs = _EV.breakeven_q(0.10, d["uni_avg"], sig, d["p_res0"])
        msgs.append("門檻（step -10% 兩平）：sigma=0 獨立下界 q >= {:.4f}；"
                    "sigma_hat={:.3f} 下 q >= {:.4f}".format(g0, sig, gs))
        if q24 >= g0:
            msgs.append("  -> 落在「連獨立下界都過」區：結論不依賴相關性假設")
        elif q24 >= gs:
            msgs.append("  -> 落在「只在實測相關性下過」區：結論依賴 sigma_hat，"
                        "獨立假設（sigma=0）下同一個 q24 會判 FAIL")
        else:
            msgs.append("  -> 落在「連實測相關性都救不了」區")

    # 寬度選擇（規格 §6 事先寫死的規則：能過門檻的最小寬度）。
    # 只在 16 / 24 之間選 —— 32 不在候選裡，見下面的備援說明。
    if ok:
        width = 16 if d["q16"] >= gate else 24
        msgs.append("寬度選擇（能過門檻的最小寬度）: {} ids/層".format(width))

    # q32 =「備援寬度」，不是寬度選擇的候選。
    # 為什麼：gate 打在 q24（規格 §6），而 q24 不過時主判準已經 FAIL，所以 q32 永遠不可能
    # 成為「能過門檻的最小寬度」——把它放進上面那個迴圈等於偷偷放寬 gate。
    # 它真正的用途是回答一個不同的問題：「q24 不夠時，多買一個 prev 源（額外一行收集）
    # 能不能救？」所以這裡獨立陳述，不併入 PASS/FAIL。這條規則同樣是事先寫死的。
    q32 = d.get("q32")
    if q32 is None:
        msgs.append("備援 q32：本輪 log 沒有 cov32 欄位（舊 binary 或未開 probe）")
    elif q24 < gate:
        msgs.append("備援 q32 = {:.4f}（需 prev 源，比 q24 貴一行收集）=> {}".format(
            q32, "可救 —— 改用寬度 32 才有資格進階段 1"
                 if q32 >= gate else "也救不了 —— 寬度買不到，只能換預測源"))
    else:
        msgs.append("備援 q32 = {:.4f}（q24 已過，不必多買 prev 源）".format(q32))

    # rows：換成歷史源後不再是寬度的證據，只留作「draft 到底有沒有跑」的診斷
    if d["rows_hist"] and max(d["rows_hist"]) == 0:
        msgs.append("note：draft rows 全是 0 => draft 預測在這個形狀下一次都沒收集到"
                    "（已知：MTP draft graph 只跑 nextn block）。寬度現在由歷史源提供，"
                    "q24 不受影響。")
    elif d["rows_hist"] and max(d["rows_hist"]) < 3:
        msgs.append("note：draft rows 最大 = {} < 3（寬度已改由歷史源提供，"
                    "此處僅供診斷 draft）".format(max(d["rows_hist"])))
    return ok, msgs


def _fmt_q(v) -> str:
    return "n/a" if v is None else "{:.4f}".format(v)


def _fmt_w(v) -> str:
    return "n/a" if v is None else "{:.1f}".format(v)


def render(groups: dict, sums: list, gate: float, want_ntok: int,
           expect_u: float = DEFAULT_EXPECT_U, expect_p: float = DEFAULT_EXPECT_P) -> int:
    if not groups:
        print("找不到任何 CGC-PREBIND-PROBE 行。"
              "（若只有 CGC-PREBIND-SUM，請開 CGC_PREBIND_PROBE_VERBOSE=1）", file=sys.stderr)
        if sums:
            print("找到 {} 行 CGC-PREBIND-SUM：".format(len(sums)))
            for s in sums[-5:]:
                print("  steps={steps} layers={layers} uni_avg={uni_avg:.2f} pred={pred} "
                      "q8={q8:.4f} q16={q16:.4f} q24={q24:.4f} q32={q32} "
                      "p_res0={p_res0:.4f} clean_pct={clean_pct:.1f}% "
                      "w24={w24} q24w={q24w}".format(
                          pred=("n/a" if s["pred_pct"] is None else "%.1f%%" % s["pred_pct"]),
                          q32=_fmt_q(s["q32"]),
                          w24=_fmt_w(s["w24"]), q24w=_fmt_q(s["q24w"]), **s))
        return 1

    print("== 按 ntok 分組（加權：q = sum(hit)/sum(uni)，只算 pred=1 的層）==")
    print("{:>5} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>7} {:>8}".format(
        "ntok", "layers", "pred%", "uni_avg", "q8", "q16", "q24", "q32",
        "p_res0", "clean_pct", "w24", "q24w"))
    for ntok, agg in groups.items():
        d = agg.as_dict()
        print("{:>5} {:>8} {:>6.1f}% {:>8.2f} {:>8} {:>8} {:>8} {:>8} {:>8.4f} {:>8.1f}% "
              "{:>7} {:>8}".format(
                  d["ntok"], d["layers"], d["pred_pct"], d["uni_avg"],
                  _fmt_q(d["q8"]), _fmt_q(d["q16"]), _fmt_q(d["q24"]), _fmt_q(d["q32"]),
                  d["p_res0"], d["clean_pct"], _fmt_w(d["w24_avg"]), _fmt_q(d["q24w"])))

    key = want_ntok
    if key not in groups:
        avail = sorted(groups)
        key = DELIVERY_NTOK if DELIVERY_NTOK in groups else avail[-1]
        print("\n（指定的 ntok={} 沒有資料，改用 ntok={}）".format(want_ntok, key))
    d = groups[key].as_dict()

    print("\n== gate 判決（ntok={}，門檻 q24 >= {:.2f}）==".format(key, gate))
    print("draft rows 分佈: {}（歷史源下僅供診斷）".format(d["rows_hist"]))
    if d["hist_hist"]:
        print("歷史年齡分佈 hist(非空歷史個數): {}".format(d["hist_hist"]))
    if d.get("sigma_hat") is not None:
        print("逐層 res: mean={:.4f} sd={:.4f} (n={}) => sigma_hat={:.3f}（由變異數估）".format(
            d["res_mean"], d["res_sd"], d["layers"], d["sigma_hat"]))
    ok, msgs = verdict(d, gate, expect_u, expect_p)
    for m in msgs:
        print("  " + m)

    stale = inputs_stale(d, expect_u, expect_p)
    if stale:
        print("\n結論: 門檻失效 — 用實測 U／p_res0 重算門檻後再判（本輪 PASS/FAIL 不作數）")
        return 3
    print("\n結論: {}".format("PASS — 進階段 1" if ok else "FAIL — 停在階段 0"))
    return 0 if ok else 2


def self_test() -> int:
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # 造 log：ntok=4 三層、ntok=2 一層，外加雜訊行
    def probe(il, ntok, uni, rows, c8, c16, c24, res, clean, c32=None, pred=1,
              hist=3, w8=8, w16=14, w24=19, w32=19):
        # pred=None => 整個欄位不出現（模擬舊 binary），不是印成 "pred=None"
        # hist=None / w8=None => 第二版欄位不出現（模擬舊 binary）
        s = "CGC-PREBIND-PROBE: il={} ntok={} uni={} rows={}".format(il, ntok, uni, rows)
        if pred is not None:
            s += " pred={}".format(pred)
        s += " cov8={:.3f} cov16={:.3f} cov24={:.3f}".format(c8, c16, c24)
        if c32 is not None:
            s += " cov32={:.3f}".format(c32)
        s += " res={:.3f} clean={}".format(res, clean)
        if hist is not None:
            s += " hist={}".format(hist)
        if w8 is not None:
            s += " w8={} w16={} w24={} w32={}".format(w8, w16, w24, w32)
        return s

    lines = [
        "llama_expert_cache: something else entirely",
        probe(0, 4, 16, 3, 0.500, 0.750, 0.875, 0.875, 0, 0.938),
        probe(1, 4, 16, 3, 0.500, 0.750, 0.875, 1.000, 1, 0.938),
        probe(2, 4, 16, 3, 0.500, 0.750, 0.875, 0.750, 0, 0.938),
        probe(0, 2, 8, 3, 1.000, 1.000, 1.000, 1.000, 1, 1.000),
        "CGC-PREBIND-PROBE: this line is malformed",
        "CGC-PREBIND-SUM: steps=3 layers=3 uni_avg=16.00 pred_pct=100.0% "
        "q8=0.500 q16=0.750 q24=0.875 q32=0.938 p_res0=0.875 clean_pct=33.3% "
        "w24=19.0 q24w=0.875",
    ]

    groups, sums = parse(lines)
    check("two groups", sorted(groups) == [2, 4])
    check("noise ignored", groups[4].n_layers == 3)

    d4 = groups[4].as_dict()
    check("q8 weighted", abs(d4["q8"] - 0.500) < 1e-9)
    check("q16 weighted", abs(d4["q16"] - 0.750) < 1e-9)
    check("q24 weighted", abs(d4["q24"] - 0.875) < 1e-9)
    check("q32 weighted", abs(d4["q32"] - 0.938) < 1e-9)
    check("p_res0 weighted", abs(d4["p_res0"] - 0.875) < 1e-9)
    check("uni_avg", abs(d4["uni_avg"] - 16.0) < 1e-9)
    check("clean_pct", abs(d4["clean_pct"] - 100.0 / 3.0) < 1e-6)
    check("rows hist", d4["rows_hist"] == {3: 3})
    check("pred_pct all", abs(d4["pred_pct"] - 100.0) < 1e-9)
    # 第二版欄位
    check("w24 avg", abs(d4["w24_avg"] - 19.0) < 1e-9)
    check("w8 avg", abs(d4["w8_avg"] - 8.0) < 1e-9)
    check("hist hist", d4["hist_hist"] == {3: 3})
    check("q24w equals q24 when all hist==3", abs(d4["q24w"] - 0.875) < 1e-9)
    check("n_warm", d4["n_warm"] == 3)

    # cov32 / pred / hist / w* 選配：沒有那些欄位的舊 log 也要能解析
    old = [probe(0, 4, 16, 3, 0.5, 0.75, 0.875, 0.875, 0, pred=None,
                 hist=None, w8=None)]
    go, _ = parse(old)
    check("q32 absent -> None", go[4].as_dict()["q32"] is None)
    check("pred absent -> treated as 1", go[4].n_pred == 1)
    check("w24 absent -> None", go[4].as_dict()["w24_avg"] is None)
    check("q24w absent -> None", go[4].as_dict()["q24w"] is None)

    # ★ pred=0 的層：不進 q 的分母，但仍貢獻 uni / p_res0 / clean
    mixed = [probe(0, 4, 16, 3, 1.0, 1.0, 1.0, 0.500, 0, 1.0, pred=1),
             probe(1, 4, 16, 0, 0.0, 0.0, 0.0, 1.000, 1, 0.0, pred=0)]
    gm, _ = parse(mixed)
    dm = gm[4].as_dict()
    check("pred=0 excluded from q denominator", abs(dm["q24"] - 1.0) < 1e-9)
    check("pred=0 still counts for uni", abs(dm["uni_avg"] - 16.0) < 1e-9)
    check("pred=0 still counts for p_res0", abs(dm["p_res0"] - 0.75) < 1e-9)
    check("pred=0 still counts for clean", abs(dm["clean_pct"] - 50.0) < 1e-9)
    check("pred_pct half", abs(dm["pred_pct"] - 50.0) < 1e-9)

    # ★ 滿歷史桶：hist<3 的層不進 q24w，但照樣進 q24
    warm = [probe(0, 4, 16, 0, 1.0, 1.0, 1.0, 0.9, 0, 1.0, hist=3),
            probe(1, 4, 16, 0, 0.0, 0.0, 0.0, 0.9, 0, 0.0, hist=1)]
    gw, _ = parse(warm)
    dw = gw[4].as_dict()
    check("q24 over all pred layers", abs(dw["q24"] - 0.5) < 1e-9)
    check("q24w only hist==3", abs(dw["q24w"] - 1.0) < 1e-9)
    check("n_warm counts hist==3 only", dw["n_warm"] == 1)

    # ★ 整輪 pred=0 => 「q 不可得」，不是 FAIL
    nopred = [probe(i, 4, 16, 0, 0.0, 0.0, 0.0, 0.875, 0, 0.0, pred=0) for i in range(5)]
    gn, _ = parse(nopred)
    okn, msgsn = verdict(gn[4].as_dict(), 0.73)
    check("no pred => q unavailable", not okn and any("q 不可得" in m for m in msgsn))
    check("no pred still reports U", any("uni_avg" in m for m in msgsn))

    # 加權必須真的以 uni 為權
    g2, _ = parse([probe(0, 4, 4, 3, 1.0, 1.0, 1.0, 1.0, 1, 1.0),
                   probe(1, 4, 16, 3, 0.0, 0.0, 0.0, 0.0, 0, 0.0)])
    check("weighting by uni", abs(g2[4].as_dict()["q24"] - 4.0 / 20.0) < 1e-9)

    # SUM 行（含第二版欄位）
    check("sum parsed", len(sums) == 1 and abs(sums[0]["q24"] - 0.875) < 1e-9)
    check("sum q32 parsed", abs(sums[0]["q32"] - 0.938) < 1e-9)
    check("sum pred_pct parsed", abs(sums[0]["pred_pct"] - 100.0) < 1e-9)
    check("sum w24 parsed", abs(sums[0]["w24"] - 19.0) < 1e-9)
    check("sum q24w parsed", abs(sums[0]["q24w"] - 0.875) < 1e-9)
    # 舊 SUM 行（沒有 w24/q24w）
    _, sold = parse(["CGC-PREBIND-SUM: steps=3 layers=3 uni_avg=16.00 pred_pct=100.0% "
                     "q8=0.5 q16=0.75 q24=0.875 q32=0.938 p_res0=0.875 clean_pct=33.3%"])
    check("old sum still parses", len(sold) == 1 and sold[0]["w24"] is None)

    # ★ sigma 由變異數估（不是由 clean_pct 反解）：同樣的 mean，res 越散 => sigma_hat 越大
    flat = [probe(i, 4, 20, 0, 0.5, 0.7, 0.8, 0.902, 0, 0.8) for i in range(9)]
    spread = [probe(i, 4, 20, 0, 0.5, 0.7, 0.8, (0.60 if i % 2 else 1.0), 0, 0.8)
              for i in range(9)]
    gf, _ = parse(flat)
    gsv, _ = parse(spread)
    check("flat res -> sigma 0", abs(gf[4].as_dict()["sigma_hat"]) < 1e-9)
    check("spread res -> sigma > 0", gsv[4].as_dict()["sigma_hat"] > 0.3)
    check("sigma reported in verdict",
          any("sigma_hat" in m for m in verdict(gsv[4].as_dict(), 0.73)[1]))

    # gate —— ★ 用一組「與定價模型自洽」的資料，否則模型檢定會先把 ok 打成 False，
    # 測到的就不是 gate 本身。
    # ★ 自洽的點改了（2026-09-23）：probe 不改行為 => clean_pct 是 **q=0** 的基線，
    #   所以要比的是 f_clean(q=0)，不是 f_clean(q24)。這組 res 全等 0.875
    #   => sigma_hat = 0 => 模型 clean = 0.875^16 = 11.81%；
    #   取 9 層中 1 層 clean = 11.11%（差 -0.7 pp）。
    consistent = [probe(i, 4, 16, 3, 0.500, 0.750, 0.875, 0.875, 1 if i < 1 else 0, 0.938)
                  for i in range(9)]
    gc, _ = parse(consistent)
    dc = gc[4].as_dict()
    check("consistent data built", abs(dc["q24"] - 0.875) < 1e-9
          and abs(dc["clean_pct"] - 100.0 / 9.0) < 1e-6)
    check("q=0 model check passes on consistent data",
          any("(差" in m and "OK" in m for m in verdict(dc, 0.73)[1]))

    okpass, msgs = verdict(dc, 0.73)
    check("gate passes at 0.875", okpass)
    okfail, msgs2 = verdict(dc, 0.90)
    check("gate fails at 0.90", not okfail)
    check("width picks 16 when q16 passes", any("16 ids/層" in m for m in msgs))
    okw, msgsw = verdict(dc, 0.80)
    check("width picks 24 when q16 fails", okw and any("24 ids/層" in m for m in msgsw))

    # ★ 門檻失效：實測 U 偏離導出門檻時用的 U => 單獨的結論，不是 PASS/FAIL
    #   這組 U=20.23（與 15.5 差 4.73 > 3）
    stale = [probe(i, 4, 20, 0, 0.500, 0.750, 0.875, 0.902, 1 if i < 3 else 0, 0.938)
             for i in range(9)]
    gs, _ = parse(stale)
    ds = gs[4].as_dict()
    check("stale detected", inputs_stale(ds, 15.5, 0.85))
    _, msgss = verdict(ds, 0.73)
    check("stale message emitted", any("門檻失效" in m for m in msgss))
    check("stale gives exact recompute cmd", any("prebind_ev.py --u 20.00" in m for m in msgss))
    # 重算後的輸入就不該再報失效
    check("stale cleared with measured inputs", not inputs_stale(ds, 20.0, 0.902))

    # 真實寬度：w24 < 24 要說清楚它不是 24 寬
    narrow = [probe(i, 4, 20, 0, 0.5, 0.7, 0.8, 0.9, 1, 0.8, w24=14, w16=11, w8=8, w32=14)
              for i in range(5)]
    gnr, _ = parse(narrow)
    _, msgnr = verdict(gnr[4].as_dict(), 0.73)
    check("narrow width reported", any("不是真的 24 寬" in m for m in msgnr))
    check("widths printed", any("|ps24|=14.0" in m for m in msgnr))

    # q32 是備援，不是寬度候選
    rescue = [probe(i, 4, 16, 3, 0.50, 0.60, 0.70, 0.875, 1 if i < 5 else 0, 0.80)
              for i in range(9)]
    gr, _ = parse(rescue)
    okr, msgsr = verdict(gr[4].as_dict(), 0.73)
    check("q24 fail stays FAIL even if q32 passes", not okr)
    check("q32 rescue reported when q24 short", any("可救" in m for m in msgsr))

    hopeless = [probe(i, 4, 16, 3, 0.40, 0.50, 0.55, 0.875, 1 if i < 3 else 0, 0.60)
                for i in range(9)]
    gh, _ = parse(hopeless)
    okh, msgsh = verdict(gh[4].as_dict(), 0.73)
    check("q32 hopeless reported", not okh and any("也救不了" in m for m in msgsh))

    okp, msgsp = verdict(dc, 0.73)
    check("q32 not needed when q24 passes", any("不必多買" in m for m in msgsp))

    # 部分 pred 要警告
    part = [probe(i, 4, 16, 3, 0.5, 0.75, 0.875, 0.875, 1 if i < 7 else 0, 0.938,
                  pred=(1 if i < 5 else 0)) for i in range(9)]
    gp, _ = parse(part)
    _, msgsp2 = verdict(gp[4].as_dict(), 0.73)
    check("partial pred warns", any("不是全層的平均值" in m for m in msgsp2))

    # 模型驗算單獨測
    bad = [probe(i, 4, 16, 3, 0.5, 0.75, 0.875, 0.875, 0, 0.938) for i in range(9)]
    gb, _ = parse(bad)
    okb, msgsb = verdict(gb[4].as_dict(), 0.73)
    check("model mismatch fails", not okb and any("模型不符" in m for m in msgsb))

    # 無效輪
    empty = Agg(4).as_dict()
    okempty, msgse = verdict(empty, 0.73)
    check("empty invalid", not okempty and "無效" in msgse[0])

    check("f_clean model r=1", abs(f_clean_model(1.0, 15.5, 1.0) - 100.0) < 1e-9)
    check("f_clean model r=p_res0",
          abs(f_clean_model(0.0, 15.5, 0.85) - 100.0 * (0.85 ** 15.5)) < 1e-9)

    # rows 全 0 => 說 draft 沒跑，且明確說 q24 不受影響
    g0, _ = parse([probe(0, 4, 20, 0, 0.5, 0.7, 0.8, 0.9, 0, 0.8)])
    _, m0 = verdict(g0[4].as_dict(), 0.80)
    check("rows all zero note", any("q24 不受影響" in m for m in m0))

    # 空輸入
    g4, s4 = parse([])
    check("empty parse", g4 == {} and s4 == [])

    print("selftest: {}/{} passed".format(ok, ok + len(fail)))
    for f in fail:
        print("  FAIL: {}".format(f))
    return 0 if not fail else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", nargs="?", help="server log 檔（省略則讀 stdin）")
    p.add_argument("--ntok", type=int, default=DELIVERY_NTOK,
                   help="要用哪一組做判決（預設 {} = 交付形狀）".format(DELIVERY_NTOK))
    p.add_argument("--gate", type=float, default=DEFAULT_GATE,
                   help="q24 門檻（預設 {}）".format(DEFAULT_GATE))
    p.add_argument("--expect-u", type=float, default=DEFAULT_EXPECT_U,
                   help="導出 --gate 時用的 uni_size（預設 {}）。實測偏離 >3 => "
                        "門檻失效，退出碼 3".format(DEFAULT_EXPECT_U))
    p.add_argument("--expect-p", type=float, default=DEFAULT_EXPECT_P,
                   help="導出 --gate 時用的 p_res0（預設 {}）。實測偏離 >0.05 => "
                        "門檻失效".format(DEFAULT_EXPECT_P))
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)

    if a.self_test:
        return self_test()
    if not a.log or a.log == "-":
        lines = sys.stdin.read().splitlines()
    else:
        with open(a.log, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    groups, sums = parse(lines)
    return render(groups, sums, a.gate, a.ntok, a.expect_u, a.expect_p)


if __name__ == "__main__":
    sys.exit(main())
