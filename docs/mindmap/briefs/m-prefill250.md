# prefill ≥ 250（交付 cell） — 技術白皮書　·　2 ② 攻關過線

> **一句話**：把 prefill 推到 ≥250 t/s——「250 / 25」裡的 250 那一半，且要在交付 cell 上成立（不是降級視窗的峰值）。

- 主題：交付里程碑　·　子目標：**不適用**（非攻關軸：約束／帳本／上界算術／入口索引／跨線產品）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

把 prefill 推到 ≥250 t/s——「250 / 25」裡的 250 那一半，且要在交付 cell 上成立（不是降級視窗的峰值）。

## 2. 判準

門檻 250 t/s（同 cell `-p 2048`）

## 3. 結果

9 次 launch ≥250，最高 296.24；乾淨視窗 283.01；同期 decode 11.49~12.20

## 4. 判定

**2 · ② 攻關過線** — prefill ≥ 250 ∧ decode 跟現在差不多（達標但未超越）

> 達標但 decode 未超越 12.57 ⇒ 只能 ②

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [① 攻關成功（pp≥250 ∧ tg>12.57）](m-total.md) | 1 | ⛔ 空 —— 最接近的一次是同 cell pp 260.41 ＋ tg 12.195（差 3%） |
| [swap 結構修復（L0–L4 + P0/P1/P2）](sys-swap.md) | 3b | launch swap 0、decode 11.49、thermal NOMINAL（commit efba7c1d5） |
| [server 窗口／box 准入（單一來源閘門）](sys-window.md) | 3b | BOX_ADMISSION_SINGLE_SOURCE 定為單一來源；SERVER_WINDOW_LEDGER 記錄逐次窗口 |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `MILESTONE_MAP_RECHECK_2026-09-25.md §5.2` |
| 備註 | 達標但 decode 未超越 12.57 ⇒ 只能 ② |
| 軸性質 | 非攻關軸：約束／帳本／上界算術／入口索引／跨線產品 |
| 對應報告 | 19 份 |

- [PREFILL250_CERTIFIABILITY_20260916.html](../../PREFILL250_CERTIFIABILITY_20260916.html)
- [PREFILL250_CONDITIONAL_DELIVERY_20260916.html](../../PREFILL250_CONDITIONAL_DELIVERY_20260916.html)
- [PREFILL250_CONDITIONAL_DELIVERY_20260916.md](../../PREFILL250_CONDITIONAL_DELIVERY_20260916.md)
- [PREFILL250_CONFIGURATION_GUIDE_2026-09-15.html](../../PREFILL250_CONFIGURATION_GUIDE_2026-09-15.html)
- [PREFILL250_DECODE25_STRATEGY_20260919.html](../../PREFILL250_DECODE25_STRATEGY_20260919.html)
- [PREFILL250_DECODE25_VERDICT_2026-09-20.md](../../PREFILL250_DECODE25_VERDICT_2026-09-20.md)
- [PREFILL250_DECODE25_WHITEPAPER_2026-09-15.html](../../PREFILL250_DECODE25_WHITEPAPER_2026-09-15.html)
- [PREFILL250_DECODE25_WHITEPAPER_20260915_1745.html](../../PREFILL250_DECODE25_WHITEPAPER_20260915_1745.html)
- [PREFILL250_DECODE25_WHITEPAPER_20260915_1810.html](../../PREFILL250_DECODE25_WHITEPAPER_20260915_1810.html)
- [PREFILL250_DECODE25_WHITEPAPER_20260915_1900.html](../../PREFILL250_DECODE25_WHITEPAPER_20260915_1900.html)
- [PREFILL250_DECODE25_WHITEPAPER_20260915_2240.html](../../PREFILL250_DECODE25_WHITEPAPER_20260915_2240.html)
- [PREFILL250_DECODE25_WHITEPAPER_20260915_2355.html](../../PREFILL250_DECODE25_WHITEPAPER_20260915_2355.html)
- [PREFILL250_DECODE_MILESTONE_20260917_1510.html](../../PREFILL250_DECODE_MILESTONE_20260917_1510.html)
- [PREFILL250_MET_AND_S1_REJUDGE_2026-09-24.md](../../PREFILL250_MET_AND_S1_REJUDGE_2026-09-24.md)
- [PREFILL250_THERMAL_TRANSIENT_20260916.html](../../PREFILL250_THERMAL_TRANSIENT_20260916.html)
- [ROADMAP_PREFILL250_DECODE25_2026-09-13.md](../../ROADMAP_PREFILL250_DECODE25_2026-09-13.md)
- [ROADMAP_PREFILL250_DECODE25_2026-09-14.html](../../roadmap-2026-09-14/ROADMAP_PREFILL250_DECODE25_2026-09-14.html)
- [ROADMAP_PREFILL250_DECODE25_2026-09-14_desktop.png](../../roadmap-2026-09-14/_shots/ROADMAP_PREFILL250_DECODE25_2026-09-14_desktop.png)
- [ROADMAP_PREFILL250_DECODE25_2026-09-14_mobile.png](../../roadmap-2026-09-14/_shots/ROADMAP_PREFILL250_DECODE25_2026-09-14_mobile.png)

---

[總目錄](index.md)　·　[HTML 版](m-prefill250.html)　·　[① 攻關成功（pp≥250 ∧ tg>12.57） →](m-total.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
