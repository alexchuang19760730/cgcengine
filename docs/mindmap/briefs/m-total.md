# ① 攻關成功（pp≥250 ∧ tg>12.57） — 技術白皮書　·　1 ① 攻關成功

> **一句話**：同一次量測裡同時達成 prefill ≥250 與 decode >12.57 t/s——只有這樣才算「① 攻關成功」。

- 主題：交付里程碑　·　子目標：**不適用**（非攻關軸：約束／帳本／上界算術／入口索引／跨線產品）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

同一次量測裡同時達成 prefill ≥250 與 decode >12.57 t/s——只有這樣才算「① 攻關成功」。

## 2. 判準

兩件事同時成立

## 3. 結果

⛔ 空 —— 最接近的一次是同 cell pp 260.41 ＋ tg 12.195（差 3%）

## 4. 判定

**1 · ① 攻關成功** — prefill ≥ 250 ∧ decode 比目前最好還要好（目前最好＝交付錨點 12.57）

> 目前無任何一格是 ①

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [prefill ≥ 250（交付 cell）](m-prefill250.md) | 2 | 9 次 launch ≥250，最高 296.24；乾淨視窗 283.01；同期 decode 11.49~12.20 |
| [swap 結構修復（L0–L4 + P0/P1/P2）](sys-swap.md) | 3b | launch swap 0、decode 11.49、thermal NOMINAL（commit efba7c1d5） |
| [server 窗口／box 准入（單一來源閘門）](sys-window.md) | 3b | BOX_ADMISSION_SINGLE_SOURCE 定為單一來源；SERVER_WINDOW_LEDGER 記錄逐次窗口 |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/MILESTONE_MAP_RECHECK_2026-09-25.md` |
| 備註 | 目前無任何一格是 ① |
| 軸性質 | 非攻關軸：約束／帳本／上界算術／入口索引／跨線產品 |
| 對應報告 | 0 份 |

- （無）

---

← [prefill ≥ 250（交付 cell）](m-prefill250.md)　·　[總目錄](index.md)　·　[HTML 版](m-total.html)　·　[decode ≥ 25（M-25） →](m-decode25.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
