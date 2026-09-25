# swap 結構修復（L0–L4 + P0/P1/P2） — 技術白皮書　·　3b ③b 實驗目標達成（可放生產）

> **一句話**：修好 swap 結構（L0–L4 ＋ P0／P1／P2），讓實驗跑得起來、不被 OOM／swap 殺掉。

- 主題：系統／記憶體　·　子目標：**不適用**（非攻關軸：約束／帳本／上界算術／入口索引／跨線產品）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

修好 swap 結構（L0–L4 ＋ P0／P1／P2），讓實驗跑得起來、不被 OOM／swap 殺掉。

## 2. 判準

超訂 4838 MiB ⇒ 目標 launch swap → 0

## 3. 結果

launch swap 0、decode 11.49、thermal NOMINAL（commit efba7c1d5）

## 4. 判定

**3b · ③b 實驗目標達成（可放生產）** — 產物已可放進生產級設置：不破壞正確性 ∧ 成本可接受 ∧ 無前置條件

> ⚠ 執行線主張、本線未複核；另判「pool 納入 wired 級管理」不建議

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [prefill ≥ 250（交付 cell）](m-prefill250.md) | 2 | 9 次 launch ≥250，最高 296.24；乾淨視窗 283.01；同期 decode 11.49~12.20 |
| [① 攻關成功（pp≥250 ∧ tg>12.57）](m-total.md) | 1 | ⛔ 空 —— 最接近的一次是同 cell pp 260.41 ＋ tg 12.195（差 3%） |
| [server 窗口／box 准入（單一來源閘門）](sys-window.md) | 3b | BOX_ADMISSION_SINGLE_SOURCE 定為單一來源；SERVER_WINDOW_LEDGER 記錄逐次窗口 |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/SWAP_STRUCTURAL_FIX_2026-09-24.md／SWAP_P0P1P2_FIX_*` |
| 備註 | ⚠ 執行線主張、本線未複核；另判「pool 納入 wired 級管理」不建議 |
| 軸性質 | 非攻關軸：約束／帳本／上界算術／入口索引／跨線產品 |
| 對應報告 | 2 份 |

- [SWAP_P0P1P2_FIX_20260924_1400.html](../../SWAP_P0P1P2_FIX_20260924_1400.html)
- [SWAP_STRUCTURAL_FIX_2026-09-24.md](../../SWAP_STRUCTURAL_FIX_2026-09-24.md)

---

← [IOCACHE 約束承認 ＋ 段邊界 S2](io-constraint.md)　·　[總目錄](index.md)　·　[HTML 版](sys-swap.html)　·　[server 窗口／box 准入（單一來源閘門） →](sys-window.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
