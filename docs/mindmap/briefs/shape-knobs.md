# shape／knob 世界模型（14–15 個旋鈕的邊界） — 技術白皮書　·　3b ③b 實驗目標達成（可放生產）

> **一句話**：建立 shape／knob 的世界模型：把 14–15 個旋鈕各自的邊界量出來，讓「改哪個旋鈕有用」有據可依。

- 主題：上界／kernel　·　子目標：**C kernel／頻寬效率**（天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

建立 shape／knob 的世界模型：把 14–15 個旋鈕各自的邊界量出來，讓「改哪個旋鈕有用」有據可依。

## 2. 判準

θ(M) 曲線 ＋ roofline ＋ 落地驗證

## 3. 結果

SHAPE_KNOB_LANDED 證實落地；SHAPE_ROOFLINE 抓到「down 是 IQ3_S 不是 IQ3_XXS」等量測事實；BANDWIDTH_CEILING 100% 掃描；MMAP_CACHE_KERNEL_25 判死（mmap 只作用在 host 側 66.42 ms）

## 4. 判定

**3b · ③b 實驗目標達成（可放生產）** — 產物已可放進生產級設置：不破壞正確性 ∧ 成本可接受 ∧ 無前置條件

> ★ BEST_SHAPE 標「唯一沒被結論覆蓋、還活著的方向」（一處 switch case）

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [K3 邊際單價（dispatch 成本）](k3-price.md) | 3b | 45.6（平均）作廢 → 邊際 0.0195%/dispatch/步（17.9 µs）⇒ 所有舊「融合能省 X%」的算術全部作廢 |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/SHAPE_WORLD_MODEL_2026-09-23.md（已判死與 3% 軸並存）` |
| 備註 | ★ BEST_SHAPE 標「唯一沒被結論覆蓋、還活著的方向」（一處 switch case） |
| 軸性質 | 天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領 |
| 對應報告 | 15 份 |

- [BANDWIDTH_CEILING_100PCT_2026-09-23.html](../../BANDWIDTH_CEILING_100PCT_2026-09-23.html)
- [BEST_SHAPE_IQ3XXS_M4_2026-09-22.md](../../BEST_SHAPE_IQ3XXS_M4_2026-09-22.md)
- [FP_ORDER_SHAPE_2026-09-22.md](../../FP_ORDER_SHAPE_2026-09-22.md)
- [FP_ORDER_TARGETS_2026-09-22.md](../../FP_ORDER_TARGETS_2026-09-22.md)
- [GPU_CEILING_STEP2_2026-09-22.md](../../GPU_CEILING_STEP2_2026-09-22.md)
- [IDEAL_SHAPE_THEORETICAL_2026-09-23.html](../../IDEAL_SHAPE_THEORETICAL_2026-09-23.html)
- [MMAP_CACHE_KERNEL_25_2026-09-21.md](../../MMAP_CACHE_KERNEL_25_2026-09-21.md)
- [SHAPE_FOR_25_2026-09-22.md](../../SHAPE_FOR_25_2026-09-22.md)
- [SHAPE_KNOB_LANDED_2026-09-22.md](../../SHAPE_KNOB_LANDED_2026-09-22.md)
- [SHAPE_OVERLAP_INSIGHT_2026-09-23.html](../../SHAPE_OVERLAP_INSIGHT_2026-09-23.html)
- [SHAPE_PARAMETER_MAP_2026-09-23.html](../../SHAPE_PARAMETER_MAP_2026-09-23.html)
- [SHAPE_ROOFLINE_2026-09-22.md](../../SHAPE_ROOFLINE_2026-09-22.md)
- [SHAPE_ROOFLINE_REVIEW_2026-09-22.md](../../SHAPE_ROOFLINE_REVIEW_2026-09-22.md)
- [SHAPE_WORLD_MODEL_2026-09-23.md](../../SHAPE_WORLD_MODEL_2026-09-23.md)
- [SHAPE_WORLD_MODEL_VISUAL_2026-09-23.html](../../SHAPE_WORLD_MODEL_VISUAL_2026-09-23.html)

---

← [G1：可達上界 / G1-G7 sweep](g1.md)　·　[總目錄](index.md)　·　[HTML 版](shape-knobs.html)　·　[IOCACHE 約束承認 ＋ 段邊界 S2 →](io-constraint.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
