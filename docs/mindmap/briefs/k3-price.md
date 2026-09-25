# K3 邊際單價（dispatch 成本） — 技術白皮書　·　3b ③b 實驗目標達成（可放生產）

> **一句話**：量出 K3 的邊際單價（多一次 dispatch 要付多少），判斷把 k 從 2 加到 3 划不划算。

- 主題：上界／kernel　·　子目標：**C kernel／頻寬效率**（天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

量出 K3 的邊際單價（多一次 dispatch 要付多少），判斷把 k 從 2 加到 3 划不划算。

## 2. 判準

量到邊際價而非平均價

## 3. 結果

45.6（平均）作廢 → 邊際 0.0195%/dispatch/步（17.9 µs）⇒ 所有舊「融合能省 X%」的算術全部作廢

## 4. 判定

**3b · ③b 實驗目標達成（可放生產）** — 產物已可放進生產級設置：不破壞正確性 ∧ 成本可接受 ∧ 無前置條件

> ★ KNOB_INVENTORY 自記：那晚的 budget sweep 無效（撤回）

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [shape／knob 世界模型（14–15 個旋鈕的邊界）](shape-knobs.md) | 3b | SHAPE_KNOB_LANDED 證實落地；SHAPE_ROOFLINE 抓到「down 是 IQ3_S 不是 IQ3_XXS」等量測事實；BANDWIDTH_CEILING 1 |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/K3_PRICE_MEASURED_2026-09-21.md` |
| 備註 | ★ KNOB_INVENTORY 自記：那晚的 budget sweep 無效（撤回） |
| 軸性質 | 天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領 |
| 對應報告 | 5 份 |

- [K3_CEILING_REAUDIT_2026-09-21.md](../../K3_CEILING_REAUDIT_2026-09-21.md)
- [K3_FOOTPRINT_AND_SHAPEKNOB_20260923_0659.html](../../K3_FOOTPRINT_AND_SHAPEKNOB_20260923_0659.html)
- [K3_PRICE_MEASURED_2026-09-21.md](../../K3_PRICE_MEASURED_2026-09-21.md)
- [KNOB_FULL_MAP_15_2026-09-22.md](../../KNOB_FULL_MAP_15_2026-09-22.md)
- [KNOB_INVENTORY_VERIFIED_2026-09-22.md](../../KNOB_INVENTORY_VERIFIED_2026-09-22.md)

---

← [cb 口徑定讞（42 vs 74）](cache-cb.md)　·　[總目錄](index.md)　·　[HTML 版](k3-price.html)　·　[k=3 的 1.43× 飄移定位 ＋ 配對認證 →](k3-swing.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
