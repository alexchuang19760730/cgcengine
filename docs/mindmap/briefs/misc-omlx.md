# OMLX verify kernel ／ 其他 verify 路線 — 技術白皮書　·　4 ④ 廢棄

> **一句話**：看 OMLX verify kernel 與其他 verify 路線能不能借來用，省下自己寫的成本。

- 主題：MTP／spec　·　子目標：**C kernel／頻寬效率**（天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領）
- 階段：已結案（廢棄／不適用）　·　④ 廢棄 ＋ 不適用

---

## 1. 目標

看 OMLX verify kernel 與其他 verify 路線能不能借來用，省下自己寫的成本。

## 2. 判準

同 cell 配對

## 3. 結果

判死（見 OMLX_VERIFY_KERNEL_VERDICT）

## 4. 判定

**4 · ④ 廢棄** — 判死／撤回／不做

> ⚠ 原歸 M；verify kernel 屬 GPU 計算／頻寬 ⇒ 改歸 C 天花板軸。

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [K0–K5 系列（融合／小 op 群）](k-series.md) | 4 | K1 不做（G2 禁區）、K4 主機側否證（gpu_union 占步時 92%）、K5 不做、L2 歸零、M-K5 不做；加 thread 只能切 K |
| [G4：元素級融合 kernel](g4.md) | 4 | 判 0 ⇒ 不寫融合 kernel（回歸斜率 0.0132 µs/numel ⇒ 融合的 gain 是 0） |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/OMLX_VERIFY_KERNEL_VERDICT_2026-09-24.md` |
| 備註 | ⚠ 原歸 M；verify kernel 屬 GPU 計算／頻寬 ⇒ 改歸 C 天花板軸。 |
| 軸性質 | 天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領 |
| 對應報告 | 1 份 |

- [OMLX_VERIFY_KERNEL_VERDICT_2026-09-24.md](../../OMLX_VERIFY_KERNEL_VERDICT_2026-09-24.md)

---

← [M3／M4／M5 離開條件（09-17 期）](m3.md)　·　[總目錄](index.md)　·　[HTML 版](misc-omlx.html)　·　[入口／索引／決策頁（非實驗） →](na-entry.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
