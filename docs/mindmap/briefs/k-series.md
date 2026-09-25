# K0–K5 系列（融合／小 op 群） — 技術白皮書　·　4 ④ 廢棄

> **一句話**：走 K0–K5 系列（kernel 融合、小 op 群），看能不能從 kernel 側拿到可觀的增益。

- 主題：上界／kernel　·　子目標：**C kernel／頻寬效率**（天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領）
- 階段：已結案（廢棄／不適用）　·　④ 廢棄 ＋ 不適用

---

## 1. 目標

走 K0–K5 系列（kernel 融合、小 op 群），看能不能從 kernel 側拿到可觀的增益。

## 2. 判準

格與格的上界（先寫死）

## 3. 結果

K1 不做（G2 禁區）、K4 主機側否證（gpu_union 占步時 92%）、K5 不做、L2 歸零、M-K5 不做；加 thread 只能切 K

## 4. 判定

**4 · ④ 廢棄** — 判死／撤回／不做

> 整條線結案；by-product 是 K3 的邊際單價（已升為 ③b）

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [G4：元素級融合 kernel](g4.md) | 4 | 判 0 ⇒ 不寫融合 kernel（回歸斜率 0.0132 µs/numel ⇒ 融合的 gain 是 0） |
| [OMLX verify kernel ／ 其他 verify 路線](misc-omlx.md) | 4 | 判死（見 OMLX_VERIFY_KERNEL_VERDICT） |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `MEMORY_PERF.md「K0–K5 結案總表」＠2026-09-21` |
| 備註 | 整條線結案；by-product 是 K3 的邊際單價（已升為 ③b） |
| 軸性質 | 天花板軸（不是活躍攻關軸）：受模型形狀與 kernel 物理約束，已知槓桿多半已證偽 ⇒ 持續證偽、只當背景約束與上界；但不能在分類裡消失，否則最大的時間塊無人認領 |
| 對應報告 | 3 份 |

- [K5_GRID_HYPOTHESIS_2026-09-21.md](../../K5_GRID_HYPOTHESIS_2026-09-21.md)
- [K5_VERDICT_2026-09-21.md](../../K5_VERDICT_2026-09-21.md)
- [M_K5_SHAPE_2026-09-21.md](../../M_K5_SHAPE_2026-09-21.md)

---

← [k=3 的 1.43× 飄移定位 ＋ 配對認證](k3-swing.md)　·　[總目錄](index.md)　·　[HTML 版](k-series.html)　·　[G4：元素級融合 kernel →](g4.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
