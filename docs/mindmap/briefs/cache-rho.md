# ρ 路線（按層批次化 prefetch） — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：走 ρ 路線：按層批次化 prefetch，用一次大批次取代多次零散小預取，減少預取本身的開銷。

- 主題：池／快取　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

走 ρ 路線：按層批次化 prefetch，用一次大批次取代多次零散小預取，減少預取本身的開銷。

## 2. 判準

qu gate（覆蓋率 ＋ 視窗）

## 3. 結果

判活：覆蓋 0.849，單獨做 14.36 t/s（+14.2%）；但視窗只有 1.30~1.59 ms，且要付 4.76 ms/步 GPU 插入 ⇒ 有前置條件

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 與 prebind 同價、不可相加

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [CGC_EB_NOFILL 診斷臂（fill 成本）](io-nofill.md) | 3a | fill 3.955 → 0.206 ms/step（−95%）⇒ 機制證；生產口徑同步 fill 僅占 step 4.7% |
| [單段提交（41 段 → 1 段）](s1-segbatch.md) | 3a | A 11.30 → B 20.73 t/s；−40.3 ms/token（45.5%），其中序列化 ~36 ms、fill 僅 3.96 ms |
| [異步 gather 流水線（單段＋miss 後台補＋局部重算）](s1-asyncgather.md) | 3a | 方案設計完成、待實施（無實測）；先跑 Stage B 量 fill 時間 vs GPU 窗口 ⇒ go/no-go |
| [prebind／方案 A（預指派 slot）](cache-prebind.md) | 3a | 判活：qu2/qu3 全過、qu1 邊緣 ⇒ 14.33 t/s（+14.0%）；提前一整步發起 ⇒ 視窗 ≈ 一步 |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/RHO_STAGE0_RESULT_2026-09-23.md／F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md` |
| 備註 | 與 prebind 同價、不可相加 |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 5 份 |

- [F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md](../../F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md)
- [GAP_SPLIT_SWAP_BIAS_2026-09-20.md](../../GAP_SPLIT_SWAP_BIAS_2026-09-20.md)
- [RHO_BATCH_REGRESSION_2026-09-24.md](../../RHO_BATCH_REGRESSION_2026-09-24.md)
- [RHO_MAXQ_SETTLED_2026-09-23.md](../../RHO_MAXQ_SETTLED_2026-09-23.md)
- [RHO_STAGE0_RESULT_2026-09-23.md](../../RHO_STAGE0_RESULT_2026-09-23.md)

---

← [accept rule／dynamic-k／n_max 調整](mtp-accept.md)　·　[總目錄](index.md)　·　[HTML 版](cache-rho.html)　·　[prebind／方案 A（預指派 slot） →](cache-prebind.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
