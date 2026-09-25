# 單段提交（41 段 → 1 段） — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：把 41 段提交合成 1 段（CGC_SEG_BATCH＋CGC_B_SCHEME），消掉段與段之間的同步與序列化開銷。

- 主題：S1／段邊界　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

把 41 段提交合成 1 段（CGC_SEG_BATCH＋CGC_B_SCHEME），消掉段與段之間的同步與序列化開銷。

## 2. 判準

同 build 同 cell 同 pool，兩臂皆無 spec（無驗收門檻）

## 3. 結果

A 11.30 → B 20.73 t/s；−40.3 ms/token（45.5%），其中序列化 ~36 ms、fill 僅 3.96 ms

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 無 hook ⇒ 無 demand fill ⇒ 正確性未證 ⇒ 不能進生產

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [CGC_EB_NOFILL 診斷臂（fill 成本）](io-nofill.md) | 3a | fill 3.955 → 0.206 ms/step（−95%）⇒ 機制證；生產口徑同步 fill 僅占 step 4.7% |
| [異步 gather 流水線（單段＋miss 後台補＋局部重算）](s1-asyncgather.md) | 3a | 方案設計完成、待實施（無實測）；先跑 Stage B 量 fill 時間 vs GPU 窗口 ⇒ go/no-go |
| [ρ 路線（按層批次化 prefetch）](cache-rho.md) | 3a | 判活：覆蓋 0.849，單獨做 14.36 t/s（+14.2%）；但視窗只有 1.30~1.59 ms，且要付 4.76 ms/步 GPU 插入 ⇒ 有前置條件 |
| [prebind／方案 A（預指派 slot）](cache-prebind.md) | 3a | 判活：qu2/qu3 全過、qu1 邊緣 ⇒ 14.33 t/s（+14.0%）；提前一整步發起 ⇒ 視窗 ≈ 一步 |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `Backup/seg_batch_s1_pairs/abba_212809.json／docs/S1_LINE_VERDICT_2026-09-25.md` |
| 備註 | 無 hook ⇒ 無 demand fill ⇒ 正確性未證 ⇒ 不能進生產 |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 7 份 |

- [S1_CAPTURE_ROUND2_20260916_2035.html](../../S1_CAPTURE_ROUND2_20260916_2035.html)
- [S1_DIAGNOSTIC_ABORT_ROOT_CAUSE_2026-09-20.md](../../S1_DIAGNOSTIC_ABORT_ROOT_CAUSE_2026-09-20.md)
- [S1_DIAGNOSTIC_ABORT_ROOT_CAUSE_20260920_0715.html](../../S1_DIAGNOSTIC_ABORT_ROOT_CAUSE_20260920_0715.html)
- [S1_FRONTIER_2026-09-18.md](../../S1_FRONTIER_2026-09-18.md)
- [S1_INSTRUMENT_AND_MEMORY_DELIVERY_20260917_0320.html](../../S1_INSTRUMENT_AND_MEMORY_DELIVERY_20260917_0320.html)
- [S1_TIMING_SEGMENT_20260917.html](../../S1_TIMING_SEGMENT_20260917.html)
- [S1_WINDOW_ANCHOR_AND_LAYER1_20260917.html](../../S1_WINDOW_ANCHOR_AND_LAYER1_20260917.html)

---

← [S1 探針臂：slot table 放 GPU（數值身分）](s1-probe.md)　·　[總目錄](index.md)　·　[HTML 版](s1-segbatch.html)　·　[S1 早期診斷系列（09-16/17） →](s1-refuted.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
