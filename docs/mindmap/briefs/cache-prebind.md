# prebind／方案 A（預指派 slot） — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：做 prebind／方案 A：預先指派 slot，消掉執行期的 slot 查表與指派成本。

- 主題：池／快取　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

做 prebind／方案 A：預先指派 slot，消掉執行期的 slot 查表與指派成本。

## 2. 判準

預先寫死 qu 門檻

## 3. 結果

判活：qu2/qu3 全過、qu1 邊緣 ⇒ 14.33 t/s（+14.0%）；提前一整步發起 ⇒ 視窗 ≈ 一步

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> ★ 方案 A 的 h=0.03 判死不適用於 S1（ids 由 GPU 端 get_rows 產生）

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [CGC_EB_NOFILL 診斷臂（fill 成本）](io-nofill.md) | 3a | fill 3.955 → 0.206 ms/step（−95%）⇒ 機制證；生產口徑同步 fill 僅占 step 4.7% |
| [單段提交（41 段 → 1 段）](s1-segbatch.md) | 3a | A 11.30 → B 20.73 t/s；−40.3 ms/token（45.5%），其中序列化 ~36 ms、fill 僅 3.96 ms |
| [異步 gather 流水線（單段＋miss 後台補＋局部重算）](s1-asyncgather.md) | 3a | 方案設計完成、待實施（無實測）；先跑 Stage B 量 fill 時間 vs GPU 窗口 ⇒ go/no-go |
| [ρ 路線（按層批次化 prefetch）](cache-rho.md) | 3a | 判活：覆蓋 0.849，單獨做 14.36 t/s（+14.2%）；但視窗只有 1.30~1.59 ms，且要付 4.76 ms/步 GPU 插入 ⇒ 有前置條件 |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/PREBIND_VERDICT_SETTLED_2026-09-24.md` |
| 備註 | ★ 方案 A 的 h=0.03 判死不適用於 S1（ids 由 GPU 端 get_rows 產生） |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 7 份 |

- [H_MEASURED_2026-09-24.md](../../H_MEASURED_2026-09-24.md)
- [H_MEASURED_A_VERDICT_2026-09-24.md](../../H_MEASURED_A_VERDICT_2026-09-24.md)
- [PREBIND_SLOT_DESIGN_2026-09-23.md](../../PREBIND_SLOT_DESIGN_2026-09-23.md)
- [PREBIND_STAGE0_RESULT_2026-09-23.md](../../PREBIND_STAGE0_RESULT_2026-09-23.md)
- [PREBIND_STAGE0_SPEC_2026-09-23.md](../../PREBIND_STAGE0_SPEC_2026-09-23.md)
- [PREBIND_TRADEOFF_2026-09-23.md](../../PREBIND_TRADEOFF_2026-09-23.md)
- [PREBIND_VERDICT_SETTLED_2026-09-24.md](../../PREBIND_VERDICT_SETTLED_2026-09-24.md)

---

← [ρ 路線（按層批次化 prefetch）](cache-rho.md)　·　[總目錄](index.md)　·　[HTML 版](cache-prebind.html)　·　[cb 口徑定讞（42 vs 74） →](cache-cb.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
