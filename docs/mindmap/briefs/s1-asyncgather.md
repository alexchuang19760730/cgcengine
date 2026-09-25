# 異步 gather 流水線（單段＋miss 後台補＋局部重算） — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：在單段提交下用 GPU presence/slot table：命中即算、miss 只寫清單並 MASK，CPU 後台異步 fill 與下一步 GPU 重疊，再只補算 miss expert（線性疊加）⇒ 拿到單段的 ×1.7~1.8 同時保住正確性。

- 主題：缺失處理　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產
- 原始方案書：[S1_ASYNC_GATHER_PIPELINE_2026-09-25.html](../../S1_ASYNC_GATHER_PIPELINE_2026-09-25.html)

---

## 1. 目標

在單段提交下用 GPU presence/slot table：命中即算、miss 只寫清單並 MASK，CPU 後台異步 fill 與下一步 GPU 重疊，再只補算 miss expert（線性疊加）⇒ 拿到單段的 ×1.7~1.8 同時保住正確性。

## 2. 判準

預先寫死：① 補算在 sampler 消費前完成的時序窗口 ② MASK＋補算 M1 bit-exact ③ 乾淨 cell r3 decode

## 3. 結果

方案設計完成、待實施（無實測）；先跑 Stage B 量 fill 時間 vs GPU 窗口 ⇒ go/no-go

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 不預測 ids（異於 prebind/ρ）、用真實 ids＋異步重疊；收益【推算】上界 ~19–20 t/s、不宣稱 25

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [CGC_EB_NOFILL 診斷臂（fill 成本）](io-nofill.md) | 3a | fill 3.955 → 0.206 ms/step（−95%）⇒ 機制證；生產口徑同步 fill 僅占 step 4.7% |
| [單段提交（41 段 → 1 段）](s1-segbatch.md) | 3a | A 11.30 → B 20.73 t/s；−40.3 ms/token（45.5%），其中序列化 ~36 ms、fill 僅 3.96 ms |
| [ρ 路線（按層批次化 prefetch）](cache-rho.md) | 3a | 判活：覆蓋 0.849，單獨做 14.36 t/s（+14.2%）；但視窗只有 1.30~1.59 ms，且要付 4.76 ms/步 GPU 插入 ⇒ 有前置條件 |
| [prebind／方案 A（預指派 slot）](cache-prebind.md) | 3a | 判活：qu2/qu3 全過、qu1 邊緣 ⇒ 14.33 t/s（+14.0%）；提前一整步發起 ⇒ 視窗 ≈ 一步 |

## 6. 與其它路線的關鍵差異

| 對象 | 它的做法／結果 | 本條目差別 |
|---|---|---|
| prebind／方案 A | 跨 token 預測 ids、加寬 24 候選、影子 38.3 ms、q24=0.354 | 不預測、用真實 ids、保持 top-8、不加寬 |
| ρ 按層批次 | 同 token 近似 gate（跳 attn）、ρ=0.86 但窗口 1.3–1.6 ms、insert 4.76 | 不做近似、miss 後台補、不依賴重合率 |
| miss-3b | 讓同步 fill 變便宜（合併 pread）、收益 ~2.9% 判死 | 讓 fill 離開同步路徑（異步）、不是變便宜 |
| miss-3c | per-expert 重算、自寫依賴 3b、舊估 +13–18% 作廢 | 異步 drain＋雙緩衝補依賴、重算只為加回、收益重測 |

## 7. 成敗點（風險 → 驗證）

| # | 風險 | 驗證 |
|---|---|---|
| 1 | 時序窗口（最關鍵）：一步 miss 的 fill＋補算要在 sampler 消費前完成 | Stage B：量 fill 時間 vs 下一步 GPU 窗口 |
| 2 | 雙緩衝同步：Metal 單隊列、fill 與當前 gather 爭用 | MAXQ 保險絲、背景 fill 主動讓步 |
| 3 | 異步 bit-exact：MASK＋補算要 M1 逐位相同 | 對 host BATCHDBG、M1 oracle |
| 4 | bg 爭用：背景 fill 搶 IO／頻寬 | 對照 bg on/off |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/S1_ASYNC_GATHER_PIPELINE_2026-09-25.md` |
| 備註 | 不預測 ids（異於 prebind/ρ）、用真實 ids＋異步重疊；收益【推算】上界 ~19–20 t/s、不宣稱 25 |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 0 份 |

- （無）

---

← [3c：per-expert 重算 kernel](miss-3c.md)　·　[總目錄](index.md)　·　[HTML 版](s1-asyncgather.html)　·　[MTP 儀器化（接進 llama-bench） →](mtp-instrument.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
