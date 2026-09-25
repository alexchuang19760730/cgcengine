# 3c：per-expert 重算 kernel — 技術白皮書　·　4 ④ 廢棄

> **一句話**：做 per-expert 重算 kernel：利用 MoE 輸出是線性疊加，只重算 miss 的那幾個 expert，而不是整層重算。

- 主題：缺失處理　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：已結案（廢棄／不適用）　·　④ 廢棄 ＋ 不適用

---

## 1. 目標

做 per-expert 重算 kernel：利用 MoE 輸出是線性疊加，只重算 miss 的那幾個 expert，而不是整層重算。

## 2. 判準

文件自寫「依賴 3b」；greedy 逐 rep 相同 ＋ 落在 13.6~14.2

## 3. 結果

不做（依賴鏈斷；上限同受 ~5% 約束；舊估 +13~18% 來自已作廢的非生產 cell）

## 4. 判定

**4 · ④ 廢棄** — 判死／撤回／不做

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [IO 請求形狀（合併 pread）](io-shape.md) | 4 | 一個 expert 三段相隔 88~115 MB ⇒ 3→1 要付 182.6× 位元組；合併邏輯早已存在（preadv）；固定開銷占 90% |
| [M-W：WORKERS 8→2](io-mw.md) | 4 | us/job −11.1%、us_per_miss −14.7%，但 t/s −0.6% ⇒ 未分離（by-product：IO 只占 step ≈9%） |
| [CGC_LAYER_AHEAD_PREFETCH（提前一層）](io-layer-ahead.md) | 4 | −7.1%（未分離、方向偏負） |
| [M-PF：背景 neighbour prefetch](io-mpf.md) | 4 | 撤回（讀入位元組 1.98×，實測最好只到 35%） |
| [S1 早期診斷系列（09-16/17）](s1-refuted.md) | 4 | 多輪被自己否證：slot owner 推論被推翻、時序推論被推翻、09-16 那七輪「第一個分歧」全部不可引用（同時踩三個盲點） |
| [3b：fill 觸發點搬出 hook ＋ batch 化](miss-3b.md) | 4 | 判死：合併邏輯早已存在、幾何 182.6×、生產口徑上界 ~9% ⇒ 收益 ~2.9% |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/STEP3_PER_EXPERT_RECOMPUTE_2026-09-24.md §3` |
| 備註 |  |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 0 份 |

- （無）

---

← [3b：fill 觸發點搬出 hook ＋ batch 化](miss-3b.md)　·　[總目錄](index.md)　·　[HTML 版](miss-3c.html)　·　[異步 gather 流水線（單段＋miss 後台補＋局部重算） →](s1-asyncgather.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
