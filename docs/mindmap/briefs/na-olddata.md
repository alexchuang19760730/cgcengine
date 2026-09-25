# 舊口徑數據報告（Gemma4／MTP_BENCHMARK_WIN8GB／TPOT 路線圖） — 技術白皮書　·　4 ④ 廢棄

> **一句話**：整理舊口徑數據報告（Gemma4、MTP_BENCHMARK_WIN8GB、TPOT 路線圖）——目的是標明它們不可再引用。

- 主題：不適用　·　子目標：**不適用**（非攻關軸：約束／帳本／上界算術／入口索引／跨線產品）
- 階段：已結案（廢棄／不適用）　·　④ 廢棄 ＋ 不適用

---

## 1. 目標

整理舊口徑數據報告（Gemma4、MTP_BENCHMARK_WIN8GB、TPOT 路線圖）——目的是標明它們不可再引用。

## 2. 判準

口徑是否仍有效

## 3. 結果

作廢：09-17 起 decode 一律 llama-bench、warm-skip 口徑（`MEMORY_PERF.md` 裁定），且 MTP_BENCHMARK_WIN8GB 屬跨啟動不可配對

## 4. 判定

**4 · ④ 廢棄** — 判死／撤回／不做

> ⚠ 含作廢數字（9.82/12.62）者，依契約 §7 不得再引用

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [M3／M4／M5 離開條件（09-17 期）](m3.md) | 4 | M3 未達（9.82 vs 門檻 15）；M4 靠一個本身壞掉的量測被否決、修好後才關閉 —— 而那個量測現已作廢 |
| [入口／索引／決策頁（非實驗）](na-entry.md) | na | 不進四級：它們是入口與索引 |
| [跨線／其他產品（Wan2.2、HarmonyOS、Windows client、Colibri、Unified IR…）](na-crossline.md) | na | 本線無實測權或非本線主題 ⇒ 不塞進四級 |
| [softpool／L4 v1v2／doublebuffer spike（09-05~09-06 期）](na-softpool.md) | 4 | 推定被 hybrid 路線取代（37/41 份零外部引用 ⇒ 已孤立） |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/CGC_TPOT_延遲分解與優化路線圖_2026-09-07.html／MTP_BENCHMARK_WIN8GB.md` |
| 備註 | ⚠ 含作廢數字（9.82/12.62）者，依契約 §7 不得再引用 |
| 軸性質 | 非攻關軸：約束／帳本／上界算術／入口索引／跨線產品 |
| 對應報告 | 11 份 |

- CGC_TPOT_\345\273\266\350\277\237\345\210\206\350\247\243\344\270\216\344\274\230\345\214\226\350\267\257\347\272\277\345\233\276_2026-09-07.html"（僅存在於分支，不在工作區）
- CGC_TPOT_\345\273\266\351\201\262\345\210\206\350\247\243\350\210\207\345\204\252\345\214\226\350\267\257\347\267\232\345\234\226_2026-09-07.html"（僅存在於分支，不在工作區）
- [AGENT_HARNESS_TECH_ARCH_MILESTONE_WHITEPAPER_20260916.html](../../AGENT_HARNESS_TECH_ARCH_MILESTONE_WHITEPAPER_20260916.html)
- Gemma4_Final_Report.md（僅存在於分支，不在工作區）
- Gemma4_Performance_Report.md（僅存在於分支，不在工作區）
- M4_SETUP_GUIDE.md（僅存在於分支，不在工作區）
- MTP_BENCHMARK_WIN8GB.md（僅存在於分支，不在工作區）
- [Gemma4_Final_Report.md](../../archive/pre-consistency-metrics-2026-09-11/Gemma4_Final_Report.md)
- [Gemma4_Performance_Report.md](../../archive/pre-consistency-metrics-2026-09-11/Gemma4_Performance_Report.md)
- [M4_SETUP_GUIDE.md](../../archive/pre-consistency-metrics-2026-09-11/M4_SETUP_GUIDE.md)
- [MTP_BENCHMARK_WIN8GB.md](../../archive/pre-consistency-metrics-2026-09-11/MTP_BENCHMARK_WIN8GB.md)

---

← [softpool／L4 v1v2／doublebuffer spike（09-05~09-06 期）](na-softpool.md)　·　[總目錄](index.md)　·　[HTML 版](na-olddata.html)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
