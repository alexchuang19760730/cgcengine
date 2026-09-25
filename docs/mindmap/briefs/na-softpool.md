# softpool／L4 v1v2／doublebuffer spike（09-05~09-06 期） — 技術白皮書　·　4 ④ 廢棄

> **一句話**：早期（09-05～09-09）softpool／L4 v1v2／doublebuffer spike 的探索，尋找池的替代實作。

- 主題：不適用　·　子目標：**不適用**（非攻關軸：約束／帳本／上界算術／入口索引／跨線產品）
- 階段：已結案（廢棄／不適用）　·　④ 廢棄 ＋ 不適用

---

## 1. 目標

早期（09-05～09-09）softpool／L4 v1v2／doublebuffer spike 的探索，尋找池的替代實作。

## 2. 判準

是否被後續版本取代

## 3. 結果

推定被 hybrid 路線取代（37/41 份零外部引用 ⇒ 已孤立）

## 4. 判定

**4 · ④ 廢棄** — 判死／撤回／不做

> ⚠ 推定；非本線判決

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [M3／M4／M5 離開條件（09-17 期）](m3.md) | 4 | M3 未達（9.82 vs 門檻 15）；M4 靠一個本身壞掉的量測被否決、修好後才關閉 —— 而那個量測現已作廢 |
| [入口／索引／決策頁（非實驗）](na-entry.md) | na | 不進四級：它們是入口與索引 |
| [跨線／其他產品（Wan2.2、HarmonyOS、Windows client、Colibri、Unified IR…）](na-crossline.md) | na | 本線無實測權或非本線主題 ⇒ 不塞進四級 |
| [舊口徑數據報告（Gemma4／MTP_BENCHMARK_WIN8GB／TPOT 路線圖）](na-olddata.md) | 4 | 作廢：09-17 起 decode 一律 llama-bench、warm-skip 口徑（`MEMORY_PERF.md` 裁定），且 MTP_BENCHMARK_WIN8GB  |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `（推定：反向引用統計 ＋ 歸檔狀態）` |
| 備註 | ⚠ 推定；非本線判決 |
| 軸性質 | 非攻關軸：約束／帳本／上界算術／入口索引／跨線產品 |
| 對應報告 | 11 份 |

- L3_Softpool_vs_L4_\346\212\200\350\241\223\346\210\220\347\206\237\345\272\246\346\257\224\350\274\203\345\240\261\345\221\212_2026-09-09.html"（僅存在於分支，不在工作區）
- CGC_Decode_Version_Milestone_2026-09-06.html（僅存在於分支，不在工作區）
- CGC_EXPERT_CACHE_SOFTPOOL_WHITE_PAPER_2026-09-05.html（僅存在於分支，不在工作區）
- CGC_SOFTPOOL_EXPERT_CACHE_L4_V1V2_2026-09-05.html（僅存在於分支，不在工作區）
- CGC_Step23_ExpertCache_ColdGuard_LogitsCallback_Repair_2026-09-06.html（僅存在於分支，不在工作區）
- CGC_Step23_RenormRouting_AB_2026-09-06.html（僅存在於分支，不在工作區）
- [CGC_Decode_Version_Milestone_2026-09-06.html](../../archive/pre-consistency-metrics-2026-09-11/CGC_Decode_Version_Milestone_2026-09-06.html)
- [CGC_EXPERT_CACHE_SOFTPOOL_WHITE_PAPER_2026-09-05.html](../../archive/pre-consistency-metrics-2026-09-11/CGC_EXPERT_CACHE_SOFTPOOL_WHITE_PAPER_2026-09-05.html)
- [CGC_SOFTPOOL_EXPERT_CACHE_L4_V1V2_2026-09-05.html](../../archive/pre-consistency-metrics-2026-09-11/CGC_SOFTPOOL_EXPERT_CACHE_L4_V1V2_2026-09-05.html)
- [CGC_Step23_ExpertCache_ColdGuard_LogitsCallback_Repair_2026-09-06.html](../../archive/pre-consistency-metrics-2026-09-11/CGC_Step23_ExpertCache_ColdGuard_LogitsCallback_Repair_2026-09-06.html)
- [CGC_Step23_RenormRouting_AB_2026-09-06.html](../../archive/pre-consistency-metrics-2026-09-11/CGC_Step23_RenormRouting_AB_2026-09-06.html)

---

← [expert cache 血統設計（09-05~09-09 期）](na-design.md)　·　[總目錄](index.md)　·　[HTML 版](na-softpool.html)　·　[舊口徑數據報告（Gemma4／MTP_BENCHMARK_WIN8GB／TPOT 路線圖） →](na-olddata.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
