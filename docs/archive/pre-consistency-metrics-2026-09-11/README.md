# ARCHIVE — 文件早於 M1/M2 一致性閘門（2026-09-11）

這些文件**沒有被刪除，也大部分沒有錯**。它們是**參考資料**。

但**裡面任何速度、記憶體、品質數字都不能直接引用**，因為它們產生的時候，
「數值是否可信」這件事**還沒有被定義過**。

## 為什麼

`M1` / `M2` 這兩個指標是在 **2026-09-11** 的
[`docs/EVAL_CONSISTENCY_METRICS_2026-09-11.md`](../../EVAL_CONSISTENCY_METRICS_2026-09-11.md)
裡才被定義的：

| 指標 | 定義 | 回答的問題 |
|---|---|---|
| **M1 numeric identity** | 每步 `row_fnv1a64` 逐位元組相同 | logits 有沒有飄 |
| **M2 decision agreement** | 每步 `argmax_token` 相同 | 選到的 token 一不一樣 |

規則（同文件）：**M2 單獨通過不代表沒有差異；M1 單獨通過也不代表決策一致。
報告一律兩者並列。** 預設 `--fail-on both`。

**在這個標準存在之前，「25 t/s」這種數字只是一個速度量測，不是一個結論。**
它沒有回答「這條跑跡是不是同一個計算」。

## 兩個具體的陷阱（都在這個 archive 裡）

### 1. `25.17 t/s · draft_accept 98.2%`（2026-09-07，TPOT 白皮書）

這是全 repo 引用最多、也最常被當成「天花板」的數字。

- M1/M2 是**四天後**才發明的 → 那個 run **不是「沒過 M1/M2」，是 M1/M2 還不存在**。
  它的證據等級是「一次速度量測」，不是「已驗證的一致性」。
- 同一份文件的 §5「已驗證死路」自己寫著：`8GB + RN_ROUTING = 25.9 t/s 但輸出壞，
  品質 0.3 → ❌ 死路`。也就是說**最快的鄰居配置是品質壞的**。
- 生產值 25.17 是另一組 arm（P2-B+P2-C+SPAC α=0.75），靠 *profile 品質分數* 達標，
  **不是**靠 M1/M2。
- 更要命的在 `EVAL_CONSISTENCY_METRICS` §7.1：後來量到的「M1/M2 全部 100%
  （跨模型）」是**同一個檔案被量了兩次**（iq3 欄位是 iq4 的 symlink），
  所以才補上 `gate_model_identity()` 這個硬閘門。

### 2. 任何把 head / 權重「跟另一顆模型比」得出的結論

MTP head 與量化 norm 的慣例都是**模型專屬**的。跟別的模型差 ~1.0
只代表「兩顆不同」，不代表這顆錯。實際因此付出過一次完整的診斷回合。

## 怎麼使用這裡的文件

可以：
- 查**設計意圖**、**參數清單**、**已排除的死路**（死路的結論仍然有效，因為它們是「更差」）。
- 查**架構與命名脈絡**。

不可以：
- 直接引用任何 tok/s、ms/token、記憶體峰值、accept % 當作「已驗證」的性能。
- 把「當時最好的配置」當成現在的天花板。

要引用數字，就要在**當前的 binary + M1/M2 閘門**下重新量一次，並把閘門結果
跟速度並列記錄。`≥20 t/s 的配置從未跑過 M1/M2` —— 這是目前最大的空白。

## 收錄範圍

判準：**git 首次 commit 日期早於 2026-09-11**（副檔名日期僅在檔案未進版控時作為備援）。
共 40 份。

| 建立日 | 文件 |
|---|---|
| 2026-08-08 | `CGC_COLIBRI_HERMES_ROUTEPOLICY_V2_INTEGRATION.md`, `CGC_COLIBRI_SINGLE_NODE_PRODUCTION_MATRIX.md`, `CGC_Gate_1.0_Whitepaper.md`, `Gemma4_Final_Report.md`, `Gemma4_Performance_Report.md`, `TURBOFIELDFARE_LOCAL_PROCESS_READY.md`, `UNIFIED_RUNTIME_IR_V0.md` |
| 2026-08-27 | `檢討報告_2026-08-26_git與build追蹤疏失.md` |
| 2026-08-29 | `CGC_COMPUTE_SHARING_ARCHITECTURE.md`, `CGC_CROSS_PLATFORM_ARCHITECTURE.md`, `CGC_ENGINE_端雲開發指導書.md`, `M4_SETUP_GUIDE.md`, `MTP_BENCHMARK_WIN8GB.md` |
| 2026-08-30 | `AGENT_HARNESS_WHITEPAPER.html`, `CGC_ARCHITECTURE_OVERVIEW.html`, `CGC_HARMONYOS_PHONE_ARCHITECTURE.md` |
| 2026-09-01 | `NATIVE_OPENAI_OPTIMIZATION_SCORECARD.html` |
| 2026-09-04 | `CGC_DEVSERVER_DEVELOPMENT_WHITEPAPER_20260904.md` |
| 2026-09-05 | `CGC_CHAT_TEMPLATE_PROFILES_2026-09-05.html`, `CGC_DEVSERVER_RELEASE_NOTES_2026-09-05.html`, `CGC_EXPERT_CACHE_COMMIT_DIGEST_2026-09-05.html`, `CGC_EXPERT_CACHE_HYBRID_DESIGN_2026-09-05.html`, `CGC_EXPERT_CACHE_INTEGRATION_DIFF_2026-09-05.html`, `CGC_EXPERT_CACHE_INVARIANTS_v1.md`, `CGC_EXPERT_CACHE_SOFTPOOL_WHITE_PAPER_2026-09-05.html`, `CGC_SOFTPOOL_EXPERT_CACHE_L4_V1V2_2026-09-05.html`, `CGC_品質優先修復說明書_2026-09-05.html` |
| 2026-09-06 | `CGC_DOUBLEBUFFER_SPIKE_DESIGN_2026-09-06.html`, `CGC_Decode_Version_Milestone_2026-09-06.html`, `CGC_Step23_ExpertCache_ColdGuard_LogitsCallback_Repair_2026-09-06.html`, `CGC_Step23_RenormRouting_AB_2026-09-06.html` |
| 2026-09-07 | `CGC_TPOT_延迟分解与优化路线图_2026-09-07.html`, `CGC_TPOT_延遲分解與優化路線圖_2026-09-07.html`, `CGC_品質漂移修復方案開發建議書_2026-09-07.html` |
| 2026-09-08 | `ExpertCache_關鍵版本技術白皮書_2026-09-08.html` |
| 2026-09-09 | `CGC_Windows_Client_Guide_2026-09-09.md`, `L3_Softpool_vs_L4_技術成熟度比較報告_2026-09-09.html`, `Wan2.2_5B_Q4_端側視頻生成開發計劃_2026-09-09.md`, `整合架構設計_llama_cpp_CGC_Wan22_2026-09-09.md`, `品质归因测试方案_2026-09-09.md` |

## 沒有被收進來、但值得注意的

- `docs/P1_Expert_流式加載_設計文檔_2026-09-11.md` —— 同一天（10:33），
  但**早於** `EVAL_CONSISTENCY_METRICS`（22:46）。它描述的 `mmap_stream` 路線
  後來已被證偽。**待決**：要不要也收進來。
- `docs/EVAL_CONSISTENCY_METRICS_2026-09-11.md` 本身是標準的定義處，永遠留在 `docs/`。

## 給未來的一條規則

> 一個數字要能被引用，必須同時附上**產生它的 binary/commit** 與**閘門判定**。
> 只有速度、沒有閘門的數字，一律標記為「未驗證」，不得當作基線。
