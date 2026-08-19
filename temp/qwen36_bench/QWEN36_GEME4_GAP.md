# Qwen3.6 vs Gemma4：基礎設施缺口完整對照（2026-08-10）

> 目標：逐項確認 gemma4 有的基礎設施 qwen36 是否也有。
> 分類：🔴 blocking（不做到不能端到端跑）｜🟡 performance（做了更快）
> 🟢 nice（可選）｜⚪ n/a（gemma4 架構專屬）

---

## 🔴 Blocking（不解決就無法端到端跑）

| # | 設施 | gemma4 用法 | qwen36 現狀 | 工作量 |
|---|---|---|---|---|
| 1 | **Pread streamer（hot pool + LRU slots）** | `PreadExpertStreamer` cache slot + pool preload + adaptive eviction | ❌ Qwen36MoERunner 整檔 mmap（全 40 層 virtual = 17GB，觸碰時 page-in → 16GB Mac 撐不住） | kernel ~10 行 + runner ~40 行 |
| 2 | **EXPERT_SLOTS 控制** | 96（pool64）→ 虛擬 9GB，RSS ~2.5GB | ❌ streamer 專屬，目前無效 | 隨 streamer 自動生效 |
| 3 | **HOT_POOL / HOT_POOL_EXPERTS / HOT_POOL_PRELOAD** | 64 experts 常駐 + sync/async 預載 | ❌ streamer 專屬，無效 | 隨 streamer 自動生效 |
| 4 | **EXPERT_READ_WORKERS** | 8 並行 pread miss | ❌ streamer 專屬，無效 | 隨 streamer 自動生效 |
| 5 | **ADAPTIVE_POOL** | LRU 可淘汰池槽 | ❌ streamer 專屬，無效 | 隨 streamer 自動生效 |

**結論**：streamer 改造是總鑰匙 — 一次解鎖 #1-#5，RSS 從 17GB → ~2.5GB，OOM/panic 消失。

---

## 🟡 Performance（做了更快）

| # | 設施 | gemma4 用法 | qwen36 現狀 | 工作量 |
|---|---|---|---|---|
| 6 | **WAKE_POLL_US=5000** | cb1 wait spin-poll，負載下 +35% decode | ❌ Qwen36ForwardRunner 有自己的 CB 鏈（embed→40 層→head），每層 `cb.waitUntilCompleted()` 無 polling | ~30 行（加 polling 到 runner 的 state 機） |
| 7 | **GPU_TIMING / EXPERT_STATS 診斷** | RealForwardRunner 內建，env 控制輸出 | ❌ Qwen36ForwardRunner 完全沒有 timing/stats | ~80 行（加 timer 封裝 + 輸出） |
| 8 | **--trust-receipt（SHA 跳過）** | TTFT −3.9s | ✅ **已接**（CLI qwen36 分支傳 `args.trustReceipt`） | 已完成 |
| 9 | **EARLY_SHARED** | cb1 期間提前 commit shared expert | ⚠️ Qwen36MoERunner 的 shared expert 在 `encodeShared` 中，但 CB 調度方式不同，需評估是否等效 | TBD（可能自然滿足） |
| 10 | **B4_HIT_ONLY_SYNC** | hit-only fetch 同步 | ❌ PreadExpertStreamer 專屬，Qwen36MoERunner 無此路徑 | 隨 streamer |
| 11 | **MTP_ADAPTIVE gate** | 冷樣本跳過 + MIN baseline + 大 margin 即關 | ❌ Qwen36 fused MTP head 尚未接進引擎 | 獨立大工程 |
| 12 | **run_prod.sh qwen36 節** | 完整生產腳本 + profile + env | ❌ 只有 CLI 直傳，無 qwen36 專用腳本 | 1 個 shell script |

---

## 🟢 Nice / Future（可選）

| # | 設施 | 說明 |
|---|---|---|
| 13 | **make_hotpool_profile.sh** | 依賴端到端通 + trace → 256-expert top-N profile（streamer 改造後才有效） |
| 14 | **Token level：tensor-ops / w8 / B4** | gemma4 專屬（不同 kernel、不同 attention 架構），不適用 |
| 15 | **FUSE_SHARED / PHASE2_CHUNK / QKV_SMEM_X** | 實驗開關，gemma4 也非生產預設 ON |
| 16 | **DIAG 類：SPEC_PROBE / UNION_STATS / EXPERT_TRACE** | RealForwardRunner 專屬診斷，qwen36 先不接 |

---

## ⚪ n/a（gemma4 架構專屬，qwen36 不適用）

| 設施 | 原因 |
|---|---|
| ATTN_TENSOROPS | B3 MPP tensor-ops attention；qwen36 用 DeltaNet + GatedAttn，架構不同 |
| ATTN_CHUNKS / ATTN_SINGLE / SKIP_ATTN_* | Gemma4 attention split path 診斷 |
| TURBO_FIELDFARE_PREFILL_EXPERT_READ | PrefillExpertReadMode，Qwen36 用 per-token produce |
| r2/r3 MODEL_BITS | Gemma4 特有；qwen36 r3/r4 已獨立 repack |

---

## ✅ 已完成的（不缺口）

| 設施 | 說明 |
|---|---|
| trust-receipt | ✅ CLI 共用 |
| MetalContext / Model.load | ✅ 共用 |
| RMSNorm / LogitProducer protocol | ✅ 共用 |
| Tokenizer（qwen36 分支） | ✅ qwen36 專用 tokenizer（sidecar 路徑） |
| ArchConfig.qwen36_35B_A3B | ✅ |
| deltanetLayers manifest | ✅ |
| mmap expert blob | ⚠️ 半套（整檔 mmap 而非 per-expert pread） |

---

## 一句話結論

```
streamer 改造（1 天）→ 解鎖 #1-#5 → 16GB Mac 跑得動
→ 接 WAKE_POLL_US（半天）→ 負載下 decode 不掉速
→ 診斷上線 → 逐優化 decode 瓶頸
→ run_prod.sh qwen36 節 + profile
→ 最後評估 fused MTP
```