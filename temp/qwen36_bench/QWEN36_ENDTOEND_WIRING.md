# Qwen3.6 端到端接線狀態（2026-08-09）

## 目標
把 Qwen3.6 35B-A3B（30 DeltaNet + 10 GatedAttn + top-8/256 MoE）接進
turbo-fieldfare 引擎，讓 CLI 可以跑真 ttft/decode。共用基礎設施（
Model.load / trust-receipt / MetalContext / RMSNorm / runRawCompletion loop）
共用，但參數按 gemma4|qwen36 分開。

---

## ✅ 已完成（全部編譯 + 測試通過）

### 1. DeltaNetRunner layerIndex 參數化
- 原實作硬編碼 `model.language_model.layers.0.*`（只能跑 layer 0）
- 加 `layerIndex: Int = 0` 參數，weight() helper 在呼叫時 re-scope
- `reset()` 清除 h/conv 狀態

### 2. 三個 runner 的 accessor + reset
- GatedAttnRunner：`yOutBuffer` accessor + `reset()`
- Qwen36MoERunner：`moeOutBuffer` accessor + mmap 零複製 expert blob
- DeltaNetRunner：`reset()`（清 h/conv state）

### 3. ManifestReader 雙格式 + ArchConfig.qwen36_35B_A3B
- magic 接受 GTURBO/TFGT；12 個欄位 optional 化 + 別名映射
- `ArchConfig.qwen36_35B_A3B`：40 層 / 256 experts / 2048 hidden
- ManifestArch 新增 optional `deltanetLayers`（qwen36 提供、gemma4 nil）

### 4. MetalContext 註冊 qwen36_forward module
- 新增 `Sources/TurboFieldfare/Metal/Qwen36/qwen36_forward.metal`
  （q36_embed / q36_residual_add / q36_fp32_to_fp16 / q36_fp16_gemv）
- shader module 名 = 檔名（`qwen36_forward`），bundle `.copy("Metal")` 自動帶入

### 5. Qwen36ForwardRunner（新檔）
- conform `LogitProducer`：`reset()` + `produce(token:position:into:)`
- 每 token：embed fp16 → 40 層（inputNorm → DeltaNet/GatedAttn → 殘差 →
  preFFN norm → MoE encodeBatch → 殘差）→ finalNorm → lm_head fp16 GEMV
- 依 manifest `deltanetLayers` 決定每層類型（fallback：每 4 層第 3 層為 GatedAttn）
- code-review 修復：embed 用 qwen36 resident 名、CB nil 時 throw、vocab 用 cfg

### 6. CLI arch 偵測 + qwen36 inline 分支
- `detectArch(modelURL)` 讀 manifest magic → `.gemma4_26B_A4B` / `.qwen36_35B_A3B`
- qwen36 分支：Qwen36ForwardRunner + RawCompletionScratch + runRawCompletion

### 7. Tokenizer qwen36 相容
- GFTokenizer 加 `Compatibility` enum（.gemma4 嚴格 / .qwen36 fail-open）
- qwen36：bos=nil→eos、pad=`<|endoftext|>`、chat template 用 `<|im_start|>`
- `loadQwen36(from:)` public loader（CLI 不需 import swift-transformers）

### 8. Qwen36MoERunner mmap（16GB 存活關鍵）
- `FileHandle.map` 是 Linux-only → 改 `Data(contentsOf: .mappedIfSafe)` +
  `makeBuffer(bytesNoCopy:)`（`UnsafeMutableRawPointer(mutating:)`）
- 效果：40 層 × 432MB 不再實體分配，OS lazy page-in → **不再 OOM**
- 6 個 Qwen36MetalTests 全過（deltaNet / gatedAttn / moe / amortized / hybrid / timing）

### 9. 端到端 smoke 進度
- arch 偵測 ✅ → model load ✅ → tokenizer ✅ → 進入 prefill
- 卡點：`chunked prefill requires a ChunkedPrefillRunner`（qwen36 runner 未
  conform ChunkedPrefillRunner；prefill 需走 `.off` 模式或讓 runner conform）

---

## 🔄 目前卡點（2026-08-09）

**prefill 模式**：runRawCompletion 預設 `.chunked`，Qwen36ForwardRunner 不
conform `ChunkedPrefillRunner` → 直接 throw。
解法（擇一）：
- a) qwen36 分支把 `runtime.prefillConfig` 換成 `.off`（逐 token produce，
  慢但正確；maxContext 短時可接受）
- b) Qwen36ForwardRunner conform `ChunkedPrefillRunner`（prefillChunked
  批次化 40 層 forward —— 對 35B 是必要的最終路，但工程量大）

---

## 下一步（正確解）：per-expert streamer 改造

kernel 目前用單一 blob + `e * expertStride` 定址（一層所有 experts 連續）。
PreadExpertStreamer 的 `loadExpert` 回傳 per-expert `(MTLBuffer, offset)` —
但 batch 內最多 8 個 expert 可能落在不同 cache slot buffer，單一
`device const uchar* blob` 無法定址。

改造方案：
1. `q36_moe_expert` kernel 接受 8 組 `(bufferIndex, offset)`（每 rank 一組）
   — 或改成 8 個 buffer 參數 + 每 rank 的 offset
2. Qwen36MoERunner 改用 `model.routedExpert(layer:expert:)`（streamer）拿
   per-expert buffer，不再整檔載入
3. **這同時讓 hot-pool / EXPERT_SLOTS / READ_WORKERS 旋鈕對 Qwen36 生效**
   ——共用基礎設施真正落地（這是本任務「共用基建」的關鍵價值）

完成後 16GB Mac 上 RSS 只追熱集（pool + LRU slots），decode 可行。

## 參數分離設計（gemma4|qwen36，run_prod.sh）

| 參數 | gemma4 | qwen36 |
|---|---|---|
| EXPERT_SLOTS | 96 (pool64) | streamer 改造後按 pool 覆蓋設 |
| HOT_POOL_EXPERTS | 64 | 需 256 experts 的 top-N profile |
| READ_WORKERS | 8 | 同上（streamer 生效後） |
| WAKE_POLL_US | 5000 | decode loop 共用後生效 |
| trust-receipt | on | ✅ 已生效（receipt 已生成） |
| MTP | off | fused MTP head（獨立評估） |

profile 生成依賴端到端可跑（trace → top-N），所以順序是：
streamer 改造 → 端到端通 → trace → profile → run_prod.sh qwen36 節。

---

# Qwen3.6 接線：gemma4 生產級設定完整清單（2026-08-09）

> 目的：把 gemma4 生產配置的**全部**旋鈕列出來，逐一標註 qwen36 適用性，
> 避免移植時漏網。來源：`turbo-fieldfare-github-official/bin/run_prod.sh`
> （全文）+ `Sources/` 代碼預設（env-gated 開關）。

## 一、run_prod.sh 顯式設定的開關（11 個）

| # | 旋鈕 | gemma4 生產值 | 作用（A/B 證據） | qwen36 現狀 |
|---|---|---|---|---|
| 1 | `TURBO_FIELDFARE_EXPERT_SLOTS` | 96（pool64）/ 64（pool32/48） | 快取槽數：16→96 讓 hit 65%→95% | ❌ 未生效（Qwen36MoERunner 整檔載入，不走 streamer） |
| 2 | `TURBO_FIELDFARE_HOT_POOL` | 1 | 釘住高頻專家：decode +25% code | ❌ 同上（kernel 單 blob 定址，無法接 pool） |
| 3 | `TURBO_FIELDFARE_HOT_POOL_EXPERTS` | 64（profile top64_code.json） | 64 = 97.1% 覆蓋 | ❌ 需 256-expert 的 top-N profile（端到端通後 trace 生成） |
| 4 | `TURBO_FIELDFARE_HOT_POOL_PROFILE` | profiles/top64_code.json | 每層熱集清單 | ❌ 無 |
| 5 | `TURBO_FIELDFARE_HOT_POOL_PRELOAD` | sync（pool64）/ async（32/48） | pool64 必須 sync（async 撞 decode 讀取 −50%） | ❌ 無 |
| 6 | `TURBO_FIELDFARE_ATTN_TENSOROPS` | 0（SDK 無 MPP，inert） | B3 tensor-ops 在此 SDK 不編譯 | n/a（Qwen36 用 split path 之外的 kernel） |
| 7 | `TURBO_FIELDFARE_EXPERT_READ_WORKERS` | 8 | decode miss 讀取深度：+11.1% decode | ❌ 未生效（整檔載入無 miss-read） |
| 8 | `TURBO_FIELDFARE_WAKE_POLL_US` | 5000 | cb1 wait spin-poll：負載下 +35% median | ❌ 未生效（Qwen36ForwardRunner 自己的 CB 等待，未接 polling） |
| 9 | `--trust-receipt` | on | 跳 SHA-256 重算：TTFT −3.9s | ✅ **已生效**（receipt 已生成） |
| 10 | `MTP_MODEL`（空=off） | 空（MTP net-negative，2026-08-08 定案） | r2/r3/r4 全負 | ⚠️ 獨立評估（fused MTP head 接受率 86.7/50.8） |
| 11 | `MODEL_BITS` | 3（r3）/ 2（r2） | 3-bit 省 44% 讀取 vs 4-bit | n/a（qwen36-r3/r4 已 repack） |

## 二、代碼預設 ON（run_prod.sh 不用設，但要知道）

這些是 gemma4 路徑的**隱藏生產開關**，prefill/decode 效能就是靠它們：

| 旋鈕 | 預設 | 作用 |
|---|---|---|
| `TURBO_FIELDFARE_EARLY_SHARED` | `!= "0"` → ON | cb1 期間提前 commit shared expert（GPU 不餓死） |
| `TURBO_FIELDFARE_B4_HIT_ONLY_SYNC` | `!= "0"` → ON | B4 hit-only fetch 同步（decode +4~6%） |
| `TURBO_FIELDFARE_SYNC_PREFETCH` | `!= "0"` → ON | 同步 prefetch |
| `TURBO_FIELDFARE_MTP_ADAPTIVE` | `?? "1"` → ON | adaptive gate（MIN baseline + 冷樣本跳過 + 大 margin 即關） |

> qwen36 目前 Qwen36ForwardRunner 不消費任何這四個 —— 它有自己的
> per-token CB 鏈（embed → 40 層 → head），需逐個評估是否套用。

## 三、記憶體家族（*mem* 相關）

| 旋鈕 | 預設 | 行為 | qwen36 相關性 |
|---|---|---|---|
| `TURBO_FIELDFARE_EXPERT_MMAP` | off（== "1" 才開） | 專家權重 mmap 零複製 lazy page-in；**與 hot pool 互斥**（`guard !useMmap`） | ✅ **已落地**：Qwen36MoERunner 用 `Data(contentsOf: .mappedIfSafe)` + `makeBuffer(bytesNoCopy:)`，16GB Mac 不再 OOM（RSS 只追工作集） |
| `TURBO_FIELDFARE_ADAPTIVE_POOL` | off（== "1" 才開） | pool 槽可淘汰（LRU 提升 miss 專家），靜態 profile 只作暖啟動 | ❌ 未生效（streamer 改造後才有意義） |
| `EXPERT_SLOTS`（見上） | 96 | 虛擬記憶體帳單：r4 expert 3.2MB → 96 slots ≈ 9GB virtual（lazy commit） | ❌ 未生效 |

**關鍵衝突提醒**：`EXPERT_MMAP=1` 會**跳過 hot pool**（`guard !useMmap`）。
qwen36 目前的整檔 mmap 是「全 lazily page-in」，等同「無 pool 的 mmap 模式」——
優點是 16GB 可跑（不會 OOM），缺點是沒有熱集優先（每 token 每層全專家 page-in）。

## 三-b、第一版漏掉的設定（2026-08-09 補掃）

第一次掃描只抓 `TURBO_FIELDFARE_*` 前綴 + run_prod.sh，以下漏網：

| 類別 | 項目 | gemma4 用法 | qwen36 現狀 |
|---|---|---|---|
| env | `HF_BASE_URL` | 遠端 tokenizer/模型下載鏡像 | ⚠️ 若走本地 tokenizer 不需；遠端下載才要 |
| env | `HF_TOKEN` | HuggingFace token（遠端下載） | ⚠️ 同上 |
| env | `TURBO_FIELDFARE_TOKENIZER_DIR` | **tokenizer 目錄覆寫**（側載 tokenizer_folder） | ⚠️ **重要**：qwen36 tokenizer 在 `tokenizer/` 子目錄，需確認此旋鈕或預設偵測有吃到 |
| CLI | `--seed` | A/B 可重現性（固定採樣） | ✅ CLI 共用（qwen36 分支也接受） |
| CLI | `--top-k` / `--top-p` | 採樣參數 | ✅ CLI 共用 |
| CLI | `--stop` | stop string（可多個） | ✅ CLI 共用 |
| CLI | `--mtp-max-draft` | MTP draft 數上限 | ⚠️ 與 fused MTP 一起評估 |
| CLI | `--quiet` | 關診斷輸出（生產乾淨輸出） | ✅ CLI 共用 |
| CLI | `--perplexity` | 品質 gate（r3/r4 對照） | ✅ CLI 共用（qwen36 品質驗證用） |
| CLI | `--routed-bits` | 生產由 `--model` 決定（MODEL_BITS 選模型） | n/a |
| 工具 | `bin/make_hotpool_profile.sh` | trace → 每層 top-N 熱集 profile | ❌ 需 streamer 改造 + 端到端通後才能用（256-expert top-N） |
| 工具 | `bin/bench_ab.sh` / `bin/b1_final_ab.sh` | A/B 基準（非生產） | 可沿用 |
| 工具 | `bin/TurboFieldfareCLI-{adaptive,b1,baseline,mmap,newkernel}` | 舊 A/B 對照 binary（非生產） | 不適用 |

**注意**：run_prod.sh 本身不含採樣旗標（溫度/重複懲罰由 caller 傳）——
生產基準慣例：`--temperature 0 --repetition-penalty 1.0`（greedy 對照）。
qwen36 若要採樣需明確傳 `--top-k/--top-p`（Qwen 慣例 top_p=0.8）。

## 三-c、qwen3.6 **dense** 已用過的設定（2026-08-09 追查）

用戶記憶屬實：qwen3.6 dense（dense-only streaming ppl benchmark）已用過以下設定，
痕跡在 repo 裡：

| 設定 | 位置 | 說明 |
|---|---|---|
| `PPL_LIMIT`（env） | `temp/qwen36_bench/qwen36_ppl_quick.py` | dense ppl 的 token 上限（64/52-token sanity 定案格式） |
| `EXPERT_MODE`（env: repack\|bf16） | `temp/qwen36_bench/q36_bf16_ab.py` | 對拍 repack int4 dequant vs 官方 bf16 的第一分歧點 |
| `GTURBO` / `HF` 路徑常數 | 同上 | r4.gturbo（dense 權重）+ 外接盤 HF（expert stream） |
| `--trust-receipt`（收益分析） | `temp/qwen36_bench/trust_receipt_compare.py` | Qwen36 18-22GB 哈希 → 省 **5-8s**（Gemma4 只 3.9s） |
| **tokenizer sidecar**（`model_dir/tokenizer/` 優先） | `app/shared/colibri_backend.py:613-619` | **這正是 qwen36 tokenizer 子目錄的處理！** app 層已自動設 `TURBO_FIELDFARE_TOKENIZER_DIR=model_dir/tokenizer` |

**重要結論**：
1. `TURBO_FIELDFARE_TOKENIZER_DIR` 的 qwen36 用法**已被 app 層（colibri_backend）內建**——
   sidecar `tokenizer/` 子目錄優先，qwen36-r4.gturbo 的 tokenizer 目錄正確命中，
   不需要手動設（CLI 直跑時才需要確認）。
2. dense ppl 的兩個 env（PPL_LIMIT / EXPERT_MODE）與 trust-receipt 分析都在
   `temp/qwen36_bench/`，是先前 dense-only 驗證的工作產物，可重跑。
3. 這些是 **Python dense benchmark 層**的設定，與引擎 env 開關不同層級——
   但證實 qwen36 路徑從 dense ppl 階段就吃 trust-receipt 與 tokenizer sidecar。

## 六、共用基礎設施改造完成（2026-08-10）

### 6.1 共用 infra 已全部接上（qwen3.6|gemma4 家族差異數據驅動）

| 基建 | 狀態 | 說明 |
|---|---|---|
| **PreadExpertStreamer**（hot pool / LRU slots / READ_WORKERS） | ✅ qwen36 已接入 | `Qwen36MoERunner.fillSlotTable` 走 `model.routedExpert` → streamer，與 gemma4 同一套 |
| **8-slot expert table** | ✅ 已用 | `q36_moe_expert_slot8` kernel + runner 8 個直接 buffer 綁定（slot==rank，nTokens=1） |
| **`q36_moe_expert` blob 路徑** | 保留（測試用） | 50-token batch 對拍仍走 blob；生產 decode 走 slot8 |
| **GPU_TIMING** | ✅ | `Qwen36ForwardRunner` 每 CB 記錄 gpuStart/gpuEnd |
| **WAKE_POLL_US / EXPERT_SLOTS / READ_WORKERS** | ✅ 生效 | EXPERT_SLOTS 由 CLI env 讀入 → `.pread(slotCount:)` |
| **trust-receipt / tokenizer sidecar** | ✅ | 共用（先前已接） |

### 6.2 關鍵修復：argument encoder 有坑，改 8 直接 buffer

第一個 slot8 實作用 `Q36BlobTable8&` struct + `makeArgumentEncoder`（gemma4 同款）→
**輸出誤差 0.052 vs 容差 0.02**（blob 路徑 bit-exact 8.9e-08）。逐層排除 streamer/offset/kernel
後定位：argument buffer 綁定。改用 **8 個直接 `device const uchar*` buffer 參數**（indices 2-9），
runner 直接 `setBuffer` → **bit-exact 4.47e-08**，測試通過。

### 6.3 對齊 gemma4 的 256-tok 基準（msg_code prompt、slots=16、READ_WORKERS=8、trust-receipt）

| 指標 | **qwen36 r3（3-bit）** | **qwen36 r4（4-bit）** | gemma4 r3 | gemma4 r4 |
|---|---|---|---|---|
| decode @256 tok | **7.55 tok/s**（33.89s） | **6.79 tok/s**（37.69s） | 22.6 | 21.5 |
| TTFT（36-tok code prompt） | 8.31s | 9.62s | ~3.5-4s | ~4.0s |
| swap | 0 增長 | 0 增長 | — | — |

> ⚠️ **重要限制**：這兩個數字是**引擎吞吐**（真實 Metal 計算/IO 時間），但 qwen36 **全模型前向輸出目前是 garbage**
> （256 tok 全是 `!` 重複）——這是先前 session 已記載的既有正確性 bug（top1-acc=0%、hidden collapse、
> layer 0 std≈0.024），**不是本次 streamer 改造造成**（kernel 層測試 7/7 全過、MoE layer3 bit-exact）。
> 修復全模型前向是下一件大事；吞吐數字可在修復後作為乾淨基線。

### 6.4 家族差異參數（共用基建、參數分開）

| 參數 | gemma4 | qwen36 | 原因 |
|---|---|---|---|
| **EXPERT_SLOTS** | 64（96 slots 含 pool） | **16**（0 swap） | qwen36 40 層 × 1.77MB/expert：64 slots × 40 層 = 4.5GB slot buffer → 16GB Mac swap 爆（實測 2.1GB swap 開始膨脹）；16 slots = 1.13GB，0 swap |
| hot pool profile | pool64-sync（97% 命中） | 無（profile 生成工具待接，streamer 已通） | qwen36 256 experts × 40 層熱集待統計 |
| topK / experts / stride | topK? / 256 / 較小 | topK=8 / 256 / 1,769,472 B | layout.json 數據驅動 |

### 6.5 驗證狀態

- `swift test --filter Qwen36MetalTests`：**7/7 通過**（含新增 `moeSlot8StreamerMatchesGolden`）
- r4 256-tok：完成無 crash、0 swap 增長、記憶體受控
- 端到端文本品質：❌ garbage（既有 bug，見 6.3）

### 6.6 下一步

1. **修全模型前向 garbage**（最高優先）：從 hidden collapse 定位（dense 權重 / DeltaNet h 累積 / 跨層整合）
2. qwen36 hot pool profile（streamer 已通 → `make_hotpool_profile.sh` 256-expert 版）
3. 非同步 expert prefetch（fetch layer L+1 與 layer L 計算重疊）——目前每 token 42 CB 同步鏈，是 6.8 tok/s 的主要 overhead

## 四、診斷/實驗（非生產，勿設）

- `EXPERT_STATS` / `EXPERT_TRACE` / `GPU_TIMING` / `GPU_TIMELINE_CSV` /
  `UNION_STATS` / `UNION_DUMP` / `SPEC_PROBE` / `MTP_DEBUG` / `MTP_ADAPTIVE_DEBUG` / `DEBUG_OFFSETS`
  → 全部是 RealForwardRunner 的診斷輸出，qwen36 runner 未接。
- `ATTN_CHUNKS` / `ATTN_SINGLE` / `SKIP_ATTN_*` / `PHASE2_CHUNK` / `QKV_SMEM_X` / `FUSE_SHARED`
  → gemma4 attention/MoE 實驗開關，qwen36 架構不同不適用。
- `MISS_PREFETCH` / `EXPERT_PREFETCH` / `EXPERT_LOOKAHEAD` → 已證偽（−41%），勿開。

## 五、qwen36 缺口的優先順序（streamer 改造是總鑰匙）

```
streamer 改造（kernel 多 buffer 定址 + per-expert load）
  → 同時解鎖 #1-#5（slots / hot pool / preload / READ_WORKERS / adaptive pool）
  → 之後才能 trace → 256-expert top-N profile → run_prod.sh qwen36 節
```

目前唯一已生效的共用基建：**trust-receipt** + **EXPERT_MMAP（mmap）** + Model.load +
MetalContext + RMSNorm + runRawCompletion loop + tokenizer(qwen36 分支)。
