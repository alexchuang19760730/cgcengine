# MTP 轉正規劃書 v1.0
### 用「ISCA'26 MoE 資料移動預測（Patterns behind Chaos）」+ Swift MoE-SpAc 把 prefetch/MTP 從淨負轉淨正

日期：2026-08-15 · 狀態：規劃（尚未開刀）· 關聯：`LLAMACPP_EXPERT_BOUNDED_RESIDENCY_FORK_方案.md` §7.2（P2 證偽）/ §8.7（MTP 放著）/ §8.10（n99c 4GiB 11.08）

---

## 0. 結論先行

**可以整合，但「整合的對象」要先修正**：

1. **`zhongkaiyu/moe_exp_placement` 不是 runtime prefetch 預測器**——它是 ISCA'26 論文的
   **離線仿真**（`zhongkai_test/` 五支分析腳本）+ SGLang 的 **EP 專家佈局**（physical→logical
   placement map）。論文對「既有 GPU 系統」的落地是 prefill-aware **expert placement**（1.25×，
   多機 EP 場景），不是單機 bounded pool 的每步預取。直接搬 repo 程式到 llama.cpp fork **不可行**。
   註：repo 目前**不在本機/外接盤**（已搜尋確認），需另行 clone 或按論文方法重現。

2. **真正可整合且直接命中問題的是論文的 Insight 1 + 2 + Swift 已存在的 MoE-SpAc 機制**：

   - **Insight 1：prefill→decode 預測**——prefill 階段的 expert 路由與 decode 高度相似，
     可用 prefill trace 預測 decode 熱集。**這是我們 P2 從未試過的信號**（P2 只試了步間/鄰層）。
   - **Insight 2：跨記憶體層級**——token 級 vs layer 級時序關係（reuse distance 不同）分層快取。
   - **Swift `MoESpAcEstimator` 已實作整套**：utility EMA 排程 + async pool swap +
     per-step look-ahead prefetch + **prefill 階段 pool 預熱**（`beginOpeningRoutedExpertStreamer`，
     正是 Insight 1 的落地）——只差移植到 C++ fork。

3. **MTP 轉正的條件 = 先讓預取轉正**：MTP 的淨負主因是 draft context 的第二個 cache 實例
   （+2.9s pread）。同一個預測器若能讓兩個 cache 的 miss 同步下降，MTP 的兩筆成本同時縮水，
   union fill 的攤薄才可能反超。**MTP 不單獨做，掛在預取轉正之後。**

---

## 0.5 P0 執行結果（2026-08-15）：prefill→decode 預測在我們機上**不成立，閘未過**

工具（已留用）：fork 加 `LLAMA_EXPERT_CACHE_TRACE=<file>` env-gated trace dump
（hook 逐層吐 routing，P/D 分相）；`tmp/p0_prefill_decode_analysis.py`（seen-set 覆蓋）、
`tmp/p0_freq_analysis.py`（頻率預測）。repo 已 clone 到 `temp/moe_exp_placement`（64MB，
`moe_placement` branch——確認是 SGLang fork + `zhongkai_test/` 離線腳本，無 runtime 預取）。

| 測量（真實文本、ngl 0、128 tok、seed 42） | qwen36 IQ3 | gemma4 IQ3_S |
|---|---|---|
| 短 prompt（6 tok）：seen-set 覆蓋 decode 熱集 K=8 | 14.1% | 67.9% |
| 長 prompt（~370 tok）：seen-set 覆蓋（恆真，全 256 exp 都見過） | 100%* | 100%* |
| **長 prompt：prefill top-K 頻率預測 decode 熱集（池命中代理）** | | |
| K=8 | 4.0% | 12.0% |
| K=43（1GiB 池槽數） | **23.2%** | **52.2%** |
| K=64 | 31.2% | 65.5% |
| 短 prompt：prefill top-K K=43 | 14.4% | 48.0% |

\* seen-set 覆蓋在長 prompt 下恆真（370 tok × 8 = 2960 次啟用可覆蓋全部 256 experts），
**無意義**——所以看頻率預測（有界池的正確指標）。

**結論**：
1. **閘（≥70%）未過**——最佳情況也只有 gemma4 K=64 的 65.5%，qwen36 只有 23%。
2. **且遠低於現行 LRU 的 91.3%**——靜態 prefill 頻率池 < 自適應 LRU，Insight 1 的
   「prefill 驅動」在單機有界池場景**不轉移**（論文是 200B-1000B 多機 EP 佈局、
   長 prefill 上千 token、24k requests 統計；我們 35B 單機 370-token prompt 的
   頻率估計太吵，decode 熱集持續演化）。
3. **prefill 信號最多只能當 SpAc EMA 的 seed**（溫啟動），不能當靜態預測器；
   但 seed 值本身也弱（23-52%），LRU 只要幾個 token 就自行溫好。
4. **P2 的「8.7% miss 靠更大 budget/更快 SSD，非排程能解」結論維持**。

**P0 決策**：Insight 1 線**終止**（不進 P1 的 SpAc 移植）。若要保留任何價值，
只剩「SpAc EMA 只做 pinned pool 決策、不預取」的零成本實驗（P1 降級為可選研究線）。

---

## 1. 現況盤點（三條線的終局狀態）

### 1.1 llama.cpp fork 的 prefetch（P2：實作完成 + 測量證偽，§7.2）

| 預測器 | 結果 | 根因 |
|---|---|---|
| 步間預測（prev-step） | **no-op** | 步間重疊 98.5%，且每層 working set 8-9 experts << 27 slots，LRU 已完整保留 |
| 層鄰預測（layer-neighbor） | **≈0 命中** | 重疊的早已在 pool；真正新進的 95% 錯 → 純污染 |
| 剩餘 miss | **8.7% 結構性不可預測**（當時結論）| 路由切換到的新 experts，便宜預測器抓不到 |

另外 v1 的 pool 背景 fill 有 **race**（bg pread 與 hook sync fill 寫同 slot → garbage；
Metal region async copy 與 bg write 並發），v2 改 page-cache 預取（`fcntl(F_RDADVISE)`）安全但無收益。
**prefetch 目前 env-gated、預設 OFF。**

### 1.2 MTP（§8.7：無加速，先放著）

- 完整 harness（llama-speculative）MTP-4 + cache：**5.34 t/s** vs 純 cache **10.77**，accept 83.3%。
- 淨負組成：① draft context 是第二個 cache（70.6% hit、8898 file_reads、+2.9s pread）；
  ② 每步多一次的 MTP 層 decode + sampling。
- union fill 只把每 token pread 8→4 次，攤薄抵不過上面兩筆。

### 1.3 Swift MoE-SpAc（已實作、只在 Swift 引擎）

`MoESpAcEstimator.swift`（arXiv:2603.09983）+ `Model.swift` 四個 hook：

1. `updateHotPoolsWithUtility(topKByLayer)` — utility top-K 取代 pinned pool（diff-based，只 pread 新的）
2. `requestHotPoolSwapsAsync` — bounded 背景佇列 async swap（不擋 decode）
3. `prefetchExpertsAsync` — per-step 前瞻 miss prefetch（預測 next-step union 進 LRU）
4. `beginOpeningRoutedExpertStreamer` — **prefill 階段就開 layer + 觸發 async pool 預熱**

成本：40×256 floats + O(n log n) topK/refresh，CPU 可忽略。env：`TURBO_FIELDFARE_MOE_SPAC=1`。

### 1.4 ISCA'26 論文 + AE repo（真實內容）

- 論文：arXiv 2510.05497，24k requests、4 個 200B-1000B MoE（含 Qwen3-235B），六個 insight。
- 對單機 16GB bounded pool **可用的兩個 insight**：
  - **Insight 1（prefill 驅動）**：prefill expert trace 與 decode 高度相似 → decode 前預載。
  - **Insight 2（跨層級）**：token 級（reuse 短，進 LRU）vs layer 級（reuse 長，進 pinned pool）。
- 不可用的四個：Insight 3-6 全是多機 EP 佈局/複製/配對分離/任務感知遷移。
- AE repo：SGLang fork + `zhongkai_test/` 五支離線分析腳本（`expert_analysis.py`、
  `expert_decode_workload_analysis.py`、`gen_expert_placement.py`、`placement_research.py`、
  `compare_placement.py`）——只做**離線路由軌跡的預測準確率測試**，不掛 runtime。

---

## 2. 為什麼之前沒轉正（機制拆解）

P2 證偽的「8.7% miss 結構性不可預測」結論**只對便宜預測器成立**：

- 便宜預測器（prev-step / layer-neighbor）都在**短時域**（1 步、1 層）內做外推；
- 而 miss 的本質是**長程熱集漂移**——對話進行中，路由熱集緩慢演化，
  「新進」的 experts 在**更早的步數或 prefill 階段早已出現過**；
- 論文的 Insight 1 正是證實：**prefill 階段出現過的 experts 就是 decode 熱集的超集**
  （在 200B-1000B 規模、24k requests 上驗證）。
- Swift SpAc 的 prefill 預熱（`beginOpeningRoutedExpertStreamer`）在 Swift 上已驗證
  「p0 hits 0% → 前幾個 token 內 pool 就緒」的改善——同一個機制可移植。

**結論：8.7% 不是不可預測，是「沒用對信號源」——應該用 prefill trace + 長期 utility，而不是 1 步/1 層的短時外推。**

---

## 3. 整合設計

```
                 ┌────────────────────────────────────────────┐
  routing trace  │  llama.cpp fork expert-cache（改）          │
  (hook 已有) ──▶│                                            │
                 │  預測器層（新增，C++）：                    │
                 │    • SpAc utility EMA（移植 Swift）         │
                 │    • prefill trace 種子（Insight 1）        │
                 │    • layer/token 分層（Insight 2）          │
                 │                                            │
                 │  執行器層（修 v1 race 後復用）：             │
                 │    • slot 狀態機 FREE→LOADING→READY        │
                 │    • 背景 pread（串行化寫入 pool）           │
                 │    • LRU 保留（現有）＋ pinned pool 由 utility│
                 │      驅動（取代純 LRU 熱池）                │
                 └────────────────────────────────────────────┘
                          │ 命中率↑ miss↓
                          ▼
        decode：ngl 0 8.82 / n99c 4GiB 11.08 → 目標 +15-30%
        MTP：draft context 與 target 共享同一 cache（消除第二實例）
```

### 3.1 信號源（全部已存在，零新增成本）

- **每步真實路由**：hook（`expert_cache_eval_cb`）已在 graph 層抓到每層 `selected_experts`
  （`CGC_MMID_PROBE` 等診斷證明可用）。
- **prefill trace**：prefill 階段的 routing 同樣經 hook 流出，落地成 per-layer seen-set +
  frequency（40×256 計數器，幾 KB）。

### 3.2 預測器（P1-P2）

1. **SpAc utility EMA**：移植 `MoESpAcEstimator` 邏輯（alpha=0.85、minScore=0.02、topK 查詢）。
   作用：決定 pinned pool 該持有哪些 experts（取代「LRU 熱池」的被動性）。
2. **prefill 種子（Insight 1）**：預測器 init 時以 prefill trace 的 seen-set 播種
   （Swift `seed(profileByLayer:)` 已有，0.5 中置信度起步）——**decode 第一步就命中熱池**。
3. **分層（Insight 2）**：reuse distance 短的 token 級 → LRU；長的 layer 級 → pinned pool。

### 3.3 執行器（P3：修 v1 race 的正確版本）

v1 race 根因：bg pread 與 sync fill 可能寫同 slot、drain 丟棄 queued fill 時狀態不一致。
正確設計（slot 狀態機，單寫者）：

- 狀態：`FREE → LOADING → READY`，`owner` 欄位在 LOADING 即鎖定；
- **pool 寫入只有一個來源**（背景預取執行緒或 hook sync fill，兩者互斥——以 per-layer mutex 保證）；
- drain/換出只能在 `READY` 狀態做（LOADING 的 slot 不可被重用）；
- Metal 路徑（n99c）下背景 pread 目標是 Metal shared buffer，完成後再 dispatch
  （保持「fill 完成 → dispatch FFN」的串行化，這是 n99c bit-identical 的前提）。

### 3.4 MTP 整合（P4，掛在預取轉正之後）

- **draft context 共享 target cache**（消除第二 cache 實例——MTP §8.7 的頭號成本）；
- verify 的 union fill 由預測器驅動（predict next-step union 再 fill，而非被動 miss-fill）；
- 接受率本身已 83-91%（P0），不是瓶頸。

---

## 4. 分刀計劃與驗證閘

| 刀 | 內容 | 驗證閘（過才進下一刀） |
|---|---|---|
| **P0** | clone `zhongkaiyu/moe_exp_placement`（moe_placement branch）+ 抓論文 HTML；用它的離線方法 + 我們的 qwen36/gemma4 真實 routing trace 跑一遍軌跡分析，確認 prefill→decode 相似度在我們模型上成立 | prefill 前 8 tok 的 seen-set 覆蓋 decode 熱集 ≥ 70% |
| **P1** | 移植 SpAc utility 排程到 fork（純 C++，`LLAMA_EXPERT_CACHE_SPAC=1`），取代 pinned pool 決策；離線用 trace 驗證 top-K 預測準確率 | top-43/layer 預測覆蓋 ≥ LRU 現況（91.3%） |
| **P2** | prefill 種子（Insight 1）：load/prefill 階段收集 seen-set → decode 前 async 預熱 pool（鏡像 Swift `beginOpeningRoutedExpertStreamer`）| 冷啟動首 8 tok 的 hit 率 ≥ 90%（現在 p0 低） |
| **P3** | 正確的執行器：slot 狀態機 + 背景 pread（`LLAMA_EXPERT_CACHE_PREFETCH=1` 重新啟用，v2 語義）| 真實文本 128 tok：hit ≥ 94%、tok/s 淨正（ngl 0：8.82→≥9.5；n99c 4GiB：11.08→≥12）|
| **P4** | MTP 轉正：draft 共享 cache + 預測驅動 union；llama-speculative 重跑 §8.7 A/B | MTP-4 端到端 ≥ 純 cache（≥10.77）|

每刀都維持 §7.2 的方法論：真實文本（非 llama-bench 隨機 token）、多輪交錯、bit-identical 對拍、
`-ngl 0` 與 `n99c 4GiB` 雙臂。

---

## 5. 風險與誠實邊界

1. **論文規模差距**：Insight 1 在 200B-1000B 多機驗證；我們的 qwen36 35B 單機、IQ3 量化——
   prefill→decode 相似度在低 bit 量化下是否保留是開放問題（P0 先用離線軌跡回答，成本最低）。
2. **8.7% miss 的量級**：就算預測 100% 命中，理論回收 = 8.7% miss × 每 miss 165µs
   （內部盤）≈ 每個 128-tok 的 run 省 ~1.8s（現 15s 級），**預測器+背景執行緒的開銷
   必須 < 這個數**，否則又是淨負——P3 的 accept 條件要嚴格。
3. **Swift 移植 ≠ 直接複製**：Swift 是 async Metal command buffer 模型；fork 是 ggml sched +
   hook 串行模型，SpAc 的 async 語義要重做（P1 的 diff 邏輯可複用，執行模型不同）。
4. **repo 現況**：AE 是離線仿真 + EP 佈局，**不是 runtime 預取**；「已放在 code 裡」目前
   在本機/外接盤都找不到（P0 第一件事就是 clone 確認）。
5. **MTP 不保證轉正**：P4 是「預取轉正後 MTP 才有機會」，不是「MTP 必然轉正」——若 P3
   預取淨正但 MTP 仍淨負，MTP 維持放著。

---

## 6. 參考

- 論文：Yu et al., *Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference*, arXiv:2510.05497（ISCA'26，v5）
- AE repo：`github.com/zhongkaiyu/moe_exp_placement`（branch `moe_placement`；`zhongkai_test/` 為離線分析腳本）
- Swift SpAc：`moeexpert/qwen3.6/.../Kernels/MoE/MoESpAcEstimator.swift` + `Model.swift` 四個 hook（arXiv:2603.09983 MoE-SpAc）
- 本 fork 現況：`LLAMACPP_EXPERT_BOUNDED_RESIDENCY_FORK_方案.md` §7.2（P2 證偽）/ §8.7（MTP）/ §8.8（位寬曲線）/ §8.10（n99c 11.08）

---

## 2026-08-18 更新（§8.82/8.83）：規劃書方向修正 + nextn 單層 graph backlog

### 現況（推翻本規劃書 v1.0 的部分前提）
1. **accept 46.4% 是 trunk 自我預測，不是 MTP head**（§8.83）——fork 的 `build_mtp_block`（qwen35moe.cpp:556-727）完整存在但**主流程未接通**，draft 用「embd 輸入 + 完整 40 層 trunk + LM head」跑 4 次完整前向。ISCA'26 預測整合（P0）已證偽、與此無關。
2. **真正的 MTP 轉正槓桿 = 接通 build_mtp_block**（draft 只跑 blk.40 單層），不是預測器、不是 harness。

### Backlog（P2，~5 工作天，詳見方案文 §8.83）
| D | 工作 |
|---|---|
| D0 | qwen36-hf MTP head 結構確認（eh_proj/enorm/hnorm 是否存在） |
| D1-2 | 補 NEXTN_* tensor（3-6 顆）進 GGUF / loader 映射 |
| D3-4 | build 主流程接「embeddings_nextn → build_mtp_block」分支（deepseek32 mtp_only 參考） |
| D5 | MTP head 真 accept 重測 + draft 41層→1層 帳本重算 |

門檻：MTP-4 t/s > no-spec × 1.3。接通前 MTP 線維持封存；接通後轉「可驗」。

### D0 完成（2026-08-18）：hf MTP head = DeepSeek-V3 式，三顆 tensor 全存在，Nail GGUF 已預裝 → D1-2 縮減為 0 天

**驗證對象**：`/Volumes/AlexZhuang/qwen36-hf`（26 shards safetensors，19 顆 MTP tensor 全在 shard 25-26）+ `models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf`（753 tensors）。

**hf → GGUF tensor 對照（D1 tensor 清單）**：

| fork 需求（build_mtp_block） | hf 名稱 | shape（hf） | Nail GGUF 名稱 | shape（GGUF） | 狀態 |
|---|---|---|---|---|---|
| `nextn.eh_proj` | `mtp.fc.weight` | [2048, 4096] | `blk.40.nextn.eh_proj.weight` | [4096, 2048] | ✅ 已有 |
| `nextn.enorm`（embedding norm） | `mtp.pre_fc_norm_embedding.weight` | [2048] | `blk.40.nextn.enorm.weight` | [2048] | ✅ 已有 |
| `nextn.hnorm`（hidden norm） | `mtp.pre_fc_norm_hidden.weight` | [2048] | `blk.40.nextn.hnorm.weight` | [2048] | ✅ 已有 |
| `nextn.shared_head_norm` | `mtp.norm.weight` | [2048] | `blk.40.nextn.shared_head_norm.weight` | [2048] | ✅ 已有 |
| KV `n_layer_nextn` | config 對應 | =1 | `qwen35moe.nextn_predict_layers` | =1 | ✅ 已有 |
| `nextn.embed_tokens` / `shared_head_head` | 無（DeepSeek-V3 式共用 trunk） | — | 缺席（TENSOR_NOT_REQUIRED） | — | fallback `model.tok_embd` / `model.output` ✅ |

**結構判定：DeepSeek-V3 式**（與 fork build_mtp_block 實作逐行對齊）——h_norm(hidden) ∥ e_norm(embd) → concat → eh_proj(2×2048→2048) → 1 層 transformer（GatedAttn：離散 q/k/v + q/k_norm + gate sigmoid）→ MoE FFN（routed + gated shared expert）→ shared_head_norm → lm_head。

**Nail blk.40 完整性**：20 顆 tensor 全齊（attn_q/k/v/o Q6_K、ffn_gate/up/down_exps、ffn_*_shexp 全組、gate_inp BF16、nextn.* 4 顆）——fork 名稱表（llama-arch.cpp:513-518）與 Nail 命名逐字吻合，**loader 開箱即讀，零 repack、零 header surgery**。

**⚠️ 兩個品質注意（D5 才見真章）**：
1. blk.40 experts 是 **Q2_K/Q3_K**（trunk 是 IQ3_XXS）——MTP block 比 trunk 更粗，可能壓低真 accept
2. MTP attn 是離散 Q/K/V Q6_K（trunk 是 attn_qkv 融合）——build_mtp_block 的 GatedAttn 分支正好對應，無結構缺口

**Backlog 修正**：D1-2（repack/loader 映射）= **0 天**（Nail 檔已內建）；剩 D3-4（build 主流程接線）+ D5（重測）≈ **3 工作天**。接線點：build()（qwen35moe.cpp:180-248）trunk 迴圈後無任何 build_mtp_block 呼叫；需在 `embeddings_nextn_masked`（draft ctx）時跳過 trunk 只跑 build_mtp_block（h input = 前一步 `t_h_nextn`）。

### Backlog 更新（2026-08-18 §8.85/8.86）：前置再清兩個，剩 D3-4 + D5

**§8.85（三處界線延伸 n_layer_all）**——blk.40 進 bounded pool（skip-load + hook 填補 + LRU）：
- loader 1278/1355 + context 1795 改 n_layer_all（adoption loop / L4 buft / remap leaf 原本就是）
- 驗證：**-ngl 99 + cache 4GiB + MTP 兩臂都跑通**（C0 exit=0；MTP accept 46.4%、hit 46.8%、RSS 10.7GB、`HOOKFIRE il=40` 觸發）
- 順帶：spec-simple 的 `-c` 預設吃 n_ctx_train 262144 → KV 5GB 是 OOM 主因之一，harness 已加 `-c 4096`（KV 80MB）

**§8.86（MTP teardown double-free）**——spec-simple 修成可靠 harness：
- 根因：`speculative-simple.cpp:81` `ctx_dft.reset(spec_init->context())` 雙重持有 → teardown 雙 llama_free → `free_tiny_botch`（ASan 確認 double-free，非 heap 損壞；與 OA_ASYNC/-fa/外接盤/§8.85 都無關）
- 修復：新增 `common_speculative_init_result::release_context()`，main 改 `reset(spec_init->release_context())`
- 驗證：**-ngl 0 無 cache / -ngl 99 + cache 的 MTP 全部 exit=0**（ASan build 亦乾淨）——spec-simple 現在可作為 D5 的正式量測 harness（修復前每次 teardown abort，accept 只能從 abort 前的統計讀）

### 剩餘 backlog（從 ~5 工作天縮到 ~3 工作天）
| D | 工作 | 狀態 |
|---|---|---|
| D0 | hf MTP head 結構確認 | ✅ 完成（DeepSeek-V3 式，D1-2 = 0 天） |
| D1-2 | NEXTN_* tensor 補檔 | ✅ 不需要（Nail GGUF 已內建 4 顆 + KV） |
| — | blk.40 進 pool（§8.85） | ✅ 完成（-ngl 99 + cache + MTP 可跑） |
| — | spec-simple teardown double-free（§8.86） | ✅ 完成（harness 可靠、exit=0） |
| D3-4 | build 主流程接「embeddings_nextn → build_mtp_block」分支 | ⏳ 唯一剩的工程（~2-3 天） |
| D5 | MTP head 真 accept 重測 + draft 41層→1層 帳本重算 | ⏳ 等 D3-4 |

門檻不變：MTP-4 t/s > no-spec × 1.3。接通後 MTP 線從「封存」轉「可驗」。

### Backlog 更新（2026-08-18 §8.87）：D3-4 翻案免工程 + D5 帳本重算——MTP 定案封存

**D3-4 翻案**：接線**本來就存在**——`build_arch_graph`（qwen35moe.cpp:155）在 `gtype==DECODER_MTP` 時回傳 `graph_mtp`，speculative.cpp:2366 已設 `ctx_type=MTP`。§8.83 的「build_mtp_block 無呼叫點」是 grep 漏了這層分派。節點數實測：**target 4087 vs draft 92**——draft 已只跑單層 MTP block。

**D5 帳本重算**（128 tok、-ngl 99 + cache 4GiB、load 2.2-2.8 同窗）：

| 臂 | t/s | accept |
|---|---|---|
| C0（no-spec） | 11.18 | — |
| MTP-4 | **7.68** | **29.6%** |
| 閾值（×1.3） | 14.53 | — |

**定案：MTP-4 −47%，遠低於閾值，正式封存（非「待驗」）。** 根因：blk.40 是 256-expert MoE block（~1.5GB，非 dense）→ 4 draft token/step ≈ 6GB weight reads；accept 29.6% < 35-40% 門檻（Q2_K/Q3_K 粗量化 + trunk IQ3_XXS 精度不匹配）。速度標的維持 Swift/turbo 線。

| 項目 | 狀態 |
|---|---|
| D3-4 接線 | ✅ 免工程（翻案，已存在） |
| D5 帳本重算 | ✅ 完成（淨負，封存定案） |
| CGC_NODE_COUNT env | ✅ 加入診斷工具集（印 ctx_type + 節點數） |

### 最終定案（2026-08-18 §8.89）：draft 深度掃描證偽——MTP 線關閉

draft 深度掃描（n-max 2/3/4，96 tok 交錯 2 輪）：c0 10.76 t/s vs d2 8.66（−19%）/ d3 7.35（−32%）/ d4 7.22（−33%）。accept 隨深度縮短上升（d2 42.5%）但無深度能翻正——**draft 結構成本（blk.40 是 256-expert MoE block，每 draft token 一次 ~1.5GB pool 讀取）是結構性淨負**，非 accept 品質問題。除非 blk.40 dense 化，MTP 不值得再投入。速度標的維持 Swift/turbo 線。
