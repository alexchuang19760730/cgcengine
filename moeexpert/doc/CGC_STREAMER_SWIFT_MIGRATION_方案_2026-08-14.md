# cgc/expert_streaming 統一框架：Swift → C/C++ 遷移方案

日期：2026-08-14
方向定案：**cgc/expert_streaming（C/C++）= 統一 per-expert per-layer 流式層（qwen36 + gemma4 雙家族），llama.cpp = compute backend（Metal/CUDA/Vulkan），Swift 引擎的流式邏輯遷移到 C/C++**，跨平台、不綁 Swift。

## 1. 為什麼這個方向成立（既有實測基礎）

| 資產 | 狀態 | 證據 |
|---|---|---|
| C GGUF parser（cgc_gguf_lite） | ✅ 已修好 | qwen35moe IQ2/IQ3 header 完整解析（ARRAY/bool/架構 KV 三 bug 已修） |
| C segment-aware per-expert 定址 | ✅ 已修好 | llama.cpp 三張量佈局（gate/up/down 分開、down 在前）9/9 byte-identical |
| C LRU slot cache + pread | ✅ 已有 | bench_gguf_fill：212 steps、warm 37ms/step |
| C repack/quantize | ✅ 已有 | cgc_repack + cgc_quantize（.gturbo 可退場，GGUF 為正典輸入） |
| 命中率模型 | ✅ 已驗證 | 3GB pool 純 LRU 82.9%（r3_trace 212 decode tokens），adaptive per-layer 大池中性 |
| compute | ⚠️ 缺 | C 框架無 Metal kernel；llama.cpp 有全套 qwen35moe + gemma4 kernel（M4 實測 25-28 tok/s） |

## 2. 目標架構（分層）

```
┌─────────────────────────────────────────────────────┐
│ 應用層（CLI/server，未來統一 launcher 只認 GGUF）        │
├─────────────────────────────────────────────────────┤
│ L2 排程層（從 Swift 遷移）                              │
│   fill-hiding（GPU 執行期間背景 pread 下一層/token）      │
│   router-early hook（top-k 提早 → 預取 union）          │
│   chunked prefill 冷池阻塞 + 後續 chunk 重疊            │
├─────────────────────────────────────────────────────┤
│ L1 流式核心（C，已 80%）                                │
│   slot table + LRU + 每層動態 slot + hot pool（JSON）   │
│   MoE-SpAc 估測器（Swift 移植）                         │
│   pread workers（segment-aware，qwen36+gemma4 共用）    │
├─────────────────────────────────────────────────────┤
│ L0 模型層（C，已 95%）                                  │
│   cgc_gguf_lite：qwen35moe.* + gemma4.* KV + per-layer │
│   佈局 → per-expert (layer, expert) → 檔案 offset       │
├─────────────────────────────────────────────────────┤
│ L3 compute（llama.cpp，插拔式）                         │
│   macOS: Metal；Linux/Windows: CUDA/Vulkan            │
│   expert 資料由 L1 pool 提供（bounded residency 或全載入）│
└─────────────────────────────────────────────────────┘
```

**關鍵決策**：Swift 的 ~24K 行 Metal kernel **不移植**——llama.cpp 的 kernel 在同顆 M4 快 4.5×（25-28 vs 5.5-5.8 tok/s），移植是負收益。要遷移的是「流式/排程邏輯」~5-7K 行：

| Swift 檔 | 行數 | 遷移內容 | C 現況 |
|---|---|---|---|
| PreadExpertStreamer.swift | 1813 | slot/LRU/hot pool/prefetch/背景執行緒 | ~60% 已有（補 profile hot pool 載入、每層動態 slot） |
| RealForwardRunner.swift（gemma4） | 3103 | decode 排程 + fill-hiding + WAKE_POLL | 0（排程層） |
| Qwen36ForwardRunner.swift | 2639 | qwen36 decode 排程 + MTP | 0（排程層） |
| Model.swift / ModelExpertIO.swift | 1177 | 層模型/IO 抽象（→ 共用佈局結構） | 部分（cgc_stream_layout） |
| MoESpAcEstimator.swift | 111 | 預取估測器 | 0 |
| ExpertAccessTrace.swift | 88 | trace 收集（feed profile 生成） | bench 已有 trace replay |

## 3. 分期（每期可獨立驗收）

### P0：gemma4 支援驗證 ✅ 已完成（2026-08-14）
- 原本的 `models/gguf/gemma-4-26B-A4B-it-UD-IQ3_S.gguf` 是 **503MB 截斷檔** → 已改用 unsloth repo 重新下載完整版（**11.29GB**，hf-mirror 續傳）
- **修了 `cgc_gguf_lite.c` 的 ARRAY 元素跳過 bug**：GGUF ARRAY 元素大小依 type 而異（BOOL/UINT8=1B、INT16=2B），原實作一律 `fseek 4` → gemma4 的 `sliding_window_pattern`（ARRAY<BOOL>, n=30）多跳 90 bytes，KV 全偏移、解析卡死。修復後：
  - gemma4 GGUF **完整解析**（PER_LAYER、128 experts/層、stride 2.3MB、hidden 2816、inter 704，`ffn_gate_up_exps` 合併佈局）
  - qwen35moe IQ2 回歸 **9/9 byte-identical PASS**（修復無破壞）
- **`.gturbo` repack 降為可選並退場**：C parser 直接吃 llama.cpp GGUF 已驗證，不再需要 repack / .gturbo / model_weights.bin（用戶定案 2026-08-14）
- **修了 `cgc_expert_streamer_gguf.c` 的 gemma4 合併佈局 segment bug**：`ffn_gate_up_exps`+`ffn_down_exps`（2 張量）原本誤入 qwen35moe 的 3 張量分支（`seg_base[0]/[1]` 取未設過的 gate/up offset = 0）→ 分支條件收斂為「gate/up 分開張量真實存在」，合併佈局走獨立 2 段分支（`seg_base=[gate_up, down, 0]`）
- **修了 data_start 偏移 bug（關鍵）**：GGUF v3 的 tensor offset 是「相對 data_start」的，但 C parser 的 seg_base/stream_offset/expert_offsets 全用 raw offset → pread 讀到早 ~15.8MB 的錯誤區域。以 gguf-py 權威 `data_offset`（656,107,808）對照 C parser raw（640,283,136）差距恰為 data_start（15,824,672）實錘。修復後 gemma4 `seg_base[1]` 與 gguf-py **完全一致**；之前「byte-identical」只是兩邊都錯得一致，現在驗證的是真實檔案佈局
- **gemma4 segment byte-identical 驗證 ✅（`tests/test_real_gemma4.c`）**：6/6 PASS——experts 0/3/127 的 gate_up（IQ3_S, 1270016 B/exp）與 down（1115136 B/exp）段，pread bytes 與 tensor 表絕對定址（data_start+raw）完全一致；qwen35moe IQ2 回歸 9/9 PASS、合成回歸 3/3 PASS
- **fill bench 泛化為雙家族通用**（`tools/bench_gguf_fill.c`）：動態層數/專家數、合併/分開佈局自動偵測、layer-0 段表 vs parser 驗證、trace 不足自動合成；gemma4 30 層/128 exp/2 段、qwen36 40 層/256 exp/3 段都跑通
  - 同款合成 trace 對照：gemma4 warm **68.7ms/step（cold 79.7）、572.4MB/step、eff ~8.3GB/s** vs qwen36 warm **56ms/step、280.5MB/step、eff ~5.0GB/s**——**raw fill 不同級**（gemma4 每 expert 2.39MB vs qwen36 0.877MB，step bytes ×2.04）
  - **真實 gemma4 routing trace（Swift 引擎 gemma4-r4 收 211 decode tokens，`/tmp/gemma4_trace.csv`）對照**：warm **31.55ms/step**（cold 46.7）、eff **18.1 GB/s**——**真實路由有 token-to-token 局部性，比合成隨機好 2.2×**（合成每步觸碰幾乎全部 experts、頁快取 churn 最壞）
  - 但**有效 fill（cache 過濾後）同級**：3GB pool 下 gemma4 IQ3 80.7% hit → 只讀 ~110MB/step → ~13ms/step，qwen36 82.9% → ~10ms（sim 已驗證）——raw streaming 數字不影響 P2-B 設計
  - bench 順手修了相位過濾 bug：`decodeProtected` 是 15 字元，原檢查寫 `==14` → 真實 trace 全被濾成 0 tokens
- **llama.cpp bench 端到端（P0 驗收，2026-08-14）**：

  | 模型 | 配置 | prefill (pp128) | decode (tg128) | 記憶體 | 結果 |
  |---|---|---|---|---|---|
  | qwen36 IQ2_XXS（10.01GiB） | `-ngl 99` Metal | **189.65 t/s** | **25.5-27.9 t/s** | free% 底 0.7%（Metal 緩衝不計 process RSS） | ✅ 可跑 |
  | gemma4 IQ3_S（10.50GiB） | `-ngl 99` 任何 Metal 配置 | — | — | — | ❌ `kIOGPUCommandBufferCallbackErrorOutOfMemory`（-3） |
  | gemma4 IQ3_S | `-ngl 0` 純 CPU | **32.02 t/s** | **16.28 t/s** | peak RSS 9.52GiB、free% 0.5% | ✅ 可跑（CPU） |

  - **gemma4 Metal 失敗根因（就是 10GB 預算的實錘）**：Metal working set 上限 11,453MB；gemma4 模型 11,286MB + KV/workspace 超出 → 任何 ngl/ncmoe/nkvo/fitt 組合都 `res=-3`（graph compute 失敗，底層 `kIOGPUCommandBufferCallbackErrorOutOfMemory`）。-ncmoe 30（全 experts 丟 CPU）也救不了——down experts（IQ4_NL）觸發 CPU_REPACK 多佔 ~4GB RAM，系統層壓力照樣 GPU OOM。qwen36 IQ2（10,751MB）剩 ~700MB 給 KV 所以過關
  - **gemma4 要上 Metal 只有兩條路**：P2-A fork bounded residency（experts 不進 Metal），或更小 quant（<10GB）
  - 純 CPU gemma4 16.28 t/s 已快過 Swift 引擎 gemma4-r4 的 12.7 tok/s（CPU BLAS 4 threads）——llama.cpp kernel 優勢顯著
- 驗收：qwen36 + gemma4 雙檔在 C parser 下 byte-identical ✅

### P1：流式核心補齊（3-5 天）
- profile hot pool 載入（既有 top*.json 直接讀）
- 每層動態 slot（adaptive，§13.147 手法；大池中性但小池有用，保留開關）
- MoE-SpAc 估測器移植
- 驗收：sim_gguf_cache.py 對同 trace 的 hit% 與 Swift 引擎實測對齊（52.2% 生產組態可重現）

### P2：compute 整合（1-2 週，最大塊）
- 兩條路：
  - **A. llama.cpp fork bounded residency**（`moeexpert/LLAMACPP_EXPERT_BOUNDED_RESIDENCY_FORK_方案.md`，~1000 行）：C streamer 當 residency manager，`kernel_mul_mm_id` +slot 表。IQ3 可跑、餘裕真實
  - **B. llama.cpp 全載入 + C streamer 只做預取**（~200 行）：多工退化已實測 ≤3%，C streamer 預取讓 OS 頁常在，16GB 上 IQ2 就是 25-28 tok/s
- **建議先 B 後 A**：B 是薄整合、立刻可用於兩家族；A 只在「模型放不下工作集」（IQ3）時才需要
- 驗收：llama-bench 同參數，C streamer 開/關對照

### P3：排程層移植（1 週）
- fill-hiding（Swift §13.97 已證 46→5ms 手法）、router-early、chunked prefill 重疊
- 驗收：128 tok 冷窗 A/B，vs Swift 引擎同設定數字

### P4：跨平台收尾（3-5 天）
- 統一 launcher（run_prod.sh 家族化改吃 GGUF + C streamer，FAMILY=qwen36|gemma4 保留）
- Linux CI（cmake + gcc，已有 build_c_gcc.sh）+ Windows（build.ps1 已存在）
- Swift 引擎退為 reference/fallback

## 4. 立即要做的兩件事（防「共存」繼續壞著）

1. **FAMILY=gemma4 目前是斷的**：統一 launcher 指向 `models/gemma4-r*.gturbo`（內部盤不存在，模型在外接盤 `/Volumes/AlexZhuang/gemma4.gturbo` 13GB）——在遷移完成前，複製回內部盤或改 launcher 路徑，二選一
2. **moeexpert/gemma4 樹的 run_prod.sh 是舊 gemma4-only 版**（0 個 FAMILY 引用），需同步統一版

## 5. 風險

- ~~gemma4 GGUF 截斷（503MB）~~ → 已解決：完整版 11.29GB 已下載、C parser 解析通過（P0 完成）；.gturbo repack 保留為備援
- llama.cpp fork（A）長期維護成本高 → 以 B 為默認，A 只在 IQ3 場景開
- Swift 排程邏輯與 llama.cpp 的 graph 語義不同（cb hook 位置）→ P3 需要對齊 top-k 提前觸發點
- 遷移期間兩套並存 → 統一 launcher 的 FAMILY 開關是唯一入口，避免雙引擎漂移

## 6. 與既有文檔的關係

- fork 方案（BOUNDED_RESIDENCY）：成為 P2-A 的實作藍圖
- 多工退化實測（多工退化_實測）：成為 P2-B 的成立依據（≤3%）
- sim_gguf_cache.py + bench_gguf_fill：P1 的驗收工具（已在 cgc/expert_streaming/tools/）

---

## 附錄：Swift TurboFieldfare 重建指令（2026-08-15 清 .build 後）

兩個 Swift `.build` 目錄已清掉換空間（3.0G，內部盤 13→16Gi free）。**成品 binary 已備份**：

| 專案 | 備份位置 | 說明 |
|---|---|---|
| gemma4 `turbo-fieldfare-github-official` | `bin/TurboFieldfareCLI-*`（5 個變體，standalone Mach-O）| .build 刪除無損失 |
| qwen3.6 `prime-agent-worktrees/turbo-fieldfare` | `bin/TurboFieldfareCLI`（新備份，自 .build/release 複製）| 原本只在 .build 內 |

**重建**（任一專案根目錄）：

```bash
swift build -c release                       # 全部 products
swift build -c release --product TurboFieldfareCLI    # 只要 CLI
swift build -c release --product TurboFieldfareServer # server 版
```

- 需要 swift.org 6.2 toolchain（此前已裝）；依賴 swift-transformers / swift-nio 由 SPM 自動解析。
- 重建約 5-10 分鐘（-c release）。
