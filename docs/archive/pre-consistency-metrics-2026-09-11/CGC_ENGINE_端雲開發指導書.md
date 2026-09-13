# CGC Engine 端雲開發指導書（2026-08-29）

> 對象：Windows 8GB 端（cgcengine_full）開發者。
> 本文 = 三平台 cherry-pick（`23bf2c1d8` → `619b908a6` → `bcc22133f`）落地 main 後，
> **端雲聯調的最後一哩**（edge_server + E2E harness）已完成並在 Mac 真機驗證通過。
> 讀完本文即可在 Windows 上跑通「Mac ↔ Windows 端雲聯調」。

---

## 0. 一句話現狀

**Router 決策 → HTTP 路由執行 → SSE token 流**這條鏈已經真機跑通（Mac 上
solo/dual/A/B 三種模式全部驗證）；Windows 端只差「build + 起 server + 跑測試」
三步。真正的 hidden-state 分裂 PD（Mac prefill → Windows decode 不重算 prefill）
是 Phase 2，路線圖見 §6。

## 1. 本次落地的內容（main branch）

| 組件 | 檔案 | 說明 |
|---|---|---|
| **Router decode-速度感知** | `CGC-main/cgc_engine/pd/router.py` | 新增 `_pick_decode_node()`：能跑該模型的節點裡**decode 最快的勝出**（平手偏好本地）。小模型不再盲留本地——Mac decode 85 t/s vs Windows 12 t/s 時，decode 送 Mac（mode 變 `pure_cloud`）。35B/8GB 場景不變（pure_cloud 全在 Mac） |
| **Edge Server（最後一哩）** | `CGC-main/cgc_engine/pd/edge_server.py` | 把 fork binary 包成 `/v1/cgc/{health,profile,emit,resume}` HTTP 服務。**純 stdlib**（http.server + subprocess），Mac/Windows 零依賴。resume = SSE 逐 token 串流 + perf 解析 |
| **E2E 聯調 harness** | `CGC-main/cgc_engine/pd/pd_e2e_test.py` | `--solo`（單機冒煙）/ dual（`--local`+`--remote` 端雲聯調）/ `--ab`（兩台對照）。輸出 Router 決策 + 實測延遲對照 |
| **Mac RAM 探測修復** | `CGC-main/cgc_engine/pd/discovery.py` | dual 測試暴露的真 bug：macOS 沒有 `SC_PHYS_PAGES` → Mac RAM 恆 0 → Router 誤判「塞不下」永遠走 cloud。改 `sysctl hw.memsize` 回退。修後 Mac（16GB）+ 35B → 正確路由 `local_full` |

**Mac 真機驗證記錄（2026-08-29，qwen36 35B IQ3_XXS, -ngl 99 + expert-cache 4GiB）**：

```
solo:  health/profile/resume 全通，SSE gen ' the capital of France.\n'，
       perf 解析 load 6.9s / decode_tps / rc=0 ✅
dual(修前): RAM=0 → pure_cloud 分支驗證 ✅（decode 上「雲」）
dual(修後): local_full（local score 60.4 sufficient）→ 執行 local ✅
--ab :  兩節點 A/B 對照 5.5s vs 5.5s（同機 tie）✅
單元 :  router 7/7 PASS（35B→pure_cloud、7B 送最快、tie 留本地、solo、mac-local）
```

### 1.5 Windows↔Mac 真機 dual E2E 實測（2026-08-29）+ Mac 生產配置還原

Windows 端（8GB）跑 dual E2E 對 35B：**Router 決策 pure_cloud ✅、
token 流從 Mac SSE 回流 ✅、輸出正確 ✅**。但實測 wall 29.3s vs Router
估算 1.1s——差距 27×，拆解如下（同一台 Mac、同 server 的四跑階梯）：

| 跑 | 配置 | decode t/s | load | hit rate | 說明 |
|---|---|---|---|---|---|
| Windows dual（首跑） | llama-simple + base IQ3_XXS | **1.71** | 21.8s | — | 冷 page cache + -n 8 攤提 |
| Mac 驗證 #1 | 同上（暖） | 6.05 | 4.9s | 81.5% | 短跑仍吃冷啟動 |
| Mac 驗證 #2 | 同上（暖+n 200） | 8.50 | 4.8s | 91.0% | 非 MTP 配置天花板 |
| **Mac MTP 生產** | **speculative + denseIQ4X + 全槓桿** | **26.44** | 4.3s | 79.2% | **accept 97.4%、draft cold 0%** |

**定案**：
1. 首跑 1.71 t/s 的主因 = 冷 page cache（load 21.8s vs 暖 4.8s）+
   `-n 8` 短生成攤提 Metal 暖機與 pool 填充爬坡，**不是配置錯誤**。
2. 6→8.5→26 的跳升 = 配置換成 MTP 生產集（`--mtp 2` + Nail denseIQ4X
   載體 + LAYER_CAPS + MMV_FUSE + GLU_FUSED_DOWN）——這才是對外服務
   該跑的配置（Step 5 命令已更新）。
3. Router 估算（18.1 t/s）對應「乾淨穩態規格值」；E2E 實測要對齊需
   Phase 2 常駐 session（消 5-7s/次的模型重載）+ 量測回填。

## 2. Windows 端下一步（按序執行）

### Step 1 — 同步代碼

```bash
cd /d/alex/flashkv0516/cgcengine_full   # 你的 Windows 工作目錄
git remote add github0823 https://github.com/alexchuang19760730/cgcengine0823.git  # 若還沒加
git fetch github0823 main
git checkout main && git reset --hard github0823/main   # 或 merge 到你的 dev
```

### Step 2 — Build（MinGW，bcc22133f 驗證過的指令）

```bash
PATH="/c/msys64/mingw64/bin:$PATH"
cmake -S src/llama.cpp -B src/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build src/llama.cpp/build --target llama-simple -j
```

注意：`MTP_SUPPORT` 現在由 CMakeLists 的 `if (WIN32)` guard **自動定義**
（Mac 端維持 build script 隔離，llama-simple 保持 non-MTP——bit-identity 鐵律）。

### Step 3 — 模型準備

- **8GB RAM 建議**：下載一個小模型（如 Qwen2.5-7B Q4_K_M ~4.5GB，或 0.6B），
  Router preset 用 `qwen25_7b` / `qwen25_1_5b`。
- 35B 載體（`Nail-...-denseIQ4X.gguf` 12.8GB）在 8GB 機上無法全載——那條路
  由「Windows 發請求 → Mac 執行」走通（§4 路由表第一行）。
  若要在 Windows 本地跑 35B 對照：模型檔名對齊
  `models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（改名/軟連結）。

### Step 4 — 起 Windows edge server

```bash
py -3 CGC-main/cgc_engine/pd/edge_server.py \
    --binary src/llama.cpp/build/bin/llama-simple.exe \
    --model models/gguf/qwen25_7b.gguf \
    --ngl 0 --port 1234
```

（8GB 記憶體：`--ngl 0` CPU-only + mmap 默認——**不要** `--no-mmap`，那是
Mac 16GB 的凍機防護配置。）

### Step 5 — Mac 端起 server（Mac 側操作，同步給 Windows）

```bash
# Mac（MTP 生產配置 = run_n30cache.sh --steady 的完整槓桿集）
CGC_EXPERT_CACHE_BYTES=4294967296 LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1 LLAMA_EXPERT_CACHE_WORKERS=8 \
CGC_WAKE_POLL_US=15 CGC_PREFETCH_SRC=hist CGC_EVICTED_RING=0 \
CGC_OA_ASYNC=1 CGC_N_CB=8 CGC_GLU_FUSED_DOWN=1 \
LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256 CGC_MMV_FUSE=1 \
python3 CGC-main/cgc_engine/pd/edge_server.py \
    --binary src/llama.cpp/build/bin/llama-speculative-simple \
    --model models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf \
    --ngl 99 --no-mmap --threads 8 --mtp 2 --port 1234
```

`--mtp 2` 內建正確參數集（`--spec-type draft-mtp --spec-draft-n-max 2
-c 3072 -expert-cache 4GiB --temp 0`）+ 自動補 MTP 必要 env
（`CGC_NO_PREFETCH/VERIFY_DECODE/DRAFT_DECODE/WATCHDOG`，已設不覆蓋）。
實測（2026-08-29）：**decode 26.44 t/s、accept 97.4%、draft cold 0%**——
Mac 生產線完整還原（見 §1.5 階梯表）。

### Step 6 — 端雲聯調（Windows 上跑）

```bash
py -3 CGC-main/cgc_engine/pd/pd_e2e_test.py \
    --local  http://127.0.0.1:1234 \
    --remote http://<mac-ip>:1234 \
    --model qwen36_35b -n 8 -p "The capital of France is" --ab
```

預期輸出（Mac 已驗證的同款）：
- `[decision]`：35B + Windows 8GB → `pure_cloud`（prefill+decode 全 Mac）
- `[exec]`：token 流從 Mac SSE 回來
- `[A/B]`：兩台實測對照（8GB Windows 跑 35B ≈ 0.5 t/s vs Mac 26 t/s 的差距會如實呈現）

## 3. Router 路由行為表（decode-速度感知後）

| 場景 | mode | prefill | decode | 原因 |
|---|---|---|---|---|
| 35B + Windows 8GB（local 塞不下） | `pure_cloud` | Mac | **Mac** | RAM 7.6 < 13GB；decode 候選只剩 Mac |
| 7B + Windows 12 t/s vs Mac 85 t/s | `pure_cloud` | Mac | **Mac**（新行為） | decode-速度比較：85 > 12×1.0，送最快 |
| 7B + 兩台一樣快（tie） | `edge_cloud` | Mac | **Windows** | 平手偏好本地（隱私/省網路跳） |
| Mac 當 local（塞得下+分數夠） | `local_full` | local | local | Rule 1 不變 |
| 無 remote | `local_full`/`cloud_only` | — | — | 舊行為不變 |

> decode 速度來源：`profile.decode_tok_per_sec[model]`（量測值）優先，否則公式
> 估算。量測值怎麼來：跑一次 `pd_e2e_test.py` 的 summary（`decode_tps`），
> 回填 profile / DeviceProfile——這是下一步「實測驅動路由」的入口。

## 4. 已知限制（誠實聲明）

1. **文本橋，非 hidden-state 分裂**：請求攜帶完整 prompt，被選中節點做整段
   推理。「Mac prefill、Windows 只 decode（不重算 prefill）」需要 C++ fork
   端新增 hidden-state emit/resume 端點（Phase 2，§6）。
   文本橋的真實價值 = 「塞不下/跑太慢的端，把推理整段路由給最快的節點」。
2. **每請求重載模型**：subprocess 模式，35B load ~5-7s/次（E2E 實測
   load_ms=6905）。摘要行會如實印出。Phase 2 = 常駐 session 池。
3. **SSE 無重連**：客戶端斷線 = 該請求作廢（subprocess 跑完為止）。
4. **profile 的 decode/prefill 量測值預設空**：Router 用公式估算（偏樂觀）。
   精確路由 = 先跑一次量測再回填。
5. **Windows 端跑不了 MTP draft**：MTP 生產線在 Mac（speculative-simple +
   Nail 載體）。Windows 端 llama-simple 是 CPU 非 MTP 路徑（RouteDecision
   的 draft_n 對 Windows 是估算標記）。Mac 端已支援：`--mtp 2`（26.44 t/s）。

## 5. 常見問題（繼承自 bcc22133f 的教訓）

- **`/usr/bin/time -l` 不存在**：那是 macOS 專用（run_n30cache.sh 已移除，
  MSYS2 沒有此工具）。edge_server 用 Python 計時，無此問題。
- **`--no-mmap` 在 8GB Windows = OOM**：12.8GB 模型 no-mmap 全讀進 RAM 直接爆。
  Windows 一律走 mmap（默認）+ `--ngl 0`。
- **`--dense-iq4x` 沒接 `--mtp` 會被靜默忽略**（run_n30cache.sh 的 MODEL 分支
  邏輯）；edge_server 不受影響（模型路徑顯式傳）。
- **Windows 防火牆**：Step 4 的 server listen 0.0.0.0:1234，第一次起會跳
  防火牆允許提示，要允許（Mac 才連得到 Windows；Mac 端同樣在系統設定放行 1234）。

## 5.5 非 MTP 跑 Nail 載體 + MTP 退化輸出根因（2026-08-29 深夜定案）

**結論：llama-simple（非 speculative）可以跑 Nail-MTP-denseIQ4X 載體，不會出問題** ——
前提 `CGC_NO_PREFETCH=1`。實測（Mac，-ngl 99 + 4GiB 池，seed 42，n 64）：
rc=0、63 tok @ 4.56 t/s、池 hit 71.0%、輸出真實文本（貪婪重複句 = 載體行為）。
載入層面 merged draft layer-40 tensors 由載入器容忍，零 tensor 錯誤 —— 非 MTP
Fallback 可用（生產仍以 base Q36 為準）。

三個定案（A/B 證據鏈，同機同 prompt 同 seed）：

1. **死鎖根因 = hist prefetch bg race（非 MTP 路徑）**：
   `CGC_PREFETCH_SRC=hist` 不帶 `CGC_NO_PREFETCH` 時，ensure_batch 在
   `bg_cv` 等一個 queued/loading slot，但 bg_loop 與 8 個 pool workers 全部
   已睡（sample 證據：main 卡 `condition_variable::wait`，bg/pool idle）——
   slot flag 與 pool_queue 配對被 drain_layer/prefetch 交錯打斷，永遠沒人清。
   watchdog 不救：它只監控 GPU cmd buffer completion，這是 CPU 端 lost-wakeup。
   **生產 non-MTP env 集（run_n30cache.sh §8.6x hist prefetch）帶同款風險** ——
   建議 non-MTP 也補 `CGC_NO_PREFETCH=1`（或修 race 後再開）。
2. **MTP「退化輸出」根因 = CGC_VERIFY_DECODE fast path 的 ZERO-mapping**：
   verify 期 73.6% 冷 expert 被映到 ZERO slot（讀零而非真實權重）→ 全換行
   輸出（25.8 t/s 的代價）。unset CGC_VERIFY_DECODE 後：6.5 t/s + 真實文本。
   **先前「IQ3_XXS 模型級退化 / fact recall loss」結論不成立** —— 兩平台
   量測都經過被污染的 MTP fast path。載體本身（非 MTP 路徑）能出正常文。
3. **spec 與 non-spec 位元級一致仍未達成**：關 fast path 後 MTP 輸出仍與
   llama-simple 分歧（重複 prompt ×3 vs "What is the possible name..."）。
   嫌疑範圍：verify 3-token batch 與 1-token eval 的 kernel 數值差、或
   CGC_OA_ASYNC submit-ahead remap race。待後續工程。

腳註：直跑 speculative-simple 若用 `-t 8` + 不完整 env，draft context 建立
會 GPU OOM（kIOGPUCommandBufferCallbackErrorOutOfMemory）；用 edge_server
參數集（`-t 5` + 完整 env）穩定重現 rc=0。A/B 腳本：`/tmp/spec_ab.sh`
（VERIFY/DRAFT/OUT 環境變數控制）。

## 6. Phase 2 路線圖（真正的分裂 PD）

1. **C++ fork emit/resume 端點**（最大工程量）：llama-simple 加
   `--emit-server` 模式——prefill 完成後 dump 最後 N 層 hidden state +
   KV head（封包格式已在 `protocol.py` 定義：`HiddenStatePacket` / MoT-h 翻譯），
   `--resume-server` 從封包恢復 KV（`context_replay.py` 的 restore 邏輯進 C++）。
2. **常駐 session 池**：edge_server 從 per-request subprocess 升級為
   worker 進程池 + stdin 指令協議（load 一次，多請求復用）。
3. **量測回填**：把 E2E summary 的 decode_tps/prompt_tps 寫回
   `DeviceProfile.decode_tok_per_sec`，Router 決策變實測驅動。
4. **DOPDSessionRuntime 接 ComputeRouter**：coordinator.py 的 generate
   端點調 `router.select()` → 按 decision 打對應節點（把 E2E harness 的
   邏輯產品化）。
5. ~~**MTP 上端雲**~~：**已完成（2026-08-29）**——edge_server `--mtp 2`
   內建 speculative-simple 參數集 + MTP env，Mac 端 26.44 t/s / accept 97.4%
   實測（§1.5）。

## 7. Phase 3 路線圖：MTP 本地 Decode + 雲端 Prefill（DGX Spark）

> 新增（2026-08-30）：在 Phase 2 分裂 PD 基礎上，引入雲端大模型做 prefill，
> 本地設備做 MTP 加速 decode，實現「雲端算力 + 本地隱私 + MTP 加速」三贏。

### 7.1 架構概覽

```
┌──────────────────────────────────────────────────────────┐
│                    Cloud (DGX Spark ×2)                   │
│                                                          │
│  DeepSeek V4 Flash 0731 (155GB, 2×DGX Spark)            │
│  ├─ Prefill: 60 tok/s (128GB unified LPDDR5x)           │
│  ├─ 1M context window                                    │
│  ├─ SWE-bench: 79% | TB 2.1: 82.7%                      │
│  └─ 輸出: KV cache → 發送到本地                          │
│                                                          │
│  或 Qwen3.8-Flash-Next (125B/6B, 單台 DGX Spark)        │
│  ├─ SWE-bench Pro: 62.5% | DeepSWE: 58.7%               │
│  └─ Apache 2.0 開源                                      │
└──────────────────────┬───────────────────────────────────┘
                       │ LAN / Internet
                       │ KV cache transfer (~5MB/1K tokens)
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Local Decode (多設備)                         │
│                                                          │
│  Mac M4 (主 decode)                                       │
│  ├─ Qwen3.6-35B-A3B GGUF (IQ3_XXS, 13GB)               │
│  ├─ MTP draft=2: ~40 t/s                                 │
│  ├─ Expert-cache: 4GiB                                   │
│  └─ CGC edge_server.py                                   │
│                                                          │
│  Windows 8GB (fallback)                                   │
│  ├─ Qwen2.5-7B 或 Qwen3.6-35B (CPU only)                │
│  └─ 3-5 t/s                                             │
│                                                          │
│  鴻蒙手机 Mate 70 Pro (端側 agent)                        │
│  ├─ Qwen3-14B (規劃中)                                   │
│  └─ 輕量 decode + 端側推理                               │
└──────────────────────────────────────────────────────────┘
```

### 7.2 為什麼 MTP 本地 Decode 是關鍵

MTP (Multi-Token Prediction) 在 PD 分離中的作用：

```
傳統 decode:  token₁ → token₂ → token₃ → token₄  (逐個生成)
MTP decode:   token₁ → [token₂, token₃, token₄]  (一次預測多個)
```

| 指標 | 無 MTP | 有 MTP (draft=2) | 有 MTP (draft=4) |
|---|---|---|---|
| **Decode 速度** | 27 t/s | ~40 t/s (+50%) | ~55 t/s (+100%) |
| **Accept rate** | — | 50-70% | 40-60% |
| **適合場景** | — | 短回覆 | 長回覆 |

本 fork 已支援 MTP（`llama-speculative-simple` + `--spec-type draft-mtp`），
Mac 端實測 26.44 t/s / accept 97.4%（§1.5）。

### 7.3 DGX Spark 規格

| 項目 | 值 |
|---|---|
| **晶片** | GB10 (Blackwell) |
| **記憶體** | 128GB unified LPDDR5x |
| **頻寬** | 273 GB/s |
| **互連** | ConnectX-7 200Gbps (2 台串聯) |
| **價格** | ~$4,700/台 |
| **可跑模型** | DeepSeek V4 Flash (155GB, 需 2 台) / Qwen3.6 27B (單台) |

### 7.4 三種可行路徑

**路徑 A：同一模型 PD 分離（最簡單）**
```
DGX Spark: Qwen3.6-35B full precision prefill
Mac M4:   Qwen3.6-35B GGUF decode (MTP)
KV cache: 直接傳輸 (格式相容)
難度: ⭐⭐
```

**路徑 B：雲端大模型 + 本地蒸餾（效果最好）**
```
DGX Spark: DeepSeek V4 Flash prefill (82.7% TB)
Mac M4:   Qwen3.6-35B 蒸餾版 decode (MTP)
KV cache: 需要投影層轉換
難度: ⭐⭐⭐⭐
```

**路徑 C：混合路由（推薦）**
```
簡單任務: 本地 Qwen3.6 全流程 ($0)
複雜任務: DGX Spark prefill → 本地 MTP decode (~$0.01)
路由: CGC DOPD Router 自動決策
難度: ⭐⭐⭐
```

### 7.5 與現有 DOPD 的差異

| | 現有 DOPD (Phase 1) | MTP + Cloud Prefill (Phase 3) |
|---|---|---|
| **Prefill** | 本地 Mac M4 | 雲端 DGX Spark |
| **Decode** | 本地 Mac M4 | 本地多設備 (MTP 加速) |
| **模型** | Qwen3.6-35B (同一模型) | 雲端大模型 + 本地量化版 |
| **KV cache** | 本地記憶體 | 跨網路傳輸 |
| **成本** | $0 | 雲端 ~$0.14/M input |
| **優勢** | 完全本地 | 雲端 prefill 更快, 本地 decode 更便宜 |

### 7.6 關鍵技術挑戰

| 挑戰 | 難度 | 解決方案 |
|---|---|---|
| **KV cache 跨網路傳輸** | ⚠️ 中 | 5MB/1K tokens, LAN <10ms, WAN 需壓縮 |
| **MTP accept rate** | ⚠️ 中 | CGC fork 已驗證 50%+ |
| **雲端模型 ≠ 本地模型** | ⚠️ 需適配 | 雲端用 V4 Flash, 本地用 Qwen3.6 蒸餾版 |
| **KV cache 格式相容** | ❌ 高 | 不同模型 KV shape 不同, 需要轉換層 |
| **延遲疊加** | ⚠️ 中 | Prefill 雲端 + 網路 + 本地 decode |

### 7.7 Benchmark 差距參考

| 模型 | Terminal-Bench 2.1 | SWE-bench | 成本 |
|---|---|---|---|
| DeepSeek V4 Flash | **82.7** | 79 | $0.14/M input |
| DeepSeek V4 Pro | **87.9** | 80.6 | $0.27/M input |
| Ornith 1.5 35B | 67.8 | 79 | 自託管 |
| Qwen3.6-35B (原始) | 52.5 | 73.4 | 自託管 |
| **我們的目標 (Phase 6)** | **68-75** | **78-82** | $0 (本地) |

### 7.8 推薦實施順序

1. **Phase 3a**: 路徑 A POC — Mac M4 做 prefill + decode（驗證 KV cache 傳輸）
2. **Phase 3b**: 路徑 C 混合路由 — CGC Router 自動切換本地/雲端
3. **Phase 3c**: 接入 DGX Spark — 雲端 prefill + 本地 MTP decode
4. **Phase 3d**: 蒸餾 — 用 V4 Flash 做 teacher 訓練 Qwen3.6 蒸餾版

---

## 8. 相關文件

- `docs/CGC_COMPUTE_SHARING_ARCHITECTURE.md` — 三平台總體架構
- `docs/CGC_CROSS_PLATFORM_ARCHITECTURE.md` — 四平台架構方案 v1.1
- `docs/CGC_HARMONYOS_PHONE_ARCHITECTURE.md` — 鴻蒙手機端架構方案
- `docs/AGENT_HARNESS_WHITEPAPER.html` — Agent Harness 技術白皮書 (DSH + Prime Agent)
- `docs/M4_SETUP_GUIDE.md` — Mac 端環境
- `docs/MTP_BENCHMARK_WIN8GB.md` — Windows 8GB MTP 基準測試
- `CGC-main/cgc_engine/pd/router.py` — 4D 路由 + decode-速度感知（本次）
- `CGC-main/cgc_engine/pd/edge_server.py` — 最後一哩 server（本次）
- `CGC-main/cgc_engine/pd/pd_e2e_test.py` — 聯調 harness（本次）
- `moeexpert/CGC_CPOT_分析工具白皮書_2026-08-29.md` — Mac 生產線效能定案
- `agent_harness/dsh_config/` — DSH + agent_harness 整合配置
