# Qwen3.8-27B Dense→MoE 轉換 + CGC Engine 整合白皮書 v1.1

**日期**: 2026-08-28  
**分支**: dev  
**目標**: 將 Qwen3.8-27B dense 模型轉換為 MoE 格式，並用 CGC engine expert-cache 在 M4 + 鴻蒙 MateBook 32GB 上高效運行

### 變更記錄

| 版本 | 日期 | 內容 |
|------|------|------|
| v1.0 | 2026-08-28 | 初版：Dense→MoE 轉換 + CGC 整合 |
| v1.1 | 2026-08-28 | +§9 AIOS 鴻蒙部署方案，更新 §4.3 分析，加入32GB MateBook 支援 |

---

## 1. 專案概覽

### 1.1 為什麼要做 Dense→MoE

| 指標 | Qwen3.8-27B Dense | Whittle-MoE (64 experts) | 改善 |
|------|-------------------|--------------------------|------|
| 總參數 | 27B | 27B (相同) | — |
| 激活參數/token | 27B | 17.8B | **-34%** |
| 理論 decode 速度 | baseline | **+50%** (更少權重讀取) | ✅ |
| 知識容量 | 27B | 27B (64×FFN) | 相同 |
| 訓練成本 | — | 只訓 router (~20MB) | 極低 |

**核心洞察**: MoE 不增加總參數，但每個 token 只讀 17.8B 權重（vs dense 27B），decode 速度理論提升 50%。

### 1.2 為什麼用 CGC Engine

CGC engine 的 expert-cache 專為 MoE 設計：
- **Skip-load**: FFN expert 權重不在 Metal allocation，按需從磁碟載入
- **Bounded pool**: 固定記憶體預算（4GB），只保留熱門 experts
- **Callback routing**: 每層 argsort 後動態選擇 experts → pool slot mapping
- **L4 zero-copy**: Pool region 直接映射到 Metal buffer，無 CPU↔GPU copy

### 1.3 16GB M4 硬體約束

```
16GB unified memory
├── 系統/OS: ~2GB
├── Model (attention + embeddings): ~4GB (Q4 quant)
├── Expert cache (4GB budget): ~4GB
├── KV cache: ~2GB
└── 剩餘: ~4GB (安全緩衝)
```

---

## 2. 架構分析

### 2.1 Qwen3.8-27B Dense 架構

```
Qwen3.8-27B:
  hidden_size: 5120
  intermediate_size: 17408
  num_layers: 48
  num_heads: 40
  head_dim: 128
  vocab_size: 151936
  
  Layer 結構 (每4層一組):
    ├── GDN (Gated DeltaNet) × 3 層  ← 線性 attention，O(1) KV cache
    └── QSA (Qwen Sparse Attention) × 1 層  ← sparse attention
  
  每層 FFN:
    gate_proj: [5120, 17408]  (gate)
    up_proj:   [5120, 17408]  (up)
    down_proj: [17408, 5120]  (down)
    總計: 5120 × 17408 × 3 = 268.4M params/layer
```

### 2.2 MoE 轉換後架構

```
Qwen3.8-27B MoE (64 experts):
  每層 FFN → 64 個專家:
    expert_0.gate_proj: [5120, 17408]
    expert_0.up_proj:   [5120, 17408]
    expert_0.down_proj: [17408, 5120]
    ...
    expert_63 (同上)
    
  Router (每層):
    gate: Linear(5120, 64)  ← 20KB/layer, 總計 ~1MB
  
  激活: top-k (動態，通常 k=8-10)
  每 token 讀取: 17.8B 權重 (vs dense 27B)
```

### 2.3 參數量分解

| 組件 | Dense | MoE (64 experts) | 倍數 |
|------|-------|-------------------|------|
| Attention (48層) | 5.0B | 5.0B | 1× |
| FFN (48層) | 12.9B | **821.4B** | **64×** |
| Embeddings | 0.8B | 0.8B | 1× |
| Router | 0 | 0.001B | — |
| **總計** | **18.7B** | **827.2B** | 44× |

### 2.4 量化後大小

| Quant | Dense | MoE | CGC skip-load 後 GPU |
|-------|-------|-----|---------------------|
| Q4_K_M | 14.7GB | 465GB | ~4GB (attn+emb) |
| IQ3_S | 10.7GB | 337GB | ~3GB |
| IQ3_XXS | 9.2GB | 281GB | ~3GB |

**關鍵**: CGC skip-load 只需載入 attention + embeddings (~4GB) 到 GPU，FFN experts 按需從磁碟載入。

---

## 3. 轉換流程

### 3.1 Stage 1: Dense→MoE Upcycling (⭐⭐ 簡單)

**目標**: 把 dense FFN 複製為 64 個專家，初始化 router

**步驟**:
```
1. 載入 Qwen3.8-27B dense 權重 (safetensors)
2. 對每一層 (layer 0-47):
   a. 複製 gate_proj → expert_0.gate_proj ... expert_63.gate_proj
   b. 複製 up_proj → expert_0.up_proj ... expert_63.up_proj  
   c. 複製 down_proj → expert_0.down_proj ... expert_63.down_proj
   d. 初始化 router: nn.Linear(5120, 64) (Xavier init)
   e. 設定 router bias = 0 (均匀分配)
3. 輸出: Qwen3.8-27B-MoE-64E safetensors
```

**關鍵細節**:
- Router 初始化用 Xavier uniform，不是隨機
- 所有 experts 初始完全相同 → router 無法區分 → 必須 Stage 2 訓練
- Scale factor: sigmoid(0.5) × topk_norm = ~0.62，防止 MoE 輸出過大

**工具**: 參考 `smolMoELM-custom/moe/upcycle.py`

### 3.2 Stage 2: Router-Healing 訓練 (⭐⭐ 中等)

**目標**: 訓練 router 均勻分配 experts，避免 dead experts

**方法**: DenseMixer (STE - Straight-Through Estimator)

```
損失函數:
  L = L_task + α × L_balance + β × L_z_loss
  
  L_task:    task loss (next token prediction)
  L_balance: load-balancing loss (強制均匀分配)
  L_z_loss:  router logits L2 正則化 (防止 logits 爆炸)
```

**超參數**:
| 參數 | 值 | 說明 |
|------|-----|------|
| 學習率 | 1e-4 | router 專用 |
| Batch size | 8-16 | M4 16GB 限制 |
| Steps | 5000-10000 | ~2-4 小時 |
| α (balance) | 0.01 | load-balancing 係數 |
| β (z_loss) | 0.001 | logit 正則化 |
| Experts 凍結 | ✅ | 只訓 router |

**數據**: 10K 通用指令 prompt（Alpaca/ShareGPT），不需要標註

**驗證**:
- Dead expert ratio < 5% (每個 expert 至少被激活 1% 的 tokens)
- Load balance std < 0.05
- Task loss 下降（不因為 router 訓練而降低輸出品質）

### 3.3 Stage 3: Anti-Loop Answer Distillation (⭐⭐⭐⭐ 困難)

**目標**: 消除 MoE 的循環/復讀問題（初代 69% 循環率 → 目標 <5%）

**核心洞察**: MoE 循環 = router 陷入局部最优，反覆選同一 expert → 相同權重 → 相同輸出

**三部分 KD 損失**:
```
L_KD = λ₁ × L_binary_KL + λ₂ × L_conditional_KL + λ₃ × L_CE

L_binary_KL:     每 token 的 binary KL divergence
                 D_KL(student || teacher) for each position
                 
L_conditional_KL: 條件 KL (考慮前文)
                  D_KL(p(x_t|x_{<t}, student) || p(x_t|x_{<t}, teacher))
                  
L_CE:            交叉熵 (teacher 作為 ground truth)
                 -log p_teacher(x_t)
```

**EOS Masking (關鍵)**:
```python
# 錯誤: 全 batch 統一 mask
mask = (tokens != eos_id)  # 會漏掉循環中的 EOS

# 正確: 逐行 mask
for i in range(batch_size):
    eos_positions = (tokens[i] == eos_id).nonzero()
    if len(eos_positions) > 0:
        first_eos = eos_positions[0].item()
        mask[i, first_eos+1:] = 0  # EOS 後不算 loss
```

**Windowed Teacher Pass**:
```
不要一次餵整個 prompt 給 teacher:
  錯: teacher(full_prompt) → 一個大 KV cache
  對: teacher(window_1) → teacher(window_2) → ... 
      每個 window 獨立生成，避免 teacher 自身的長文循環
```

**數據**: 10K-50K prompt + dense 模型生成的 response（teacher output）

**工具**: 自行實現（無現成框架），參考 Tulu 3 / dolomite 的蒸餾代碼

### 3.4 Stage 4: 評估與調優

**Benchmark 對標**:

| 指標 | Dense 基線 | MoE 目標 | 測試工具 |
|------|-----------|---------|---------|
| MMLU | ~75% | ≥73% | lm-eval-harness |
| HumanEval | ~65% | ≥60% | pass@1 |
| GSM8K | ~80% | ≥78% | 8-shot |
| 循環率 | <1% | <5% | 自定義 100 條長文本 |
| 推理速度 | baseline | **+30-50%** | tok/s |

---

## 4. CGC Engine 整合

### 4.1 Expert-Cache 適配

```
Qwen3.8-27B MoE + CGC Expert-Cache:
  n_expert: 64 (vs Qwen3.6 的 256)
  n_expert_used: 8-10 (top-k)
  pool capacity: 取決於 budget
  
  Budget 4GB:
    expert_size = 17408 × 5120 × 3 × 0.5625 = 150MB/layer
    slots = 4GB / (150MB × 48 layers) = 0.55 → 0 slots ❌
    
  Budget 8GB:
    slots = 8GB / (150MB × 48) = 1.1 → 1 slot/layer ❌
    
  問題: 150MB/expert 太大，pool 放不下幾個
```

### 4.2 記憶體優化方案

**方案 A: 減少 expert 數量**
```
64 experts → 16 experts (top-4 routing)
  expert_size = 150MB (不變)
  但每層只需 16 個 slot → pool 更小
  
  Budget 4GB: 4GB / (150MB × 48) = 0.55 → 還是不夠
```

**方案 B: 更小量化 (IQ2)**
```
IQ2 量化: expert_size = 150MB × (0.3125/0.5625) = 83MB/layer
  Budget 4GB: 4GB / (83MB × 48) = 1.0 → 1 slot/layer ❌
```

**方案 C: 凍結 attention 到 CPU ( aggressive skip-load)**
```
Attention 也 skip-load → GPU 只保留 embeddings
  GPU: 0.8GB (embeddings only)
  Cache: 14GB - 0.8GB - 2GB = 11.2GB
  Slots: 11.2GB / (150MB × 48) = 1.5 → 1 slot/layer ❌
```

**方案 D: 不用 CGC，直接 mmap (最簡單)**
```
Model 12GB (IQ3_S) → mmap → OS page cache 管理
  每 token 讀 17.8B × 0.4 bytes = 7.1GB
  M4 bandwidth: 120 GB/s
  理論 max: 120 / 7.1 = 16.9 t/s
  
  實際: ~10-15 t/s (受 mmap page fault 影響)
```

### 4.3 建議方案: 按記憶體選擇

**16GB 機器（M4 基礎款）**:
```
150MB/expert × 64 experts × 48 layers = 460 GB total FFN
→ 無法全載，必須 skip-load + mmap 混合
→ 分層策略: 前 12 層 CGC skip-load，後 36 層 mmap
```

**32GB 機器（M4 Max / 鴻蒙 MateBook）**:
```
12.7GB 模型 + 4GB cache + 4GB 系統 = 20.7GB < 32GB ✅
→ 記憶體充裕，不需要 aggressive skip-load
→ 直接 CGC expert-cache 4GB budget，全模型 mmap + cache
→ CPU decode: ~0.5-1 t/s（活躍參數量的物理限制）
```

**結論**: 32GB 機器上 expert-cache 主要價值是**加速 cold expert 首次載入**（hit rate 60-80%），而不是省記憶體。速度瓶頸是 17.8B active params 的 bandwidth 需求，不是 cache。

---

## 5. M4 實戰配置

### 5.1 硬體需求

| 配置 | 適用場景 | 預估速度 |
|------|---------|---------|
| M4 16GB | Stage 1-2 訓練 | 5-10 t/s |
| M4 Pro 24GB | Stage 1-3 全流程 | 10-15 t/s |
| M4 Max 32GB | 完整訓練 + 評估 | 15-20 t/s |

### 5.2 環境搭建

```bash
# 1. 安裝依賴
pip install torch transformers accelerate
pip install densemixer  # Router 訓練
pip install mlx mlx-lm  # M4 優化
pip install safetensors  # 權重格式

# 2. 下載 dense 權重
huggingface-cli download Qwen/Qwen3.8-27B --local-dir models/qwen3.8-27b-dense

# 3. 下載 GGUF 量化版 (用於評估)
hf download huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF \
  Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf \
  --local-dir models/gguf/
```

### 5.3 訓練流程

```
Day 1-2: Stage 1 Upcycling
  python upcycle.py --input models/qwen3.8-27b-dense \
                     --output models/qwen3.8-27b-moe-64e \
                     --num_experts 64

Day 3-5: Stage 2 Router Training  
  python train_router.py --model models/qwen3.8-27b-moe-64e \
                         --data alpaca_10k.json \
                         --steps 10000 \
                         --lr 1e-4

Day 6-12: Stage 3 Anti-Loop KD
  python anti_loop_kd.py --teacher models/qwen3.8-27b-dense \
                         --student models/qwen3.8-27b-moe-64e \
                         --data teacher_outputs_50k.json \
                         --epochs 3

Day 13-14: Stage 4 Evaluation
  python evaluate.py --model models/qwen3.8-27b-moe-64e-v2.1 \
                     --benchmarks mmlu,humaneval,gsm8k \
                     --loop_test 100_prompts.json
```

---

## 6. 風險與限制

### 6.1 技術風險

| 風險 | 概率 | 影響 | 緩解 |
|------|------|------|------|
| 循環率壓不下去 | 高 | 模型不可用 | Stage 3 必須做到位 |
| M4 訓練 OOM | 中 | 訓練中斷 | 用 gradient checkpointing |
| MoE 輸出品質差 | 中 | 低於 dense 基線 | 調 α/β 參數 |
| Router 收斂慢 | 低 | 訓練時間長 | 用 warmup + cosine LR |

### 6.2 已知限制

1. **Answer Distillation 未開源**: logic65 的 v2.1 核心創新無公開代碼，需自行實現
2. **M4 訓練速度慢**: 27B 模型在 M4 上訓練比 GPU 慢 10-50x
3. **評估不完整**: 沒有標準 MoE benchmark，只能用 dense 基線對標
4. **8% 循環率**: 即使做到 v2.1 水平，仍有 8% 循環（dense <1%）

### 6.3 成功標準

- ✅ MoE 模型能在 CGC engine 上跑通（無 crash）
- ✅ 推理速度 > dense 30%+
- ✅ 循環率 < 5%
- ✅ 下游 benchmark ≥ dense 95%

---

## 7. 時程規劃

| 週 | 任務 | 交付 |
|----|------|------|
| W1 | Stage 1 Upcycling + CGC 適配 | MoE 模型 + 基本跑通 |
| W2 | Stage 2 Router Training | Router 訓練完成 |
| W3-4 | Stage 3 Anti-Loop KD | v2.1 模型 |
| W5 | Stage 4 Evaluation + 優化 | Benchmark 報告 |

**總預估**: 5 週（M4 Max 32GB）/ 8 週（M4 16GB）

---

## 8. AIOS 鴻蒙 MateBook 14 部署方案

### 9.1 目標硬體

| 項目 | 規格 |
|------|------|
| 設備 | 鴻蒙 MateBook 14 (G4AU042K7) |
| SoC | 麒麟9030 + Maleoon 935 (UMA) |
| RAM | **32 GB** unified memory |
| OS | HarmonyOS NEXT (Linux kernel, aarch64) |
| CPU | ARMv8.2-A, NEON + SVE |

### 9.2 為什麼 32GB 改變了一切

16GB 機器上 MoE 的瓶頸是**記憶體不夠**（12.7GB 模型 + 4GB cache + 系統 = OOM）。32GB 機器上記憶體充裕，但**活躍參數量仍是物理限制**：

```
16GB: 裝不下 → OOM（根本跑不了）
32GB: 裝得下 → 但 17.8B active = 6 倍計算量 → 速度仍是 Qwen3.6 A3B 的 1/6
```

| 指標 | Qwen3.6 A3B (3B active) | MoE (17.8B active) |
|------|------------------------|-------------------|
| 32GB 記憶體 | ✅ 充裕 | ✅ 充裕 |
| Expert cache 4GB | ✅ 容易 fit | ✅ fit |
| CPU decode 預估 | **~3-5 t/s** | **~0.5-1 t/s** |
| Prefill (1K) | ~1-2 t/s | ~0.2-0.5 t/s |
| Quality | Good | **Excellent** |

### 9.3 部署包結構

```
AIOS/harmonyos/
├── build.sh          # 麒麟9030 CPU-only NEON build
├── run.sh            # 雙模型切換 + expert-cache
├── benchmark.sh      # Thread sweep 對比
└── README.md         # 完整部署指南
```

### 9.4 Build Flags

```
Metal=OFF, Vulkan=OFF, OpenCL=OFF
BLAS=OFF (MUST — IQ3 garbled output)
Accelerate=OFF
CPU_REPACK=OFF (MUST — IQ3 tensor boundary)
OpenMP=OFF
MTP_SUPPORT=ON (CGC expert-cache + MTP)
Arch: -march=armv8.2-a -mtune=cortex-a720 (Kirin 9030 NEON/SVE)
```

### 9.5 使用方式

```bash
# Build
./build.sh /path/to/llama.cpp-source

# Qwen3.6 A3B（速度優先）
./run.sh -m qwen36 -n 128 -p "The capital of France is"

# MoE（品質優先）
./run.sh -m moe -n 128 -p "The capital of France is"

# Benchmark 雙模型
./benchmark.sh
```

### 9.6 Expert-Cache 配置

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `LLAMA_EXPERT_CACHE_ENABLE` | 1 | 啟用 expert cache |
| `LLAMA_EXPERT_CACHE_BUDGET` | 4GB | Pool 預算 |
| `LLAMA_EXPERT_CACHE_WORKERS` | 8 | I/O workers |
| `LLAMA_EXPERT_CACHE_ALLOW_NGL` | 1 | 允許 GPU layers + cache |
| `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0` | 1 | Skip layer 0 in pool |

### 9.7 記憶體預算（32 GB 機器）

| Config | Model | Cache | System | Free |
|--------|-------|-------|--------|------|
| Qwen3.6 A3B | 12 GB | 4 GB | 4 GB | 12 GB ✅ |
| MoE Q3_K_S | 12.7 GB | 4 GB | 4 GB | 11.3 GB ✅ |

### 9.8 選擇建議

| 場景 | 推薦 | 原因 |
|------|------|------|
| 速度優先 | **Qwen3.6 A3B** | 3B active, 3-5 t/s |
| 品質優先 | **MoE** | 17.8B active, dense-quality |
| 雙模型 | **兩者都裝** | 用戶自選 |

---

---

## 9. 附錄

### 8.1 參考資源

- **DenseMixer**: github.com/yaof20/DenseMixer (Router 訓練)
- **smolMoELM-custom**: github.com/pranavktrpl/smolMoELM-custom (Upcycling)
- **Whittle-MoE**: huggingface.co/logic65/Qwen3.8-Whittle-MoE-27B-A17.8B
- **CGC Engine**: 本 repo expert-cache fork

### 8.2 關鍵文件路徑

```
src/llama.cpp/src/llama-context.cpp    # expert_cache_on_topk callback
src/llama.cpp/src/llama-expert-cache.cpp  # pool management
src/llama.cpp/src/llama-graph.cpp       # build_moe_ffn (remap leaf)
src/llama.cpp/ggml/src/ggml-backend.cpp  # segmented dispatch
scripts/run_n30cache.sh                 # benchmark harness
AIOS/harmonyos/                         # 鴻蒙 MateBook 部署包
```

---

*白皮書由 CGC Engine Team 編撰，最後更新：2026-08-28*
