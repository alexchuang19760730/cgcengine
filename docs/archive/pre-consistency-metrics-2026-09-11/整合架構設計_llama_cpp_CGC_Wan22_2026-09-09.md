# 整合架構設計：llama.cpp + CGC Expert Engine + MTP + Wan 2.2 視頻生成

> 版本：v1.0  
> 日期：2026-09-09  
> 目標：在統一項目架構下整合 LLM 推理與視頻生成，共享技術積累與資源管理

---

## 一、整合目標與原則

### 1.1 整合目標

1. **統一項目結構**：LLM 推理與視頻生成在同一個項目下，共享模型管理、腳本、文檔
2. **技術复用**：CGC Expert Cache 的核心思想復用到視頻擴散模型的 MoE
3. **統一啟動管理**：一個入口管理 LLM 服務和視頻生成服務
4. **資源隔離**：兩個系統互不幹擾，可獨立啟動/停止
5. **漸進式整合**：先共存，再逐步深度整合

### 1.2 整合原則

- **不重複造輪子**：視頻生成用成熟的 ComfyUI，不自研擴散模型推理引擎
- **技術思想复用**：Expert Cache、內存管理、性能優化的思路復用，但實現獨立
- **接口標準化**：兩個系統通過標準 API（HTTP）交互，不直接耦合
- **配置統一**：統一的配置文件格式和環境變量管理

---

## 二、統一目錄結構

```
flashkv-devserver/
├── models/                          # 統一模型目錄
│   ├── gguf/                        # LLM 模型（Qwen 3.6 等）
│   │   ├── Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
│   │   ├── Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
│   │   └── Wan2.2-TI2V-5B-Q4_K_M.gguf    # 擴散模型也放這裡（統一管理）
│   ├── vae/                         # VAE 模型
│   │   └── Wan2.1_VAE.safetensors
│   ├── text_encoder/                # 文本編碼器
│   │   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors
│   └── lora/                        # LoRA 模型（LLM 和視頻通用）
│
├── src/
│   └── llama.cpp/                   # 原有 llama.cpp + CGC Expert Engine + MTP
│       ├── src/                     # C++ 源碼
│       ├── build/                   # 編譯產物
│       └── examples/                # 示例
│
├── video-gen/                       # 視頻生成模塊（新增）
│   ├── ComfyUI/                     # ComfyUI 安裝目錄
│   │   ├── main.py
│   │   ├── custom_nodes/            # 自定義節點（含 ComfyUI-GGUF）
│   │   ├── models/                  # 模型目錄（軟鏈接到統一 models/）
│   │   │   ├── diffusion_models -> ../../models/gguf
│   │   │   ├── vae -> ../../models/vae
│   │   │   └── text_encoders -> ../../models/text_encoder
│   │   └── output/                  # 生成輸出
│   ├── workflows/                   # 自定義工作流
│   │   ├── wan22_5b_q4_basic.json
│   │   └── wan22_5b_q4_optimized.json
│   └── scripts/                     # 視頻生成腳本
│       ├── start_comfyui.sh
│       ├── test_generation.py
│       └── batch_generate.py
│
├── scripts/                         # 統一腳本目錄
│   ├── start_llm_server.sh          # 原有：啟動 LLM 服務
│   ├── start_video_gen.sh           # 新增：啟動視頻生成服務
│   ├── start_all.sh                 # 新增：同時啟動兩個服務
│   ├── stop_all.sh                  # 新增：停止所有服務
│   └── status.sh                    # 新增：查看所有服務狀態
│
├── cgc-expert-cache/                # CGC Expert Engine 核心（可被兩邊复用）
│   ├── core/                        # 核心算法（slot table、remap、LRU）
│   ├── llm-adapter/                 # LLM 適配器（llama.cpp 集成）
│   └── diffusion-adapter/           # 擴散模型適配器（ComfyUI 自定義節點）
│
├── docs/                            # 統一文檔
│   ├── ExpertCache_關鍵版本技術白皮書.html
│   ├── Wan2.2_5B_Q4_端側視頻生成開發計劃.md
│   └── 整合架構設計_llama_cpp_CGC_Wan22.md  # 本文檔
│
└── config/                          # 統一配置
    ├── llm_server.yaml              # LLM 服務配置
    ├── video_gen.yaml               # 視頻生成配置
    └── shared.yaml                  # 共享配置（模型路徑、內存限制等）
```

---

## 三、兩個系統的職責邊界

### 3.1 LLM 系統（llama.cpp + CGC + MTP）

**職責**：
- 文本生成、代碼生成、推理問答
- Expert Cache 優化 MoE LLM 推理
- MTP（Multi-Token Prediction）投機解碼加速
- 提供 OpenAI 兼容 API

**技術棧**：
- C++ / llama.cpp
- Metal GPU 加速（Mac）
- 自研 Expert Cache slot table
- 自研 MTP draft 模型

**端口**：8080（默認）

### 3.2 視頻生成系統（ComfyUI + Wan 2.2）

**職責**：
- 文本生成視頻（T2V）
- 圖像生成視頻（I2V）
- 擴散模型去噪推理
- MoE Expert 管理（後續可復用 CGC 思路）

**技術棧**：
- Python / PyTorch
- ComfyUI 工作流引擎
- MPS / CUDA GPU 加速
- GGUF 量化模型加載

**端口**：8188（默認）

### 3.3 交互方式

兩個系統通過 **HTTP API** 交互，互不直接調用：

```
用戶請求
    │
    ├─→ LLM 服務 (:8080) ──→ 文本/代碼生成
    │
    └─→ 視頻生成服務 (:8188) ──→ 視頻生成
         ↑
         │ 可選：調用 LLM 服務優化 Prompt
         └──────────────────────────┘
```

**可選的深度整合**：視頻生成服務可以調用 LLM 服務來優化用戶輸入的 Prompt（把大白話轉成高質量視頻生成 Prompt）。

---

## 四、CGC Expert Engine 技術复用路徑

### 4.1 當前狀態

CGC Expert Engine 目前深度集成在 llama.cpp 中，用於優化 MoE LLM 的 expert 加載和推理。核心組件：
- Slot Table：expert 到 pool slot 的映射
- Remap ID：top-k 路由到 slot id 的轉換
- LRU Eviction：冷 expert 淘汰策略
- Prefetch：預測性加載下一輪可能用到的 expert

### 4.2 復用難點

Wan 2.2 的 MoE 與 LLM 的 MoE 有本質差異：

| 維度 | LLM MoE（Qwen） | Wan 2.2 MoE |
|------|----------------|-------------|
| 路由粒度 | token 級（每個 token 選 expert） | 時間步級（每個去噪步選 expert） |
| 路由變化 | 層間變化大，局部性中等（~60-70%） | 階段內幾乎不變，局部性極高（~95%+） |
| 推理框架 | llama.cpp（C++） | PyTorch / ComfyUI（Python） |
| Expert 加載 | 按需加載（token 路由動態） | 按階段加載（時間步分組） |

### 4.3 分階段復用策略

#### Phase 1：思想复用（當前階段）
- 把 Expert Cache 的**設計思路**用於優化 Wan 2.2 的 expert 加載
- 在 ComfyUI 工作流層面實現：預加載熱門 expert、分階段加載
- 不改動 CGC Engine 源碼，獨立實現

#### Phase 2：核心算法复用（中期）
- 把 Slot Table、LRU Eviction 等核心算法抽成獨立庫（`cgc-expert-cache/core/`）
- 用 Python 重寫核心算法，供 ComfyUI 自定義節點調用
- LLM 側繼續用 C++ 實現，兩側共享算法設計但獨立實現

#### Phase 3：深度整合（長期）
- 實現 `cgc-expert-cache/diffusion-adapter/`：ComfyUI 自定義節點
- 自動檢測 Wan 2.2 的 MoE 路由規律
- 動態管理 expert pool，優化內存使用
- 與 LLM 側共享配置和監控接口

### 4.4 MTP 技術的適用性分析

**MTP（Multi-Token Prediction）是自回歸模型的投機解碼技術**，通過一個小的 draft 模型預測多個 token，然後用大模型並行驗證，實現加速。

**對於擴散模型（Wan 2.2）**：
- ❌ MTP 不適用：擴散模型不是自回歸的，沒有"逐 token 生成"的概念
- ✅ 類似思想可借鑒：擴散模型有"少步蒸馏"技術（如 4-step、8-step 蒸馏），用小模型預測去噪軌跡
- 🔄 後續可考慮：集成 Wan 2.2 的 Lightning 蒸馏 LoRA，實現 4-8 步快速生成

---

## 五、統一服務管理

### 5.1 啟動腳本

**start_all.sh**：同時啟動 LLM 服務和視頻生成服務
```bash
#!/bin/bash
# 啟動 LLM 服務（後台）
./scripts/start_llm_server.sh &
LLM_PID=$!

# 啟動視頻生成服務（後台）
./scripts/start_video_gen.sh &
VIDEO_PID=$!

echo "LLM 服務 PID: $LLM_PID (端口 8080)"
echo "視頻生成服務 PID: $VIDEO_PID (端口 8188)"
echo "按 Ctrl+C 停止所有服務"

wait
```

**start_llm_server.sh**：原有 LLM 服務啟動腳本（保持不變）

**start_video_gen.sh**：新增視頻生成啟動腳本
```bash
#!/bin/bash
cd video-gen/ComfyUI
source venv/bin/activate
python main.py --listen 0.0.0.0 --port 8188
```

### 5.2 狀態監控

**status.sh**：查看所有服務狀態
```bash
#!/bin/bash
echo "=== 服務狀態 ==="
echo ""

# 檢查 LLM 服務
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ LLM 服務：運行中 (端口 8080)"
else
    echo "❌ LLM 服務：未運行"
fi

# 檢查視頻生成服務
if curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
    echo "✅ 視頻生成服務：運行中 (端口 8188)"
else
    echo "❌ 視頻生成服務：未運行"
fi

echo ""
echo "=== 內存使用 ==="
memory_pressure | head -5
```

---

## 六、整合路線圖

### Phase 0：共存（當前，第 1 週）
- [x] 統一目錄結構
- [x] ComfyUI 安裝在 `video-gen/`
- [x] 模型軟鏈接統一管理
- [ ] 統一啟動腳本
- [ ] 兩個服務可獨立運行，互不幹擾

### Phase 1：基礎整合（第 2-4 週）
- [ ] 統一配置文件
- [ ] 統一狀態監控腳本
- [ ] 視頻生成可選調用 LLM 服務優化 Prompt
- [ ] 統一日志格式和位置
- [ ] 文檔整合

### Phase 2：技術复用（第 5-10 週）
- [ ] CGC Expert Cache 核心算法抽象為獨立庫
- [ ] Python 版本核心算法實現
- [ ] ComfyUI 自定義節點實現 Expert Cache for Diffusion
- [ ] 性能對比基準（有/無 Expert Cache）

### Phase 3：深度整合（第 11-24 週）
- [ ] 統一 Web UI（一個界面管理 LLM 和視頻生成）
- [ ] 統一模型管理界面
- [ ] 統一監控和報警
- [ ] 自動化測試和 CI/CD
- [ ] 一鍵安裝包（Mac .app / Windows .exe）

---

## 七、風險與注意事項

### 7.1 內存資源競爭
- **問題**：LLM 服務和視頻生成服務同時運行時，16GB 內存可能不夠
- **解決**：
  - 預設不同時運行兩個大模型
  - 提供 `start_all.sh` 但明確警告內存風險
  - 配置文件中設置每個服務的內存上限
  - 自動檢測：如果內存不足，提示用戶停止另一個服務

### 7.2 端口衝突
- **問題**：兩個服務都可能想用 8080 或其他常用端口
- **解決**：
  - LLM 服務固定 8080
  - 視頻生成服務固定 8188
  - 配置文件中可修改，啟動時檢查端口是否被佔用

### 7.3 技術棧碎片化
- **問題**：C++（llama.cpp）+ Python（ComfyUI）+ Bash（腳本），維護成本高
- **解決**：
  - 明確每個技術棧的職責邊界
  - 核心算法用偽代碼/文檔統一描述，兩側獨立實現
  - 統一的 API 接口標準
  - 優先保證穩定性，不急於重構

### 7.4 MTP 不適用於擴散模型
- **問題**：用戶可能期望 MTP 也能加速視頻生成
- **解決**：
  - 文檔中明確說明 MTP 是自回歸模型技術，不適用於擴散模型
  - 提供替代方案：少步蒸馏 LoRA（4-step、8-step）
  - 長期可研究擴散模型的投機解碼技術

---

## 八、總結

本整合架構的核心思想是**"共存優先、思想复用、漸進整合"**：

1. **當前階段**：兩個系統獨立運行，共享目錄結構和模型管理
2. **中期階段**：CGC Expert Cache 的核心算法抽象復用，優化視頻生成的 MoE
3. **長期階段**：統一 UI、統一管理、一鍵安裝，做成消費級產品

**關鍵原則**：不自研擴散模型推理引擎（用成熟的 ComfyUI），不強行耦合兩個系統（通過標準 API 交互），不為了整合而整合（每一步整合都要有明確的價值）。

---

*文檔結束*
