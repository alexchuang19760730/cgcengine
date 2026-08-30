# C 引擎「bounded 3GB + Metal MoE compute」可行性評估（2026-08-14）

設計前提：16GB M4、stream llama.cpp GGUF（IQ2/IQ3）、bounded ~3GB resident、experts 的 GEMM 在 Metal 上算（不放 CPU）、單 GPU 不 PD。

## 一、記憶體：✅ 達成（這是設計唯一確定成立的軸）

| 項目 | 數值 |
|---|---|
| 3GB pool 可容納 slots | IQ2 = 3674 / IQ3 = 3096（全模型 10240） |
| 工作集（8 slot/層 × 40 層） | ~320 slots ≈ 268-317MB |
| hot pool 餘量 | ~3300 slots（top-frequency profile，對照 Swift 的 pool 96-112 就達 72-85% hit） |
| 有效 fill（85% hit） | IQ2 ≈ 42MB/step、IQ3 ≈ 51MB/step ≈ **10-13ms/step**（@4GB/s） |

- Swift 引擎實測 2.5-4GB resident 已證明這個量級可行（r3q4 縮到 2.5GB 後仍有高 hit）
- 3GB resident + OS 4-5GB + 其他 app 3-5GB = 10-13GB < 16GB → **真餘裕**（對照 llama.cpp 全 offload free% 9%）
- 10-13ms 的 fill 可藏在 GPU 後（Swift 的 fill 已實證藏住）→ 不疊加到 step 時間

## 二、品質：⚠️ 只有 IQ3_XXS「不減」

| 檔位 | bpw | 對照現行生產（r3q4la_ga_e4h3 ≈ 3.2-3.5bpw） |
|---|---|---|
| IQ2_XXS | 2.06 | **降級** |
| IQ3_XXS | 3.06 | ≈ 等價（品質不減） |

關鍵：**IQ3_XXS（13.2GB）恰是 llama.cpp 在 16GB 跑不動的檔位**（Metal OOM res=-3、CPU-only 9.69 且 12.3GB resident 無餘裕）。所以「品質不減 + 多工共存」的組合只有 streaming 做得到 → 這是設計的獨特價值，但**必須以 IQ3_XXS 為目標模型，不是 IQ2**。

## 三、速度：⚠️ 天花板由 Metal kernel 品質決定，不是 streaming

兩個已實測錨點：
- **Swift 級 kernel**（我們自己寫的）：GPU ~92ms/step → 5.5-5.8 tok/s
- **llama.cpp 級 kernel**（成熟調校）：GPU ~39ms/step → 25-28 tok/s（全 offload）

streaming 只動記憶體軸，不動 compute 軸。分兩情境估：

| 情境 | GPU/step 推估 | fill 重疊 | 預估 tok/s |
|---|---|---|---|
| C+Metal streaming IQ3，Swift 級 kernel | ~95-115ms（IQ3 張量比 r3 大：dense 是 Q5_K/Q6_K） | 藏住（10-13ms） | **6-9** |
| C+Metal streaming IQ2，Swift 級 kernel | ~85-100ms | 藏住 | **7-10** |
| C+Metal streaming IQ3，llama.cpp 級 kernel | ~45-55ms | 部分重疊 | **15-20** |
| C+Metal streaming IQ2，llama.cpp 級 kernel | ~40-50ms | 部分重疊 | **18-24** |
| （對照）llama.cpp 全 offload IQ2 | 39ms | — | 25-28（無餘裕） |
| （對照）llama.cpp -ncmoe | — | — | 6.47（CPU experts 陷阱） |

## 四、結論

1. **可行性：記憶體 ✅ / 品質 ✅（限 IQ3_XXS）/ 速度 ⚠️ 取決於 kernel**
2. **誠實預估**：以我們自己的 kernel 工程實績（Swift 5.5-5.8）為基準，C 引擎自寫 Metal kernel 的合理預期是 **6-10 tok/s**——比全 offload 慢但**保有真餘裕**，且是 16GB 上跑 IQ3 品質的唯一路徑
3. **要突破 10 直上 15-24，唯一辦法是拿到 llama.cpp 級 kernel 品質**——而這不是「C 引擎重寫 kernel」，是 **fork llama.cpp 讓 expert 張量走 bounded residency**（複用其全部 Metal kernel，只加一層 per-expert 管理）。自寫 C+Metal 引擎只會重現 Swift 的 5.5-8 檔位，成本翻倍

## 五、建議路線（排序）

1. **不建自寫 C+Metal 引擎**（重現 Swift 檔位，ROI 低）
2. 若要「品質不減 + 多工共存」：評估 **fork llama.cpp 的 qwen35moe 加 expert 張量 bounded residency**——工作集中在 llama.cpp 的 model/graph 層，kernel 全複用；預估 15-20 tok/s（IQ3）＋真餘裕
3. 短期務實替代：llama.cpp 全 offload IQ2（25-28）**關掉其他 app 用**；需要共存時退回 MLX 6.18GB 或 Swift 引擎
