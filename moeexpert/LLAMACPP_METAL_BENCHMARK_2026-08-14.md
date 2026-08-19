# llama.cpp + Metal Benchmark（M4 16GB）— 2026-08-14

## 背景

C 框架（cgc_engine）的 compute 盤點結果：
- **compute 存在但 Windows/CUDA-only**：kernel 庫（gemm_cpu/attention/linear/norm/quant/rope/sampling + KDA）+ DeepEP/grouped_gemm（cloud_sglang CUDA 路徑）
- **CGC 的 Metal backend 幾乎是空的**：`cgc_metal_backend.mm` 只有 KDA attention shader（126 行），`src/kernels/metal/` 空目錄，**沒有 MoE GEMM Metal kernel**
- **PD scheduler 是雙 GPU 假設** → 本機單 GPU 不需要（已確認）
- C streamer（GGUF per-expert pread + cache + segment-aware）已在 macOS 編譯跑通（本日修完 4 個 parser bug + 3 段式佈局）

**macOS 定案**：compute 交給 llama.cpp + Metal（它有 qwen35moe 全套 kernel：GatedAttn/DeltaNet/MoE），C streamer 只負責 expert IO；單 GPU 不 PD 分離。

## 硬體事實

- M4 16GB，`recommendedMaxWorkingSetSize = 11453.25 MB`（GPU 工作集上限）
- 內部 SSD（models/gguf/），llama.cpp build `679da73`
- 系統空閒 free ≈ 11.2GB；IQ2 全 offload 時 free pages → 30MB（≈0，零餘裕但穩定）

## Benchmark 結果（llama-bench build 679da73，-p 128 -n 128，r=1）

### IQ2_XXS（10.76GB 檔 / 10.01 GiB / 2.0625 bpw）

| 設定 | pp128 (t/s) | tg128 (tok/s) | 記憶體 |
|---|---|---|---|
| CPU-only（-ngl 0） | 12.66 | 12.87 | mmap ~10GB RSS |
| auto-fit（-fitt 512） | 168.5 | 26.98 | ~10.9GB GPU WS |
| **全 offload（-ngl 99）** | **170.9-175.5** | **25.4-27.8** | ~10GB GPU WS，free≈0 |
| 專家拆分（-ngl 99 -ncmoe 128） | 65.5 | 6.47 | ~5GB GPU + ~4.5GB CPU experts |
| （參考）我們 Swift 引擎 | ~27s TTFT | 5.5-5.8 | ~3GB resident |

### IQ3_XXS（13.2GB 檔 / 12.29 GiB / 3.0625 bpw）

| 設定 | pp (t/s) | tg (tok/s) | 記憶體 |
|---|---|---|---|
| CPU-only（-ngl 0） | 5.61 (pp64) | 9.69 (tg64) | mmap ~12.3GB RSS |
| 任何 Metal offload（-ngl 99 / -fitt） | ❌ res=-3 | ❌ | 12.29GiB > 11.45GB WS |

## 關鍵發現

1. **llama.cpp + Metal 全 offload IQ2 = 25-28 tok/s，本機可行且不 OOM**（之前 16GB OOM 是疊 DFlash draft 的情況，純 IQ2 剛好塞進 11.45GB）。比我們 Swift 引擎快 **4.5×**。
2. **`-ncmoe`（llama.cpp 原生專家拆分 = streaming 方向的內建版）只有 6.47 tok/s**——CPU 側 expert GEMM 是瓶頸。任何「experts 放 CPU、其餘 Metal」的架構都會撞同一堵牆。
3. **IQ3 全 offload 失敗（12.29 > 11.45）**——這才是 streaming 真正該解的場景（更高品質 + 放不下）。llama.cpp CPU fallback 9.69 tok/s 仍比我們的引擎快。

## 記憶體真相（2026-08-14 補充）：11.45GB 工作集確實激進，且 margin 救不了

### 實測（memory_pressure free% 正確指標）

| 設定 | tg128 | free% 最低 | 結論 |
|---|---|---|---|
| -ngl 99 全 offload | 24-28 | **9%** | 系統極度吃緊 |
| -fitt 1024（1GB 餘裕） | 22.8 | — | 餘裕是假的 |
| -fitt 2048（2GB 餘裕） | 22.8 | — | 餘裕是假的 |
| -fitt 4096（4GB 餘裕） | 16.8-20.1 | **9%** | 餘裕是假的 |
| -ngl 99 -lm dio | 24.6 | **9%** | 繞過 page cache 也一樣 |

### 機制
- Metal 開 `use shared buffers`（零拷貝共享）：offload 的權重就是 mmap 頁本身，**整個 10GB 檔案頁都是 resident**，不隨 offload 比例縮減
- `-fitt` margin 只限 GPU 工作集分配，不釋放檔案頁；`-lm dio` 只改載入路徑
- **10GB 模型在 16GB 機 = 無論如何佔 ~10GB**，只剩 ~6GB 給 OS + 其他 app → 同時開瀏覽器/IDE 必 compression/swap，實測 tok/s 會遠低於 headless 數字

### 結論
- benchmark 數字全是 **headless 前提**，不能當「開著其他 app」的生產數字
- 真正要「模型 + 其他 app 共存」只有兩條路：
  1. **更小模型**（如 MLX 6.18GB → 留 ~10GB 餘裕）
  2. **expert streaming（bounded ~3GB resident）**——唯一能在 16GB 上跑 10GB 模型且保有真實餘裕的方式；**這加強了 C streamer 的價值主張**（不是「放不下」，是「放得下但要留餘裕」）
- streaming 前提不變：experts 的 GEMM 必須上 Metal（-ncmoe CPU 陷阱 = 6.47 tok/s）

## 對 C 引擎架構的定案

- **結論：最划算的路不是把 Swift 引擎移植成 C，而是直接用 llama.cpp 全 offload（25-28 tok/s）**——C streamer 只有在「模型 > 11.45GB 工作集」時才有價值（IQ3+）。
- 若 C 引擎要跑 streaming：experts 的 GEMM 必須上 **Metal**（不能像 -ncmoe 放 CPU），否則就是 6.47 tok/s 的陷阱。這需要自寫 MoE Metal kernel（CGC 的 `kernels/metal/` 目前是空的）——正是我們 Swift 引擎做過的事，但我們的 kernel 比 llama.cpp 慢 4.5×，所以自寫 kernel 的 ROI 很低。
- **優先建議**：streaming 線先擱置；直接以 llama.cpp -ngl 99 全 offload IQ2（25-28 tok/s）為生產基線。IQ3 若要跑，只有 CPU-only 9.69 tok/s 一途，或等 M5/更大記憶體機器。
