# cgc_expert_streaming

MoE expert streaming 模組（原 `cgc_engine/cpp/expert_streaming`，2026-08-14 整理至此）。

## 結構

```
expert_streaming/
  include/    # 公開 header（C + C++ 相容層）
  src/        # C 核心 + C++ wrapper + torch 整合
  tests/      # C/C++/Python 測試
  tools/      # bench_gguf_fill（真實檔 fill 延遲）、sim_gguf_cache（命中率模擬）、ctypes 綁定
  tools/scratch/  # Windows 時代的一次性除錯腳本（保留參考）
  CMakeLists.txt
```

## 功能

- `cgc_gguf_lite`：輕量 GGUF v3 header 解析（已修 ARRAY 欄位順序 / 小整數寬度 / 型別依 type 跳過 / qwen35moe KV keys）
- `cgc_expert_streamer`：per-expert pread 串流 + LRU slot cache + hot pool pin
- `cgc_expert_streamer_gguf`：llama.cpp 佈局解析（**雙家族 segment-aware**：qwen35moe 3 段 gate/up/down 分開、gemma4 2 段 gate_up/down 合併；已修 data_start 偏移——GGUF offset 相對 data_start，segment 定址輸出檔案絕對位置）
- `cgc_expert_compute`：streamed 權重的零拷貝視圖橋接
- `cgc_pd_scheduler`：prefill/decode 分層排程（雙 GPU 假設，單 GPU 機不需要）
- `cgc_repack*`：safetensors → per-layer GGUF repack
- C++ 相容層（`expert_streamer.*`、`expert_compute.*`、`pd_expert_scheduler.*`）
- torch/DeepEP 整合（`expert_compute_torch.cpp`，**CUDA-only**，macOS 請改用 llama.cpp+Metal）

## macOS 驗證（2026-08-14）

gcc/clang 可編譯；合成回歸 3/3 PASS；真檔 segment byte-identical 驗證：
`test_real_qwen35moe`（qwen36 IQ2，3 段）9/9 PASS、`test_real_gemma4`（gemma4
IQ3_S，合併 2 段）6/6 PASS——pread bytes 與 GGUF tensor 表**絕對定址**（data_start+raw）完全一致。

## CMake 建置與測試

```bash
cmake -S . -B build
cmake --build build -j 4
ctest --test-dir build          # 合成回歸 3/3

# 真檔 segment 驗證（可選，需真實 GGUF）
cmake -S . -B build-real -DCGC_REAL_GGUF_TESTS=ON \
      -DCGC_QWEN36_GGUF=/path/to/Qwen3.6-...-IQ2_XXS.gguf \
      -DCGC_GEMMA4_GGUF=/path/to/gemma-4-...-IQ3_S.gguf
cmake --build build-real -j 4
ctest --test-dir build-real     # 8/8：3 合成 + real_qwen35moe 9/9 + real_gemma4 6/6
                                #     + streamer_real_gemma4 消費端端到端
                                #     + bench_fill_qwen36/gemma4（fill 頻寬 + 段表 vs parser 定址）
```

編譯範例（無 CMake）：

```bash
gcc -O2 -I include -o /tmp/test_cgc_gguf_integration \
  tests/test_cgc_gguf_integration.c \
  src/cgc_expert_streamer.c src/cgc_expert_streamer_gguf.c \
  src/cgc_pd_scheduler.c src/cgc_expert_compute.c src/cgc_gguf_lite.c -lm
```

## 相關

- compute：CUDA/DeepEP（Windows/NVIDIA）或 **llama.cpp + Metal**（macOS，見
  `moeexpert/LLAMACPP_METAL_BENCHMARK_2026-08-14.md`：IQ2 全 offload 25-28 tok/s）
- `cgc_moe_engine/`（兄弟目錄）：torch MoE 算子（DeepEP dispatch + DeepGEMM + combine，CUDA-only）
