# Expert Streamer 兩條解析路徑評估（2026-08-14）

評估對象：
- **路徑 A**：`cgc_engine/cpp/expert_streamer.h/.cpp`（header-only，llama.cpp gguf_context 整合版）
- **路徑 B**：`cgc_engine/cgc/expert_streaming/` 的 `cgc_gguf_lite.c` + `cgc_expert_streamer_gguf.c`（自包含輕量 parser + segment-aware streamer）

## 實測結果（路徑 A 對真實 IQ2_XXS）

### 1. 無法對當前 llama.cpp 編譯（已移除 3 個 API）
- `gguf_get_data` — 已移除（現代用 ggml tensor）
- `gguf_kv` / `gguf_find_kv` — 型別不公開（現代用 `gguf_find_key` + `gguf_get_val_u32`）
- `gguf_get_tensor_nb` — 已移除（維度改用 `gguf_get_tensor_ne`）
- 修補 5 處才強制編譯過：`gguf_kv` stub、`gguf_get_data` stub、`gguf_get_tensor_nb` stub

### 2. .h 與 .cpp 徹底失同步（這對檔從未一起編過）
- `LayerInfo`：header 用 `*_info` 指標、實作用 `*_tensor_id` int
- `ExpertSlice.dims`：header 不存在，實作卻寫入
- `offsets_`：header 值型別 `const gguf_tensor_info*`、實作存 int
- `top_k_`：private 但 UnifiedExpertStreamer 直接存取

### 3. 強制編譯後對 IQ2 的實測（全部失敗）

```
[PerLayerAdapter] Initialized: 40 layers, hidden=0, inter=0, experts=128, top_k=8
detected layout: 1 (PER_LAYER)
num experts (layer 0): 128        ← 錯誤（真實 256，KV key 讀不到）
load ok: 0/12                      ← 全部專家載入失敗
```

功能層 5 個 bug：
1. **`.weight` 後綴沒剝**：`role == "ffn_down_exps"` 但實際 role 是 `"ffn_down_exps.weight"` → 張量永不匹配（連 down 都找不到）
2. **找 `ffn_gate_up_exps`（合併張量）**：真實 llama.cpp qwen35moe 是 `ffn_gate_exps` + `ffn_up_exps` 分開 → gate/up 永不匹配
3. **KV key 只認 `gemma4.*`**：qwen35moe 檔是 `qwen35moe.*` → hidden=0, inter=0, experts=128 全錯
4. **`tensor_rows/cols` 用 `nb`（stride）當維度**：語義錯誤（應是 `ne[0]=cols, ne[1]=rows`）
5. **Vulkan zero-copy 設計**：macOS 無此路徑

## 對照（路徑 B，我們已修好的）

| 項目 | A（llama.cpp 整合版） | B（cgc_gguf_lite） |
|---|---|---|
| 編譯 | ❌ 對現代 llama.cpp 5 處錯誤 | ✅ gcc/clang macOS 直接編 |
| IQ2/IQ3 解析 | ❌ 0/12 載入失敗 | ✅ 9/9 byte-identical |
| 3 張量佈局（gate/up/down 分開 + down 在前） | ❌ 假設合併張量 | ✅ segment-aware |
| KV keys | ❌ 只認 gemma4.* | ✅ qwen35moe.* + gemma4.* fallback |
| 依賴 | llama.cpp（整檔 mmap，10GB resident） | 自包含（零依賴） |
| 記憶體 | 與 streaming 目的矛盾（需整檔駐留） | bounded slot cache |

## 決策

- **保留路徑 B（cgc_gguf_lite + cgc_expert_streamer_gguf）**，它是唯一在 macOS 上能正確讀取 llama.cpp IQ2/IQ3 的實作
- **路徑 A 不保留**：死程式碼（從未編譯成功、API 已過時、無法讀真實檔案）。`cpp/` 根的 `expert_streamer.h/.cpp`、`expert_streamer_gpu.*` 建議刪除或封存
- 可借鑑的只剩 A 的 C API 形狀（`expert_streamer_create/load_expert`）——若日後要為 cgc streamer 開 C API，可參考其簽名
