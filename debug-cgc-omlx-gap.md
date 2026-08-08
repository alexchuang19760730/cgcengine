# Debug Session: cgc-omlx-gap [OPEN]

## Symptom
- `Gemma4-E4B Q2_K` 在本機 M4 16GB 上：
  - 裸 `llama.cpp` 量到 `Generation: 26.6 t/s`
  - `edge_first_proxy on_bypass` 量到 `decode 25.3 tok/s`
- 這說明 proxy 應用層稅不是主差距。
- 目前需要再抓一組 `CGC/oMLX + FlashMoE` 的同口徑本地基線，拆清楚它與目前 `expert_data_plane` 的差異究竟來自：
  1. execution layer 已內嵌
  2. 還是我們仍停留在外圍控制面

## Session Goal
- 量出 `CGC/oMLX + FlashMoE` 的同口徑本地 baseline
- 以 runtime evidence 比對：
  - `bare llama.cpp`
  - `edge_first_proxy`
  - `CGC/oMLX + FlashMoE`
- 判定下一步應該：
  - 繼續補 `expert_data_plane`
  - 或把 streaming 下沉進 target runtime

## Falsifiable Hypotheses
1. `CGC/oMLX + FlashMoE` 目前根本沒有走到真正的 `DFlashEngine` / `FlashMoE` 執行路徑，而是在 fallback 到 `llama.cpp` 或 `mlx_lm`。
2. `CGC/oMLX + FlashMoE` 即使有執行，也因 model reference / quantization format 不匹配，沒有形成可與 `E4B GGUF` 同口徑比較的本地基線。
3. `CGC/oMLX + FlashMoE` 的主要優勢確實來自 execution-layer 內嵌（paged cache / continuous batching / MLX executor），而不是 proxy 外層少掉幾個控制面步驟。
4. 我們目前 `expert_data_plane` 的成本主體不在 decode 內核，而在 request 級 `_build_plan / prefetch / ensure_loaded / frontier bookkeeping`。
5. 若 `CGC/oMLX + FlashMoE` 對 `E4B` 根本不適用，則現在應該先用一顆真 MoE 模型做同口徑對照，否則會把「模型格式限制」誤判成「執行層優勢」。
