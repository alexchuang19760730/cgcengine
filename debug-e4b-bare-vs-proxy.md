# Debug Session: e4b-bare-vs-proxy [OPEN]

## Goal
- 量出 `Gemma4-E4B Q2_K` 在本機 M4 16GB 上：
  1. 裸 `llama.cpp` 的 decode 基線
  2. `edge_first_proxy` 的應用層稅
  3. `expert_data_plane` 與 `oMLX + FlashMoE` 的結構性差異

## Bare llama.cpp Result
- 命令口徑：
  - `ngl=all`
  - `flash_attn=on`
  - `n_batch=256`
  - `n_ubatch=128`
  - `threads=10`
- 以記憶體內抓取 stdout/stderr 的方式執行 `llama-cli`
- 觀測到的內建摘要：
  - `Prompt: 88.4 t/s`
  - `Generation: 26.6 t/s`

## Proxy Comparison
- 目前最佳穩定 `proxy on_bypass` 基線：
  - `TTFT 180.4ms`
  - `decode 25.3 tok/s`
- 對比裸 `llama.cpp`：
  - `bare decode = 26.6 tok/s`
  - `proxy decode = 25.3 tok/s`
  - 目前應用層稅約落在 `~1.3 tok/s`，量級上屬於小尾差

## Structural Difference
- `oMLX + FlashMoE` 本地路徑是資料面執行層：
  - `use_flashmoe=True` 時，`local_infer.py` 先嘗試 `DFlashEngine`
  - 只有 `DFlash` 失敗時才退回 `llama.cpp` / `mlx_lm`
  - 參考：
    - [local_infer.py:L934-L1044](file:///Users/alexchuang/Documents/flashkv0516/app/edge_engine/local_infer.py#L934-L1044)
    - [dflash.py:L1047-L1115](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/Backend/oMLX/omlx/engine/dflash.py#L1047-L1115)
- `DFlashEngine` 自帶：
  - paged cache
  - SSD cache
  - continuous batching
  - MLX executor 內直接 `stream_generate`
- 我們現在的 `expert_data_plane` 主要是控制面：
  - request 開始時 `_build_plan()`
  - 排 prefetch
  - 同步 `_ensure_loaded(current_keys)`
  - 記錄 frontier / cache hit / prefetch hit
  - 參考：
    - [expert_data_plane.py:L1016-L1088](file:///Users/alexchuang/Documents/flashkv0516/app/shared/expert_data_plane.py#L1016-L1088)

## Current Conclusion
- `E4B` 現在和裸 `llama.cpp` 的差距不大，主問題不是 proxy 應用層稅。
- `expert_data_plane` 和 `oMLX + FlashMoE` 差很多，主因不是單一參數，而是它們所在層級不同：
  - `oMLX + FlashMoE` 是模型 forward 內部的資料面執行
  - `expert_data_plane` 目前仍偏向 request 級控制面與外圍預載

## Next Step
- 針對 `CGC/oMLX + FlashMoE` 再抓一組同模型可比基線。
- 確認 `expert_data_plane` 若要追上 `oMLX + FlashMoE`，需要下沉到哪個 execution layer。
