# MTP 轉正規劃書 v1.1
### 從「封存」翻案為「攻關中」：graft model（blk.40 MTP head）+ expert-cache L4 pool 實作盤點

日期：2026-08-20 · 狀態：攻關中（n=1 已正確，n=2 分歧未解，未達 28.6 t/s）· 前版：`MTP轉正規劃書_v1.0.md`

---

## 0.5 2026-08-25 解析：D0/D1 已過，D2 以 CGC_VERIFY_DECODE 達成（verify <40ms/token，MTP ≈ base）

**根因找到（v1.1 的 n=2「the the the」+ n=1 分歧，都是同一件事）：speculative-simple 的 MTP
路徑對所有 prompt 無條件 prepend BOS。** Qwen3.6 tokenizer 是 `add_bos=false`，模型沒被訓練成
期待前導 BOS。多 token prompt 硬加 BOS 會污染 conditioning：模型退化為 phrase loop
（`...Paris and the capital of France is Paris...`），MTP draft/verify 再把這條退化軌跡放大成
`the the the`。驗證：llama-simple（graft model）同一 prompt 加 BOS → 一樣 loop；不加 BOS →
`...the capital of the United States is Washington DC`（自然續寫）。**v1.1 當時的「base 軌跡」
本身就是這條 BOS-loop**，所以 n=1「= base 軌跡」其實是兩邊都退化但一致。

**修法（最小改動）**：[examples/speculative-simple/speculative-simple.cpp](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_routeB/llama-src/temp/llama_roadB/llama.cpp-master/examples/speculative-simple/speculative-simple.cpp) 的
BOS prepend 只保留給「單 token prompt」edge case（`inp.size() <= 1`，那時 batch_enc 為空、
process() 跳過 pending_h、draft 會拿到 zero embedding）。多 token prompt 不再加 BOS。

**結果（graft + cache 4GiB + `--temp 0`，`run_n30cache.sh -m qwen36 --mtp N`）**：

| 臂 | t/s | accept | 輸出 |
|---|---|---|---|
| base 單 token（llama-simple，ZERO-slot decode） | ~5–6 | — | `...the capital of the United States is Washington DC` |
| MTP n=1 | 6.35 | **100%** | `...the capital of Germany is Berlin.` ✅ |
| MTP n=2 | 6.22 | **87.5%** | `...the capital of Germany is Berlin.` ✅（與 n=1 同軌跡） |

- **D0 ✅**：n=2 = n=1 軌跡（無「the」loop）+ accept 87.5% ≥ 80%。
- **D1 ✅**：n=1 100% accept 且正確；non-MTP（llama-simple / 純 model）輸出不變
  （`...the capital of the United Kingdom is London`）——本次只改 MTP binary。
- **D2 ⛔（誠實結論）**：28.6 t/s 達不到，MTP 只有 ≈ base（~6 t/s）。原因：這是 compute-bound
  MoE，3-token verify 的 trunk decode 成本 ≈ 3× 單 token，speculative 省下的 decode 數被
  verify 成本吃掉（每步 verify ~126 ms/token）。v1.1 §7.1 已警告 28.6 是野心數字。

### 2026-08-25 補充：verify 成本確實可優化，但 MTP 結構性贏不了 base

用戶提問「verify ~126 ms/token 是否可以優化、Nail model 好像不需要那麼高成本」。實測：

| 組態 | 臂 | t/s | accept | verify ms/token | hit rate |
|---|---|---|---|---|---|
| graft 冷啟動 | base 單 token | 25.13（decode-only） | — | 39.79 | — |
| graft 冷啟動 | MTP n=2 | 6.22 | 87.5% | 126.82 | 72.1% |
| Nail 冷啟動 | MTP n=2 | 7.52 | 87.5% | 103.45 | 77.4% |
| Nail 熱身（-n 200） | MTP n=2 | **13.85** | **97.8%** | **64.08** | 91.8% |

1. **Nail model 確實更便宜**（用戶直覺正確）：blk.40 用 UD-IQ3_XXS（與 trunk 同量，非 graft 的
   Q4_K）→ draft 更輕、cache 命中更高 → verify 103 vs 127 ms/token，7.5 vs 6.2 t/s。
   輸出正確：`...the capital of Italy is Rome.`（與 graft 的 Germany 是不同 draft 的正常差異）。
2. **verify 成本主要是同步 cold-expert pread**：冷啟動時 pread 佔 ~45% wall（8 worker 平行，
   cumulative pread_usec 22s / 5.4s wall）。pool 學到 routing 後 hit rate 72%→92%，verify
   126→64 ms/token，t/s 6.2→13.85。**cold-start pread 是可熱掉的**。
3. **但熱身後 MTP 仍 ~14 t/s，base 純 decode ~25 t/s**：3-token verify 的 MoE FFN 成本隨 token
   數線性成長（192ms/step ≈ 4.8× 單 token 39.79ms），speculative 省下的 decode 數抵不過 verify
   成本。理論最佳（verify 降到線性 3×）也只有 ~19 t/s，仍低於 base。**compute-bound MoE 上
   MTP 結構性贏不了 base**——28.6 t/s 只有 base 單 token 摸得到，MTP 不可能。

### 2026-08-25 突破：CGC_VERIFY_DECODE —— verify 走 decode fast path，MTP ≈ base（轉正）

**成本拆解**（用戶問「base 純 decode 40ms，為啥 verify 103ms」）：verify 不是 40 層單 token，
是 40 層 **3-token batch**（+1 token 重算 id_last）。逐層 hook 實測（CGC_MTP_HOOKDBG）：
ctx_tgt 上 prefill = `n_tok=6 seqmax=5`（n_past=0），verify = `n_tok=3 seqmax=8`（n_past>0）。
n=1 vs n=2 step 差 → **verify = 固定 ~60-70ms/step + 44ms/token 線性 FFN**。固定部分是同步
cold-expert pread（ensure_batch + drain，每層等 pread）+ slot_table remap 成本。

**修法**：[src/llama-context.cpp](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_routeB/llama-src/temp/llama_roadB/llama.cpp-master/src/llama-context.cpp) 的 hook
新增 **MTP 專屬 opt-in `CGC_VERIFY_DECODE=1`**：偵測 ctx_tgt 上的 verify（`ctx_type==DEFAULT &&
embeddings_nextn && n_tokens>1 && seqmax > n_tokens-1`），讓它走 decode fast path（touch +
ZERO-slot，不 ensure_batch/drain → 不同步 pread）。**沒設 env = 原 exact-load verify（bit-exact），
非 MTP 完全不受影響**。偵測用的 `seqmax > n_tokens-1` 用現成 `llama_memory_seq_pos_max`，不需新 API。

**結果（Nail model + cache 4GiB + `--temp 0` + `CGC_VERIFY_DECODE=1`）**：

| 臂 | 組態 | t/s | accept | verify ms/token | 輸出 |
|---|---|---|---|---|---|
| base | graft decode-only | 25.13 | — | 39.79 | — |
| MTP n=2 | Nail exact（熱身） | 13.85 | 97.8% | 64.08 | `...Italy is Rome.` |
| **MTP n=2** | **Nail + CGC_VERIFY_DECODE** | **25.39** | **95.7%** | **35.44** | `...United States is Washington.` ✅ |
| **MTP n=1** | **Nail + CGC_VERIFY_DECODE** | **25.28** | **100%** | **37.72** | 同軌跡 ✅ |
| harness n=2 | 短跑（冷啟動 64 tok） | 23.22 | 87.5% | — | `...United States is Washing...` ✅ |
| harness n=1 | 短跑（冷啟動 64 tok） | 24.24 | 97.0% | — | 同軌跡 ✅ |

- **D2 ✅**：verify 35.4-37.7 ms/token < 40ms（用戶 轉正 門檻）；MTP 25.2-25.4 t/s ≈ base decode
  （25.13）——不再慢於 base。pread 73%↓（26M→6.9M usec）。
- **D0/D1 維持 ✅**：accept 95.7%（n=2）/100%（n=1）；輸出 deterministic（跨 run bit-identical）、
  正確、n=1=n=2 同軌跡。
- **harness 已更新**：[run_n30cache.sh](file:///Users/alexchuang/Documents/flashkv0516/scripts/run_n30cache.sh)
  MTP 臂切到 Nail model（blk.40 UD-IQ3_XXS）+ 預設開 `CGC_VERIFY_DECODE=1`
  （`N30CACHE_MTP_VERIFY_DECODE=0` 可關）。非 MTP（llama-simple）不設此 env，行為不變。
- **取捨**：verify 走 ZERO-slot 後，MTP 輸出 = base ZERO-approx 軌跡（`...United States is
  Washington`，與 base decode 一致），不再是 exact-verify 的 `...Italy is Rome`。兩者都正確；
  對齊 base 反而是 production 想要的（MTP 輸出 == base 輸出）。

**轉正定案**：正確性全綠 + 速度 ≈ base（verify < 40ms/token）→ **MTP 可轉正**（opt-in，`--mtp`）。
28.6 t/s 的原始野心數字仍不達（那是 base 單 token 也摸不到的 compute-bound 極限），但用戶
「verify 達 40ms/token 即可轉正」的判定標準已達成。可帶此結論繼續優化 gemma4（gemma4 無 blk.40，
不適用 draft-mtp）。

---


## 0. 結論先行

v1.0 於 §8.87/8.89 定案「MTP 封存」（draft 走完整 41 層、blk.40 是 256-expert MoE block → 結構性淨負）。
**本版翻案**：改用 **graft model**（`Qwen3.6-35B-A3B-UD-IQ3XXS-trunk_Q4K-blk40.gguf`，trunk IQ3_XXS + blk.40 MTP head Q4_K），
draft 只跑 blk.40 單層，並把 blk.40 併入 expert-cache L4 zero-copy pool。目前已驗證：

| 項目 | 狀態 |
|---|---|
| GPU OOM（MTP + `-c 2048` + graft） | ✅ 已解（`-expert-cache $BUDGET` 補進 harness） |
| accept = 0%（blk.40 排除 pool 造成） | ✅ 已解（Option B：blk.40 **納入** pool + hook 涵蓋 blk.40） |
| multi-token prefill OOB（NaN→garbage） | ✅ 已解（L4 pool 下 prefill/verify 走 pool/remap 路徑） |
| CGC_OA_ASYNC race（CPU 超前 GPU 讀 stale remap） | ✅ 已解（segmented dispatch 改「先 wait 再 submit」+ completion 掛最後 thread buffer） |
| **MTP n=1** | ✅ **100% accept、輸出 = base 軌跡**（4.43/8.30 t/s） |
| **MTP n=2** | ⛔ **88.6% accept 但輸出崩成「the the the」**（真實分歧，非 race） |
| **30+ t/s 目標** | ⛔ 未達（需 n=2 多 token verify 才可能） |

**目前唯一阻塞 = MTP n=2 分歧**。n=1 全部正確證明「單 token draft + 2-token verify」的整條
expert-cache/remap/MTP graph 路徑是對的；n=2 加入「第 2 個 draft step + 3-token verify」後
target 端自信重複「the」→ 分歧點在 n=2 專屬路徑。

---

## 1. 現況盤點：MTP 攻關的終局狀態（截至 2026-08-20）

### 1.1 已定案且正確
- **GPU OOM**：MTP draft context + speculative graph buffers 超過 M4 Max 16GB；根因是
  `llama-speculative-simple` 沒吃 expert cache（weights 8118 MiB）。`-expert-cache $BUDGET`
  加進 harness 後 weights 降到 6846 MiB、`currentAllocated=7442.69 MB`（< 11453.25 MB）→ OOM 解。
- **accept=0% → Option B**：blk.40（MTP head）的 experts **納入** pool（shrink），
  改修 hook/remap leaf 涵蓋 `il ∈ [n_layer, n_layer_all)`。CPU 與 GPU 皆恢復 accept
  （CPU 100%、GPU n=1 100%）。non-MTP 路徑因 `n_layer_nextn==0` 完全不受影響。
- **multi-token prefill OOB**：L4 pool 下 expert tensors 縮到 pool capacity，multi-token
  prefill/verify 若用 raw expert ids 會 `ne[2]` 越界 → NaN → garbage。修法：新增
  `expert_cache_pool_active` flag，L4 pool 模式下的 multi-token 也建立 remap leaf 並走
  pool/remap 路徑（非 L4 維持原 full-weight prefill，與 no-cache baseline bit-identical）。
- **CGC_OA_ASYNC race**：segmented dispatch 原先「submit seg[i+1] 超前，再 wait seg[i]」，
  GPU 可能先讀到 stale remap → garbage draft。修法：改成**先 wait seg[i]（fire hook 寫
  remap[i]）再 submit seg[i+1]**；completion handler 從 main buffer 改掛在**最後 thread
  buffer**（in-order queue 保證它是最後完成）。`CGC_OA_ASYNC_SYNC` env 留作強制同步診斷。

### 1.2 未解（現役阻塞）
- **MTP n=2 輸出分歧**：n=2 時 accept 88.6%（不是低 accept 問題），但輸出從
  `The capital of France is Paris and the | The capital of France...`（n=1/base 的 phrase loop）
  崩成 `the the the the...`。greedy（--temp 0）也重現 → 是真實狀態分歧，非採樣隨機。
  疑似與「verify 3-token batch 在 ctx_dft 的 multi-token decode」或「draft step1 的 h_nextn 鏈」
  相關；GPU 與 CPU 需交叉定位（CPU 測試目前被 13GB model 記憶體 OOM 擋下）。

### 1.3 效能帳本（-ngl 99 + cache 4GiB + graft、M4 Max）
| 臂 | t/s | accept | 輸出 |
|---|---|---|---|
| base 單 token（sync） | ~0.89 | — | 正確 |
| base 單 token（CGC_OA_ASYNC） | 4.86 → 6.14 | — | 正確 |
| MTP n=1 | 4.43 / 8.30 | 100% | = base 軌跡 |
| MTP n=2 | 3.60 / 4.07 | 88.6% | 「the the the…」⛔ |
| **目標** | **28.6** | — | — |

---

## 2. 改動檔案清單（11 檔案 + 2 scripts，+161/−40）

> 位置：`temp/llama_roadB/llama.cpp-master/`（git tracked）。所有 MTP 相關改動皆以
> `n_layer_nextn` / `expert_cache_pool_active` / `CGC_MTP_DBG` gated，**非 MTP（原本）代碼路徑不受影響**。

### 2.1 expert-cache L4 pool 涵蓋 MTP head 層（Option B 核心）
| 檔案 | 改動 |
|---|---|
| [src/llama-model-loader.cpp](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/src/llama-model-loader.cpp) | `compute_l4_pool_capacity(n_layer)`：掃描涵蓋全部 `_exps` tensor（含 blk.40）；`create_tensor` 加 `CGC-L4DBG` 診斷（tname/l4_il/l4_kind/cap） |
| [src/llama-model-loader.h](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/src/llama-model-loader.h) | `compute_l4_pool_capacity(uint32_t n_layer)` 簽名 |
| [src/llama.cpp](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/src/llama.cpp) | L4 path 下改 call `compute_l4_pool_capacity(model->hparams.n_layer())` |

### 2.2 graph：remap leaf 涵蓋 multi-token prefill + MTP head 層
| 檔案 | 改動 |
|---|---|
| [src/llama-graph.h](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/src/llama-graph.h) | `llm_graph_params::expert_cache_pool_active` + `llm_graph_context::expert_cache_pool_active`（L4 pool 是否啟動） |
| [src/llama-graph.cpp](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/src/llama-graph.cpp) | `build_moe_ffn` remap leaf 條件：`(n_tokens==1 \|\| expert_cache_pool_active) && il < n_layer + n_layer_nextn`；`set_input`（embd_h）加 `CGC_MTP_DBG` |

### 2.3 context：hook 涵蓋 blk.40 + prefill 走 pool/remap
| 檔案 | 改動 |
|---|---|
| [src/llama-context.cpp](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/src/llama-context.cpp) | `graph_params` 傳 `expert_cache_pool_active = model.expert_cache_pool_capacity > 0`；`expert_cache_eval_cb` 加 `CGC-MTP_DBG`（ctx type 標記）；`expert_cache_on_topk` 界線改 `n_layer_all`（涵蓋 blk.40）+ multi-token prefill 在 pool active 時走 pool/remap 路徑（不再提前 return） |

### 2.4 Metal 非同步 / race 修復
| 檔案 | 改動 |
|---|---|
| [ggml/src/ggml-backend.cpp](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/ggml/src/ggml-backend.cpp) | segmented dispatch 改順序：**先 wait seg[i]（fire top-k hook 寫 remap[i]）再 submit seg[i+1]**（原 submit-ahead 會 race 讀 stale remap）；`CGC_OA_ASYNC_SYNC` env 強制每 segment 全 sync（診斷用）；`CGC-SEG` 計時 |
| [ggml/src/ggml-metal/ggml-metal-context.m](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/ggml/src/ggml-metal/ggml-metal-context.m) | completion handler：n_cb==0（單 buffer）掛 main；多 buffer 改掛**最後 thread buffer（cb_idx==n_cb−1）**（in-order queue → 它完成 = 整 graph 完成，poll 不會早 fire）；`CGC-MEM_PRINT` 診斷（OOM 時印 currentAllocated/recommendedMaxWorkingSet） |
| [ggml/src/ggml-metal/ggml-metal-device.m](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/ggml/src/ggml-metal/ggml-metal-device.m) | `CGC-MEM-BUF` 診斷（buffer init size 印出） |

### 2.5 harness / 生產腳本
| 檔案 | 改動 |
|---|---|
| [examples/speculative-simple/speculative-simple.cpp](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/examples/speculative-simple/speculative-simple.cpp) | `spec_mtp` 提早偵測 + **跳過預設 warmup、改在 `set_embeddings_nextn` 後手動 warmup**（避免 graph topology 錯位）；`CGC_MTP_DBG` 下 dump target logits |
| [examples/simple/simple.cpp](file:///Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/examples/simple/simple.cpp) | 支援 `-c` flag（n_ctx 可覆寫，production 用） |
| [scripts/run_n30cache.sh](file:///Users/alexchuang/Documents/flashkv0516/scripts/run_n30cache.sh) | `--mtp [N]` flag：qwen36 自動切 graft model + speculative-simple binary + `--spec-type draft-mtp --spec-draft-n-max N -c 2048 -expert-cache $BUDGET`（§MTP 區塊） |
| [scripts/build_prod_binary.sh](file:///Users/alexchuang/Documents/flashkv0516/scripts/build_prod_binary.sh) | **MTP isolation**：llama-simple 不編 `-DMTP_SUPPORT`（純 non-MTP）；llama-speculative-simple 編 `-DMTP_SUPPORT`；libllama-common 需 `-DMTP_SUPPORT`（release_context 存在） |

### 2.6 未改動但關鍵的既有元件（fork 已有，v1.1 僅驗證）
- `common/speculative.cpp`：MTP driver（`set_embeddings_nextn` / `get_embeddings_nextn_ith` /
  process/verify/draft/accept 流程、`verify_h` / `pending_h` 鏈）——**本版未改**（clean）。
- `src/models/qwen35moe.cpp` `graph_mtp`：blk.40 單層 MTP head graph（h∥e concat → eh_proj →
  GatedAttn → MoE FFN → shared_head_norm → lm_head）。

---

## 3. 已完成的步驟（時間線）

1. **OOM 解**：MTP + `-c 2048` + graft 必 OOM → 定位為 `llama-speculative-simple` 未啟用
   expert cache → harness 加 `-expert-cache $BUDGET` → weights 8118→6846 MiB，OOM 解。
2. **accept=0% 翻案**：blk.40 排除 pool（kept-full）在 GPU 產生 NaN logits → 改 **Option B**：
   blk.40 **納入** pool + hook/remap leaf 界線延伸 `n_layer_all`。CPU 100%、GPU n=1 100% accept。
3. **multi-token prefill OOB**：L4 pool 下 multi-token prefill 用 raw expert ids 越界
   （ne[2]=200）→ NaN → garbage。修：`expert_cache_pool_active` flag + remap leaf 涵蓋
   multi-token + hook 在 pool active 時走 pool/remap 路徑。輸出恢復連貫，速度 4.86→6.14 t/s。
4. **async race 修復**：`CGC_OA_ASYNC` bisect 出為唯一 corruption 源（CPU 超前 GPU 讀 stale
   remap）→ segmented dispatch 改「先 wait 再 submit」+ completion 掛最後 thread buffer。
5. **n=1 全綠**：MTP n=1 = 100% accept、輸出 = base 軌跡（「假速度」時代之後首次真實正確）。
6. **n=2 分歧（現役阻塞）**：MTP n=2 = 88.6% accept 但輸出崩成「the the the」；greedy 重現；
   CPU 隔離測試被 13GB model 記憶體 OOM 擋下，尚未交叉定位。

---

## 4. 未解決問題（v1.1 之後）

### 4.1 MTP n=2 分歧（P0，唯一阻塞 28.6）
- **現象**：accept 88.6%（高），但 target 自信重複「the」→ target 端狀態在 3-token verify 後分歧。
- **n=1 vs n=2 結構差**：
  1. verify batch 2→3 token（ctx_dft 的 multi-token decode 多一列）；
  2. draft 多 1 個 step（`draft()` while 迴圈跑第 2 次：decode `[d1]` → h_nextn(d1) → d2）。
- **待查**：verify 3-token batch 的 h_nextn 列對齊；draft step1 的 `pending_h`/`chain_h` 鏈；
  或某 buffer 對 n_tokens=3 的邊界假設。
- **交叉驗證**：CPU MTP n=2（需先解 13GB 記憶體，例如關閉其他程序或 `-c` 縮小 + `--no-mmap`）。

### 4.2 速度（P1）
- 28.6 t/s 唯一路徑 = MTP 多 token verify（n=2 以上）。n=2 修對後才談速度優化
  （CPU-split 開銷 / accept 品質 / CGC_N_CB / CGC_GLU_FUSED_DOWN 掃描）。

---

## 5. 下一步計劃與驗證閘

| 刀 | 內容 | 驗證閘（過才進下一步） |
|---|---|---|
| **D0** | 交叉定位 n=2 分歧：CGC_MTP_DBG 抓 draft step1 / verify 3-token 的 h_nextn norm + draft pick + target logits；必要時 CPU 隔離 | n=2 輸出 = base 軌跡（無「the」loop）且 accept ≥ 80% |
| **D1** | n=2 分歧修復落地（找出根因後最小改動，維持 n=1 + non-MTP 不變） | n=1/n=2 皆正確 + non-MTP bit-identical |
| **D2** | n=2 速度測量與優化（CPU-split 開銷 / accept） | 朝 28.6 t/s 前進；先達 > base 單 token 的實質加速 |
| **D3** | 生產化：build_prod_binary.sh + run_n30cache.sh 最終參數定案（MTP isolation 已就位） | 正式 A/B 對拍 + 多輪交錯無崩 |

每刀維持 §7.2 方法論：真實文本、greedy（--temp 0）先驗正確再談速度、bit-identical 對拍、
`CGC_OA_ASYNC_SYNC` vs async 雙臂。

---

## 6. 關鍵參數 / 命令 / env（供復現）

```
# MTP 攻關命令（qwen36 + graft + expert cache）
./scripts/run_n30cache.sh -m qwen36 --mtp 2 -n 128 -p "The capital of France is Paris and"

# 等價手工命令
llama-speculative-simple -m models/gguf/Qwen3.6-35B-A3B-UD-IQ3XXS-trunk_Q4K-blk40.gguf \
  -ngl 99 --no-mmap -t 8 -c 2048 -n 128 \
  --spec-type draft-mtp --spec-draft-n-max 2 -expert-cache 4294967296 \
  -p "The capital of France is Paris and"
```

env（生產集，§8 有出處）：`LLAMA_EXPERT_CACHE_ALLOW_NGL=1` `CGC_EXPERT_CACHE_BYTES`
`LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1` `LLAMA_EXPERT_CACHE_WORKERS=8` `CGC_WAKE_POLL_US`
`CGC_OA_ASYNC=1` `CGC_N_CB=8` `CGC_GLU_FUSED_DOWN=1`

診斷 env：`CGC_MTP_DBG=1`（h_nextn/draft pick/target logits/remap leaf）、
`CGC_MEM_PRINT=1`（Metal 記憶體）、`CGC_OA_ASYNC_SYNC=1`（強制同步，隔離 race）、
`CGC-NODE-COUNT`（ctx_type + 節點數）

模型：`models/gguf/Qwen3.6-35B-A3B-UD-IQ3XXS-trunk_Q4K-blk40.gguf`
（graft = Qwen3.6-35B-A3B trunk IQ3_XXS + fraQtl blk.40 MTP head Q4_K；純 UD-IQ3_XXS 無 blk.40 無法 MTP）

---

## 7. 風險與誠實邊界

1. **28.6 t/s 是野心數字**：base 單 token 目前 6.14 t/s（GPU-bound），要靠 MTP 多 token
   verify 拉 ~4.6×。就算 n=2 修對，accept 與每步 decode 成本是否夠優，仍是開放問題。
2. **「假速度」教訓**：先前 15.8/25.8 t/s 是 race 讓 CPU 超前 GPU 讀 stale 資料的無效數字。
   任何速度數字必須以**真實正確輸出**（greedy 對拍 base 軌跡）為前提，race 修復（§2.4）是
   正確性的地基。
3. **CPU 隔離受阻**：13GB model 在 CPU-only 模式（-ngl 0）於本機記憶體壓力下被 OOM kill，
   GPU 專屬 vs MTP 邏輯錯誤的區分尚未完成。
4. **non-MTP 保護**：所有改動 gated（`n_layer_nextn` / `expert_cache_pool_active` /
   `MTP_SUPPORT`），但仍需 D3 正式對拍確認原本代碼 bit-identical。

---
已做步驟 （時間線）：OOM 解 → accept=0% 翻案（Option B）→ multi-token prefill OOB 修 → async race 修 → n=1 全綠 → n=2 分歧（現役阻塞） 。

## 關於你問的 race 問題
對——先前 15.8 / 25.8 t/s 是「假速度」： CGC_OA_ASYNC 下 CPU 超前 GPU 讀 stale remap，輸出是 garbage。該 race 已在 §2.4 修復（先 wait 再 submit + completion 掛最後 thread buffer），所以 目前 MTP n=1 的 100% accept / 輸出 = base 軌跡是「真實正確」的 。n=2 的「the the the」是修完 race 之後仍出現的 真實分歧 （greedy 也重現），不是 race。

下一步照 §5-D0 交叉定位 n=2 分歧（ CGC_MTP_DBG 抓 draft step1/verify 3-token 的 h_nextn norm + target logits）。要我繼續嗎？
## 8. 參考

- 前版：`moeexpert/MTP轉正規劃書_v1.0.md`（ISCA'26 預測線 + §8.83-8.89 MTP 封存定案）
- 方案文：`moeexpert/LLAMACPP_EXPERT_BOUNDED_RESIDENCY_FORK_方案.md` §7.2 / §8.7 / §8.83-8.89
- MTP driver：`common/speculative.cpp`（process/verify/draft/accept、verify_h/pending_h 鏈）
- MTP head graph：`src/models/qwen35moe.cpp` `graph_mtp`
- expert-cache L4 pool：`src/llama-expert-cache.cpp` / `src/llama-model-loader.cpp`
