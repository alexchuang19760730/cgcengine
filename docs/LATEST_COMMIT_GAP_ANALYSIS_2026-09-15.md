# 最新 commit 缺什麼 — prefill 250 / decode 25 + M2 路徑 bit-identical

**日期** 2026-09-15 · **分支** `demo/sweet-spot-windows-fix` · **HEAD** `41195d5a3`
**模型** `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（13.66 GB，256 experts/layer）
**結論（先講）**：prefill 250 **已量到但沒被 commit 造成**，decode 25 **完全沒到**，
而 M1/M2/M3 的 19/19 **是在自己跟自己比**——參考 oracle 本身就是 expert-cache-ON 的產物。
你那組 NOGATHER A/B 才是目前唯一有鑑別力的證據，它把故障 100% 鎖在 hook（remap/pool/gather）那一側。

---

## 1. 最近 10 個 commit

| # | commit | 主題 | 性質 |
|---|---|---|---|
| 1 | `41195d5a3` | perf(m2): n_batch=6144 sweet spot + prefill 250+ tok/s configuration | 記錄 + env 放行 |
| 2 | `b8220555f` | perf(m2): double-buffer slab fill — I/O 與 graph build 重疊（+28% prefill） | 效能 |
| 3 | `987e597d6` | perf(m2): contiguous slab read — 每 (layer,kind) 一次 pread 而非 256 次 | 效能 |
| 4 | `fe6c0fbf2` | feat(m2): prefill whole-layer streaming + `--m2` oracle mode | 新路徑 |
| 5 | `81725f30f` | fix(server): graceful shutdown + pipefail-safe cleanup | 穩定性 |
| 6 | `baaa37926` | WIP: M2 prefill streaming + CGC_POOL_SPLIT + provenance framework | 新路徑（大） |
| 7 | `e8b85393d` | feat(matrix): 註冊 Ornith 1.5 35B 為第三組 (base, head) | 量測 |
| 8 | `1ca502384` | fix(m1): pool cap 決定 prefill chunk，不能隨便動 | 量測方法學 |
| 9 | `dd0ad9694` | fix(flip_rate): 用 two-ruler API 打分 | 量測方法學 |
| 10 | `db460acef` | feat(mtp): per-carrier draft acceptance + head 身分閘 | 量測 |

`41195d5a3` 的實際改動（去掉重建的 dylib）：

- `src/llama.cpp/src/llama-context.{cpp,h}` — +79/+17
- `src/llama.cpp/src/llama-expert-cache.{cpp,h}` — +85/+8
- `scripts/run_server.sh` — +11（把 `CGC_M2_PROFILE` / `CGC_M2_DB_DISABLE` 加進 env allowlist）
- `scripts/check/knifeedge_matrix.py` — +5（`args.m2` → `getattr(args,"m2",False)`）
- `docs/PREFILL250_CONFIGURATION_GUIDE_2026-09-15.html` — +347（新文件）

**沒有一個是新的數值修正**。而 264.78 tok/s 那筆量測在 `Backup/cgc_logs/llama_server_20260915_011146.log`，
時間戳 01:11，比 commit（01:45）早 34 分鐘。也就是說：**這個 commit 是「把 250 寫進文檔」，不是「做出 250」。**

---

## 2. 目標 1：prefill 250 tok/s / decode 25 tok/s

### 2.1 Prefill — 量到了，但條件很窄 ✅(有條件)

```
0.29.639.579 I slot print_timing: prompt eval time = 17720.16 ms / 4692 tokens
             (3.78 ms per token, 264.78 tokens per second)
```

| n_batch / n_ubatch | prefill tok/s | 狀態 |
|---|---|---|
| 2048 / 512 | 102.36 | 基線 |
| 4096 / 4096 | 227.15 | +122% |
| **6144 / 6144** | **264.78** | **最佳** |
| 8192 / 8192 | — | GPU OOM |

必要條件（缺一不可）：`CGC_PREFILL_STREAM=1` + `CGC_GATHER_SLAB_CAP=256` +
`n_batch=n_ubatch=6144` + 8 GiB pool + 4692-token 冷啟動 prompt。

還差什麼：

1. **`run_server.sh` 的預設不是這組配置。** commit 後 02:24–02:50 那批 log 的 prefill 只有
   3.94–19.40 tok/s（`llama_server_20260915_02*.log`）——都是短 prompt 的預設 run。
   250 是「要人手動開一組 env 才拿得到」，不是 branch 的預設行為。
2. **8192 OOM 沒有處置**，6144 是在 OOM 邊界上撿到的 sweet spot，沒有保護/降級邏輯。
3. 量測方法學有坑（指南 §4.1 自己寫了）：API 的 `prompt_per_second` 只算未命中 cache 的 token，
   不能直接用。這條沒進 harness，只在文檔裡。

### 2.2 Decode — 沒到，缺口約 2× ❌

| 來源 | decode t/s |
|---|---|
| 今日 log 峰值（`20260915_021201`，20-token prompt / 7-token 生成） | **18.89** |
| 今日 log 典型區間（6 條以上樣本） | 6.34 – 13.01 |
| roadmap M3 自己的離場條件 | MTP-off ≥ **15** |
| 2026-09-14 實測 MTP-off | 8.87 |
| 2026-09-14 實測 MTP-on（nail carrier，accept 70.39%） | 6.74（**淨損失 1.8×**） |

所以 **25 t/s 還差 ≈ 1.3–2×，連 roadmap 自己設的 15 t/s 中繼門檻都沒過**。
roadmap 指名的兩根柱子都還立著：

- **W2（每 token 計算）**：T=1 是 92 ms/token、T=2.6 是 104 ms/token → verify 完全沒吃到批量紅利。
  MTP 額外成本的 **84% 在這裡**。
- **W3 的殘餘**：pool 143 → 179 slots 讓 I/O 少 3.7×，decode 只 +9% → decode 已不是 I/O bound，
  只能從 compute 側砍（M3 的 per-layer `eval()` 同步、MoE gather 融合、batched-union gather）。

---

## 3. 目標 2：M1/M2/M3 對「正確 M2 oracle」bit-identical

### 3.1 你那組 A/B 的鑑別力（這是目前最有力的證據）

| 配置 | step 0 argmax | logit |
|---|---|---|
| 8GB cache（gather ON） | 271 | 14.22 |
| NOGATHER（gather OFF） | **198** | **27.18** |
| 無 expert cache（基線） | **198** | **27.18** |

兩點判讀：

1. **差 13 logits，不是 ulp。** 27.18 → 14.22 這個量級不可能是浮點結合律/縮減順序造成的
   （那個量級是 ~1e-6）。它是**語義錯誤**：要嘛讀到錯的 expert 權重（最可能是全零／stale），
   要嘛 id 映射錯。這跟 WIP 註解裡量到的「slot0[0..15]=00 for expert 161」方向一致。
2. **NOGATHER 關掉的是什麼？** `llama.cpp:365/417`：`expert_cache_skip_load` 與
   `expert_cache_active` 同時被關。也就是說權重回到**全量 resident**，且
   **remap / pool / gather hook 完全不跑**。它 == 基線，代表
   loader、GGUF index、pread、sampler、MTP、template 全部乾淨。
   → **故障 100% 在「expert cache 這一側」，別再往別處找。**

   ⚠️ 但 NOGATHER **一次動了兩個變數**：`skip_load`（resident ↔ 串流）和
   `expert_cache_active`（hook 開 ↔ 關）。目前引擎**沒有只關 hook、保留 skip_load 的 env**
   （`llama-model.h` 的 gate 是 `expert_cache != nullptr && n_gpu_layers() <= 0 && !NOGATHER`，
   單一條件）。所以嚴格說，這組 A/B 把故障縮到「skip_load ∪ hook」，還沒縮到「hook」。
   下一步必須把這兩個變數拆開（見 Step 2 第 0 項）。

### 3.2 但 gate 顯示 19/19 — 因為它是自證的 ⚠️

`scripts/run_server.sh:587` 無條件帶 `-expert-cache "$BUDGET"`，harness 沒有任何
「關掉 expert cache」的 launch path。後果：

| 參考檔 | 產生時機 | step 0 |
|---|---|---|
| `ref_iq3_pool8gb.jsonl` | 09-13 03:49，cache **ON**（sidecar: head `c14785bda`） | 248046 / 15.53 |
| `ref_iq3_pool8gb_M2.jsonl` | 09-14 16:13，cache **ON**（sidecar: head `baaa37926`, cap=8） | 248046 / 15.53 |
| `ref_iq3_pool8gb_M2_6144.jsonl` | 09-15 01:14，cache **ON** | 248046 / 15.53 |
| `Backup/knifeedge_matrix/` 裡的 cache-OFF 參考 | **不存在** | — |

所以 `combo_iq3_pool8gb_m2_final_verify.json` 裡的
`m1_numeric_identity = 19/19 / m2_decision_agreement = 19/19 / m3_topk_set_agreement = 19/19`
**只證明了 pool-size 不變性（4/6/8 GiB 互相一致），完全沒有證明等價於真值。**

兩個讓這件事更難被發現的點：

- 指南 §4.3 的驗證指令帶了 `--allow-model-mismatch --allow-stale-oracle`
  ——正好把唯二會擋下這種錯的兩個 guard 關掉了。
- `union_fit` gate 回報 `gather: {exercised: false, gather_steps: 0}`、
  `union_max=64 / usable=142` ——在 cap 8 的 sweep 裡 gather 路徑**根本沒被走到**，
  所以「M1 19/19」連 gather 路徑的覆蓋都沒有。

### 3.3 最關鍵的戰略事實：最快的那條路 = 壞掉的那條路

`llama-context.cpp:4205` 起的 M2 prefill streaming：`n_tokens > cgc_pool_max_tokens()` 時
不分流到 pool，而是填一塊 **whole-layer slab（全部 256 experts）**、把 FFN tensor 指過去、
`ne[2]` 設成 `n_expert`、讓 `mul_mat_id` 吃 **raw expert id（無 remap）**。
這就是 lifting n_batch clamp 的機制，也就是 264.78 tok/s 的來源。

換句話說：**M2 路徑 ≡ slab/gather 路徑。你現在是拿一條已知非 bit-identical 的路徑在跑 250 tok/s。**
在 gather 路徑修好之前，「250 tok/s + M1 100%」這兩件事不能同時宣稱。

### 3.4 WIP（未提交）已試過、但都沒關掉的三個假說

| 假說 | probe / 修法 | 結果 |
|---|---|---|
| 權重 bytes 沒到 GPU（`--load-mode none` → storageModePrivate，CPU pread 寫不到 GPU） | `CGC_POOL_SYNC` / `CGC_POOL_GPU_SYNC` / `CGC_EXPERT_GPU_SYNC` 三處 `ggml_backend_tensor_set` | 仍錯（271 → 248046，還是 ≠ 198） |
| remap id 沒到 GPU（hook 在 graph 中途寫 CPU 端，Metal 早在 split 開頭就 copy 完） | 無條件 `ggml_backend_tensor_set(remap, ...)` | 仍錯 |
| gather 用 `llama_expert_cache_ensure`（走 `cache->map` + `fill_slot`，不寫 pool slot） | 換成 `ensure_slot` + `fill_pool_direct` | 仍錯 |
| MM vs MV kernel（`ne21_mm_id_min`） | `CGC_FORCE_MV_ID` | 仍錯（排除 kernel 選擇） |

四次都沒關掉，表示**單一「同步」或「換 kernel」都不是根因**，需要換判別設計（下節）。

---

## 4. 建議的下一步（按順序，前面的沒做完不要往後走）

### Step 0 — 先生出真值參考（缺這個，後面每一個「修好了」都無法判定）

```bash
# expert cache 整個關掉（BUDGET=0 → 無 index → 全量 resident），同一支 binary、同一個 probe
CGC_SERVER_EXPERT_CACHE_BYTES=0 CGC_SERVER_CTX=8192 ./scripts/run_server.sh
# 或最小侵入：LLAMA_EXPERT_CACHE_NOGATHER=1（你已證明它 == 基線）
```
把 dump 存成 `Backup/knifeedge_matrix/ref_iq3_pool8gb_NOCACHE_M2.jsonl` 並寫 sidecar
（cap / model digest / launch sig / source stamp）。

### Step 1 — 把 gate 從「pool 互比」改成「絕對比真值」

- `knifeedge_matrix.py` 加 `--oracle-abs`（或 `--baseline-nocache`）：每個 combo 都去比
  cache-OFF 參考，而不是比另一個 cache-ON dump。
- sidecar 增加 `expert_cache: on|off` 欄位；**cache-ON 的參考直接拒絕當絕對真值**。
- 指南 §4.3 那兩個 `--allow-*` 從慣例指令裡拿掉。

### Step 2 — 用一個最小實驗把「權重錯」和「id 錯」分開

**第 0 項（先做，成本最低）**：加一個只關 hook、保留 skip_load 的 env
（例如 `LLAMA_EXPERT_CACHE_NOHOOK=1`，只把 `expert_cache_active` 設 false、
`expert_cache_skip_load` 維持 true）。這樣就有第四條臂：

| 臂 | skip_load | hook | 預期 |
|---|---|---|---|
| 基線（BUDGET=0） | off | off | 198 ✓（已知） |
| NOGATHER | off | off | 198 ✓（已知） |
| **新：skip_load ON + hook OFF** | **on** | **off** | 若 = 198 → 串流進來的 bytes 是對的，故障在 hook；若 ≠ 198 → 故障在 skip_load/串流層 |
| 現況：8GB gather ON | on | on | 271 ✗（已知） |

這一格就能把嫌疑範圍從「兩個變數」砍成「一個」，比再多十次 `fprintf` 都有用。

然後才是一次 forward 的權重/id 對打：

1. gather 之後，把 slab 的前 N bytes 從 GPU 讀回來（`ggml_backend_tensor_get`），
   跟直接用 `pread` 從 GGUF 同一 offset 讀出的 bytes 比。**相等 → 權重沒問題，故障在 id/幾何。**
2. 若權重對，就把 hook 寫的 remap 值 vs `mul_mat_id` 實際從 `src[2]` 讀到的值對打
   （`CGC_MMID_MV_DBG` 已經在讀 `ids_cpu`，把它和預期 raw id 比）。
3. 若兩者都對，剩下唯一嫌疑是 `ne[2]` / `nb[2]` 幾何（gather 把 `ne[2]` 改成 union，
   而 baseline 是 256）——量 `wt->ne[2]` 在 prefill step 與 decode step 的值。

### Step 3 — 才回到效能

- prefill：把 6144 組態寫進 `run_server.sh` 的 M2 profile（現在只在文檔裡），
  並加 8192 OOM 的降級。
- decode：這是 1.3–2× 的量級差，不是調參能補的。roadmap M3 的三個工作項
  （per-layer `eval()` 同步、MoE gather 融合、batched-union gather）至少要做掉前兩個；
  M4 的拒絕取樣是 MTP 從淨損失翻正的前置條件。

---

## 5. 檢查清單：最新 commit 還缺什麼

**正確性（擋在所有效能數字前面）**

- [ ] cache-OFF 的絕對參考 oracle（Step 0）
- [ ] gate 改成絕對比較；cache-ON 參考不得當真值（Step 1）
- [ ] sidecar 記 `expert_cache: on|off`；`--allow-stale-oracle` 不再出現在慣例指令
- [ ] 加 `LLAMA_EXPERT_CACHE_NOHOOK=1`（只關 hook、保留 skip_load），把兩個變數拆開（Step 2 第 0 項）
- [ ] 權重 vs id 的判別實驗（Step 2）
- [ ] gather/slab 路徑修到 step 0 argmax = 198 / logit = 27.18（與基線逐位元一致）
- [ ] 修好後重跑 4/6/8 GiB × M2，並**強迫 gather 被走到**（現在 `exercised: false`）

**效能**

- [ ] decode 25 t/s：現況峰值 18.89、典型 6–13 → 缺 1.3–2×，roadmap M3/M4 都還沒做完
- [ ] prefill 250：已量到 264.78，但要寫進 `run_server.sh` 預設 + 8192 OOM 降級
- [ ] MTP 目前淨損失 1.8×（6.74 vs 8.87 t/s），拒絕取樣沒做之前 MTP 不算資產

**工程衛生**

- [ ] WIP 裡那批 `fprintf(stderr, ...)` debug（`CGC-ENSURE-SLOT` / `CGC-FILL-VERIFY` /
      `CGC-POOL-DATA` / `CGC-REMAP-SYNC`）是無條件輸出的，進 commit 之前要嘛 env-gate、
      要嘛整段拿掉——`run_server.sh` 的 env allowlist 之外，`getenv` 沒 gate 的會直接噴滿 log。
- [ ] `ne21_mm_id_min` 被 `CGC_FORCE_MV_ID` 改成 1000000，是 debug 開關，確認沒有殘留進預設。

---

## 6. 證據索引

| 內容 | 路徑 |
|---|---|
| 264.78 tok/s 原始量測 | `Backup/cgc_logs/llama_server_20260915_011146.log` |
| commit 後預設配置的速度（3.94–19.40 tok/s） | `Backup/cgc_logs/llama_server_20260915_02*.log` |
| gate 記錄（19/19，gather exercised:false） | `Backup/knifeedge_matrix/combo_iq3_pool8gb_m2_final_verify.json` |
| 參考 oracle 的 provenance | `Backup/knifeedge_matrix/ref_iq3_pool8gb_M2.jsonl.cap` |
| 你那組 A/B 的 dump | `/tmp/oracle_nogather.jsonl`（198/27.179）、`/tmp/oracle_nocache_ids.jsonl`（198/27.179）、`/tmp/oracle_8gb_*.jsonl`（271/14.216、248046/15.389） |
| NOGATHER 的 gate 定義 | `src/llama.cpp/src/llama.cpp:363-365, 415-417` |
| M2 prefill streaming（slab repoint + raw ids） | `src/llama.cpp/src/llama-context.cpp:4205-4215` |
| cap 不變性 = partition 敏感（不是 rounding） | `docs/CAP_INVARIANCE_LOCALIZATION_2026-09-14.md` |
| roadmap / M3 離場條件 | `docs/roadmap-2026-09-14/ROADMAP_PREFILL250_DECODE25_2026-09-14.html` |

**誠實的但書**：§3.1 的 A/B 是同一 prompt、同一 binary 的單點對比，不是 19-row sweep；
`/tmp/oracle_*` 這批 dump 沒有 sidecar，確切的 env 組態沒有被記錄下來，
所以 271 / 248046 / 2487 三種「錯法」各自的觸發條件無法從檔案本身回溯——
這正是 Step 0「先建帶 provenance 的真值」要解決的問題。

---

## 7. 已執行的改動（2026-09-15 12:1x，未提交）

### 7.1 無條件 fprintf（原 Task 1）

工作區裡那 7 處（`CGC-ENSURE-SLOT` / `CGC-FILL-VERIFY` / `CGC-FILL-JOB` /
`CGC-FILL-POOL-DIRECT` / `CGC-COLLECT-DBG` / `CGC-POOL-DATA` / `CGC-REMAP-SYNC`）
在我動手前已從工作區消失（被還原，stash 裡也沒有），所以無需處理。
HEAD 裡還活著的是同一家族的 `CGC-M2-DBG`（`llama-context.cpp:4216`，每次 run 無條件噴 10 行，
在 prefill 熱路徑上）——已改成 `CGC_M2_DBG` env-gated，並加進 `run_server.sh` 的 env allowlist
（不在 allowlist 裡的 `CGC_*` 會被靜默丟掉，等同「沒設」）。

### 7.2 第四臂 `LLAMA_EXPERT_CACHE_NOHOOK=1`（原 Step 2 第 0 項）

`src/llama.cpp/src/llama.cpp`：

- 新增 `no_hook` 解析（數值解析，`=0` 是關）。
- `expert_cache_skip_load` **不動**（NOHOOK 保留它原來的值）。
- `expert_cache_active` 加上 `&& !cgc_no_hook` → hook / remap / pool / slab repoint 全關，
  ids 維持 raw、`ne02` 維持 `n_expert`。
- `l4_path` 也加上 `&& !cgc_no_hook`。**這一步是必要的**：`l4_path` 開著時 loader 會把 expert
  tensor 縮到 pool 容量（8 GiB → 143 experts），而填這些 slot 的正是 hook；hook 關掉 + tensor 縮小
  = 讀到從沒被寫過的權重，那一臂量到的是空氣。關掉 `l4_path` 後 tensor 保留完整 256 experts。

判讀方式：

| NOHOOK 結果 | 結論 |
|---|---|
| == 基線（198 / 27.18） | 權重 bytes 與 CPU residency 都沒問題 → 故障在 remap / pool / slab repoint |
| != 基線 | 故障在 hook 之下（residency、buffer type、loader 的 CPU-buffer 放置） |

補充：先前以為 `skip_load` 是「不載入權重」，實際讀 `llama-model-loader.cpp:1323` 後確認它
**只是把 expert tensor 放到 CPU buffer**（`buft = ggml_backend_cpu_buffer_type()`），權重照樣從
GGUF 讀進來。所以 NOHOOK 是一臂合法的實驗，不是必然崩壞的組合。

型別：`llama-model.h` 新增 `bool expert_cache_no_hook`，供 provenance 蓋章。

### 7.3 cache-OFF 真值參考（原 Step 0）+ 絕對閘門（原 Step 1）

`scripts/check/knifeedge_matrix.py`：

- 新增 `expert_cache_state(extra_env, pool_bytes)` → `on` / `off` / `off_nogather` / `on_nohook`。
  **從 launch 參數判定，不看檔名**——叫 `*_NOCACHE*` 但其實是帶 budget 跑出來的 dump 比沒有更糟。
- `_write_oracle_meta(..., expert_cache=...)` → sidecar 寫 `expert_cache` 欄位；
  沒給就寫 `"unknown"`，讓「沒記錄」與「記錄為 off」保持可區分。
- 新增 `_truth_guard(ref_path)`：cache-ON 或 NOHOOK 的參考一律拒絕當真值；
  **且 `--allow-stale-oracle` / `--allow-model-mismatch` 都無法繞過**——那兩個旗幟回答的是
  「這份參考舊不舊 / 是不是同一個檔案」，跟「這份參考是不是對照組」是兩個問題，
  把它們混在一起正是 19/19 的成因。
- `oracle_gate` / `oracle_recompare` 新增 `absolute=`；verdict 會蓋
  `comparison: "absolute" | "relative"`，之後沒有任何 19/19 能在不知道是哪一種的情況下被讀。
- 新增 CLI：`--oracle-abs`（絕對比較）、`--no-expert-cache`（budget 0 的對照臂）。
- `launch_pool_bytes(args, gb)`：`--no-expert-cache` → 0 bytes；三處 launch 呼叫全部改走它。
- `--no-expert-cache` 時跳過 pool feasibility gate（budget 0 沒有 pool geometry 可判），
  改用 `verdict: GROUND_TRUTH`。
- `record_provenance` 的 `pool` 加 `expert_cache` 與 `ground_truth` 欄位。

既有的 46 個 sidecar 已回填 `expert_cache: "on"`（附 `expert_cache_source` 說明；
依據是 `run_server.sh:587` 無條件帶 `-expert-cache`）。3 個沒有 sidecar 的參考
（`ref_iq3_pool8gb_M2_6144.jsonl`、`ref_iq3_pool8gb_STALE_pre-d222a1d6e.jsonl`、
`ref_iq4_pool8gb.jsonl`）會被擋在 `truth_unknown`——這是正確行為，它們本來就不能當對照組。

### 7.4 自測

新增 `scripts/check/oracle_truth_gate_selftest.py`（離線，不開 server、不載模型），
27 條全 PASS，其中 4 條專門釘住「兩個 `--allow-*` 不能繞過真值檢查」。
`scripts/check/feasibility_gate_selftest.py` 仍 ALL PASS。

### 7.5 怎麼跑

**注意：用 `/opt/homebrew/bin/python3`，不要用 managed python。** managed python 3.13.12 沒有 numpy，
`pool_feasibility` 會 import 失敗，可行性閘門只印 `UNKNOWN / No module named 'numpy'` 而不擋——
一個看起來像通過、其實沒算的閘門。系統 python3 有 numpy 2.5.2。

```bash
# 1) 生真值（expert cache OFF）。慢——沒有 bounded residency，不要拿這個數字當效能。
/opt/homebrew/bin/python3 scripts/check/knifeedge_matrix.py \
    --models iq3 --pools 8 --pool-cap 8 --m2 --gates-only \
    --no-expert-cache --dump-oracle
#   → Backup/knifeedge_matrix/oracle_iq3_pool8gb_nocache.jsonl
#     sidecar 會寫 expert_cache=off（--tag 未給時自動補 nocache，不會覆蓋同名 cache-ON dump）
#   若被 --min-free-pct 擋下：那是真的記憶體不足，先清 machine 再跑，不要用降門檻硬闖。

# 2) 用真值做絕對比較（cache-ON 的候選 vs cache-OFF 的對照）
python3 scripts/check/knifeedge_matrix.py \
    --models iq3 --pools 4,6,8 --pool-cap 8 --m2 --gates-only \
    --oracle-ref Backup/knifeedge_matrix/oracle_iq3_pool8gb.jsonl \
    --oracle-abs --kill-existing

# 3) 第四臂：只關 hook、保留 skip_load
LLAMA_EXPERT_CACHE_NOHOOK=1 ...（或 --extra-env LLAMA_EXPERT_CACHE_NOHOOK=1）
```

### 7.6 併入的另一批證據（另一路排查，2026-09-15 12:06）

同一時段的另一路排查把**資料層**幾乎全部排除了，這批結論直接改寫下一步的優先序：

- pool 資料、pread、remap 值、tensor 的 `data`/`buffer` 全部正確；
  `ggml_backend_tensor_get` 讀回來 GPU 端資料與 CPU 一致 → **「權重沒到 GPU」這個假說死了**。
- `dst_cur` 的 `[expert][token]` vs `[token][expert]` 佈局疑慮：原碼就是對的，修改等價。
- `down_combine` 只在 `n_tokens == 1` 時被呼叫 → prefill 走標準路徑，這條線索是假的。
- `kernel_mul_mv_id` 確實被呼叫（把輸出強制歸零後結果改變，證明它真的在算）。
- prefill 期間 `n_tokens == 2` 是正常的，`ne2=[8,2]` 不是異常。

剩下的嫌疑因此從「資料」上移到「**ids / 幾何 / kernel 參數**」：
`ne02`（slot 數 vs 256）、`nb02`（stride）、`ne20=`{8}、`ne21=`{2} 在
pool 路徑（`ne02 = n_slots` + remap 後的 slot id）與 M2 slab 路徑（`ne02 = 256` + raw id）
之間的交換。建議下一步先用 `CGC_MMID_MV_DBG` 把這四個數字在兩條路徑上各印一次對打，
再決定要不要進 Metal kernel。

⚠️ 待澄清：該路排查記「乾淨 baseline 問題仍然存在（argmax=248046）」，
但 11:52 的 `/tmp/oracle_nocache_ids.jsonl` 是 198 / 27.179。
兩者可能指不同配置（cache-ON vs cache-OFF），**必須用 7.5 的第 1 步重跑一次才能定案**——
這也正是 Step 0 存在的理由。

### 7.7 本次實跑結果：管線通了，量測沒跑完

`--no-expert-cache` 這一臂已端到端驗證到「server 真的以 budget=0B 起來」：

```
[start] .../Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf  port=8080  ctx=8192  ngl=99  budget=0B
```

且 sidecar 正確蓋章（下面這份是當次產出，因 server 未完成已刪除，此處留格式參考）：

```json
{"cap": "8", "expert_cache": "off",
 "launch": {"expert_cache": "off", "extra_env": [], "mtp": "1"}, ...}
```

`_truth_guard()` 對它回傳 `None`（接受為對照組）；對任何既有 cache-ON 參考回傳
`not_ground_truth`。**也就是說：Step 0 的「能不能產生可判定真值」這件事已經成立**，
只差真的把那 19 行 dump 跑出來。

本次沒跑完的兩個原因，都跟程式碼無關：

1. **記憶體**：本回合開始時 42% free，跑完一次失敗的載入後掉到 **11%**，
   harness 的 `--min-free-pct`（預設 40%）正確地拒絕了。這是真的壓力，不是誤報——
   13.66 GB 模型在 16 GB 機器上沒有餘裕。**不要用降門檻硬闖**：撐過載入也會在 swap 上跑，
   量出來的東西跟「pool 差異」無法區分。重開機或清出記憶體後再跑。
2. **`ps` 權限**：harness 用 `ps` 監控/回收 server（`server_procs` / `wait_health`），
   在我的執行環境裡被禁。這只影響我這邊，不影響你本機。

所以 **Step 0 的最後一步要你在本機跑** 7.5 的第 1 條指令。跑完後：

```bash
# 看真值的 step 0（cache-OFF 的答案到底是 198 還是 248046——7.6 那筆矛盾要靠這個定案）
head -1 Backup/knifeedge_matrix/oracle_iq3_pool8gb_nocache.jsonl | python3 -m json.tool | head -20

# 然後做絕對比較
/opt/homebrew/bin/python3 scripts/check/knifeedge_matrix.py \
    --models iq3 --pools 4,6,8 --pool-cap 8 --m2 --gates-only \
    --oracle-ref Backup/knifeedge_matrix/oracle_iq3_pool8gb_nocache.jsonl \
    --oracle-abs
```
