# mul_mat_id 幾何 vs ids 對打探測（2026-09-15）

## 為什麼要這個探測

expert cache 的兩條路徑在 `src0`（FFN expert 權重張量）上裝的是**不同的 `ne[2]`**：

| 路徑 | 觸發條件 | `ne02` | ids 的意義 |
|---|---|---|---|
| pool | `n_tokens <= CGC_POOL_MAX_TOKENS` | `slots_per_layer`（71 @4GB / 143 @8GB） | **slot index**（remap 後） |
| M2 whole-layer slab | `n_tokens > CGC_POOL_MAX_TOKENS` 且 `CGC_PREFILL_STREAM=1` | `n_expert`（256） | **raw expert id**（此分支不建 remap leaf） |

而 ids 是由 `llama-context.cpp` 的 host hook 寫的 —— 跟設定 `ne02` 的
`cgc_apply_expert_geometry()` **不同檔案、不同時機**。兩者一旦錯配：

- `ne02 = slots` + raw ids（0-255）→ 越界讀到別層的儲存 → 垃圾 token
- `ne02 = 256` + slot ids → 讀到「真實存在但錯誤」的 expert → logit 偏低、argmax 偏移

兩種都不會崩，都會通過「資料層」檢查（pread 成功、pool 非零、GPU 讀回與 CPU 一致），
因為被讀的那些 bytes 本身是好的 —— 錯的是**選了哪一塊**。

---

## 工具

### 1. `CGC_MMID_MV_DBG`（`ggml-metal-ops.cpp`，env-gated）

放在 MM / MV 分支判斷**之前**，兩條路都吃得到。`=1` 印一次完整 pass（150 筆）、`=2` 印 512 筆、`=3` 印 4096 筆。

每行內容：

```
CGC-MMID name=ffn_moe_gate-0 type=iq2_s
  | src0 ne=[2048,512,143] nb01=656 nb02=335872
  | ids ne=[8,2] nbi1=32
  | dst ne=[512,8]
  | ids=[64,0,125,112,6,89,100,39,73,1] id_min=0 id_max=125 id_oob_vs_ne02=0
  | fp(4096B/row) id64:cca0dc8042b74885 id0:f7e985a014aacf71 ...
  | src0_data=0x336cce300 src0_offs=502211328 ids_offs=0
```

四個對打數字就是 `ne02` / `nb02` / `ne20` / `ne21`。另外兩個是判讀的關鍵：

- **`id_oob_vs_ne02`**：任何 id ∉ [0, ne02) 就 +1。這是「幾何與 ids 直接錯配」的一票否決指標。
- **`fp`**：該 id 在 `src0` 上 `id*nb02` 處前 4096 bytes 的 FNV-1a64。
  這個數字讓「輸出不一樣」可以升級成「bytes 不一樣」。

### 2. `scripts/check/mmid_geometry_probe.sh`

多臂、同 prompt：`gather` / `gatherslab` / `nogather` / `nohook` / `nocache`。
產出 `Backup/mmid_probe/<arm>_<ts>.{log,mmid,resp.json}`。

### 3. `scripts/check/mmid_pool_vs_gguf.py`

把 `fp` 拿去跟 **GGUF 檔案**直接比對。

- pool 路徑：用 `CGC_EXACT_PATH_DBG=1` 印的 `uni[k]=expert -> slot` 表反查 slot→expert
- slab 路徑：`ne02 >= n_expert_file` 時 id 就是 raw expert id，直接查

```bash
/opt/homebrew/bin/python3 scripts/check/mmid_pool_vs_gguf.py Backup/mmid_probe/p250_*.log
```

---

## 實測結果

### Run A — 4GB pool，200-token prompt（pool 路徑）

```
src0 ne=[2048,512,71]  ids=[64,6,5,4,12,1,2,39,...]  id_max=64  id_oob_vs_ne02=0
```

- 150 筆 / 600 個 id-row，`id_oob_vs_ne02` **全為 0**
- pool vs GGUF：**36/36 OK**
- 輸出：`是的，巴黎是`（正確）

### Run B — 8GB pool，200-token prompt（pool 路徑）

```
src0 ne=[2048,512,143]  ids=[64,0,125,112,6,89,100,39,73,1]  id_max=125  id_oob_vs_ne02=0
```

- 512 筆 / 2048 id-row，oob 全 0；pool vs GGUF **39/39 OK**

### Run C — 8GB pool，`-b 6144 -ub 6144 -c 8192`，7343-token prompt（**兩條路都進了**）

| ne02 | records | id_max | oob | zero fps |
|---|---|---|---|---|
| 143（pool） | 240 | 142 | 0 | 2 |
| 256（slab） | 272 | 255 | 0 | 2 |

- pool vs GGUF：**941 筆 / 0 mismatch**（含 slab 路徑）
- 輸出：`Paris is the capital`（正確）
- prefill **198.96 tok/s**（7343 tok / 36.9 s）；第二個 chunk 206.66 tok/s

### 跨 run 交叉驗證（最有力的一條）

4GB 與 8GB 是兩次**獨立**的 run，slot 編號完全不同，但同一個 expert 的 fp 逐位元相同：

| expert | 4GB slot / fp | 8GB slot / fp |
|---|---|---|
| 64 | 64 `cca0dc8042b74885` | 64 `cca0dc8042b74885` |
| 161 | 6 `f7e985a014aacf71` | 0 `f7e985a014aacf71` |
| 125 | 5 `ec1059731d16886f` | 125 `ec1059731d16886f` |
| 112 | 4 `90b0ca95d38adf43` | 112 `90b0ca95d38adf43` |

slot→expert 的**恆等捷徑**（`e < slots` 時 `slot = e`）也被證實是對的：
71 slots 時 `39→39, 64→64, 73→0, 89→1, 125→5, 161→6`；143 slots 時 `64→64, 125→125, 112→112`。

---

## 結論

### 已否決

1. **「ids 與 ne02 錯配」不是根因** —— 2648+ 個 id（pool）+ 2176 個（slab）全部 `id_oob_vs_ne02 = 0`。
2. **「pool 內容錯」不是根因** —— 941 筆 fp 與 GGUF 逐位元相符，且跨 pool size、跨 run 穩定。
3. **「權重沒到 GPU」不是根因**（併入另一路排查的結論）—— CPU 端指紋本來就對。

### 新發現：極少數 expert row 讀到全零

零值 FNV 常數 = **`b43a063055adc383`**。

| run | 全零 id-row / 總數 | 分佈 |
|---|---|---|
| 4GB | 2 / 600（0.3%） | 全部 `il=1`，gate+up 為零、**down 不為零** |
| 8GB | 6 / 2048（0.3%） | 全部 `il=1` |
| p250 | 4 筆（pool 2 + slab 2） | — |

同 slot、同層，gate/up 是空的但 down 有資料 → **kind 專屬的 cold-slot**：
該層 gate/up 的 pool region 還沒填完就被 kernel 讀走，讀到全零。
發生在 prefill 之後的第一個 decode step。

這條跟「layer 0 差異 0.5%、逐層放大」的觀察相容：單一 expert 貢獻歸零，
在 layer 0 只動搖整體統計量的小數點後兩位，但誤差會在 40 層裡累積。

### 關於「數值精度累積」的結論 —— 不成立

> 8GB 輸出 0.3484 / NOGATHER 0.3467（差 0.5%），逐層放大，故為精度累積

**0.5% 不可能是浮點運算順序造成的。** FP32 重排的相對誤差在 1e-7 量級，
std 會吻合到 5-6 位有效數字；0.5% 大了五個數量級。
更何況本次實測證明：**同一個 expert 在同一套程式下的 bytes 是逐位元確定的**
（跨 pool size、跨 run 的 fp 完全相同）。既然輸入 bytes 相同、kernel 相同，
輸出就必須 bit-identical；觀測到 0.5% 差異，代表**輸入的 ids 或某一塊權重不同**，
不是累積誤差。上面那個全零 row 就是一種具體機制。

---

## 尚未覆蓋的路徑（下一步）

現有 4 次 run **都沒有重現**使用者回報的 271 vs 198。差異點依可疑度排序：

1. **MTP / speculative verify** —— 我的 probe 直接起 binary，沒開 MTP。
   使用者的配置是 `MTP=1 + denseIQ4X`。MTP verify batch 的 `n_tokens = n_draft+1 > 1`，
   remap 要對**所有 token 的 union** 寫入，走的是另一條 union 分支
   （`CGC_POOL_MAX_TOKENS` 的註解明確說 verify 批次會流經 pool 路徑）。
   **這是目前最可疑、也完全沒被檢查過的一條路。**
2. **`--jinja` + 不同 chat template** 造成 prompt token 數不同 → 決定走 pool 還是 slab。
3. **`CGC_OA_ASYNC` / `CGC_MV_FUSE` 等 fused 路徑**：`ggml_metal_op_mul_mat_id` 在 fuse 命中時
   會**提前 return**，我的 probe 在 fuse 判斷之後，所以 fused 路徑完全沒被看到。

建議下一步：

```bash
# MTP verify 路徑（最可疑）
CGC_PROBE_POOL_GB=8 CGC_MMID_MV_DBG=2 bash scripts/check/mmid_geometry_probe.sh gather
# 但要在 probe 腳本裡加上 MTP（denseIQ4X draft model + -ngld 99），
# 或直接改用 scripts/run_server.sh（它預設 MTP=1）並帶 CGC_MMID_MV_DBG=1
CGC_MMID_MV_DBG=2 CGC_SERVER_PROFILE=prefill250 ./scripts/run_server.sh
```

另外應把 `id_oob_vs_ne02 > 0` 與「讀到全零 row」做成**總是開啟的斷言**
（成本：每個 mul_mat_id dispatch 讀 64 個 int32，在 decode 上約 2k 次讀取/token，可忽略），
而不是只在 probe 開啟時才看得到。

---

## `run_server.sh` 的 `prefill250` profile（新增）

`CGC_SERVER_PROFILE=prefill250 ./scripts/run_server.sh` 現在會定案成：

```
CTX=8192  BATCH=6144  UBATCH=6144  PREFILL_STREAM=1  SLAB_CAP=256
```

（乾跑驗證過；全部用 `${VAR+x}` 守門，顯式環境變數永遠贏。）

注意 250 tok/s 的三件事必須**同時**成立，缺一不可：

- `n_batch = n_ubatch = 6144` → prefill 一次吃滿，不被切碎
- `CGC_PREFILL_STREAM=1` → 大 chunk 走 whole-layer slab 而不是 pool
- `CGC_GATHER_SLAB_CAP=256` → slab 裝得下全部 256 個 expert，否則 streaming 會被拒

本次 probe run（7343-token prompt、冷啟動）實測 **198.96 tok/s**，第二 chunk 206.66 tok/s ——
與 264.78 的差距來自 prompt 長度與 cache 冷熱，不是配置沒生效。
