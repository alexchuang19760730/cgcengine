# 「kernel 優化」的正確題目與施工計畫 —— Occupancy Project（2026-09-21 14:2x）

> 使用者問：「能，但它是『在維持逐位元相同的前提下把 Metal 的 quant GEMV 與整條 elementwise 重寫到接近峰值頻寬』的專案級工作 —— 你有方案嗎？」
> 本篇回答：**有，但題目要先改一個字。** 不做「重寫到接近峰值頻寬」（只剩 3.3×，而且大搬運者已經貼屋頂），
> 要做「**修正 dispatch 幾何／occupancy**」（獨立探針實測 13.6×，且 G2 相容）。
> 本計畫用到的每一個數字都註明來源；未實測的部分一律標「估」。

---

## 0. 一句話

| | |
|---|---|
| **錯誤題目** | 把 GEMV 重寫到接近峰值頻寬 |
| **正確題目** | 把每一個 kernel 的 dispatch 幾何改成能餵飽 GPU 的形狀 |
| **最重要的一條實測** | 同一個 1 MiB 冷載荷，grid 從 32 threads 拉到 8192：**168 µs → 12.4 µs（13.6×）**，而 payload 一個位元組都沒變（§EN-389） |
| ⚠️ **但 13.6× 不能當承諾** | 它是**合成 streaming payload 的探針上界**。真實 op 能吃多少，取決於「今天 grid 是不是已經把輸出元素吃滿」——那正是 K0 要量的第一件事（見 §2 的產出表） |
| **步時天花板（估）** | ~3.5–4×（由真實 roofline 反推，**這才是可引用的那個數字**；13.6× 只說明 mechanism 的量級） |
| **25 t/s 需要的量** | 步時 ×0.5，**落在天花板以內** |
| **是否碰 G2** | 主線 **不碰**，但**禁區要寫死**：永久禁用 `nxpsg`／`N_SIMDWIDTH`／K 分組，verify 全程鎖 `CGC_MM_BITIDENT=1`（見 §3.2.3） |
| **規模** | 4–6 週一人，**且前 2 週不必重建主 binary** |

---

## 1. 證據鏈（三個來源，彼此獨立）

### (a) 步級：不是算力、也不是頻寬 → 是 latency/occupancy（`joint_capture_20260920_2144.json`）

```
每步 RAM 流量 1.34 GB    GPU busy = 97.90 ms    ⇒  有效 13.6 GB/s  ≈ 峰值 ~11%
節點數 nodes_all=4076 / nodes_work=2455    ⇒  約 1621 個節點不產生任何工作（busy 0.00%）
```

### (b) op 級：每一個節點都在付一次固定成本（`attn_moe_split.py ops` 讀既有的 `20260919_191400.log`）

| OP | 佔 busy% | 節點 | µs/節點（推導值） |
|---|---:|---:|---:|
| `MUL_MAT` | 15.4 | 426 | 35–57 |
| `ADD` / `MUL` / `UNARY` / `CPY` / `GET_ROWS` | ~45 | ~1300 | 31–44 |
| `RMS_NORM` | 7.6 | 130 | 57–92 |
| `GATED_DELTA_NET` | 2.8 | 30 | 91–146 |

自洽檢查：**50 µs × 2455 nodes ≈ 123 ms ≈ 整個步時**
⇒ **步時 ≈ 每個節點付一次固定成本，跟它搬多少 bytes 幾乎無關。**

### (c) Metal 獨立探針：那筆固定成本的來源是 grid（`Backup/phase_decomp/metal_bw_roofline.m`，§EN-389，不起 llama）

| 數字 | 值 |
|---|---|
| DRAM 冷讀屋頂（≥4 MiB） | **≈91 GB/s**（規格 120，實測 streaming 76%） |
| 0.5 MiB 純 float4 冷讀＋加 | 75 GB/s |
| 0.5 MiB **3-bit 解量化＋MAC**（≈真實 GEMV 屋頂） | **45 GB/s** |
| 空 dispatch（同一 CB 內連續） | **3.28 µs** |
| commit + waitUntilCompleted 往返 | **164 µs** |
| **1 MiB 冷載荷 grid 32 → 8192** | **168 µs → 12.4 µs（13.6×）** |
| 64 KiB 冷載荷 grid 32 → 2048 | 11.5 µs → 1.14 µs（10.1×） |

**三條推論：**

1. 大搬運者（`MUL_MAT_ID` 60–73 GB/s）**已經貼著甚至超過** 45 GB/s 的解量化屋頂 ⇒ 那一腿沒得修。
2. 真正的主詞是 grid，量級 13.6×，比頻寬那 3.3× 大 4 倍。
3. ⚠️ 空 dispatch 只有 3.28 µs，但觀測到的每節點成本是 30–90 µs ⇒
   **「每節點成本」裡面一定有一個比 dispatch 更大的成分還沒被指名**（§2 的 K0 就是在指名它：GPU 執行時間 vs 主機端 encode/submit）。

---

## 2. K0（半天，0 行 kernel）：先把題目最後一個未知數關掉

**K0 唯一目的：把「每節點 30–90 µs」拆成 GPU 執行時間與主機端成本兩項。**
這個拆解決定未來四周走哪一條主線，**所以它是必須的第一步，不是可選的漂亮的**。

```sh
# 六個 env 全部已在 scripts/run_server.sh 的 allowlist 內（已查證），不用 bypass launcher
CGC_GPU_NODES=1 CGC_GPU_OPS=1 CGC_CB_N_MAIN=1 \
CGC_GPU_NODES_TRACE=1 CGC_GPU_TIMING=1 CGC_MM_DBG=1 \
  bash scripts/run_server.sh --profile prod25   # 須先確認 8080 空、無他線量測
```

- `CGC_GPU_OPS=1` 的 `uni` 欄 = **只含該 op 的 command buffer 上實測的精確 µs/node**（正控制，非按節點數推導）。
- `CGC_CB_N_MAIN=1` 把主 CB 的 `MAX(64, 0.1*n_nodes)` 下限打掉 ⇒ 更多 solo CB ⇒ `uni` 覆蓋率上升。
  ⚠️ 不要同時把 `CGC_SERVER_N_CB` 拉大：2026-09-18 實測 n_cb=127 會卡死 command buffer 建立，可用區間 ≤ ~16。
- `CGC_MM_DBG=1` 印 `MMDBG <family> ne11 ne00 ne01 type0 type1` ⇒ **直接拿到每一個 mul_mat 的形狀與它選了哪一族 kernel**。
- 解析器現成：`scripts/check/gdn_split.py`（已支援 `uni` 與最小二乘）＋ `scripts/check/attn_moe_split.py ops`。
- **K0 要跑兩份：baseline 與 `CGC_MM_BITIDENT=1`**（2026-09-21 新增）。
  因為 §3.2.3 規定 K1 全程鎖 BITIDENT ⇒ 我們**事先得知道這道鎖要付多少**；
  若它一開就吃掉 20%，K1 的收益是從「扣掉 20% 之後」才開始算的。
  ⇒ K0 的產出表要出成一對（baseline vs BITIDENT），差額就是保護逐位元相同要付的價錢。

**K0 產出（一張表，逐 op）：**

```
op | 形狀(ne00,ne01,ne11,M) | bytes/節點 | 實測 grid(TG × threads) | uni µs/節點 | 有效 GB/s | %roof | 步時佔比 | ★ threads/輸出元素
```

**★ 最後那一欄是 13.6× 能不能變現的仲裁欄**（2026-09-21 由使用者指出）：
合成探針拿得到 13.6×，前提是「原本 grid 太小、輸出維度還沒被切滿」。
⇒ K0 必須對每一個 op 算 `總 threads ÷ 輸出元素數`：

- **比值 ≥ 1（每個輸出元素已經分到一條以上 thread）** ⇒ 輸出軸已被切滿；再加 thread 只能去碰 K（禁區）
  ⇒ 13.6× **對它不適用**，它該走 K3（融合）或認列；
- **比值 ≪ 1（很多輸出元素擠在同一條 thread 裡）** ⇒ 輸出軸還有得切 ⇒ K1 對它有效，再進 K1-a 掃真實形狀。

⇒ **13.6× 是 mechanism 的量級，不是 K1 的收益預測。** K0 這一欄把「哪些 op 吃得到」先圈出來，
K1 的收益上界才會是一個加權數字，而不是一個盼望。

**決策表（寫死在跑之前，不做事後解釋）：**

| K0 結果 | 判定 | 主線 |
|---|---|---|
| `uni` ≈ 30–90 µs 且 `threads/輸出元素` ≪ 1 | **GPU 執行真的慢，而且輸出軸還有得切** ⇒ grid 假說成立 | **K1（occupancy）** |
| `uni` ≈ 30–90 µs 但 `threads/輸出元素` ≈ 1 | 慢在別處（多趟＋barrier、inside-kernel 串行），輸出軸已滿 | 該 op 改走 K3／單點優先 |
| `uni` ≪ 30 µs（例如 3–8 µs） | GPU 早就做完了，剩下的時間在主機端 encode / submit / commit+wait | **K4（主機側）**，K1 降優先 |
| 混合（大那一坨在 GPU、小那一坨在主機） | 兩條都走，先砍大的一坨 | K1 與 K4 並行 |
| 某幾個 op 單獨就佔走 ≥15% 步時 | 那個 op 直接單點突破，不必等全面計畫 | 專案化那一個 op |

⚠️ **K0 跑之前不能開始寫任何 kernel。** 這一條是本計畫最便宜也最容易為了「有進度感」而被跳過的一步，而它的代價是四週。

---

## 3. K1（主線，4–6 週）：修正 dispatch 幾何

### 3.1 為什麼它 G2 相容（這條是整套方案的胸口）

逐位元相同要求的不是「別動 kernel」，而是：

> **每一個輸出元素的部分和，仍由同一條串列加法鏈、在同一個資料型別裡產生。**

在此約束下可以任意改：thread↔輸出元素的映射、**threadgroup 大小、grid 大小、每群 simdgroup 數、
訪存合併寬度、dequant 查表的實作方式**、以及「中間值留在 register / threadgroup 而不落 DRAM」。
不能改的只有：split-K 後相加、任何 tree reduce（`simd_sum`／蝴蝶樹）、accumulator 型別、
把原本分開的 `mul+add` 換成 `fma`（或反之）、scale 的套用位置。

**而「加大 grid」正是沿輸出列／輸出元素切，不碰歸約軸。**

### 3.2 repo 內已有先例（不是我的推理，是這個 repo 寫死過的事實）

`ggml-metal-ops.cpp:2539-2560` 的 `CGC_MM_BITIDENT` 註解寫得很白：

> 「mul_mv's per-element mapping depends only on (tsrc0, ne00) — **ne11 only sets the grid row count** —
> so per-row results are M-invariant」

也就是：**「grid 的列數不影響逐位元結果」這件事，本 repo 為了 bit-identical 已經依賴過一次。**

⚠️ **2026-09-21 14:4x 由使用者更正，我逐行核對原始碼後重寫。**
上面那段註解其實講了**兩件事**，初版只取了對計畫有利的那一件。完整的兩件是：

1. ✅ **安全的一半**：沿輸出維度加 thread／threadgroup，且**每個輸出元素自己跑完整條 K 串列加總** ⇒ bit-exact。
2. ❌ **被禁的一半**：small-batch mat-mv family 依 `ne11`(=M) 選 `nxpsg`，而 `nxpsg` 會改 K 方向的歸約樹。
   註解原文：「The per-row K-reduction tree therefore changes with the batch size, so the SAME logical row
   yields ULP-different values.」

### 3.2.1 `nxpsg` 為什麼是硬禁區（原始碼，不是類比）

`ggml-metal.metal:4067-4085`，`kernel_mul_mv_ext_*_impl` 的列內歸約是一棵**照 `nxpsg` 逐層開關的二元樹**：

```metal
if (nxpsg >= 32) { sumf[ir1] += simd_shuffle_down(sumf[ir1], 16); }
if (nxpsg >= 16) { sumf[ir1] += simd_shuffle_down(sumf[ir1],  8); }
if (nxpsg >=  8) { sumf[ir1] += simd_shuffle_down(sumf[ir1],  4); }
if (nxpsg >=  4) { sumf[ir1] += simd_shuffle_down(sumf[ir1],  2); }
if (nxpsg >=  2) { sumf[ir1] += simd_shuffle_down(sumf[ir1],  1); }
```

⇒ **`nxpsg` 一改，樹的高度就改**；而它還是由 host 端 `ggml-metal-ops.cpp:2604-2637` 拿 `ne11`(=M) 選出來的
⇒ **今天的引擎什麼都不動，就會隨 batch 換樹。**

### 3.2.2 ⚠️ 鎖 `CGC_MM_BITIDENT=1` 買到的是「M-不變」，**不是「沒有 K 共享」**

核對 `mul_mv_q1_0_f32_impl` 的結果：它本身也把 K 分給多條 lane——

- `ggml-metal.metal:3737-3742`：`ix = tiisg/8`、`il = (tiisg%8)*16`、stride `N_SIMDWIDTH/8`（=4）
  ⇒ **照樣 4 條 lane 共享一列的 K**；
- `ggml-metal.metal:3760`：`const float tot = simd_sum(sumf[row]);` ⇒ 一樣是 tree reduce。

它之所以 M-invariant，只因為這個分割由 `(ne00, N_SIMDWIDTH)` 決定、**永遠不看 ne11**。
⇒ K1 要保護的不變量請寫成這一句（比「不碰歸約軸」更可執行）：

> **K 分割必須永遠是 `(tsrc0, ne00, N_SIMDWIDTH)` 的函數；
> 永遠不得是 M 的函數，也不得是新引進的 grid 形狀的函數。**

實務後果：**K1 在輸出軸加 thread 時若順手動了 `N_SIMDWIDTH` 或 `tiisg/8` 那個分組，照樣破位元——
而那一步看起來完全像是「只加 thread」。**

### 3.2.3 禁區清單（K1 施工規格的硬約束）

| 項目 | 判定 | 根據 |
|---|---|---|
| `nxpsg` | ❌ **永久禁用** | 改 `simd_shuffle_down` 樹形（`:4067-4085`），且由 ne11=M 選出 |
| `N_SIMDWIDTH`、`ix=tiisg/8`、`il=(tiisg%8)*16`、stride `N_SIMDWIDTH/8` | ❌ **永久禁用** | mul_mv 家族的等價開關（`:3737-3742` ＋ `:3760`） |
| `r1ptg` | ⚠️ **機制上安全，仍列禁區** | 每列有獨立 `sumf[ir1]` 累加器、不碰歸約；但它與 nxpsg 一起決定 pipeline variant 且由 ne11 選出 ⇒ 碰它就走回 M-dependent 路徑，而收益 ~0 ⇒ 當成便宜的保險 |
| `nsg`／沿輸出軸加 threadgroup | ✅ 允許（每次仍過 117/117） | 只決定「哪個 simdgroup 算哪一列」，列內 lane 分割不變 |
| verify ／ prefill-tail 路徑 | ✅ **K1 全程鎖 `CGC_MM_BITIDENT=1`** | 不讓 M=3–4 掉回 small-batch family；ULP drift 在 prefill tail 翻 argmax 是有前科的 |

### 3.3 施工順序（每一步都可退回）

| 步驟 | 做什麼 | 在哪 | 成本 |
|---|---|---|---|
| **K1-a microbench 先行** | 把既有的 `metal_bw_roofline.m` 長成 `metal_grid_sweep.m`：針對**真實形狀**（down: IQ1_S, K=2048/N=512；gate/up: IQ4_NL；norm: K=2048, T=1/4）**掃 grid**，秒級迭代 | `Backup/phase_decomp/` | 2–3 天 |
| **K1-b 加 A/B 旋鈕** | 今天 grid 由 pipeline 變體名決定（`mul_mv_ext_f32_f32_r1_<nxpsg>`，function constant `FC_mul_mv_nsg/nxpsg`）⇒ **沒有 env 可 A/B**（§EN-389 已踩）。加一個 `CGC_GRID=<policy>` 把 K1-a 找到的幾何做成可切換 | `ggml-metal-ops.cpp` 幾何計算處（2562–2666 與各 op 的 dispatch 行） | 1–2 天 |
| **K1-c 逐 family 回寫** | 一次只改一個 family，改完過數值閘門 | 同上 | 各 1–3 天 |
| **K1-d 端到端配對** | `scripts/check/paired_ab.py`，AB／BA，熱閘門視窗確認 | `scripts/check/` | 每次 0.5–1 天 |

**每一個進 mainline 的改動都必須自帶 A/B 旋鈕** —— 這是本次明確學到的教訓：
沒有旋鈕就沒辦法證明是自己變快，只能證明「重建之後不一樣快」。

### 3.4 天花板怎麼算（含假設，K0 之後要換成實測）

```
1.34 GB/步，其中約 62% 走「解量化 + MAC」 ⇒ 0.83 GB ÷ 45 GB/s  = 18.4 ms
其餘      0.51 GB 走單純 elementwise stream ⇒ 0.51 GB ÷ 75 GB/s =  6.8 ms
                                            屋頂地板  ≈ 25 ms
今天 GPU busy 97.9 ms  ⇒  天花板 ≈ 3.5–4×
```

⇒ **25 t/s 只需要「步時 ×0.5」，也就是這個天花板的一半不到。**
（目標函數一律用「步時」，不用 t/s：t/s 還乘著「每步吐幾個 token」（MTP accept），那是另一個變數，不該混進來。）

### 3.5 例外：沒有 parallelism 軸可以開的那一類

`RMS_NORM` 這類「歸約軸 = hidden、輸出只有 T 列」的 op **先天沒有 grid 可開**（7.6% busy、只搬 ~64 KB）。
⇒ 那一類只能靠**融合**（把多 op／多列併進同一個 dispatch），而它的天花板 G4 已經量過：**11.5%**。
⇒ K1 不要浪費時間試圖用 grid 修它；它歸 K3。

---

## 4. K2 / K3 / K4（與 K1 同層，按 K0 結果排序）

| Stage | 內容 | 既有基礎 | 天花板（已知） |
|---|---|---|---|
| **K2 bytes 那一腿** | K0 若發現有大筆 bytes 是可以不搬的（CPY、GDN conv state、KV maintenance、layout 造成的副本）⇒ 直接拿掉 | §EN-387 已指名 120 個冗餘 CPY（≤22%），尚待定點 | ≤22%（未定點） |
| **K3 融合 RMS_NORM 類** | 多 op 併 dispatch，串序累加保 G2 | `docs/G4_FUSION_INVENTORY_2026-09-20.md` | **11.5%**（< G4 門檻 12.8%） |
| **K4 主機側** | K0 若判定時間花在 encode/submit ⇒ 消掉逐 node 的 `waitUntilCompleted`、CB 數量權衡、encoder 重用 | 已量：commit+wait 往返 **164 µs**；既有 `-np` per-job overhead 計畫 | 未知，但**只要 GPU 那一腿被砍成功，它就會變成新的主項** |

⚠️ **K1 成功的副作用要提前預期**：GPU busy 從 97.9 ms 掉到 ~30 ms 之後，
今天只佔幾 ms 的主機端／池 IO（`cb` 9.2% 那一腿）會瞬間變成主項。
所以 K1 與 K4 不是「先後」關係，是「誰先撞牆就先修誰」。

---

## 5. 工作隔離（不做這一步就不能動工）

已知的共用資源 hazards（2026-09-17 的教訓）：**`cmake --build` 會蓋掉別條線正在 map 的 `.dylib`**（tree 上現在有 872 行未提交的引擎改動）。
這也是過去幾次「明明有方向卻不能動工」的唯一原因。解法是把 builder 隔離：

```sh
cd /Users/alexchuang/Documents/flashkv-devserver
git worktree add /Users/alexchuang/Documents/flashkv-kernel -b kernel/occupancy
# 該 worktree 有自己的 src/llama.cpp/build ⇒ 重建不再碰別人的 .dylib
```

- 無 submodule（`src/llama.cpp` 是 in-tree，會跟著過去）⇒ worktree 自足。
- 剩餘磁碟 43 GiB，但要先把 `build/` 整個重建一次（或 rsync 一份現有 build 過去省時間）。
- 量測仍然視窗共享：**動手前後各跑 `git status --porcelain -uall`，起 8080 前先看有沒有他線的量測行程。**

---

## 6. 門檻、退場與判準（寫死，跑之前就簽）

| 項目 | 規則 |
|---|---|
| **數值閘門（檢定力無限）** | 任何進 mainline 的 kernel 改動必須 **117/117 逐位元相同**。仲裁者是閘門，不是推理。這是唯一不必擔心檢定力的閘門：它是**精確比較**，不是統計量。 |
| **效能門檻** | 單一 family 的改動要 ≥ **3% 步時**下降才值得進 mainline |
| **⚠️ 門檻必須跟 reps 一起訂**（2026-09-21 由使用者指出：單臂 ±27% 噪音下，「≥3%」不綁 reps 就形同虛設） | ① 先用 **5 對 AB/BA** 當 pilot，量 per-pair ratio 的殘留 SD；② 需要的對數 `n ≥ ceil((k·SD / 3%)²)`，`k = 3`；③ `n > 15 對` ⇒ **提高門檻或放棄這個改動**，不要用跑不完的 reps 拖著；④ 一次 session 塞不下就**跨 session 跑**（這台機器今天連 4 臂都跑不完，熱配額是硬限制），但每個 session 都要重跑 sentry 自證窗口 HEALTHY；⑤ 配對降低的是 **SD，不是 n** —— `paired_ab.py` 證明過它能消掉熱漂移，但不能代替 reps。 |
| **證明方式** | 一律配對（`paired_ab.py`，AB／BA），**不用單臂點估計** |
| **旋鈕義務** | 每個改動都要能用 env 開關，否則不能合併 |
| **退場** | 若 K1-a 的 microbench 在真實形狀上拿不到 ≥2×（對比今天的幾何）⇒ 停手，報告，不要做 K1-c |
| **已知風險** | ① 真實 op 是「多趟＋barrier」（norm 要兩趟）而非純 grid 問題；② occupancy 拉高後撞 45 GB/s 屋頂，收益提前飽和；③ 為了把中間值留在 register／threadgroup 而壓低 occupancy，融合反而更慢；④ CPU/GPU 主項翻轉。 |

---

## 7. 與既有的其他路線怎麼比（為什麼值得開）

今天已實測、且不用碰 G2 的手段：**融合 ≤11.5%、prefetch 輸在 bytes、IO 形狀 ≤1%、超訂 ~6%，合計 ~19%**。

⇒ 那 19% 是「確定的但不足以達標」；K1 是「不確定但天花板 3.5–4×」。
兩者不互斥：**K1 是把主項打掉，其餘是把剩下的零頭收乾淨。**
順序上 K0 → K1-a 是必須先做的，因為它決定「其餘那些」到底還值不值得做。

---

## 8. 現在可以做什麼（三選一，需要使用者批准）

1. **跑 K0**（約 0.5–1 小時，不需要重建、不改碼）：目前機器是空的（8080 無 listener、無他線 llama 行程、free 76%）⇒ **窗口現在是開的**。
2. **開 worktree**：先把 builder 隔離出來，之後任何重建都不再卡別人（約 10 分鐘＋一次完整 build）。
3. **維持只出規劃**：本份先作為文件落地，等乾淨窗口再一次做 1+2。

---

## 9. 本篇的所有數字與其來源

| 數字 | 來源 | 狀態 |
|---|---|---|
| 1.34 GB/步、GPU busy 97.90 ms、13.6 GB/s | `Backup/phase_decomp/joint_capture_20260920_2144.json` | 實測 |
| nodes 4076/2455、op 佔比表 | `Backup/cgc_logs/llama_server_20260919_191400.log` | 實測（µs/節點為按節點數**推導**） |
| 一個 dispatch = 0.0736%/步 | `docs/G4_FUSION_ANSWER_2026-09-20.md` | 實測（配對，n=1/臂） |
| 屋頂 45/75/91 GB/s、空 dispatch 3.28 µs、commit+wait 164 µs、grid 13.6× | §EN-389，`Backup/phase_decomp/metal_bw_roofline.m` | 實測（**獨立 Metal 探針，不含 llama**） |
| 各 op 的 bytes/節點、1.34 GB 的 62/38 分配 | 由幾何（hidden 2048／inter 512／experts 256／layers 41）**估** | ⚠️ 估測，K0 要用實測換掉 |
| 120 冗餘 CPY ≤22% | §EN-387（門牌號與數量對，但「冗餘」與「22%」兩處待更正） | 待定點 |
| 「grid 列數不改逐位元結果」 | `ggml-metal-ops.cpp:2539-2560` 既有註解 | repo 既有事實 |

---

## 10. 機率評估：這條路打到 25 t/s 的可能性（2026-09-21 15:5x 追加；15:5x 依「統一 llama-bench」重算）

> 主觀機率，但每個乘數綁一個可觀測量，上修／下修條件預先註冊（§10.6）。
> 目的：讓「要不要投 4–6 週」變成帶賠率的決定，而不是願望。

### 10.0 口徑（唯一，不得混用）

**所有 t/s 一律 llama-bench**（使用者約定，2026-09-18 起；2026-09-21 重申）。

| 工具 | 儀器 | 能不能拿來算 t/s |
|---|---|---|
| `scripts/check/prod_profile.py`、`profile_duo.py`、`paired_ab.py` | **llama-bench** | ✅ 唯一可引用 |
| `scripts/check/http_duo.py` | HTTP | ❌ 只研究儀器間差異 |
| `scripts/check/decode_sweep.py`、`llama-server` log | **server** | ❌ ms 級步時**只做機制診斷** |
| `Backup/phase_decomp/poolsize_ab.py` | **server（起 server 聽 port）** | ❌ 同上 |

- **禁止跨儀器相乘**：llama-bench 的 t/s × server log 的 ms ⇒ 結果不可引用（本節上一版就犯過，已撤銷）。
- **引用格式**：數字 ＋ cell 口徑（`--reps`／`--batch`／`--ctx-size`／`--warm-skip`／`--spec-type`），缺一不可引用。

### 10.1 需要多少（純 llama-bench 比值）

| 項 | 值 | 來源 |
|---|---|---|
| 今天交付 decode | **12.57 t/s** | `Backup/prod_profile/prod_profile_decode_20260920_2041.json`（llama-bench；cell = `--prompt 0 --gen 128 --depths 512 --batch 512 --ctx-size 4096 --warm-skip 64 --spec-type draft-mtp`，reps 3） |
| 25 t/s 需要 | **1.99×** | 兩端同一 cell 的**純比值** |

⚠️ **上一版犯的錯（已撤銷）**：曾用 12.57（llama-bench）× 85.65 ms（server log）推「1.08 token/步」
⇒ **跨儀器相乘**，結論不可引用。「MTP 今天貢獻多少」降為**待確認項**（見下）。

⚠️ **12.57 是否真的含 verify，尚未定案**：`decode_sweep.py:325` 註解稱「llama-bench cannot speculate」，
而 prod_profile 的 cell 帶 `--spec-type draft-mtp`。不影響 1.99× 這個比值（兩端同 cell），
但影響「MTP 軸還能不能加」⇒ 列為 **K0 附帶確認項**（關 MTP 再跑一次同一 cell 即可）。

### 10.2 已知槓桿的天花板（**標注儀器**；不同儀器不可混加）

| 槓桿 | 上界 | 儀器 | 能否用於 t/s 外推 |
|---|---|---|---|
| 融合（減少 dispatch 數） | ≤11.5%（最大單一集群 5.3%） | **llama-bench**（`paired_ab.py`） | ✅ **可用** |
| 拿掉記憶體超訂 | +5.8%（3/3 同號） | **server**（poolsize_ab 起 server 聽 port） | ⚠️ 待以 llama-bench 重測 |
| 冗餘 bytes（120 CPY） | ≤22% | 未定點 | ❌ 不可用 |
| per-op command buffer 數 | **0%** | server | ✅ 是否定結論，可用 |
| 權重搬運（即使免費） | 8.8% 步時 | server log（122 ms 步時，§EN-380） | ⚠️ **機制診斷**：只證明「頻寬不是槓桿」，**不是一條可引用的 t/s 折扣** |

⇒ **嚴格只算 llama-bench 可用的那一項：融合 ≤11.5%。**
⇒ 上一版「已知槓桿打滿 ≈ 1.36×」**降級為機制上界估**，不是可引用的 t/s 預測。

### 10.3 缺口（改用倍率，不用 ms）

`1.99 ÷ 1.115 = 1.785` ⇒ **K1 必須獨自交付 ~1.79×（砍掉 ~44% 步時）**。

（上一版寫 1.46×，是因為把 server-log 來的折扣也算進去了；**統一 llama-bench 之後門檻變嚴，不是變鬆。**）

### 10.4 機率（重算後下修）

```
P(25) = P(K0 顯示主導 op 真有 occupancy 問題) × P(K1 吃到 ≥1.79×) × P(其他槓桿也中)
      ≈ 0.40          × 0.12          × 0.50   ≈ 2.4%
```
加上「還有一個成分沒被指名」的尾部（G4 的 0.0736%/dispatch 與 2455 node 線性外推得 181% > 100%
⇒ 那個矛盾本身就是有東西沒被指名的證據）⇒ **+4–7%**。

⇒ **P(25) ≈ 5–10%**（統一 llama-bench 前是 10–15%，**下修**）。

### 10.5 中央情境（比機率更重要）

**15–17 t/s（1.2–1.35×）** —— 仍跨過舊的 14–16 封版的下緣，是一場實質勝利。
⇒「成功」的定義要先講好，否則 4–6 週後會把一場勝利報導成失敗。

### 10.6 會讓我改口的觀測（預先註冊，不得事後解釋）

| 方向 | 條件（K0 產出表要同時滿足） |
|---|---|
| **上修到 25%+** | ① 前 10 大 op 的 `uni` ≥ 30 µs；② 那些 op 的 `threads ÷ 輸出元素` ≪ 1；③ 合計 ≥ 50% 步時 |
| **下修到 <3%** | `threads ÷ 輸出元素` ≥ 1，或 `uni` ≪ 10 µs ⇒ K1 沒有對象 ⇒ **25 當場死，連 20 都難** |

### 10.7 結論：K0 的價值不是推進，是**算賠率**

半天、0 行 kernel。它把一個 4–6 週專案的勝率從「不知道」變成 **5–10%** 或 **<3%**。
**在 3% 與 10% 之間，值不值得動工是兩個不同的決定。**

### 10.8 統一 llama-bench 之後產生的兩個待辦（都不是 K0 的前置）

1. **重測「記憶體超訂 +5.8%」**：poolsize_ab 走的是 server 路徑 ⇒ 要麼改成 llama-bench 臂
   （`profile_duo.py` 加 big/small 兩臂，改 `-expert-cache` ARGV），要麼明確標為「機制線索，不進 t/s 表」。
2. **確認 12.57 含不含 verify**：同一 cell 跑一次關 MTP 的對照；順便把「MTP 軸還能不能加」定案。

---

## 11. K0 已跑完（2026-09-21 16:2x）：**命中下修條件，K1 結案為「不做」**

> 完整報告：`docs/K0_RESULT_2026-09-21.md`（4 臂 server 量測、0 行 kernel、0 重建）。
> 本節只做一件事：把 §10 的預註冊判決表**填結果**，不修改判決規則本身。

### 11.1 判決

| §10.6 預註冊的條件 | 實測 | 命中 |
|---|---|---|
| 下修：`threads ÷ 輸出元素` ≥ 1 | **16–128**（每一個觀測到的 decode 形狀） | ✅ **命中** |
| 下修（另一條）：`uni` ≪ 10 µs | 13 µs 固定／11–25 µs 邊際 | ❌ 未命中 |
| 上修：前十 op `uni` ≥ 30 µs 且 `threads÷輸出元素` ≪ 1 且合計 ≥ 50% | 前者部分成立、**後者反向** | ❌ 未命中 |

⇒ **P(25)：5–10% → <3%**；**中央情境：15–17 → ~14–15 t/s**；**K1 建議不做。**

### 11.2 為什麼 13.6× 不適用（機制已指名）

那個合成探針是**純 streaming payload、沒有歸約軸**。真實 GEMV 的 32 條 lane 就是在**切 K**，
而輸出軸早已被完全切開（每列一個 simdgroup）⇒ 要再加 thread 只能切 K ＝ §3.2.3 的永久禁區。
（`nsg` 加大也不增加總 thread 數，只是把更多列塞進同一個 threadgroup。）

### 11.3 三條附帶收穫（都不是 K1 的收益，但都有價值）

1. **上鎖的價錢 ≈ 4–5%**（`CGC_MM_BITIDENT=0` vs `=1`，四指標同號）。
   ⚠️ 且 `run_server.sh:2187` **預設就強制 BITIDENT=1** ⇒ 這 4% **今天已經在付**，
   而 G2 禁止退掉它。這是 bit-exactness 的定價，不是待榨的收益。
2. **K4（主機側）被否證**：`gpu_union` = 步時的 92%（102.0 / 111.6 ms），`busy_sum/union = 1.17`
   ⇒ GPU 跨度吃掉整個步，主機端沒在空轉。
3. **免費的重現性讀數**：同配置連跑兩次，`gpu_union` 差 **0.2%**（102.0 vs 101.8）
   —— 遠優於 t/s 的 ±27%。⇒ **`gpu_union` 是比 t/s 更靈敏的配對指標**，值得進 `paired_ab.py`。

### 11.4 方案狀態

| 階段 | 處置 |
|---|---|
| K0 | ✅ **已完成，結論為負** |
| **K1（grid 幾何）** | ❌ **結案：無對象，不做**（4–6 週的投資不再建議） |
| K3（融合） | ⬆ **升為唯一主力**：≤11.5%，是目前唯一有 llama-bench 口徑上界的槓桿 |
| K2（冗餘 bytes） | 保留：≤22%，但連「是不是真的冗餘」都還沒證明 |
| K4（主機側） | ❌ 否證 |

### 11.5 方法論遺產（`uni` 欄不可用）

`CGC_GPU_OPS` 的 `uni`（unique/isolated 節點時間）**正控制沒有開火**：24 個 op 裡 23 個是 `-1`。
⇒ 方案 §2 的 K0 產出表那一欄**不能靠 `uni` 填**；後續要量 per-op 成本得換儀器
（`CGC-NSM` 的 per-buffer `dur_ns` ＋ node 區間是目前可用的替代，但 per-kind 最小二乘會 collinear）。
