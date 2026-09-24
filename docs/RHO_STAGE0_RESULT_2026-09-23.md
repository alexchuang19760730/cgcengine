# ρ 實測：提前一層發起 fill 的重合率（2026-09-23 實跑）

結論一句話：**ρ 量到了，`cov_uni = 0.854`（ntok=4，2040 層，skip=0），過門檻 0.398
而且超過窗口飽和點 0.70**。這是本專案到目前為止第一個「造重疊窗口」方向裡**通過 gate** 的。
同一輪還修掉並重測了 prebind —— **上一輪「prebind 結案 = 不做」的判決作廢**（儀器 bug，見 §5）。

log：`/tmp/prebind_probe/run_2026-09-23_145819.stderr.log`
判決：`python3 scripts/check/rho_probe_parse.py <log>`（selftest **21/21**）

> ### ⚠ 2026-09-23 16:45 追加：本輪量的是「準度」，**沒有量到窗口**（位置 bug，已修）
>
> 那一輪的影子 router 建在 `attn_post_norm` **之後**（`qwen35moe.cpp` 舊 :216 之後）。
> ggml 的**執行序 = 建圖序** ⇒ 那個 node 在 GPU 上排在 `attn_norm` / attn / residual /
> post_norm **全部後面**，離真 gate (`build_layer_ffn`) 只差幾行 ⇒ **提前量 ≈ 0**。
>
> - **`cov_uni = 0.854` 仍然成立**：它是「近似 logits vs 真實 logits」的比較，與該 node
>   排在圖裡哪裡無關（同一組輸入、同一組權重）。§3 的 PASS 不用撤回。
> - **但「窗口」那一維在本輪是完全空的**：從這一輪的 log **讀不出任何 lead**。
>   拿舊位置的 log 去量「可發起時點在 segment 的百分位」會得到 ~0，然後用「建圖順序」
>   的理由把方向殺掉 —— 與 prebind 那個 token-0 儀器 bug 是同一類假陰性。
> - **已修**（2026-09-23 16:45）：影子分支移到 `inpSA = inpL` 正下方，成為 layer L 的
>   第一個算子；nextn 的 `inpSA` get_rows 也一併提前，避免多一顆 node、並保證 ρ 用到的
>   張量與真實路線逐位元相同。數學一字未改。舊位置保留為對照臂 `CGC_RHO_PROBE_LATE=1`
>   （同一支 binary、同一組權重、ρ 應完全相同，只有 GPU 位置不同）。⇒ 詳見 §8。
>
> ✅ **2026-09-23 17:3x 補完（§9）**：新／舊位置各跑 3 趟，lead 實測
> **EARLY 1.31~2.27 ms vs LATE 0.41~1.27 ms**（影子從 segment 的 80.4% 搬到 45.1%）。
> 代回定價：cb=42.04（同 regime）⇒ **14.39 t/s（+14.5%）**，且這一格已被實測關起來
> （視窗不綁）。「逐位元相同」那個判據做不到，已換成噪音帶判據 —— 見 §9.1。

---

## 1. 量的是什麼

真實 route(L) 的輸入是 `norm(h after attn(L))`。若在 **MoE core(L−1) 一結束**就改用
「還沒過 attn(L) 的殘差」`inpSA` 走同一個 post-attn norm + 同一個 `gate_inp` matmul，
就能在 MoE core(L−1) 之後立刻發起 fill(L)，讓 CPU 的 pread 落在 GPU 還在忙的那段時間裡。

兩個量：

| 量 | 定義 | 用途 |
|---|---|---|
| `rho_tok` | 每個 token 的 \|近似 top-8 ∩ 真實 top-8\| / 8 | 路由本身像不像（診斷） |
| **`cov_uni`** | \|近似 union ∩ 真實 union\| / \|真實 union\| | **能省多少 fill ⇒ gate 打這個** |

與 prebind 的本質差別：prebind 是**跨 token 猜**；這是**同一步、同一批 token、只差一個
submodule 的真實 matmul** ⇒ 不必加寬 ⇒ 幾乎沒有過度抓取（見 §4）。

### 怎麼量的（不改行為）

在 `qwen35moe.cpp` 的層迴圈裡加一條**影子分支**：

```cpp
rho_pre    = build_norm(inpSA, attn_post_norm, ...)   // 同一個 norm，同一個 gate weight
rho_logits = build_lora_mm(ffn_gate_inp, rho_pre)
cb(rho_logits, "cgc_rho_logits", il)                  // 命名後 eval cb 才認得
ggml_build_forward_expand(gf, rho_logits)              // 進圖，但不接回主圖
```

`expert_cache_eval_cb` 把它的值讀出來，`expert_cache_on_topk` 拿真實 top-k 比。
**它不觸發任何 fill／ensure／prefetch**，純量測。開關 `CGC_RHO_PROBE`。

---

## 2. 實測（ntok=4 交付形狀，2040 層）

| | 值 |
|---|---|
| `cov_uni` | **0.8542** |
| `rho_tok` | 0.8257 |
| `uni_avg`（真實 union） | 19.36 |
| `pred_uni_avg`（預測 union） | 19.97 |
| skip（沒量到的層） | **0** |

各形狀（同一支 log）：

```
ntok=1  layers=40    rho_tok=0.8281  cov_uni=0.8281
ntok=2  layers=1080  rho_tok=0.8247  cov_uni=0.8412
ntok=3  layers=200   rho_tok=0.8369  cov_uni=0.8647
ntok=4  layers=2040  rho_tok=0.8257  cov_uni=0.8542   <= 交付形狀
```

⇒ 殘差流局部性很強：**跳過 attn 不換 token，只差一個 submodule，top-8 就有 82.6% 相同。**

---

## 3. 判決

門檻是**看到數字之前**就寫死的（§EN-464）：

| 門檻 | 值 | 結果 |
|---|---|---|
| step −3% | `cov_uni >= 0.164` | 過 |
| **step −10%** | **`cov_uni >= 0.398`** | **過（2.1 倍餘量）** |
| 窗口飽和 | `cov_uni > 0.70` | **過 ⇒ 再準也換不到時間** |

⇒ **PASS（step −10%）**。而且因為已過飽和點，**瓶頸從「準不準」變成「窗口有多大」**。

### 上界（視 `cb` 口徑）

每層 fill = `cb/40`；窗口 = segment 的後 40%（MoE core 佔 60.1%）= 1.30 ~ 1.59 ms。
移出量 = `min(窗口, cov × 每層 fill)`，再減插入的 gate+topk 4.76 ms/步：

| `cb` 口徑 | 每層 fill | 移出/層 | 淨省 ms/步 | 換算 |
|---|---|---|---|---|
| 42.04（同 run） | 1.051 | 0.898（覆蓋受限） | 31.15 | **14.38 t/s（+14.4%）** |
| 74.18（分解式） | 1.855 | 1.300（窗口受限） | 47.24 | **15.53 t/s（+23.5%）** |
| 74.18 + span 口徑窗口 | 1.855 | 1.584（覆蓋受限） | 58.60 | **16.46 t/s（+30.9%）** |

⇒ **+14% ~ +31%（12.57 → 14.4 ~ 16.5 t/s）**。這個區間的寬度本身是未閉合項：
兩個儀器對同一個 `cb` 差 1.8×（42 vs 74），還沒有判決誰對。

---

## 4. 對照：prebind（同一輪重測）

`rho` 路線不只比較準，而且**幾乎不過度抓取** —— 這是它勝出的關鍵：

| 方案 | 覆蓋率 | 預測 union / 真實 union | 每步 slot touch（實際需 774） |
|---|---|---|---|
| **rho 路線** | **0.854** | **1.03×** | 799 |
| prebind qu1（1 步） | 0.589 | 0.90× | 680 |
| prebind qu2（2 步） | 0.690 | 1.27× | 952 |
| prebind qu3（3 步） | 0.726 | 1.53× | 1152 |

prebind 要做到 0.69 的覆蓋，得抓 **1.27×** 的量（多 178 次 pread/步 + LRU 污染）；
rho 路線做到 0.854 只抓 **1.03×**。**rho 全面壓過 prebind。**

---

## 5. ★ 上一輪 prebind 的「FAIL」作廢（儀器 bug）

`docs/PREBIND_STAGE0_RESULT_2026-09-23.md` 與 `MEMORY.md` 寫的
「q24 = 0.354 vs 門檻 0.579 ⇒ FAIL，差 39% ⇒ 結案不做」**是錯的**。

根因：probe 只存了 `ids[0..k−1]`，也就是 **token 0 那一行**。但 MTP decode 的一個 step
的 ntok=4 是 **4 個不同位置**的待驗證 token，相鄰兩個 step 的 token **完全不相交**
⇒ 「上一步的 token 0」是**另一個 token**，不是同一個 token 的上一個狀態。

修正後（存全部 token 行）：

| 口徑 | 數值 |
|---|---|
| 舊（只 token 0，重現第一輪的錯） | `qt8=0.3613 qt16=0.3950 qt24=0.4414`（寬 13.2） |
| **新（整步 union）** | **`qu1=0.5887`(寬 17.0) `qu2=0.6901`(寬 23.8) `qu3=0.7257`(寬 28.8)** |

⇒ 同一支 binary 上兩種口徑都印（`CGC-PREBIND-SUM2`），舊的留著對照，避免「修好了」
被當成「數字變好看」。**結論：prebind 不是「證偽」，是「比 rho 差」** —— 覆蓋較低而且
過度抓取 1.27~1.53×。

---

## 6. 未決 / 風險（下一步）

1. **窗口的實際大小沒在本輪量到。** 1.30~1.59 ms 是既有 segment 形狀量測推的
   （MoE core 佔 60.1%）。影子分支真正的「可發起時點」在 segment 的哪個百分位，
   要另外量 —— 它現在是上界的主導項。
2. **本輪 t/s 不可引用**：圖裡多一個 norm + 一個 matmul，而且 `CGC_TD_CB` 會把 async
   pipeline 序列化。只取幾何／命中率。
3. **插入成本 4.76 ms/步**是照既有 route 的實測估的；真的做還要加一次 host read
   （256×ntok floats/層）。
4. `cb` 的 42 vs 74 口徑分歧沒解決 ⇒ 上界只能給區間。
5. **`qu1 = 0.589` 這個數字值得單獨看**：寬度 17.0 < 真實 union 18.8，等於「只用上一步
   的 union、不擴寬」就有 0.589 —— 比第一輪那個 0.354 高 66%。

---

## 7. 怎麼重跑

```bash
./scripts/check/prebind_probe_run.sh          # 會自己帶 CGC_RHO_PROBE / CGC_TD_CB
python3 scripts/check/rho_probe_parse.py /tmp/prebind_probe/run_<ts>.stderr.log
```

⚠ **`CGC_TD_CB=1` 是必須的**：這個 repo 走 CGC 分段派送器（`ggml-backend.cpp` ~2490/3010），
只把「每段最後的 top-k 節點」送進 eval callback（`ask=false`）；其它節點只有在
`CGC_TD_CB` 存在時才逐段轉發。不開它，`CGC-RHO-CAP` 會是零行而 skip=100%（實測過）。
它是張量名過濾器，設成 `1` 就不會真的 dump。

⚠ **stamp 判據用「有前進」不要用「兩邊相等」**：`CGC_TD_CB` 的逐段轉發會把同一節點
轉發多次（實測 shadow_stamp=3 vs hook_stamp=2），嚴格相等會把 100% 的層判成 stale。

---

## 8. ★ 位置修正（2026-09-23 16:45）：探針的位置就是它要量的那個量

### 8.1 這次的錯是什麼

要量的不只是「ρ 準不準」，還有「**能在多早發起**」。後者完全由影子 node 在 GPU 上的
落點決定，而那個落點在建圖時就定死了：

| | 舊位置（`attn_post_norm` 之後，舊 :216） | 新位置（`inpSA = inpL` 正下方） |
|---|---|---|
| 圖中順序 | attn_norm → attn → residual → post_norm → **rho** → gate | **rho** → attn_norm → attn → … → gate |
| 距真 gate | 幾行（≈ 0 算子） | 整個 attn(L) 的全部算子 |
| 能量到什麼 | **只有準度** | 準度 **＋ 提前量** |
| ρ 值 | 0.854 | **應完全相同**（同一組輸入、同一組權重、無 attn 依賴） |

⇒ **準度結論被位置影響的程度是零，窗口讀數被位置影響的程度是全部。**
這正是它危險的地方：一次其實無法回答問題的實驗，看起來「跑完了、有數字、過 gate」。

### 8.2 改了什麼（只動 `src/llama.cpp/src/models/qwen35moe.cpp`）

1. 影子 router 上移到 layer L 的最前面（緊貼 `ggml_tensor * inpSA = inpL;`）。
2. nextn 的 `inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids)` 一併上移。
   它原本和 `cur` 那行寫在一起、在 attn **之後**；它與 attn 無依賴，上移不改值，
   但能讓 ρ 用到**與真實路線同一個**（已 mask 的）`inpSA`，且不新增任何 node。
3. 舊位置保留成對照臂：`CGC_RHO_PROBE_LATE=1`。
4. `src/llama.cpp/src/models/qwen3next.cpp` 裡那份**誤加的副本已移除**（本專案是
   `general.architecture = qwen35moe`）；留一份只會在下次改主角時靜默分歧。

驗證：`clang++ -c` 單檔（`-Wall -Wextra -Wpedantic`）零警告，兩個 arch 都過；**沒重連 dylib**。

### 8.3 下一步那一趟怎麼跑

位置 A/B（證明 lead 真的從位置來，而不是從 ρ 來）：

```bash
./scripts/check/prebind_probe_run.sh                       # 新位置（early）
CGC_RHO_PROBE_LATE=1 ./scripts/check/prebind_probe_run.sh  # 舊位置（late）對照
python3 scripts/check/rho_probe_parse.py /tmp/prebind_probe/run_<ts>.stderr.log
```

判據：**兩趟的 `cov_uni` 必須相同**（逐位元），而 `cgc_rho_logits` 的 GPU 時間戳位置不同。
若 `cov_uni` 變了 ⇒ 位置影響了數學 ⇒ 上面「無依賴」的前提錯了，要回頭查，不要直接報 lead。

要讀 GPU 時間戳不需要新寫儀器：`ggml_metal_cgc_gpu_take_cb` 已經給 per-command-buffer 的
`GPUStartTime` + node range（`CGC_GPU_NODES`）。兩個坑：
① 預設 `n_nodes_0 = MAX(64, 0.1N)` 會把 node 埋掉 ⇒ 要 `CGC_CB_N_MAIN=1` ＋ 加大 `CGC_N_CB`
   （⚠ `CGC_N_CB=127` 會卡死 Metal 的 command-buffer 建立，見 `run_server.sh:1743`；
     可用上限 ≈ 16 ⇒ buffer 還是 ~5 個 node 寬 ⇒ lead 只能給區間）；
② **只取 START 不取 duration**（全 VIEW 的 buffer 報過 592 µs/node 的假值）；
③ **要印 range 裡的全部名字**：只印第一個的話，`cgc_rho_logits` 會被埋在 5-node buffer 中間，
   看起來像「影子節點根本不在圖裡」（2026-09-23 17:12 實測踩到）。
   為此在 `ggml-backend.cpp` 加了一條**新行標** `CGC-NSCB`（新開關 `CGC_GPU_NODES_START=1`），
   不改成既有的 `CGC-NSM`，以免打破 `gdn_split` / `per_op_slice_parse` / `attn_moe_split`。
那一趟**不引用 t/s**（圖裡多一個 matmul，`CGC_TD_CB` 還會序列化 pipeline）。

---

## 9. ★ 位置 A/B 實測（2026-09-23 17:1x–17:3x）：lead 量到了

同一支 binary、同一組權重、同一個數學，只換 `CGC_RHO_PROBE_LATE` 的有無
（EARLY = 影子在 `inpSA = inpL` 下方＝新位置；LATE = attn 之後＝舊位置）。每趟 ~45 s，
EARLY 3 趟 / LATE 3 趟。

```bash
# EARLY
CGC_SERVER_N_CB=16 CGC_CB_N_MAIN=1 CGC_GPU_NODES=1 CGC_GPU_NODES_START=1 \
    CGC_PREBIND_WORKDIR=/tmp/rho_lead_early ./scripts/check/prebind_probe_run.sh
# LATE：同上，多一個 CGC_RHO_PROBE_LATE=1
python3 scripts/check/rho_probe_parse.py <log>   # 準度
python3 scripts/check/rho_lead_parse.py  <log>   # lead（本節新增，selftest 14/14）
```

### 9.1 準度：沒有逐位元相同，但差在噪音帶內

| arm | 各趟 `cov_uni`（ntok=4） | 平均 | 同臂散佈 |
|---|---|---|---|
| EARLY | 0.8528 / 0.8655 / 0.8616 | 0.8600 | 0.0127 |
| LATE  | 0.8392 / 0.8337 / 0.8490 | 0.8406 | 0.0153 |

- ⚠ **「逐位元相同」這個判據做不到，是我上一輪（§8.3）寫錯了。** 解碼軌跡（MTP 接受數）每次跑
  都不一樣 ⇒ ntok=4 的層數 2000/1840/2120 vs 1840/1680/1880 ⇒ **token 母體不同** ⇒ `cov_uni`
  本來就會動，跟位置無關。
- 可用的判據是「**落在軌跡噪音帶內，且兩臂都在門檻同一側**」：兩臂差 0.019，與同臂散佈
  0.013~0.015 同階 ⇒ 無法區分；兩臂都 ≥ 0.83，離門檻 0.398 有兩倍以上、都過飽和點 0.70
  ⇒ **§3 的 PASS 不受位置影響**。

### 9.2 提前量：位置從 80% 搬到 45%，悲觀下界 ×3.3

取 102-node 的 segment（≈ 一層）族群，中位數：

| arm | 影子 node 在 segment 的位置 | 真實 `ffn_moe_argsort` 位置 | lead 區間（悲觀, 樂觀） | segment wall span |
|---|---|---|---|---|
| EARLY | **45.1%** | 99.0% | **+1.31 ~ +2.27 ms** | 8.0 ms |
| LATE  | **80.4%** | 99.0% | **+0.41 ~ +1.27 ms** | 6.2 ms |

（悲觀下界 = `start(argsort buffer) − end(rho buffer)`，樂觀上界 = `end(argsort) − start(rho)`；
buffer 有 ~5 個 node 寬 ⇒ 只能給區間。三趟各自一致。）

⇒ **舊位置的 lead 不是 0，是 0.41~1.27 ms**（我上一輪寫「≈ 0」是推的，實測沒那麼慘）；
但**新位置把它變成 1.31~2.27 ms** —— 悲觀下界翻了 3.3 倍。

### 9.3 代回定價（`scripts/check/window_joint_ev.py`，模型沒改）

| | cb = 42.04（**同 regime 口徑**） | cb = 74.18（跨 regime，**僅供對照**） |
|---|---|---|
| EARLY（lead 1.31~2.27） | **14.39 t/s（+14.5%）** | 15.56 ~ 16.50（+23.8~31.2%） |
| LATE（lead 0.41） | 13.19 t/s（+4.9%） | 13.19 t/s（+4.9%） |
| 結構上限（lead = ∞） | 14.39（**EARLY 已達到**） | 16.50 |

★ **cb=42.04 那一格現在是實測關起來的，不再是推的**：`cov·F = 0.860 × 1.051 = 0.904 ms/層
< 1.31 ms` ⇒ **視窗裝得下全部需求 ⇒ 視窗不綁**，該 regime 的上界完全由覆蓋率決定
⇒ 14.39（+14.5%）就是它，再搬位置也上不去。這正是 §EN-467「**cb 先、視窗後**」的實測收尾。

### 9.4 ⚠ 未解釋：EARLY 的 segment wall span 比 LATE 長 29%（8.0 vs 6.2 ms）

三趟一致，不是噪音。它可能只是這趟重儀器（n_cb=16、`CGC_TD_CB`、多一顆 matmul）的排程副作用，
也可能代表「把影子搬到層首」本身有代價 —— 後者會把 9.3 的數字整個打掉。
⇒ **絕對 lead 不能直接搬到 delivery regime**。按比例縮到 delivery 的 6.2 ms/層 ⇒
lead ≈ 1.1~1.4 ms，仍 > `cov·F` = 0.904 ⇒ **cb=42.04 的 14.39 不變**；cb=74.18 掉到 ~14.9。

### 9.5 想用 t/s 分辨它 —— 失敗，而且失敗得很乾脆（噪音太大）

不帶儀器（無 `CGC_TD_CB`／`CGC_GPU_NODES`／n_cb=16，但仍帶 `CGC_RHO_PROBE=1` 使兩臂都有那顆
多出來的 matmul）各跑 2 臂，讀 `avg_ts`：

| EARLY | LATE |
|---|---|
| 4.95 / 2.21 t/s | 3.98 / 2.18 t/s |

單臂散佈 **2.2×**（±38%）⇒ 兩臂平均差 +16% 完全在噪音裡。要分辨 29% 大約要 **每臂 ~30 趟**
（≈ 45 分鐘盒時）。**不做**：9.6 的敏感度分析顯示這個問題在 cb=42.04 下根本不改變結論。

### 9.6 敏感度：那 29% 就算全是真的，結論也不動

| 情境 | lead | cb = 42.04 | cb = 74.18 |
|---|---|---|---|
| 實測原值 | 1.31 ~ 2.27 ms | **14.39（+14.5%）** | 15.56 ~ 16.50 |
| 假設 29% 是位置代價（×0.71） | 0.93 ms | **14.39（+14.5%）** | 14.46（+15.1%） |
| 按 delivery 6.2 ms/層 正規化 | 1.10 ~ 1.40 ms | **14.39（+14.5%）** | 14.93 ~ 15.84 |

原因：`cov·F = 0.904 ms/層`，**只要 lead 不掉到 0.904 以下，視窗就不綁** ⇒ 要讓 cb=42.04 那一格
動搖，位置的代價得超過 31%。⇒ **「cb=42.04 下 ρ 的上界是 14.39（+14.5%）」這句話對那個未解釋的
29% 免疫**；受影響的只有跨 regime 的 cb=74.18 那一格（本來就不該引用）。

### 9.7 這一節改變了什麼、沒改變什麼

- **改**：lead 從「推的 0 / 1.30~1.59 ms」變成**實測的區間**（EARLY 1.31~2.27、LATE 0.41~1.27）；
  位置修正的價值被量化（cb=42.04：13.19 → 14.39；cb=74.18：13.19 → 15.56~16.50）。
- **不改**：§3 的 PASS（`cov_uni` 0.84~0.86，與位置無關）、「不要做疊加」、「cb 先視窗後」。
- **新增的待辦**：① 那 29% 是什麼（低優先，對結論免疫）；② `cb` 的 42 vs 74 仍然沒有定讞
  ⇒ 它現在是**唯一**還能移動 14.39 這個數字的東西。