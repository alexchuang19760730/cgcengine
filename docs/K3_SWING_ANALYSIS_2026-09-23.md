# k=3 為什麼會飄 1.43×：變異數分解 + 靜態分析（任務 2 / 任務 4）

日期 2026-09-23 · 分工文件中的【WorkBuddy】區（**靜態分析，零 GPU**）。
對象現象：`docs/CERTIFIED_WINDOW_K_AB_2026-09-23.md` §4——k=3 同一份工作在 5 次 launch 上讀到
8.59–12.25 t/s（1.43×），k=2 只有 11.21–12.83（1.14×）。

---

## 0. 一句話

把「飄」拆成兩層之後，它不是一個問題，是兩個：

```
                  單一 request 的噪音     每次 launch 被指定的基準
k=2                     10.54%                   0.00%   (F=0.62 < 1)
k=3                     12.21%                  13.16%   (F=4.49, df 4/10)
```

**k=3 有一個專屬的「開 process 時才被決定」的 13.2% 項；k=2 完全沒有。**
兩組的逐 request 噪音幾乎一樣（10.5% vs 12.2%）⇒ 差的全部在 launch 那一層。

這直接解釋了為什麼 ABBA 配對買不到東西：配對只能扣掉**兩組共有**的項，而這一項是 k=3 獨有的
⇒ 配對 sd 只能等於 √2 × 單臂 sd（實測 2.28 vs 2.18 t/s，吻合）。

---

## 1. 分解怎麼做的

一 arm = 一 launch = 3 個 counted request（`Backup/k_abba_2026-09-23/{abba,gated}/*.json`
都已存了逐 request 的 `decode_tps`），所以單因子隨機效應可以直接套：

```
MS_within  = SS_within  / (n(m-1))    期望 = σ²_within
MS_between = SS_between / (n-1)       期望 = σ²_within + m·σ²_launch
σ²_launch  = max(MS_between − MS_within, 0) / m
```

`max(…, 0)` 是刻意的地板：MS_between < MS_within 會偶然發生，意思是「沒測到」，
不是「負變異數」。印負數只會讓人拿去相減。

工具：**`scripts/check/k_swing_decompose.py`（selftest 18/18）**

```bash
python3 scripts/check/k_swing_decompose.py --selftest
python3 scripts/check/k_swing_decompose.py \
        --dir Backup/k_abba_2026-09-23/abba --dir Backup/k_abba_2026-09-23/gated --groups k2,k3
```

它同時印三件判決用的東西：**launch 級 sd**、**一個 arm mean 的總散佈**、
以及「單靠 request 噪音原本該散多少」——最後那個數字是配對有沒有用的直接證據。

### 1.1 順便量到：arm 自己把自己跑熱

| | 3 個 request 單調下滑的比例 | first→last 跌幅 |
|---|---|---|
| k=2 | **4/5** | 17 / 2 / 22 / 18 / 24 % |
| k=3 | 1/5 | 7 / 21 / 18 / 6 / 23 % |

k=2 有 4/5 臂單調下滑，最多掉 24% ⇒ **一條 arm 的 ~70 s 自己就在發熱**，
arm mean 有一部分是「第 1 個 request 時盒子有多涼」決定的。這是在 launch **之內**的漂移，
任何 launch 前的閘門都管不到它。

---

## 2. 靜態分析：ntok=4 到底多了什麼

先確認 k 怎麼變成 token 數：`server-context.cpp:4158`
`GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1)` ⇒ **k=3 → 一個 verify step 4 個 token，
k=2 → 3 個**。旁證 `llama-context.cpp:4796` 的註解 `ntok {2:1, 3:2, 4:107, 8:18}`，
與 09-23 log 裡的 `CGC-PHASE-DBG: … phase=VERIFY … n_tokens=4`。

### 2.1 相位閾值：**不是**它（已否證）

`llama-cgc-phase.h:133-143` 用 `n_tokens <= decode_width` 選 DECODE（池）／PREFILL（整層 slab）。
若 width==3，k=3 的 4-token 就會整步走 PREFILL，那會是天差地遠的兩張圖。

但既有 log 直接把它否證了：`CGC-PHASE-SPLIT: cap=8 routable=142 slots / top_k=8 ->
decode graph width=8 tokens`，09-23 log 的 `CGC-SHAPE … width=8 routable=142`
也是 8。**3 和 4 都 ≤ 8 ⇒ 兩條 arm 同一張 DECODE 圖。**

### 2.2 真正過的門檻：**kernel 換了一顆**（這是新的、硬的發現）

`ggml-metal-ops.cpp` 的 small-batch `mul_mv_ext` 進入條件是**分類型**的：

| 類型群 | 門檻 | 行號 |
|---|---|---|
| F32 / F16 / BF16 / Q8_0 / IQ4_NL / Q4_0 … | `ne11 >= 2 && ne11 <= 8` | `:2569-2581` |
| **Q4_K / Q5_K / Q6_K / Q2_K / Q3_K** | **`ne11 >= 4 && ne11 <= 8`** | **`:2583-2590`** |

⇒ **Q_K 家族在 ntok=3 走不到 ext，在 ntok=4 才進得去。** 這個模型有 Q2_K×2、Q3_K×1、Q6_K×1
（arm JSON 的 `head_types`）。

同一個 ext kernel 內部也換 PSO：`nxpsg` 由 `ne11 < 3` 決定（`:2607-2612`，3→16、4→8），
`r1ptg` 由 `ne11` 決定（`:2618` 預設、`case 4:` 在 `:2627`）。PSO 是 runtime 編出來的
（`GGML_METAL_EMBED_LIBRARY=ON` ⇒ `ggml-metal-device.m:125/225-226`）。

**結論：k=2 與 k=3 不是「同一件事做寬一點」的乾淨 A/B —— 它同時是一次 kernel 替換。**
任何 k2−k3 的 t/s 差都把兩件事混在一起。（順帶：這條路上的 `nsg = 2` 是寫死的
`ggml-metal-ops.cpp:2604`，`CGC_MMV_NSG` 根本摸不到它。）

### 2.3 「SYNCFILL 計數相同」並沒有排除 IO

`llama-context.cpp:6497` 印的是 `CGC-SYNCFILL: … cold_filled=%zu` —— **是請求數
（`cold.size()`），不是真正落盤的讀取數**。真正的讀在 `llama-expert-cache.cpp:35` 的
`pread`（檔描述子 + `FILE*`，`:33`），服務時間取決於 OS 頁快取。

⇒ **「175 次請求、四次 arm 都一樣」和「每次請求的服務時間差 1.43×」完全可以並存。**
`CERTIFIED_WINDOW_K_AB` §4.2 用計數相同推出「不是池做了什麼」—— 那一步是對的，
但推不到「不是池花多久」。

---

## 3. 候選機制，排序 + 否證條件

| # | 機制 | 為什麼輪到它 | 什麼觀測會否證 |
|---|---|---|---|
| **H1** | **池 IO 服務時間（頁快取命中 vs NVMe）** | 池是 `pread`（`:35`）；盒子 reclaimable 實測在數十秒內擺 **1463↔9869 MB**（`FOOTPRINT_HALFSTEP` §3）⇒ 每次開 process，池檔有多少頁還在頁快取裡是**不一樣的**；k=3 每步 union 更大（4×8=32 vs 3×8=24），突發更長、尾巴更暴露 | **抓 `pread_usec`/arm**：不跟 arm mean 走就死 |
| **H2** | **ntok=4 專屬的那顆 ext PSO 對記憶體壓力更敏感** | `r1ptg=4`、`nxpsg=8` 的 threadgroup 形狀與 ntok=3 不同（§2.2），佔用/暫存/in-flight load 都不同，對「盒子還剩多少可用」的敏感度可以不一樣 | **同一個乾淨窗口下，`--tokens` 軸的 null cell 寬度在 ntok=4 明顯大於 ntok=3** ⇒ H2 得分；一樣寬 ⇒ 死 |
| **H3** | launch 當下的**功率／DVFS 上限**，OS 的 level 標籤沒抓到 | 10 條 arm **全部**是 `thermal_at_launch.level = 0`（全部 NOMINAL），而 k=3 照樣飄 ⇒ 閘門是必要條件但不是這個變數 | 錄一條 arm **過程中的**頻率／功率軌跡（不是 launch 的一個戳） |
| ~~H4~~ | pool 分段劃分不同 | **已否證**：k=3 四臂 SEG 全是 33（只有 k=2 出現過 33 / 36 兩種） | — |
| ~~H5~~ | 閘門等待時間 | **已否證**（原文件 §4.1，第 5 對殺掉） | — |

### 3.1 一個已經存在的計數器，我們卻沒開

`llama-expert-cache.cpp:432` 會印
`CGC-RIG-SNAPSHOT mode=%d reads=%zu read_bytes=%llu pread_usec=%llu fill_usec=%llu …`
—— **`pread_usec` 正是把 H1 從假設變成數字的那個量**，而且它本來就在樹上。

但：`grep CGC-RIG-SNAPSHOT Backup/cgc_logs/llama_server_2026092[23]*.log` ⇒ **零命中**；
09-23 的 log 只有 `CGC-SYNCFILL / HOOK / SCAFFOLD / COLD / SEG / WARM / PRE / POST / SHAPE / TOPK /
PHASE / MMID / CPU / ADD`。**這個計數器在 09-23 那一輪根本沒被記下來。**
（09-13 時代的 log 有 `CGC-RIG-SNAPSHOT`，但格式是舊的、沒有 `pread_usec`。）

⇒ **最便宜的下一步不是跑更多 arm，是把這一行打開。**

---

## 4. 單一 process 內的量法（任務 2 要的產出）

### 4.1 先講一件壞消息：per-request 切 k **目前做不到**

`server-schema.cpp:198` 是 `#if 0`，而 `speculative.n_max` 那個 per-request 欄位就在裡面的
`:200`（上游註解：*"to keep things simple, we disable speculative parameter adjustments for now"*）。
`grep -n "\.n_max" tools/server/*.cpp` 只剩 `server-context.cpp:3209`（全域）。

⇒ **不能在一個 process 內交替 k=2 / k=3。** 要嘛改碼重編（會動到別條線的 dylib，要協調），
要嘛接受 launch 級那一項存在、改用別的辦法對付它。

### 4.2 因此正確的策略是「讓它可觀測」，不是「用協定消掉它」

| 做法 | 解決什麼 | 代價 |
|---|---|---|
| **(a) 每 arm 抓 `CGC-RIG-SNAPSHOT`（`pread_usec` / `read_bytes`）** | 直接把 H1 判生死 | 零 GPU 成本；可能只需把那行打開或調週期 |
| **(b) 一條 arm 從 3 個 request 加到 ~12 個** | arm mean 的 SE 從 7.05% 降到 ~3.5%，launch 項以截距的形式浮出來；順便拿到更密的發熱曲線 | 每 arm 多 ~3 分鐘 |
| **(c) op 級：`mmid_shapes.py --tokens 1,2,3,4`（單一 process）** | 拿 per-token 邊際成本曲線；**重點看 `g(4)−g(3)` 是否明顯大於 `g(3)−g(2)`** —— 大就代表第 4 個 token 的代價異常（§2.2 的 kernel 換手會在這裡現形） | 需要乾淨窗口（門檻已裝好，不 admit 就 rc=2） |
| **(d) 翻 `server-schema.cpp:198` 的 `#if 0`** | launch 級項**整項消失**，k2/k3 從此可在單 process 內 ABBA | 要重編 ⇒ 會蓋掉別條線的 dylib，必須協調窗口 |

**(c) 就是原文件 §10 說的那件事**，而它現在有個更具體的讀法：不要只看絕對值，看**二階差分**。

### 4.3 不改碼的話，價格是多少

```
一個 arm mean 的總散佈 = √(13.16² + 7.05²) = 14.93%
要把 k2 vs k3 定到 ±3%          ⇒  n = (14.93/3)² ≈ 25 臂/組 ≈ 25×2×70 s ≈ 58 分鐘盒時
相比之下 k=2 那一側：6.09%      ⇒  n = 4 臂/組
```

⇒ **「先把 launch 項消掉／觀測到」比「多跑 arm」便宜一個數量級。** 這是這份分析的實際用途。

---

## 5. 任務 4：還活著的 knobs，排優先級

`docs/KNOB_FULL_MAP_15_2026-09-22.md` 已把 15 條逐一核過。這裡只給**還活著的**與**今天新增的**：

| 優先 | knob / 軸 | 狀態 | 下一步 |
|---|---|---|---|
| **1** | `#5 NSG × MoE**（`MUL_MAT_ID`） | 唯一還有量級的（%峰值 20–36% vs dense 76–91%），但**五次都被窗口擋下** | 等 begin/end 都 admit 的窗口；不要用 `--allow-busy` |
| **2** | **新的：launch 級項本身**（H1/H2） | **今天才被定位**，而且它有 13.2% 的量級、解釋了 k=3 全部的散開 | (a) 抓 `pread_usec`；(c) 看 `--tokens` 的二階差分 |
| 3 | `#10 WIN_PIN` | 已被 09-21 `IO_PATH_AB` 自然實驗降優先（⚠ 跨 run，非結案） | 同場配對才准講話 |
| 4 | `#3 N_CB` | 有把手、無先驗 | 需 ≥10% 先驗才值得 |
| 5 | `CGC_MMV_FUSE` | **與既有「實測更慢、777/800 divergent」記錄打架** | 先解衝突再碰 |
| — | GDN、prefetch、graph 分段、tile/unroll/vector、tree speculation、`n_batch`/`n_ubatch` | **已判死／不存在／被夾死** | 不要再排 |

新增的認知（不在原本 15 條裡）：**k 本身不是一個乾淨的 knob**——k=2↔k=3 同時換 kernel（§2.2）。
所以「掃 k」讀到的差，既包含「寬 verify 划不划算」，也包含「那顆 PSO 好不好」。
這兩件事要分開，只能靠 (c) 的 op 級曲線，或靠 (d)。

---

## 6. 誠實欄

- **n 很小**：5 臂 × 3 request。F(4,10) 的 0.05 臨界是 3.48，k=3 得 4.49 —— 過了，但只過一點；
  k=2 的 F=0.62 是「沒測到」不是「證明為零」（要證明為零需要多得多的臂）。
  這份文件給的是**量級與不對稱性**，不是定論。
- **H1/H2/H3 沒有被這份文件的任何數字判定。** 分解只說「有一個 launch 級的項、13.2%、只有 k=3 有」；
  它是什麼，要等 (a) 或 (c)。
- **`CGC-RIG-SNAPSHOT` 的 `pread_usec` 我沒有在 09-23 的 binary 上確認它會不會真的印出來**——
  我只確認了它在原始碼裡（`:432`）且在所有 09-23 log 裡零命中。可能是沒開、也可能是這條路徑沒走到。
- **§2.2 的 kernel 替換是讀碼讀出來的，沒有跑起來驗。** 要驗的是 (c)：ntok=4 那個 cell 的
  null 寬度 / %峰值是否與 ntok=3 明顯不同。
- **`union=64`（09-23 log 的 `CGC-SHAPE … union=64 topk=8 M=8`）我沒有追到底**：
  它看起來是 M×top_k 的上界而非當步實際值，若真如此則 MoE 的 GEMM 形狀對 ntok=3/4 **是同一個**
  （那會讓「k=3 的 MoE 做更多功」這個直覺失效）。這條沒查完，**不要當結論用**。
- 全程零 GPU、零 build、沒有動別條線的檔案。

---

## 7. 交給執行 Agent 的 handoff

1. **最便宜的一步**：確認 `CGC-RIG-SNAPSHOT`（含 `pread_usec`）能不能在現有 binary 上印出來；
   能的話，把它接進 arm 產物，重跑 4–6 臂 k=3。若 `pread_usec` 與 arm mean 同向 ⇒ H1 成立。
2. **乾淨窗口時**：`mmid_shapes.py --tokens 1,2,3,4 --reps ≥3`，報 **二階差分**
   `g(4)−g(3)` vs `g(3)−g(2)`，以及各 cell 的 null 寬度。
3. **不要在熱機上跑、不要用 `--allow-busy`、不要跨 session 比。** 這三條不因本文件而改變。
4. 若 1、2 都做完而 H1/H2 皆否證 ⇒ 剩 H3，那需要的是過程中的功率軌跡，不是更多 arm。


---

## 7. 2026-09-23 03:55 追加：三個數字被我自己算錯 / 講錯，結論要重排

有人問「所以結論是後續用 k=2 四臂/組？」。**不是。** 追下去發現三處要更正，而更正後
最該做的實驗不是「挑一個 k」，是「把 k=2 vs k=3 那個 16% 認證掉」。

### 7.1 「4 臂」是標準誤，不是信賴區間；而且它是點估計

工具的 `n = (總散佈 / 目標%)²` 把**標準誤**壓到 3%，不是把 **95% CI 半寬**壓到 3% —— 差 1.96² ≈ 3.8 倍。
已改成兩個數字都印：

| | 點估計 | 95% 上界（見 7.2） |
|---|---|---|
| k=2 | SE 3% ⇒ **4 臂**；95% CI 3% ⇒ **16 臂** | SE ⇒ **15**；CI ⇒ **58** |
| k=3 | SE 3% ⇒ **25 臂**；95% CI 3% ⇒ **95 臂** | SE ⇒ **148**；CI ⇒ **568** |

⇒ **n=5 臂根本不足以給這個問題定價。** 點估計與上界差 5–6 倍，中間任何數字都無法排除。

### 7.2 ★ k=2 的 `0.00%` 不是「證明為零」，它的 95% 上界是 ~10%

`max(MS_b − MS_w, 0)` 在 MS_b ≤ MS_w 時一律回 0，而「0」會被讀成一個發現。它不是。
在模型下 `F_obs / (1 + m·λ) ~ F(df1, df2)`，`λ = σ²_launch / σ²_within`，反解 F 的**下尾**就得到
「資料排除不掉的最大 λ」。工具已加 `launch_sd_upper()`（純 stdlib，`f_cdf` 走不完全 beta）。

- **k=2：點估 0.00%，95% 上界 9.98%**
- **k=3：點估 13.16%，95% 上界 35.78%**

⚠ 我第一版用 Simpson 積分算 `F_0.05(4,10)` 得到 **0.6126（錯）**，正確是 **0.1677**
（兩種方式交叉驗證：本檔 `f_cdf` 自身，以及 `1 / F_0.95(10,4)`；而 `F_0.95(4,10)=3.478`
剛好等於本檔在用的 `F_CRIT_DEFAULT=3.48`，是獨立的第三個錨）。
錯的那個值會讓 k=2 的上界看起來是 **0.67%** 而不是 **9.98%** —— 差 15 倍，而且方向有利於
「k=2 很乾淨」這個我本來就想相信的結論。**已把 0.1677 釘進 selftest（25/25）。**

### 7.3 ★★ 真正被漏掉的事：k=2 平均比 k=3 快 16.4%，而它是「雙峰」不是「吵」

| rep | k=2 | k=3 | 差 |
|---|---|---|---|
| r1 | 12.23 | 8.59 | **+3.64** |
| r2 | 11.21 | 10.72 | +0.49 |
| r3 | 12.83 | 9.09 | **+3.74** |
| r4 | 12.20 | 11.48 | +0.72 |
| gated | 12.20 | 12.25 | −0.06 |

配對 mean diff **+1.71 t/s（+16.4%）**，sd 1.83，n=5 ⇒ **t=2.08，df=4，p≈0.11 ⇒ 未認證**。

但重點在**形狀**：k=3 有 3/5 次掉到 8.59–10.72，2/5 次正常（11.48 / 12.25）。
⇒ **§1 那個 13.16% 的 launch 級項，和這個 16.4% 的平均差，是同一件事**：
k=3 不是「每一步都慢一點」，而是**有些 launch 掉進一個慢狀態**，把平均拖下去。
而變異數項本來不該 bias 平均 —— 它會 bias，正因為慢狀態是單邊的、不是對稱噪音。

### 7.4 k 是可以改的，而且**不用 rebuild**

- `llama-bench.cpp:666` 有 **`--spec-draft-n-max`**（`:1301`/`:1354` 解析，`:2685` 寫入
  `params.speculative.draft.n_max`），預設 3，範圍 1..16。
- `run_server.sh:436` `SPEC_DRAFT_N_MAX="${CGC_SERVER_MTP_N_MAX:-3}"`。
- `scripts/run_n30cache.sh:86` **預設就是 2**。

⇒ 交付 cell 換 k=2 是一行旗標的事。**但這不代表可以無條件換**（不同 PSO，§2.2）。

### 7.5 重排之後的建議順序

1. **先認證那個 16.4%**（配對 sd 1.83、diff 1.71 ⇒ 約 **7 對 ABBA**，現在已有 5 對 ⇒ 再 2 對）。
   這同時回答「交付 cell 該不該換 k」和「台架該用哪個 k」，而且比 25 臂便宜一個數量級。
2. 若成立 ⇒ 交付 cell 換 k=2（一行旗標），台架問題自動解決；**此時**才談 k=2 要幾臂。
3. 若不成立 ⇒ k 只是噪音源，才回到「k=2 當便宜 screen」，而且 winner 仍要回 k=3 確認一次
   （PSO 不同，排名不可直接搬）。
4. **`pread_usec`（`llama-expert-cache.cpp:432`）仍然是最便宜的一步** —— 它針對的是
   「k=3 為什麼會掉進慢狀態」，也就是 §7.3 那個雙峰的成因，比多跑 arm 更接近根因。

### 7.6 誠實欄（本節）

- n=5 對，配對 t=2.08 未達 df=4 的 2.776 ⇒ **16.4% 仍未認證**。
- 「7 對」是用 z=1.96 估的；用 t 迭代後 n=7 ⇒ df=6、t=2.447 ⇒ 仍落在 7，但沒有餘裕。
- gated 那一對（−0.06）是否被視為同一母體，我沒有查；它若屬不同條件，上面 5 對要重算。
- §7.2 的上界是**變異數比**的信賴界，不是「k=2 的 launch 項的 95% 分佈」——用語要小心。
