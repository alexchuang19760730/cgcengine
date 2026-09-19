# 雙軸 profile：prefill / decode 必須同時列出（2026-09-18）

## 0. 這份文件在做什麼

使用者要求一件事：**給一個在 NOMINAL 下 prefill 與 decode 都最高的 profile 組合，門檻 prefill ≥ 250、
decode ≥ 12，而且從此任何 profile 的數字都必須同時列出兩軸。**

本文的答案是「組合是什麼」＋「量到多少」＋「哪一半還沒成立」。所有數字都標明儀器與熱態。

---

## 1. 結論：推薦組合

```sh
CGC_SERVER_PROFILE=prefill250 ./scripts/run_server.sh
```

它解析出的實際設定（全部來自 `run_server.sh`，不是這裡抄的）：

| 旋鈕 | 值 | 為什麼是它 |
|---|---|---|
| `ctx` | 8192 | prefill250 的預設 |
| `batch` / `ubatch` | 5632 / 5632 | 一次吃 5632 token，不被切碎；**6144 在 8 GiB 池上 5 次裡死 5 次** |
| `CGC_EXPERT_CACHE_BYTES` | 8 GiB | |
| `CGC_SERVER_MTP` | 1（denseIQ4X 模型） | 見 §3：關掉它兩軸都更慢 |
| `CGC_PREFILL_STREAM` | 1 | 大 chunk 走 whole-layer slab 而非 pool |
| `CGC_GATHER_SLAB_CAP` | 256 | slab 裝得下全部 256 expert，否則 streaming 被拒 |
| `CGC_SPAC` / `ALPHA` | 1 / 0.75 | decode 工作集駐留；09-16 實測 d512 +17% |
| `CGC_SERVER_OA_ASYNC` | 1 | |

**沒有任何 override。** 這是重要的：`prefill250` 本身就是那個組合，加 override 只會變差（§3）。

**門檻判定（2026-09-18）：prefill ✅ · decode ❌。**

- prefill：**292.34**（llama-bench pp2048，NOMINAL）→ **達標**，而且是本日唯一過 250 的數字。
- decode：六臂 llama-bench 落在 **7.03 – 10.80**，全程 NOMINAL 的兩臂是 **10.80 / 7.74**
  → **未達 12，而且連「現在到底是多少」都只能給範圍，不能給點**（§5：噪音 ±1.9 > 效應）。

⇒ 所以「250/12+」這個組合今天**拿不出來**。能交付的是「prefill 達標、decode 低於門檻」。
把 12 寫進任何結論都會是引用舊數字，而那正是「不要再退回去」要防的事。
⚠️ 也**不要**把「decode 退步了」寫進結論 —— 那是同一個錯誤的另一面（§5 已否證其可歸因性）。

---

## 2. 量到的數字（兩軸發射皆 NOMINAL）

儀器：`scripts/check/profile_duo.py`（llama-bench），**每一臂發射前等熱態回 NOMINAL**。
2026-09-18 16:16–16:26。

| profile | prefill t/s（pp2048） | decode t/s（d512, -b512, n128） | bar 250/12 |
|---|---|---|---|
| **`prefill250`**（MTP=1，生產預設） | **292.34**（315.41 / 302.34 / 282.34） | **9.12**（8.02 / 9.78 / 8.46） | prefill ✅ · decode ❌ |

出處：`Backup/prod_matrix/duo_prefill250_20260918_1615.json`

同一個 profile 在**服務路徑**（HTTP，同一個 server process 量兩軸）：

| profile | prefill t/s | decode t/s | bar 250/12 |
|---|---|---|---|
| `prefill250`（HTTP，prompt 4500 tok） | 201.35（219.13 / 201.35 / 拒跑 ×2） | 8.52（3.60 / 8.13 / 8.52 / 10.27） | ❌ ❌ |

出處：`Backup/prod_matrix/http_duo_prefill250_20260918.json`

⚠️ 兩個數字**不能直接和 llama-bench 那列比**：HTTP 那一輪的 prompt 是 4500 token，llama-bench 的
`prefill-house` 是 2048 token，而 prefill t/s 對 prompt 長度很敏感（09-16 在 2873 token 的 HTTP 臂上
量到 275–278）。所以「HTTP prefill 201」不是「prefill 退步」，是**不同長度的量**。

decode 那一列有個不能略過的形狀：**3.60 → 8.13 → 8.52 → 10.27 逐輪上升**（四次請求發射時都是
NOMINAL）。這是專家池在逐輪填入，不是熱態。⇒ 8.52 這個「平台值」**低估**穩態；末輪 10.27 比較接近。
即便如此，兩台儀器今天都沒有回到 12。

---

## 3. 「關掉 MTP 換 decode」——已被否證

這是最自然的猜想（09-16 記的 10.78/10.91、12.36、12.95 全部標註 `CGC_SERVER_MTP=0`）。實測：

| | prefill | decode | 模型 |
|---|---|---|---|
| MTP=1（預設） | **292.34** | **9.12** | …MTP-UD-IQ3_XXS-**denseIQ4X**.gguf |
| `CGC_SERVER_MTP=0` | 227.44 | 8.56 | Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf |

**兩軸都更差。** 而且要注意：`CGC_SERVER_MTP=0` 換的是**模型檔**（`run_server.sh:154-160`），
不只是旗標 —— 所以這不是同一個受試對象的 A/B，而是兩個不同模型的比較；結論只能是
「MTP=0 沒有比較快」，不能說「MTP 這個機制值多少」。

出處：`Backup/prod_matrix/duo_prefill250_mtp0_20260918.json`

---

## 4. 為什麼不是 `prod25`：結構上量不到

不需要跑 GPU，`--dry-run` 就顯示 `compat()` 直接拒絕：

```
prod25 / prefill-house → -p 2048 does not fit -b 8
                         (GGML_ASSERT n_tokens_all <= n_batch; 引擎把 n_batch 夾在 cgc_pool_max_tokens())
prod25 / decode        → profile pins no BATCH 且沒有 CGC_PREFILL_STREAM，
                         池路徑夾 n_batch<=8，cell 的 -b 512 不會是有效 batch
```

`prod25` 的池路徑被 `cgc_pool_max_tokens()` 綁在 `n_batch <= 8`（`llama-graph.h:18-37` 的註解自己
把它綁在 MTP 上），而 whole-layer slab 只有 prefill250 有。**結論：250/12+ 的候選只剩 prefill250 血統。**

---

## 5. 「decode 退步」——已用 binary bisect 否證（不是退步，是臂內升溫）

疑點：09-16 記 10.78/10.91，今天同一形狀只有 9.12（MTP=1）／8.56（MTP=0，且是**同一個模型檔**）。

**bisect 方法（不必重建）**：產物 144 檔全在版控裡，所以 `git archive <commit> src/llama.cpp/build/bin`
就能取出該提交當年編出的 engine。⚠️ 取出後必須 `install_name_tool -rpath` 把絕對 rpath 改指到 /tmp 那份
（123 檔）—— 否則 `@rpath` 會回到真的 build 目錄載**當前的** dylib，bisect 等於沒做。
這條路也避開了重建：重建會**蓋掉別條 session 正在 map 的產物**。

交錯 A（舊 `09a998d10`, 09-17 20:58）／B（HEAD）×2，全部從 NOMINAL 發射：

| arm | 引擎 | 臂內 worst | platform | samples |
|---|---|---|---|---|
| A1 | 舊 | NOMINAL 109/109 | **10.80** | 8.25 / 10.84 / 10.75 |
| B2 | 新 | MODERATE（0 HEAVY） | 10.34 | 7.80 / 10.23 / 10.44 |
| A2 | 舊 | HEAVY ×17 | 8.92 | 7.39 / 8.95 / 8.90 |
| B1 | 新 | HEAVY ×33 | 8.58 | 7.56 / 8.50 / 8.65 |
| **B3** | 新 | **NOMINAL（clean）** | **7.74** | 6.91 / 8.32 / 7.16 |
| B4 | 新 | HEAVY | 7.03 | 6.92 / 7.39 / 6.68 |

### ❌ 我中途下錯的一個結論（保留記錄）

四臂時數字看起來與「臂內 worst」完美單調對應（NOMINAL→10.80、MODERATE→10.34、HEAVY→8.92/8.58），
我也一度把它寫成規則。**第 5 臂 B3 否證了它**：B3 全程 NOMINAL（clean）卻只有 **7.74**，
比 worst=HEAVY 的 A2（8.92）還低。⇒ **熱態不是主控變數。**

### 真正的結論：**這個退步無法歸因，因為儀器的噪音比效應大**

六臂散佈 **7.03 – 10.80（±1.9 t/s）**，全部從 NOMINAL 發射。而「疑似退步」的量是
10.78 → 9.12 ≈ 1.7 t/s —— **效應小於噪音**。三個被測的變數（引擎版本、發射熱態、臂內 worst）
**沒有一個能排出六臂的順序**。

**噪音源已定位到第四個變數：記憶體。**

```
vm.swapusage: total = 15360.00M  used = 14134.12M  free = 1225.88M   ← swap 92% 滿
Pageins: 1,413,205,148
```

16 GB 機器上放 13.6 GB 模型 ＋ 專家池 ＋ compute buffer ⇒ 每次跑有多少被換出，決定那一臂的速度。
這也解釋了先前記錄的「同一道命令 prefill 散佈 2.25×」。

⇒ **正確述句：「疑似 −15% 退步」未獲證實，也未獲排除 —— 在 swap 92% 滿的機器上，
llama-bench decode 的單臂精度不足以分辨 15%。** 要裁決它，得先把記憶體壓力變成受控變數
（關掉其他 session、釋放 swap、或改用不受換頁影響的指標），再做配對設計。

### 順帶解除的嫌疑（不必再重建 bisect）

`CGC_DOWN_COMBINE` 是 env-gated（須 `=1`）且未設 ⇒ 融合 kernel 根本沒開；
`CGC_IDS_STRIDE`（`17d675cb8` 寬度 8→64）只動 debug 常數；
`605452177` 自述 decode path untouched 且有 58.25% 接受率覆現佐證。
⇒ 11 個 `src/` 提交裡沒有明顯的行為改變者，與「效應小於噪音」一致。

（`profile_duo.py` 仍加註 `clean = launch 與 worst 皆 NOMINAL`。它不能保證數字，
但**沒有它的臂更不可引用** —— 它排除的是「發射時就已經熱」這種明確不可用的樣本。）

---

## 6. 還沒成立的一半：decode ≥ 12 只存在於另一台儀器

記錄上 **decode ≥12 從未在 llama-bench 出現過**（該儀器的上限約 10–11）。所有 12 以上的數字都來
自 **HTTP 服務路徑**：12.36（`decode_bench`）、12.95（`p25-gputime`，n=124）。

所以「250/12+」能不能成立，**取決於用哪台儀器** —— 而歷來 prefill 用 llama-bench、decode 用 HTTP，
兩台儀器的數字被並排放進同一張表，這正是「兩個從未同時成立的數字」的來源。

為此新寫了 `scripts/check/http_duo.py`：同一個 server process 量兩軸，讀 server 自己的
`timings.prompt_per_second` / `predicted_per_second`，每次請求前等 NOMINAL。

**本日跑完了，答案是：服務路徑也沒有 12。** 平台值 decode 8.52、末輪 10.27；prefill 那一軸有兩輪
因為熱態停在 MODERATE 被閘門拒跑（不是量到低，是**沒量**）。

⇒ 結論：**「decode ≥ 12」在今天這個 build 上，兩台儀器都沒有重現。** 記錄上的 12.36 / 12.95
屬於 09-16 的狀態，而 §5 的紅旗正是同一個方向 —— 現在能引用的是 **9–10.3**。

---

## 7. 新約定（機械化，不是口號）

從此每一個 profile 的數字都必須兩軸並列。落點是兩支工具，閘門寫在程式裡：

| 工具 | 儀器 | 閘門 |
|---|---|---|
| `scripts/check/profile_duo.py` | llama-bench | 每次發射**前**等 NOMINAL；等不到就拒跑（不發射） |
| `scripts/check/http_duo.py` | HTTP 服務路徑 | 每次請求**前**等 NOMINAL |

為什麼要新工具而不只用 `prod_matrix.py`：**它只記錄熱態、不等**。所以四格連跑時第 1 格 NOMINAL、
後 3 格 HEAVY —— 本日下午那一輪就是這樣，四個數字裡三個不能用，而唯一抓到它的是事後人眼讀標籤。

```sh
python3 scripts/check/profile_duo.py --profiles prefill250 --cells prefill-house,decode --reps 3 \
    --json Backup/prod_matrix/duo.json --md docs/duo.md
python3 scripts/check/http_duo.py --profile prefill250 --reps 4 \
    --json Backup/prod_matrix/http_duo.json
```

---

## 8. 下一步（按能裁決的程度排序）

1. **把記憶體壓力變成受控變數（最高優先，且是其他一切的前置）**：swap 已用 14134/15360 MiB
   （92% 滿），單臂散佈 ±1.9 t/s ⇒ **現在任何 decode A/B 都分辨不出 15%**。要做的是
   釋放 swap／確認沒有其他 session 佔 GPU 與記憶體／或改用不受換頁影響的指標，再做配對設計。
   **在此之前不要下「退步」或「改善」的結論。**
2. **§5 的 bisect 已做完，不必再重建**：引擎版本無法解釋散佈（舊引擎也有 8.92、新引擎也有 10.34），
   `src/` 裡也沒有明顯的行為改變者。**繼續 bisect 是浪費。**
3. **儀器已統一為 llama-bench**（使用者裁決）⇒ 第 2、3 點關於 HTTP 的對齊工作**降為非必要**；
   `http_duo.py` 只在研究「儀器間差異」時才用。
4. **prefill 是可信的那一半**：292.34（pp2048，NOMINAL）。若需要與記錄的 275–278 比，
   記得那是 2873-token 的 HTTP 臂，長度不同。
