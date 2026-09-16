
---

## §W. 儀器對比：llama-bench vs decode_bench（同 env、同模型、交錯、帶散熱讀數）

起因：路線圖那組 decode 9.16/9.45 標籤寫「llama-bench」，而我的 12.95 出自 decode_bench
（HTTP）。「不同儀器比大小」是這個專案最常踩的錯，所以直接量。

**做法**：`Backup/run_instrument_compare.sh`（＋ `_p2.sh`）。llama-bench 這邊的 env **不是手寫的**，
用 `--arms 'prod25:CGC_SERVER_MTP=0;CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1'` ⇒ 逐字等於
`decode_sweep --arms p25-gputime` 的 env，兩邊都經 `run_server.sh CGC_DUMP_ENV=1` 解析。
`MTP=0` 會讓 profile 載入**非 MTP 模型**（`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`），兩邊一致 ⇒ 公平。

### 量到的（全部同場、交錯、每臂都有發射前散熱讀數）

| # | 儀器 | n | t/s | 發射溫度 |
|---|---|---|---|---|
| 1 | llama-bench | 24（r=3） | **7.21 ± 1.29** | NOMINAL |
| 2 | decode_bench | 24（w0） | **18.74**（min 7.13／max 20.83） | NOMINAL |
| 3 | llama-bench | 128（r=3） | **9.42 ± 0.96** | NOMINAL |
| 4 | decode_bench | 128（w1） | **12.36**（12.06–12.95） | NOMINAL |
| 5 | llama-bench | 128（pass2） | **9.32 ± 0.76** | NOMINAL→MOD |
| 6 | decode_bench | 128（pass2） | **10.64** | MODERATE（全程 HEAVY） |
| 7 | llama-bench | 128（MTP 模型） | **8.75 ± 0.95** | HEAVY |
| 8 | llama-bench | 128（**--no-warmup**） | **9.40 ± 1.19** | HEAVY |
| 9 | llama-bench | 24（**r=8**） | **7.88 ± 1.12** | MODERATE→HEAVY |
| 10 | llama-bench | **256** | **9.45 ± 1.15** | HEAVY |

LB n=128 的三次獨立量測 9.42／9.32／9.38 ⇒ 離散 **1.1%**。

### 主結論：**差距的大半是「第一個 rep 是冷的」，不是引擎也不是時基**

`llama-bench` 的 tg **warmup 是 `test_gen(ctx, 1, ...)` ＝ 1 個 token**（`llama-bench.cpp:2392`），
而 context／pool 每 instance 建一次 ⇒ 池從**空**開始，第一個 rep 付 compulsory 冷填
（它自己的歸因：`compulsory=4437/5195 = 85.4%`）。`avg_ts` 把那個 rep 平均進去；`decode_bench --warmup 1`
**恰好丟掉對應的那一輪**。

逐 rep（DECPROF `ntok=1`，每 8 步印一次）：
- n=128 r=3 → r1=**120** r2=86 r3=93 ms/token ⇒ rep1/平台 = **1.35×**
- n=256 r=3 → r1=**116** r2=87 r3=87 ⇒ **1.34×**（同一個冷成本攤在更長的 rep 上）
- n=24 r=8 → r1=**181** r2=132 r3=97 r4=93 r5≈r8≈95-109 ⇒ **1.73×**，平台要 ~4 個 rep

⇒ **平台值**（把 rep1 丟掉）：n≥128 都是 **87–89 ms/token ＝ 11.3 t/s**，對 decode_bench 的暖值
（12.36 t/s ＝ 80 ms）是 **1.10×**，不是標題的 1.32×。**n=24 的 2.6× 降到約 1.9× 就降不下去**：
LB 的平台在 n=24 也只有 ~98–104 ms（9.8 t/s），DB 是 48–54 ms（18.7–20.8 t/s）。這一段**未歸因**。

### 三個寫下來的預測，兩個死掉（這是本輪最該記的）

| 預測 | 結果 |
|---|---|
| `-n 24 -r 8` 會把均值推上去（reps 會養熱池） | **只 +9%**（7.21→7.88），且該臂在 HEAVY（本該更低）⇒ 不算證實 |
| `-n 128 --no-warmup` 會掉到 ~7（warmup 在養池） | **9.40 vs 9.38 ＝ 0.2%** ⇒ **no-op**，證實原始碼讀到的「warmup 只有 1 token」，同時**推翻 harness 自己 `--no-warmup` 說明**那句「warmup 讓池變熱」 |
| `-n 256` 會升向 DB 的 per-token 成本 | **9.45 vs 9.38 ＝ +0.7%**（且在 HEAVY）⇒ **否證**。三個可調旋鈕（reps／warmup／長度）**都動不了它** ⇒ 殘差**不是攤銷問題** |

### 結構性事實（讀原始碼，可證）

1. **llama-bench 對 MTP 是瞎的**：`sampler|speculat|draft|MTP` 在 `llama-bench.cpp` **零命中**。
   `test_gen` ＝ `llama_decode(1 token); llama_synchronize(ctx); token = rand()%n_vocab;`
   ⇒ 沒有取樣器就沒有投機迴圈。25 t/s 唯一剩下的槓桿是 MTP ⇒ **llama-bench 不可能當這個目標的
   instrument of record**（量到的 8.75 只是「MTP 模型不跑投機」）。
2. **兩個 token 流給池的壓力結構不同**（同引擎、同池、同 env）：
   LB 隨機 id → hit 96.0%、**compulsory 85.4%**、超訂層 **7/40**、讀 5.48 GiB；
   伺服器連貫文本 → hit 92.6%、**compulsory 51.0%／capacity 49.0%**、超訂層 **40/40**、讀 **18.35 GiB**。
   **命中率高的那一邊反而慢** ⇒ 「hit% 解釋差距」已被這組數字殺掉，也說明**計數器不能跨行程比**（§eng-mh-0041）。
3. **`predicted_ms` 的視窗起點是第一個「取樣完成」的 token**（`server-context.cpp:4092`，在
   `n_decoded == 1` 內）⇒ 用 `N` 除以 `N−1` 步 ⇒ 樂觀 `N/(N−1)`（n=24 時 +4.3%、n=124 時 +0.8%）。
4. llama-bench 的 `n_ctx = n_prompt + n_gen + n_depth` ⇒ 這批是 **128**，伺服器是 **4096**。

### decode 的散熱分離（§V 說「還沒量到」的那一項，本輪有了第一個）

同 env、同模型、同 n：**NOMINAL 12.36 → HEAVY 10.64（−14%）**。llama-bench 側
NOMINAL 9.42/9.38 →（MTP 模型 ＋ HEAVY）8.75 **有兩個變數同時動（模型＋熱）⇒ 不可歸因**。
⇒ decode 對熱壓**有**反應，但幅度遠小於 prefill（250 vs <212）。**每個等級 n=1 ⇒ 仍然只記錄、不閘門。**

### 路線圖 §1 那張表的標籤

`docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md` §1 的指令欄寫
`decode_sweep.py --profile prod25 --arms p25-gputime,... --n-predict 120`，但表格那列標成
**「llama-bench decode 9.16 / 9.45」**——表格與它自己的指令互相矛盾。今日 LB n=128 的
**9.42/9.32/9.38 把 9.45 夾在中間**，而 decode_bench 讀 12.36–18.74 ⇒ 那兩個值**可重現為
llama-bench，不可重現為 decode_bench** ⇒ 標籤大概對，但**該文件本身無法稽核這個歸屬**
（`llama_bench_matrix.py` 是 09-15 17:42 `d3cbe9744` 才引入的，文件沒記載用過它）。
我上一份白皮書引用「那組出自 llama-bench」是**繼承了一個無法稽核的標籤**。

### 順手修掉的一個真缺陷

`llama_bench_matrix.py` 把 stderr 日誌寫成 `llama_bench_<tag>.stderr.log`，而用
`PROFILE:ENV=...` 形式時 **tag 就是整個 spec 字串** ⇒ `-n 24` 與 `-n 128` 撞同一個檔名，
先跑的那份日誌**被靜默覆蓋**（那正是專家池自己計數器的唯一落點）。已改成 `tag` 淨化 ＋ 形狀後綴；
`--json` 路徑是每臂不同的，所以 JSON 證據本來就沒受影響——**這正是它沒被提早發現的原因**。
副作用：它也覆蓋掉了一份 09-15 的 `llama_bench_prod25.stderr.log`。
