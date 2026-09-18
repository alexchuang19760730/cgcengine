# 口徑 A/B：llama-bench（權威）vs run_server.sh 服務路徑 —— 2026-09-18 18:16–18:24

**一句話**：同一份 env、同一個模型、同一個 8 GiB 生產 pool、兩側都 **MTP off**、全程 **NOMINAL**，
`llama-bench` 量到 **8.89／9.13**，而走 `run_server.sh` 的服務路徑量到 **10.74／11.89** ⇒
**改用 run_server.sh，數字會高 +21%～+30%。**

記錄裡的同類對照（`MEMORY_PERF.md:227,327`）是 **10.79–10.89 vs 12.24–12.59 ⇒ +13%～+15%**。
⇒ **這個差不是常數**：今天 bench 那一側比記錄低 **17%**，而服務路徑只低 **4%**。
任何「用一個固定倍率把 10.x 換算成 12.x」的做法，今天會錯 2 倍。

---

## 判準（先寫下來，再量）

1. **只比 decode**。prefill 那一半這次**量不到**（原因見 §5，是工具的限制不是速度問題）。
2. **env 必須同源**。`llama_bench_matrix.py` 的 env 不是自己寫的，是跟 `run_server.sh` 拿的
   （它自己的 docstring 逐字寫 *ONE SOURCE OF TRUTH*）⇒ 兩臂的 `SERVER_ENV` 由同一支腳本解析。
3. **兩側都 MTP off**，否則比到的是 MTP 而不是口徑。
4. **只在 NOMINAL 下取樣**；`llama-bench` 臂附 `thermal hist`，服務臂每次請求前等 NOMINAL。

## 環境（量測當時）

| 項 | 值 |
|---|---|
| 熱壓 | **NOMINAL（level 0）** 全程；A1 臂 hist = NOMINAL 114/114 |
| `memory_pressure -Q` | **81–82%**（`run_server.sh` 的 memory guard 讀這個；門檻 35 ⇒ 放行） |
| `vm_stat` free | 0.45–1.02 GB |
| swap | **used 13.7 GB / 15.4 GB（約 91%）** |
| 其他量測行程 | 無（8080 空、無 `llama-server`、無驅動）。開工前有 abort 閘門。 |

## 1. 六臂結果

| 臂 | 儀器 | 讀數 | thermal | 備註 |
|---|---|---|---|---|
| A1 | llama-bench | **9.13 ± 1.14** | NOM 114/114 | prod25-stream，`-b 512 -ub 512 -p 0 -n 128 -d 512 -r 3` |
| B2 | HTTP（run_server.sh） | **11.89**（11.92／11.61／11.89） | 每次請求前 NOMINAL | rep1 暖機 8.13（丟棄） |
| ~~C3~~ | HTTP | **失敗** | — | server 被外部 SIGTERM 打死，見 §5 |
| D4 | llama-bench | **8.89 ± 3.41** | NOM 79 / MOD 46 | 同上形狀 |
| E5 | HTTP（補跑，前景） | **10.74**（10.74／10.46／12.04） | 每次請求前 NOMINAL | rep1 暖機 8.13（丟棄） |

**合併**：bench `median 9.01`（n=2）；HTTP 六個去暖機樣本 `{11.92, 11.61, 11.89, 10.74, 10.46, 12.04}`，
`median 11.75`，極差 10.46–12.04（**15%**）。
⇒ **比值 11.75 / 9.01 = 1.30（+30%）**；逐臂配對 A1→B2 = **+30.2%**、D4→E5 = **+20.8%**。

兩個非靠運氣的細節：**兩個 HTTP 臂的暖機 rep1 幾乎一模一樣（8.1332 vs 8.1290）**——
那個「載入後第一支請求」的狀態是可重現的，而且它**比穩態低 32%**（8.13 vs 11.75）。
若只跑 rep1 就下結論，會把服務路徑量成 8.1，**比 bench 還低**。

## 2. 對照記錄：差不是常數

| | 今天（18:16–18:24） | 記錄（09-17／09-18 04:39） | 今天 vs 記錄 |
|---|---|---|---|
| llama-bench（prod25-stream，標準形狀） | **9.01** | **10.79–10.89** | **−17%** |
| HTTP 服務路徑（prod25，MTP off） | **11.75** | **12.24–12.59** | **−4%** |
| 比值 | **+30%** | **+13~15%** | **差 2 倍** |

★ 兩側用的是**同一個模型檔**（`Nail-…-denseIQ4X.gguf`）、**同一個 8 GiB pool**、
**同一支腳本解析的 env**、**同一個熱態**（NOMINAL）。唯一沒對齊的是**記憶體狀態**：
本機 swap 已用 13.7 GB（91%）。
⇒ 可引用的述句：**「今天 bench 側掉 17%、服務側掉 4%」**；
**不可**說「口徑差就是 30%」——它昨天是 14%，今天 30%。

## 3. 可引用 vs 不可引用

**可引用（本檔新增）**
- 同 env／同模型／同 pool／MTP off／NOMINAL 下，**服務路徑比 llama-bench 高 +21%～+30%**（今天）。
- **服務路徑的間內離散 15%**（10.46–12.04，n=6）；**bench 的間內離散 2.7%**（8.89–9.13，n=2）。
  ⇒ 今天「哪個儀器比較穩」與 09-17 裁定時**相反**（那次是 llama-bench 三次差 1.1%）。
- **載入後第一支請求 ≈ 8.13**，兩臂一致，比穩態低約 32%。

**不可引用**
- 任何把今天的 11.75 當「12.62 重現」的說法——**12.62 是 MTP on**，本輪兩側**都 MTP off**。
- prefill：本輪**沒有**服務路徑的 prefill 數字（§5）。

## 4. 每個數字的出處

```sh
# bench 臂（標準形狀，MEMORY_PERF.md:233）
python3 scripts/check/llama_bench_matrix.py --arms prod25-stream \
    --prompt 0 --gen 128 --depths 512 --reps 3 --json /tmp/cal/A1_bench.json

# 服務臂（同一份 env；MTP off；prod25 的 CTX=4096）
python3 scripts/check/http_duo.py --profile prod25 \
    --extra-env "CGC_PREFILL_STREAM=1;CGC_GATHER_SLAB_CAP=256;CGC_SERVER_MTP=0" \
    --prefill-predict 16 --decode-predict 128 --reps 4 --json /tmp/cal/B2_http.json
```

- `llama-bench` 實際 argv（由產物／`ps` 讀到）：
  `llama-bench -m …/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf -ngl 99 --load-mode none
   -t 8 -expert-cache 8589934592 --cache-type-k q8_0 --cache-type-v q8_0 -b 512 -ub 512
   -p 0 -n 128 -d 512 -r 3 -o json` ⇒ **pool = 8 GiB** ✓、`-p 0` ⇒ **只量 decode**。
- 服務側數字取自 server 自己的 `timings.predicted_per_second`（**不是客戶端碼錶**），
  prompt 10 token、`n_predict 128`、`/v1/completions`。
- **原始產物留在本機** `Backup/caliber_20260918/`（`A1_bench`、`D4_bench`（tag=`prod25-stream`）、
  `B2_http`、`C3_http`、`E5_http`（`profile=prod25`）＋各自的 `.txt` 原文輸出）。
  ★ **`Backup/` 是 `.gitignore:396` 明確排除追蹤的區域**（"Backup/（1.8G 手動備份區，明確排除追蹤）"）
  ⇒ **它不隨 repo 走**，所以本檔把可引用的讀數**全部內嵌**、把重跑指令寫在 §4 —— 記錄要能自足，
  不能依賴一個不會被 clone 到的目錄。`C3_http.json` 的 4 個 rep 全是 `null`：那正是 §5② 那一臂，
  **留著是有意的**，它是「失敗長什麼樣」的證據。
- server log：`Backup/cgc_logs/llama_server_20260918_1817*.log`、`…_181929.log`、`…_182245.log`
  （同樣在 `Backup/` 底下，同上：本機）。

## 5. 附帶發現（具名，都是真缺陷）

**① `http_duo.py` 的 prefill 軸在 `prod25` 上跑不了。** 每一筆 prefill 都回 400，服務端原文
（`Backup/cgc_logs/llama_server_20260918_181734.log:753`）：

```
E srv send_error: request (4500 tokens) exceeds the available context size (4096 tokens)
```

`http_duo.py` 的 prefill prompt 是那段中文 ×60 ⇒ **實測 4500 token**；`prod25` 的 `CTX=4096`
（`scripts/run_server.sh:463`）。它的預設 `--profile` 是 `prefill250`（大 ctx）⇒ **這個儀器
prefill 那一半是為大 ctx profile 寫的**。本輪因此只有 decode 可比。
要量服務路徑的 prefill，得換大 ctx 的 profile，或把 prompt 縮到 <4000 token。

**② C3 臂的 server 被外部 SIGTERM 打死（不是崩、不是 watchdog、不是 OOM）。** 服務端原文
（`…_181929.log:383`）：

```
[CGC] Received SIGTERM — initiating graceful shutdown (…)
```

排除法：in-process watchdog 走的是 `GGML_ABORT`（`ggml-metal-context.m:863-866`），不是 SIGTERM；
`llama-server` 當時在正常出 token（`n_decoded=80`）；機器沒有 OOM 徵兆；8080 事後乾淨、無殘留。
**最可能**：本輪的 ABBA 是用 `nohup` 啟動的，**沒有新開 session/process group** ⇒ 宿主側
回收兄弟命令時，整個進程組一起收到 SIGTERM（`run_server.sh` 自己就是為此才寫了 `--detach`
＋ `os.setsid()`，見 `scripts/run_server.sh:40-47`）。
**做法**：長量測要麼**前景跑**（E5 就是這樣補回來的，一次成功），要麼新開 session。
⚠ 這是**假設**：日誌裡看不到發送方，只有「收到」那一行。但它可證偽——E5 改用前景就沒再發生。

## 6. 我的判斷（是推論，不是記錄裡的話）

`12.62 vs 10.9` 那個 +15.8% 的差，**不是一個可攜的「口徑常數」**。今天的資料顯示：
**兩條路對環境的敏感度不同**（bench −17% vs 服務 −4%），所以當環境一變，這個比值就從 14% 漂到 30%。
若要繼續用「權威口徑」這個框架，**必須同時報兩個口徑、且每次重跑**，否則任何跨日的比值都不可比。

## 7. 沒做的（具名）

- **prefill 的服務路徑數字**：沒有（§5①，工具限制）。
- **MTP on 的服務側數字**：沒有。本輪為了隔離口徑，兩側刻意都 MTP off。
- **`-b 512` 與 server 自身 batch 設定是否等價**：未查。`llama-bench` 的 argv 帶 `-b 512 -ub 512`，
  而 server 走 profile 的 batch 預設 ⇒ **這是本比較唯一沒有完全對齊的旋鈕**，要更硬就得先核它。
- **環境的哪一項造成 bench 掉 17%**：未定位（swap 91% 是嫌疑，未證）。
- **D5**：本輪改動 0 個 `src/`（只有 `docs/`）⇒ oracle 那一半不適用。
