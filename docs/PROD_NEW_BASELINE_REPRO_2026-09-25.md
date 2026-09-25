# prod-new（MTP off）基準 —— 復現嘗試與判決（2026-09-25）

> **規則（operator 2026-09-25）**：實測若後續接手的人無法復現 ⇒ **丟廢棄**。
> 本檔就是「依他的環境復現」的那一次嘗試，連同兩個讀數的判決與**復現所需的全部身分**。
>
> ⛔ **一句話**：同一盒況連跑兩輪，**兩輪都不可引用，而且失敗在不同條款上**
> —— run1 熱污染（worst HEAVY 63/162）、run2 rep 散度 **89.7%**。
> ⇒ 本輪兩個讀數**廢棄**；卡上錨點家族今日**未取得可引用複現**。

---

## 1. 身分（下一個人要比對的就是這些）

| 項 | 值 |
|---|---|
| repo HEAD | `65c76b8c7`（branch `demo/sweet-spot-windows-fix`） |
| `libllama.0.0.578.dylib` | `2c19145d327165cf4cf651298cd67c24` |
| `libggml-base.0.19.0.dylib` | `9c4ff970ebd206f2f4d79f4b7a31ee34` |
| `libggml-metal.0.19.0.dylib` | `920e4ad5e19c5fd344281efefe3abac8` |
| `llama-bench` | `28ae5feed73660da9fa45ec5db626e54` |
| model | `models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`，md5 `1644f1dc3133e3cceb4eb50b0eb82216`，13 663 116 512 B |

## 2. 配方（不複製，只引用）

**形狀／cell／profile 一律以 `docs/PROD_NEW_TEST_CARD_2026-09-24.md` §2／§3／§7 為準**（SSOT）。
本輪實際跑的是它的預設入口：

```
python3 scripts/check/commit_bench.py --record-only \
    --workdir <dir> --json <dir>/result.json
```

展開後（測試卡 §2 的形狀）：

```
llama-bench -m models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf \
  -ngl 99 --load-mode none -t 8 -expert-cache 8589934592 \
  --cache-type-k q8_0 --cache-type-v q8_0 -b 5632 -ub 5632 \
  -p 2048 -n 128 -d 512 -r 3 -o json --warm-skip 64
```

**MTP off 已由 `--arms prod-new` 的 profile 保證**（命令列沒有 `--spec-type`；測試卡 §3 `CGC_SERVER_MTP=0`）。

## 3. 兩輪讀數（同 cell、同 build、相隔 15 分鐘、第二輪先冷卻 420 s）

| | run1 `18:18:19→18:19:50` | run2 `18:32:42→18:34:15` |
|---|---|---|
| prefill `pp2048` | **228.39 ± 9.31** t/s | **292.01 ± 11.18** t/s |
| decode `tg128` | **11.59** t/s（11.81 / 11.39 / 11.57） | **9.35** t/s（11.77 / **3.95** / 12.34） |
| rep 散度 | **3.6%** ✅ | **89.7%** ❌ |
| hit% | 96.3 | 96.1 |
| thermal | launch NOMINAL，**worst HEAVY**，`{NOMINAL 86, MODERATE 13, HEAVY 63}` | launch NOMINAL，worst MODERATE，`{NOMINAL 127, MODERATE 43}` |
| swap | 4978 → 7886 MiB，**growth +2908**，max 8570 | 5462 → 8305 MiB，**growth +2843**，max 8609 |
| `min_free` | **14.1 MiB** | **14.2 MiB** |
| attribution | **both**（thermal=HEAVY, swap_growth=2908） | **swap**（swap_growth=2843） |
| 產物 | `Backup/mtpoff_base/run1_20260925_181950.json` | `Backup/mtpoff_base/run2_20260925_183415.json` |

## 4. 判定（三條既有規則，逐條套用）

| 規則 | 出處 | run1 | run2 |
|---|---|---|---|
| **污染門**：`HEAVY` 或 `swap growth > 500 MiB` ⇒ 只能當診斷價，不可進錨點 | 測試卡 §5.1 | ❌ 兩條都踩（HEAVY＋growth 2908） | ⚠ swap growth 2843 > 500 |
| **三態判決**：thermal worst 非 NOMINAL，或 rep 散度 > 12% ⇒ **UNRELIABLE ⇒ 作廢** | `ASSERTION_PROTOCOL` §4.1 | ❌ worst HEAVY | ❌ 散度 89.7% |
| **複現門**：decode 落在 **±0.8 同帶**內即可複現（卡面錨點 `11.90 ± 0.15`） | 測試卡 §6 | ✅ `11.90−11.59 = 0.31`（值同帶） | ❌ `11.90−9.35 = 2.55`（不同帶） |

**⇒ 判決**：run1 **值可複現、狀態不可引用**（污染）；run2 **不可複現（UNRELIABLE）**。
**⇒ 本輪的兩個 t/s：廢棄**（依 2026-09-25 規則）。
**⇒ 卡上錨點家族（`12.17 / 11.55 / 11.49` / 卡面 `11.90 ± 0.15`）：今日未取得可引用複現**，標「未復現（待乾淨盒況）」，不改其原文。

## 5. 診斷：為什麼復現不了 —— 前置條件在**起跑前**就已不合格

1. **起跑 swap 4978 / 5462 MiB**，而契約 §3.3 的起跑門檻是 **≤ 2048 MiB** ⇒ 兩輪都在門檻外起跑，本來就不該產出數字。
2. **兩輪各自再灌 +2.9 GB swap**（模型 13.6 GB ＋ 8 GiB pool 在 16 GB 機器上是靜態超訂 4838 MiB），
   而 §5.1 的污染門是 **+500 MiB** ⇒ 當時看起來像「這個 cell 按設計就會踩污染門」。
   ⛔ **【已證偽 2026-09-25】**：重開機後同一個 cell（launch swap 0 MiB）**只成長 +84.7 MiB**（§9.2 run4）。
   ⇒ 那 +2.9 GB **不是 cell 的性質，而是起跑時已存在的 swap 壓力**被這一輪推著再吃。
   正確述句：**這個 cell 對「起跑時已經有 swap」極度敏感**，不是它自己必然污染。
3. **`min_free` 兩輪都只剩 14 MiB** ⇒ 機器被推到牆邊；同樣的牆在 run1 表現成 thermal HEAVY，在 run2 表現成
   一個 **3.95 t/s** 的離群樣本（散度 89.7%）。
4. **swap 不會自己回收**：兩輪之後 `used = 8137 / 9216 MiB`（total 還被 macOS 從 6144 擴到 9216）。
   ⇒ **下一輪的起跑條件只會更差**，這不是「等一會兒就好」。

★ **附帶發現（規則的漏洞）**：現行 gate 只檢查 `swap_used ≤ 2048`，但 macOS 會**動態擴張 swap total**
（6144 → 9216），而真正傷害讀數的是**這一輪自己的 growth**（+2.9 GB）。⇒ 起跑 gate 應該改成
「`swap_used + 預期 growth ≤ swap_total` **且** `free` 有餘裕」，否則 gate 會在 swap 被擴張時放行一個
註定污染的 run（本輪就是這樣放行的）。

## 6. 要讓它能復現，起跑前必須同時成立的條件（缺一即不該產出數字）

| # | 條件 | 檢查方式 |
|---|---|---|
| 1 | 零殘留 llama 行程（含鄰居） | `pgrep -fl "llama-server\|llama-bench"` |
| 2 | `swap_used ≤ 2048 MiB` **且**能容納本輪 growth（見 §5★：以 total 與 free 一起判） | `sysctl -n vm.swapusage` |
| 3 | thermal **NOMINAL 貫穿全程**（不是只有起跑那一瞬） | 產物的 `thermal.worst`，不是 `thermal.launch` |
| 4 | rep 散度 ≤ 12% | 產物的 samples |
| 5 | 沒有別條線在跑（本輪期間 repo 有 21 個他人未提交檔） | `git status`＋行程表 |

**複現述句的正確寫法**（本輪示範）：
`prefill=X t/s / decode=Y t/s` ＋ attribution 行 ＋ `thermal worst` ＋ swap growth ＋ 產物路徑。
⚠ **只寫 `thermal launch=NOMINAL` 是不夠的** —— run1 的 launch 是 NOMINAL、worst 是 HEAVY。

## 7. 影響清單

- 本輪兩個讀數：**廢棄**（run1 污染、run2 UNRELIABLE）。
- 卡上錨點家族：**標「未復現（2026-09-25）」**，原因＝盒況前置不成立（§5），**不是**數字本身被否證。
- 任何拿 `11.59 / 9.35 / 228.39 / 292.01` 當基準的後續比較：**不成立**（母體不合格）。
- 這兩個產物**仍保留**，用途只有一個：**作為「盒況不合格」的示範樣本**與 gate 的測資 —— 不是速度基準。

---

## 8. 存檔指紋與「重開機後的第一步」

### 8.1 指紋（重開機後先驗這一張；任何一格不同 ⇒ 內容被動過，先查再說）

| 檔 | md5 | 大小 |
|---|---|---:|
| `Backup/mtpoff_base/run1_20260925_181950.json` | `b9a454c473c3dbe332968b5b63ec2492` | 39 872 B |
| `Backup/mtpoff_base/run2_20260925_183415.json` | `67aadc087c48ebd9438744c4fed2312c` | 41 970 B |
| `Backup/mtpoff_base/run1_20260925_181950.log` | `7adb747a9d733e024a426d4d7f66d4f4` | 2 499 B |
| `Backup/mtpoff_base/run2_20260925_183415.log` | `345416786bac2fa4046e3c0e23df4951` | 2 575 B |
| `docs/PROD_NEW_BASELINE_REPRO_2026-09-25.md` | 見 `Backup/mtpoff_base/MANIFEST.md5`（檔不列自身指紋：一改就失效） | — |
| `…/run1_workdir/llama_bench_prod-new_p2048_n128_d512_r3.json` | `31b8d19e94ba9d012f63dc3d6466be0d` | 2 693 B |
| `…/run2_workdir/llama_bench_prod-new_p2048_n128_d512_r3.json` | `0a5f08b8d064bf43d152d161e96b4781` | 2 695 B |

重開機前最後一筆狀態：`swap = 7374 / 8192 MiB`、thermal NOMINAL、llama 行程 0。（重開機後 swap 應該歸零。）

### 8.2 重開機後的第一步（照 §6 的五條前置，然後只跑一輪）

```sh
# 1. 驗指紋（§8.1）
md5 -q Backup/mtpoff_base/run1_20260925_181950.json   # 應為 b9a454c4…

# 2. 五條前置：任何一條不過就不要產出數字
pgrep -fl "llama-server|llama-bench"          # 應為空
sysctl -n vm.swapusage                          # used 應 ≤ 2048 MiB（契約 §3.3）；重開機後通常歸零
python3 scripts/check/thermal_pressure.py       # 應為 NOMINAL
git status --short | wc -l                      # 別條線是否在跑（非 0 要先確認）

# 3. 只跑一輪（detached，形狀＝測試卡 §2）
python3 scripts/check/commit_bench.py --record-only \
    --workdir /tmp/mtpoff_base3 --json /tmp/mtpoff_base3.json

# 4. 判定只看這兩格（不是 thermal.launch）
#    thermal.worst 必須 NOMINAL　＋　samples 散度 ≤ 12%　⇒ 才可引用；否則再進 §4 表
```

**引擎不需重建**：只要 §1 的四顆 binary 指紋相同，這一輪與 run1/run2 就是同一個母體。
**若重開機後 `launch swap` 仍 > 2048**：那是有別的行程在佔記憶體，**先處理它、不要開跑** ——
否則只會再得到一個 §5 那樣的污染樣本。

---

## 9. 重開機後的兩輪：選項才是主因

重開機後：`swap = 0.00 MiB`、thermal NOMINAL、零 llama 行程、MANIFEST 11/11 ✅、HEAD 不變、引擎指紋與 §1 相同。

### 9.1 run3（`harness bench`，**未加** `--fixed-fill-seed`）—— 整輪不可引用，但 `4.13` 是**冷池讀數**不是雜訊

| | 值 |
|---|---|
| prefill / decode | 292.37 ± 15.77（`307.92/292.78/276.39`）／ **9.22 ± 4.41**（`4.13/11.85/11.67`） |
| thermal | launch NOMINAL，**worst HEAVY**，`{NOMINAL 102, MODERATE 45, HEAVY 20}` |
| swap | launch **0.0** → end 962.5，growth **+962.5**，min_free 39.3 |
| 散度 | decode **83.7%** |
| 判定 | §5.1 污染（HEAVY＋growth 962.5）＋ §4.1 UNRELIABLE ⇒ **不列為 steady 錨點**；`4.13` 另立為 **cold 診斷值**（見 §9.5） |

★ 但它的 `platform_ts`（丟第一 rep 的均值）decode = **11.76**、prefill = 284.59 ⇒ **同帶**。
⇒ 「整輪不可引用」與「平台值可複現」是兩件事，**報告必須指名是 `avg_ts` 還是 `platform_ts`**（09-19 文件 §3.2 已要求）。
產物：`Backup/mtpoff_base/run3_20260925_184701_noseed.{json,log}`。

### 9.2 run4（matrix 入口，**加** `--fixed-fill-seed 1`）—— ✅ 乾淨、可引用

```
python3 scripts/check/llama_bench_matrix.py --arms prod-new \
  --prompt 2048 --gen 128 --depths 512 --reps 3 --ctx-size 0 --warm-skip 64 \
  --fixed-fill-seed 1 --workdir <dir> --json <dir>/result.json      # ← workdir 必須先 mkdir -p
```

| | 值 |
|---|---|
| **prefill** | **304.09 ± 9.03**（`313.27/303.77/295.22`，散度 5.9%）；`platform_ts` = **299.50** |
| **decode** | **11.64 ± 0.30**（`11.80/11.83/11.30`，散度 **4.6%**）；`platform_ts` = **11.56** |
| hit% | 96.2（misses 4929、reads 82572） |
| thermal | launch NOMINAL，**worst NOMINAL**，`{NOMINAL 146}` —— **全程 NOMINAL** |
| swap | launch 898.06 → end 982.75，growth **+84.7 MiB**，min_free 54.1 |
| 判定 | §5.1 **無污染**（無 HEAVY、growth ≤500）＋ §4.1 **clean**（worst NOMINAL、散度 4.6%）＋ §6 **同帶**（`\|11.90−11.64\| = 0.26`） |

⇒ **這是本日第一個可引用的 MTP-off 基準**：

```
prefill=304.09t/s decode=11.64t/s (mtp=off prod-new p2048 n128 d512 r3 --fixed-fill-seed 1)
thermal: launch=NOMINAL worst=NOMINAL hist={NOMINAL:146} ; swap: 898→983 MiB (growth +84.7)
```

產物：`Backup/mtpoff_base/run4_20260925_190210_seed1.json`（＋`run4_llama_bench_raw.json`、`.log`）。

### 9.3 選項的 A/B（同一盒況、同 cell、只差一個旗標）

| 旗標 | run3 | run4 |
|---|---|---|
| `--fixed-fill-seed` | 0（預設＝**每 rep 換一條隨機流**） | **1** |
| decode σ | 4.41（散度 83.7%） | **0.30（散度 4.6%）** |
| 同帶性 | 整輪不同帶（平台值同帶） | **整輪同帶** |

### 9.5 那個低樣本是**冷/穩兩種 regime**，不是儀器噪音（2026-09-25 事後修訂）

前一版把它寫成「離群樣本」⇒ **那是錯的標籤**。它是「**該 rep 的計時窗自己付掉了 compulsory fill**」的那種讀數，
也就是 repo 早已命名的 `cold` regime。欄名照 `docs/DECODE_STEADY_BASELINE_2026-09-19.md` §3.3 的既有約定：
`decode_tps_cold` / `decode_tps_steady` —— **不得用一個 `avg_ts` 蓋過兩者**（跨 regime 平均出來的數不可引用）。三條證據：

| # | 證據 | 讀法 |
|---|---|---|
| 1 | run3 與 run4 的**總 I/O 幾乎相同**：misses `5007 vs 4929`、`file_reads 82797 vs 82572`、hit `96.1 vs 96.2%`、pread 執行緒時間 `1942.9 vs 1890.6 s`、`resident 6197.04 MiB` 兩者相同 | ⇒ 差的**不是填充量，是填充落點**：一輪把 fill 付在窗內（cold）、一輪付在窗外（steady） |
| 2 | 低樣本**不一定在第一個 rep**：run2 = `11.77 / **3.95** / 12.34`（第 2 個）、run3 = `**4.13** / 11.85 / 11.67`（第 1 個） | ⇒ 它是「該 rep 的窗重新路由」的性質，不是「第一個 rep 一定冷」⇒ **regime，不是位置** |
| 3 | run4（seed=1）**第 1 個 rep 就是 11.80** | ⇒ `--fixed-fill-seed` 自己的說明寫了機制：「reseed at the top of EVERY rep … the expert cache can reach steady state」＝暖機那段預熱的正是被計時的那段路由 |

**機制**：無 seed ⇒ 每個 rep 的暖機路由 ≠ 該 rep 的計時路由 ⇒ 窗內付 compulsory fill；有 seed ⇒ 兩者相同 ⇒ 窗內 fills ≈ 0。
兩次冷值量級一致（`3.95` 在**頁快取已暖**的 run2、`4.13` 在**重開機後頁快取冷**的 run3）⇒ 主因是**專家池**；OS 頁快取是加重項（此歸屬為推測，未單獨量）。

**可引用性**：`4.13` 只能當**診斷值**（單樣本、bench 隨機流、且該輪 thermal worst=HEAVY 無法定位到哪個 rep）——
它回答的是「首次遇冷要付多少」的量級（−65% vs steady），**不是** KPI；交付路徑由 `run_server.sh` 的預熱請求吸收它（測試卡 §7）。
**未做**：一個刻意的 cold 臂（`--fixed-fill-seed 0`、n≥3、全程 NOMINAL）——那才是可引用的 `decode_tps_cold`。

⇒ 依 `docs/DECODE_STEADY_BASELINE_2026-09-19.md` §3.2「`--fixed-fill-seed` 設為 bench 預設（σ −68%）」，
**這一格應該是預設值，卻不是**。另兩件要一起記：

- `--no-warmup` **保持關**（＝暖機 ON）是對的：matrix 自己的說明寫「warmup is what leaves the expert pool hot」。
- `--prompt-file`（真實文本）**本輪未用** —— 它會換母體（`ROUTE_OVERLAP_CROSSPROMPT §1.1`：隨機填充的路由不是真實語言分佈），
  要用得跟 §1 的錨點家族分開標，不能混在同一列。

### 9.4 本輪復現出兩個工具缺陷

| # | 缺陷 | 證據 |
|---|---|---|
| 1 | `harness.py bench`（最新 commit 的統一入口）**沒 expose** `--fixed-fill-seed` / `--prompt-file` | `grep -n "fixed-fill-seed\|prompt-file" scripts/check/harness.py` **零命中** ⇒ 用它量出來的必然是 09-19 已診斷過的 σ 膨脹形狀（run3 就是） |
| 2 | `llama_bench_matrix.py` **不會自建 `--workdir`** | 本輪第一次執行：llama-bench **跑完**，卻在收尾寫 `.stderr.log` 時 `FileNotFoundError` ⇒ **量測做了、產物沒了**（`FILL_COST_MEASURED §7.6` 已記過同一條） |

---

## 10. run5（使用者指定的原命令：`--warm-skip 64`、無 seed）—— 同帶的診斷價，不是錨點

```sh
python3 scripts/check/llama_bench_matrix.py --arms prod-new \
  --prompt 2048 --gen 128 --depths 512 --reps 3 --ctx-size 0 --warm-skip 64 \
  --workdir <dir> --json <dir>/result.json      # workdir 要先 mkdir -p（§9.4 缺陷 2 未修）
```

| | 值 |
|---|---|
| **prefill** | **292.24 ± 8.48**（`300.43/292.79/283.49`，散度 5.8%）；platform 288.14 |
| **decode** | **11.22 ± 0.45**（`11.2248/10.7564/11.6626`，散度 **8.0%**）；platform 11.21 |
| `warm_skip` | 64，且引擎輸出 `n_gen=64` ⇒ **確實套用**（不是 §9.5 那類「欄位有值、實際沒生效」） |
| 逐樣本 regime | **無冷樣本**（10.76–11.66 連續）⇒ steady 讀數，不必拆欄 |
| thermal | launch NOMINAL、worst **MODERATE**、hist `{NOMINAL 126, MODERATE 31}` |
| cache | hit 95.8%、misses 5384（**compulsory 4358＝81%**、capacity 1026＝19%）、reads 83937、resident 6197.04 MiB、worst_layer 2（209 distinct > slots）、layers_over_slots 6 |
| swap | **產物沒有 swap block**（matrix 記錄缺這欄）⇒ 手動量：起跑 `271.94/1024 MiB` → 收尾 `4549.81/6144 MiB`，**系統級 growth ≈ +4278 MiB**（跑完 0 個 llama、無鄰居，歸屬本輪可接受，但這是自己的前後讀數、非產物欄位） |
| 判定 | §4.1 散度 8.0% ≤12% ✓；**§5.1 污染門踩到（swap growth 4278 > 500）** ⇒ **同帶的診斷價，不是錨點** |

**與同 cell 兩格並比**：run4（`--fixed-fill-seed 1`）**11.64 / 304.09**；run5（無 seed）**11.22 / 292.24**
⇒ decode −3.6%、prefill −3.9%，落在 09-19 §3.2 記載的啟動序漂移量級內。錨點家族（11.03 / 11.79 / 12.20 / 12.27）中 run5 落在**低端但同帶**（`|11.90−11.22| = 0.68 ≤ 0.8`）。

⇒ **今天 prod-new cell 兩次重現都在 11.2–11.6**，與卡面 `11.90 ± 0.15` 差 0.3–0.7；這個 cell 在本機的**可重現地板**與**卡面錨點**之間約有 5% 的系統性缺口，且兩次都伴隨 GB 級 swap 成長。

產物：`Backup/mtpoff_base/run5_20260925_193721_prodnew_ws64.json`（＋raw json、driver log、`run5_workdir/`）。
